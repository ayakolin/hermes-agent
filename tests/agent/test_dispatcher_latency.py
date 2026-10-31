"""``dispatcher.latency.*`` reaches a dispatcher agent once, at build time (agent/dispatcher_latency.py).

Real ``AIAgent`` construction against a temp ``HERMES_HOME`` with a real ``config.yaml``; the only seam
faked is the creation-time dispatcher decision (``agent_is_dispatcher``), which belongs to the
session-mode package. Contracts:

* a dispatcher agent freezes the configured numbers; any other agent has ``None`` and every helper then
  answers "legacy behaviour";
* a deferring dispatcher gets a ``PreparedCompaction`` even with ``compression.prepare_ahead`` off, unless a
  memory provider supplies compaction context (every prepared summary would be discarded), and an explicit
  ``prepare_ahead: true`` is the user's choice and is honoured regardless;
* ``0`` / ``false`` in config restore the legacy behaviour per key.
"""
from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.dispatcher_latency import (
    DispatcherLatency, dispatcher_latency, dispatcher_streams_deltas, freeze_dispatcher_latency,
    memory_budget_s, pre_llm_call_budget_s, skip_recall_for_turn, wants_prepared_compaction,
)
from agent.memory_manager import MemoryManager
from agent.memory_provider import MemoryProvider
from agent.turn_context import _collect_pre_llm_call_context
from hermes_cli.plugins import PluginManager
from run_agent import AIAgent


class _SuppliesContext(MemoryProvider):
    @property
    def name(self):
        return "supplies-context"

    def is_available(self):
        return True

    def initialize(self, session_id, **kwargs):
        pass

    def get_tool_schemas(self):
        return []

    def on_pre_compress(self, messages):
        return "insight"


def _write_config(home, text):
    (home / "config.yaml").write_text(text, encoding="utf-8")


def _build(monkeypatch, tmp_path, *, dispatcher, config="", memory_manager=None):
    _write_config(tmp_path, config)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("agent.dispatcher_session.agent_is_dispatcher", lambda _agent: dispatcher)
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
        patch("agent.model_metadata.get_model_context_length", return_value=200_000),
        patch("agent.context_compressor.get_model_context_length", return_value=200_000),
    ):
        return AIAgent(
            api_key="test-key-1234567890", base_url="https://openrouter.ai/api/v1", model="test/model",
            quiet_mode=True, skip_context_files=True, skip_memory=memory_manager is None,
            memory_manager=memory_manager, max_iterations=2,
        )


def test_a_dispatcher_agent_freezes_the_configured_numbers(monkeypatch, tmp_path):
    agent = _build(monkeypatch, tmp_path, dispatcher=True, config=(
        "dispatcher:\n  latency:\n    memory_budget_s: 0.5\n    pre_llm_call_budget_s: 1.5\n"
        "    defer_ceiling_ratio: 7\n    stream_deltas: false\n"
    ))
    assert dispatcher_latency(agent) == DispatcherLatency(
        memory_budget_s=0.5, pre_llm_call_budget_s=1.5, defer_compaction=True,
        defer_ceiling_ratio=1.0,  # a ratio above the window would defer past what the provider accepts
        stream_deltas=False,
    )
    assert (memory_budget_s(agent), pre_llm_call_budget_s(agent)) == (0.5, 1.5)
    assert dispatcher_streams_deltas(agent) is False


def test_defaults_apply_when_the_section_is_absent(monkeypatch, tmp_path):
    latency = dispatcher_latency(_build(monkeypatch, tmp_path, dispatcher=True))
    assert latency == DispatcherLatency(2.0, 3.0, True, 0.85, True)


def test_a_non_dispatcher_agent_has_no_latency_and_every_helper_answers_legacy(monkeypatch, tmp_path):
    agent = _build(monkeypatch, tmp_path, dispatcher=False, config="dispatcher:\n  latency:\n    memory_budget_s: 0.5\n")
    assert agent._dispatcher_latency is None and dispatcher_latency(agent) is None
    assert (memory_budget_s(agent), pre_llm_call_budget_s(agent)) == (0.0, 0.0)
    assert not dispatcher_streams_deltas(agent) and not skip_recall_for_turn(agent)
    assert agent.context_compressor.prepared_compaction is None


def test_test_doubles_never_look_like_dispatchers():
    assert dispatcher_latency(MagicMock()) is None and memory_budget_s(MagicMock()) == 0.0
    agent = SimpleNamespace(_dispatcher_session=True)
    freeze_dispatcher_latency(agent, {})
    assert dispatcher_latency(agent) is not None
    other = SimpleNamespace(_dispatcher_session="yes")  # the flag must be the real True
    freeze_dispatcher_latency(other, {})
    assert other._dispatcher_latency is None


def test_a_deferring_dispatcher_gets_prepared_compaction_without_prepare_ahead(monkeypatch, tmp_path):
    agent = _build(monkeypatch, tmp_path, dispatcher=True)
    assert agent.context_compressor.prepared_compaction is not None


def test_defer_compaction_false_restores_the_legacy_compressor(monkeypatch, tmp_path):
    agent = _build(monkeypatch, tmp_path, dispatcher=True, config="dispatcher:\n  latency:\n    defer_compaction: false\n")
    assert agent.context_compressor.prepared_compaction is None


def test_a_provider_that_supplies_compaction_context_gets_no_prepared_pass(monkeypatch, tmp_path):
    manager = MemoryManager()
    manager.add_provider(_SuppliesContext())
    agent = _build(monkeypatch, tmp_path, dispatcher=True, memory_manager=manager)
    assert agent._memory_manager is manager and dispatcher_latency(agent) is not None
    assert agent.context_compressor.prepared_compaction is None  # no pass would be wasted starting


def test_explicit_prepare_ahead_is_honoured_even_with_such_a_provider(monkeypatch, tmp_path):
    manager = MemoryManager()
    manager.add_provider(_SuppliesContext())
    agent = _build(
        monkeypatch, tmp_path, dispatcher=True, memory_manager=manager,
        config="compression:\n  prepare_ahead: true\n",
    )
    assert agent.context_compressor.prepared_compaction is not None


@pytest.mark.parametrize(("latency", "prepare_ahead", "expected"), [
    (None, False, False), (None, True, True), (DispatcherLatency(2.0, 3.0, True, 0.85, True), False, True),
    (DispatcherLatency(2.0, 3.0, False, 0.85, True), False, False),
])
def test_wants_prepared_compaction_table(latency, prepare_ahead, expected):
    assert wants_prepared_compaction(SimpleNamespace(_dispatcher_latency=latency), prepare_ahead) is expected


def test_a_failing_config_load_still_leaves_a_dispatcher_agent_with_defaults(monkeypatch):
    """The trims themselves never fail an init: an unreadable config freezes the defaults, not None."""
    monkeypatch.setattr("agent.dispatcher_session.agent_is_dispatcher", lambda _agent: True)
    monkeypatch.setattr("hermes_cli.config.load_config_readonly",
                        lambda: (_ for _ in ()).throw(OSError("unreadable config")))
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
        patch("agent.model_metadata.get_model_context_length", return_value=200_000),
        patch("agent.context_compressor.get_model_context_length", return_value=200_000),
    ):
        agent = AIAgent(api_key="test-key-1234567890", base_url="https://openrouter.ai/api/v1",
                        model="test/model", quiet_mode=True, skip_context_files=True, skip_memory=True,
                        max_iterations=2)
    assert agent._dispatcher_latency == DispatcherLatency(2.0, 3.0, True, 0.85, True)


def test_the_turn_s_pre_llm_call_hook_reads_the_frozen_budget(monkeypatch, tmp_path):
    """Wiring: the frozen budget reaches the turn's hook dispatch, and a callback slower than it is
    abandoned rather than allowed to hold the first token (no suppression: see tests/hermes_cli/test_hook_budget.py)."""
    agent = _build(monkeypatch, tmp_path, dispatcher=True,
                   config="dispatcher:\n  latency:\n    pre_llm_call_budget_s: 0.3\n")
    assert pre_llm_call_budget_s(agent) == 0.3
    hold, calls = threading.Event(), []
    # Safety net: if the budget were ignored the collector blocks until this releases; keep a failing run
    # short (the release lands long after the budget either abandons the callback or the assertion fires).
    threading.Timer(2.0, hold.set).start()

    def slow_plugin(**_kwargs):
        calls.append("slow")
        hold.wait(timeout=10)
        return {"context": "late"}

    manager = PluginManager()
    monkeypatch.setattr("hermes_cli.plugins._delivery_manager", lambda: manager)
    # This runner's spawned workers may carry the safe-worker policy, under which lifecycle skips hooks.
    monkeypatch.setattr("agent.safe_worker_policy._policy", None)
    monkeypatch.setattr("hermes_cli.plugins._resolve_hook_callback_timeout", lambda: 30.0)

    def collect():
        return _collect_pre_llm_call_context(
            agent, effective_task_id="t1", turn_id="turn-1", original_user_message="hi",
            messages=[{"role": "user", "content": "hi"}], conversation_history=None,
        )

    collect()  # warm the observer's imports: a process-level one-time cost, not per-turn dispatch
    manager._hooks["pre_llm_call"] = [slow_plugin]
    try:
        # No hook_budget here: the collector must open the frozen budget itself, or this call blocks.
        assert collect() == "" and calls == ["slow"]
    finally:
        hold.set()
