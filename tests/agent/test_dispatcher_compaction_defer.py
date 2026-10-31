"""Compaction deferral on a dispatcher session (agent/dispatcher_latency.py, the three automatic decision sites).

Only the summary LLM wait is removed. While a background prepared pass is still writing the summary and the
request is under ``max(trigger, ratio * window)``, the automatic sites skip compaction for now; the real
compaction that follows (once the summary is ready) splices it and writes no second summary. Everything
else keeps the synchronous behaviour: a ready summary, no pass at all, a request past the ceiling, no
dispatcher latency, ``defer_compaction`` off, a memory provider that supplies summary context, and a
provider-proven overflow.

The sites are driven through their real entry points with a real ``ContextCompressor`` subclass whose
background pass is held by an event (the summariser is stubbed at its model boundary, like
tests/agent/test_prepared_compaction.py). No test sleeps.
"""
from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent import prepared_compaction as pc
from agent.context_compressor import COMPRESSED_SUMMARY_METADATA_KEY, ContextCompressor
from agent.dispatcher_latency import DispatcherLatency, should_defer_compaction
from agent.memory_manager import MemoryManager
from agent.memory_provider import MemoryProvider
from agent.prepared_compaction import PreparedCompaction
from agent.turn_context_compaction import run_turn_start_compaction
from agent.turn_preflight import PreflightGateVerdict, compress_after_tool_results, run_preflight_compression
from tests.waiting import wait_until

TRIGGER, WINDOW = 20_000, 40_960
CEILING = int(0.85 * WINDOW)  # 34_816: above the trigger, so there is a deferral band
IN_BAND, PAST_CEILING = 25_000, 36_000
LATENCY = DispatcherLatency(
    memory_budget_s=2.0, pre_llm_call_budget_s=3.0, defer_compaction=True, defer_ceiling_ratio=0.85,
    stream_deltas=True,
)


class _Reached(Exception):
    """Raised by the recording double once a decision site calls ``_compress_context``."""


class _StubCompressor(ContextCompressor):
    """Summariser stubbed at its model boundary; background passes wait for ``gate``."""

    def _generate_summary(self, turns_to_summarize, focus_topic=None, memory_context="", bypass_cooldown=False):
        worker = threading.current_thread().name == "compaction-prepare"
        if worker:
            from agent.auxiliary_client import AuxiliaryExplicitCancellation, _aux_interrupt_cancel_requested

            wait_until(lambda: self.gate.is_set() or _aux_interrupt_cancel_requested(), what="pass released")
            if not self.gate.is_set():
                raise AuxiliaryExplicitCancellation()
        self.calls.append({"worker": worker})
        body = f"Summary v{len(self.calls)} of {len(turns_to_summarize)} turns."
        self._previous_summary = body
        return self._with_summary_prefix(body)


class _SuppliesContext(MemoryProvider):
    """External provider that overrides ``on_pre_compress``: its insights feed every summary."""

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


def _transcript() -> list:
    msgs = [{"role": "system", "content": "sys"}]
    msgs += [{"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i} " + "z" * 1200} for i in range(30)]
    return msgs + [{"role": "user", "content": "current ask"}]


class _Scenario:
    def __init__(self, monkeypatch, latency, memory_manager):
        with patch("agent.context_compressor.get_model_context_length", return_value=WINDOW):
            cc = _StubCompressor(model="test/model", protect_first_n=3, protect_last_n=5, quiet_mode=True)
        cc.context_length, cc.threshold_tokens, cc.tail_token_budget = WINDOW, TRIGGER, 2500
        cc.calls, cc.gate, cc._session_id = [], threading.Event(), "S1"
        cc.prepared_compaction = PreparedCompaction()
        cc.last_prompt_tokens = 21_000  # real usage inside the prepare band, above the trigger
        self.cc, self.msgs, self.thread = cc, _transcript(), None
        self.labels, self.warned, self.tokens, self.stop = [], [], IN_BAND, True
        self.agent = SimpleNamespace(
            context_compressor=cc, compression_enabled=True, api_mode="chat_completions", model="m",
            session_id="s1", max_compression_attempts=3, _compression_feasibility_checked=True,
            _request_pressure_anchored=True, _dispatcher_latency=latency, _memory_manager=memory_manager,
            _emit_status=lambda _msg: None, _safe_print=lambda *_a: None,
            _warn_context_overflow_blocked=lambda *args: self.warned.append(args),
            _compress_context=self._compress_context,
        )
        # The request-size estimate: big before a compaction, small after it.
        monkeypatch.setattr(
            "agent.turn_context._preflight_request_tokens",
            lambda _agent, messages, _prompt: self.tokens if len(messages) > 15 else 5_000,
        )

    def _compress_context(self, messages, system_message, *, approx_tokens=None, trigger=None, **_kw):
        self.labels.append(trigger)
        if self.stop:
            raise _Reached(trigger)
        return self.cc.compress(messages, current_tokens=approx_tokens), "sys"

    def start_pass(self):
        self.thread = pc.maybe_prepare(self.agent, self.msgs, origin="tool-batch")
        assert self.thread is not None and self.cc.prepared_compaction.state() == "running"

    def finish_pass(self):
        self.cc.gate.set()
        self.thread.join(5.0)
        assert self.cc.prepared_compaction.state() == "ready"

    def turn_start(self):
        return run_turn_start_compaction(
            self.agent, messages=self.msgs, system_message="sys", active_system_prompt="sys",
            conversation_history=None, current_turn_user_idx=len(self.msgs) - 1, user_message="current ask",
            effective_task_id="task",
        )

    def pre_api(self, *, overflow=False):
        verdict = PreflightGateVerdict(
            action="fallthrough", pending_moa_prepared_request=None, messages=self.msgs,
            active_system_prompt="sys", conversation_history=None, api_call_count=1, compression_attempts=0,
            final_response=None, failed=False, _turn_exit_reason=None, _compression_timeout_exhausted=False,
            _preflight_compression_blocked=False, _provider_overflow_recovery_pending=False,
            _last_preflight_pressure=0,
        )
        return run_preflight_compression(
            self.agent, verdict, compressor=self.cc, request_pressure_tokens=self.tokens,
            provider_overflow_preflight=overflow, defer_preflight=lambda _tokens: False,
            moa_prepared_request=None, system_message="sys", user_message="current ask",
            max_compression_attempts=3, effective_task_id="task",
        )

    def post_tool(self):
        return compress_after_tool_results(
            self.agent, messages=self.msgs, system_message="sys", user_message="current ask",
            active_system_prompt="sys", conversation_history=None, compression_attempts=0,
            max_compression_attempts=3, effective_task_id="task", final_response=None,
            turn_exit_reason=None, current_turn_user_idx=len(self.msgs) - 1,
        )


@pytest.fixture
def scenario(monkeypatch):
    made = []

    def build(latency=LATENCY, memory_manager=None):
        made.append(_Scenario(monkeypatch, latency, memory_manager))
        return made[-1]

    yield build
    for scn in made:  # never leave a held background pass behind
        scn.cc.gate.set()
        if scn.thread is not None:
            scn.thread.join(5.0)


# ------------------------------------------------------------------ turn start (preflight)


def test_turn_start_waits_for_the_running_summary_then_splices_it_once(scenario):
    scn = scenario()
    scn.start_pass()

    out = scn.turn_start()  # the summary is still being written: nothing is compacted this turn
    assert scn.labels == [] and out.messages is scn.msgs and not out.compressed and not out.blocked
    assert scn.cc.calls == []

    scn.finish_pass()
    scn.stop = False
    out = scn.turn_start()  # ready: the ordinary compaction runs and splices the prepared summary
    assert scn.labels == ["turn_start_threshold"] and out.compressed
    assert [c["worker"] for c in scn.cc.calls] == [True]  # one summary, written on the background thread
    assert sum(1 for m in out.messages if m.get(COMPRESSED_SUMMARY_METADATA_KEY)) == 1
    assert scn.warned == []


def test_turn_start_defers_after_a_grown_window_too(scenario, monkeypatch):
    """The hook follows the LAST ``should_compress``: the grown-window re-check as well as the first."""
    scn = scenario()
    scn.start_pass()
    monkeypatch.setattr("agent.conversation_loop._maybe_grow_local_window", lambda *_a: 65_536)
    monkeypatch.setattr("agent.turn_context_compaction._apply_grown_window", lambda *_a: None)
    scn.turn_start()
    assert scn.labels == [] and scn.warned == []


@pytest.mark.parametrize("case", [
    "past-the-ceiling", "no-latency", "defer-off", "provider-context", "ready", "no-pass",
])
def test_turn_start_compacts_now_when_deferring_is_not_the_right_call(scenario, case):
    latency = {"no-latency": None, "defer-off": DispatcherLatency(2.0, 3.0, False, 0.85, True)}.get(case, LATENCY)
    manager = None
    if case == "provider-context":
        manager = MemoryManager()
        manager.add_provider(_SuppliesContext())
    scn = scenario(latency=latency, memory_manager=manager)
    if case == "past-the-ceiling":
        scn.tokens = PAST_CEILING
    if case != "no-pass":
        scn.start_pass()
    if case == "ready":
        scn.finish_pass()
    with pytest.raises(_Reached):
        scn.turn_start()
    assert scn.labels == ["turn_start_threshold"]


# ------------------------------------------------------------------ pre-API


def test_pre_api_defers_unless_the_provider_proved_an_overflow(scenario):
    scn = scenario()
    scn.start_pass()
    assert scn.pre_api().action == "fallthrough" and scn.labels == []  # the request goes out as it is

    with pytest.raises(_Reached):  # the provider proved it does not fit: compact now, pass or no pass
        scn.pre_api(overflow=True)
    assert scn.labels == ["pre_api"]


def test_pre_api_compacts_with_no_pass_running(scenario):
    scn = scenario()
    with pytest.raises(_Reached):
        scn.pre_api()
    assert scn.labels == ["pre_api"]


# ------------------------------------------------------------------ post-tool


def test_post_tool_defers_without_a_blocked_warning_or_a_spent_attempt(scenario):
    """A deferred turn is neither a compaction nor a blocked one: no attempt is spent, and no
    'attempts exhausted' overflow warning is raised for the over-trigger request."""
    scn = scenario()
    scn.start_pass()
    verdict = scn.post_tool()
    assert scn.labels == [] and verdict.compression_attempts == 0 and not verdict.end_turn
    assert scn.warned == []


def test_post_tool_compacts_with_no_pass_running(scenario):
    scn = scenario()
    with pytest.raises(_Reached):
        scn.post_tool()
    assert scn.labels == ["post_tool"]


# ------------------------------------------------------------------ the predicate


def test_the_ceiling_is_the_larger_of_the_trigger_and_the_window_ratio(scenario):
    scn = scenario()
    scn.start_pass()
    assert should_defer_compaction(scn.agent, CEILING - 1) and not should_defer_compaction(scn.agent, CEILING)
    scn.cc.threshold_tokens = 38_000  # a trigger above the ratio: the trigger is the ceiling
    assert should_defer_compaction(scn.agent, 37_999) is True
