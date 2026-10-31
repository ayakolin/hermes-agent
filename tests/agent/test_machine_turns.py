"""Machine-text turns skip recall (agent/machine_turns.py, agent/dispatcher_latency.py).

A completion or process notice is text Hermes wrote. On a dispatcher session recall keyed on it is noise and
costs memory budget, so neither the turn's own prefetch nor the next turn's queued prefetch runs for it; the
provider's per-turn tick still does. The kinds are judged by what the real gateway produces for a wake
event, not by one literal, and the agent-side set is pinned to the gateway's.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent.dispatcher_latency import DispatcherLatency
from agent.machine_turns import MACHINE_TEXT_DISPLAY_KINDS, is_machine_text_display_kind
from agent.turn_context import _memory_turn_start_and_prefetch
from gateway.platforms.event import MessageEvent
from gateway.response_filters import MACHINERY_DISPLAY_KINDS, display_kind_for_event
from run_agent import AIAgent

CARD = {"display_kind": "async_delegation_complete", "display_text": "Executor finished: report ready"}


def _latency(budget: float = 2.0) -> DispatcherLatency:
    return DispatcherLatency(
        memory_budget_s=budget, pre_llm_call_budget_s=3.0, defer_compaction=True,
        defer_ceiling_ratio=0.85, stream_deltas=True,
    )


class _Memory:
    """Records what the turn start asked of the memory manager."""

    def __init__(self):
        self.ticks, self.prefetches, self.bounded, self.queued, self.synced = [], [], [], [], []

    def on_turn_start(self, turn, query, **_kw):
        self.ticks.append(query)

    def prefetch_all(self, query, session_id=""):
        self.prefetches.append(query)
        return "RECALL"

    def begin_turn_bounded(self, turn, query, *, budget_s, recall, session_id="", **_kw):
        self.bounded.append((query, recall))
        return "RECALL" if recall else ""

    def describe_recall(self):
        return ""

    def sync_all(self, user, assistant, **_kw):
        self.synced.append(user)

    def queue_prefetch_all(self, query, session_id=""):
        self.queued.append(query)


def _agent(display_kind, latency):
    return SimpleNamespace(
        _memory_manager=_Memory(), _user_turn_count=3, session_id="s1", _dispatcher_latency=latency,
        _turn_display_kind=display_kind, _emit_status=lambda _msg: None,
    )


# What the real gateway persists for each kind of turn it can start.
WAKE_EVENTS = {
    "wake notice": MessageEvent(text="[done: abc]", internal=True, metadata={"notification_category": "x"}),
    "wake with producer card": MessageEvent(text="[done: abc]", internal=True, metadata=dict(CARD)),
}


@pytest.mark.parametrize("name", list(WAKE_EVENTS))
def test_real_wake_events_are_machine_text(name):
    kind = display_kind_for_event(WAKE_EVENTS[name])
    assert kind in {"internal_notification", "async_delegation_complete"}
    assert is_machine_text_display_kind(kind)


def test_a_human_message_is_not_machine_text():
    assert display_kind_for_event(MessageEvent(text="what is on my calendar")) is None
    assert not is_machine_text_display_kind(None)


def test_agent_side_set_covers_the_gateway_machinery_kinds():
    assert MACHINERY_DISPLAY_KINDS <= MACHINE_TEXT_DISPLAY_KINDS
    assert "hidden" in MACHINE_TEXT_DISPLAY_KINDS  # a completion persisted with its presentation suppressed


@pytest.mark.parametrize("name", list(WAKE_EVENTS))
def test_wake_turn_ticks_but_skips_recall_on_a_dispatcher(name):
    agent = _agent(display_kind_for_event(WAKE_EVENTS[name]), _latency())
    assert _memory_turn_start_and_prefetch(agent, "[done: abc] the report is ready") == ""
    assert agent._memory_manager.bounded == [("[done: abc] the report is ready", False)]


def test_human_turn_recalls_on_a_dispatcher():
    agent = _agent(None, _latency())
    assert _memory_turn_start_and_prefetch(agent, "what did we decide about the launch") == "RECALL"
    assert agent._memory_manager.bounded == [("what did we decide about the launch", True)]


def test_legacy_path_is_unchanged_without_a_latency():
    """A non-dispatcher session (no latency) still recalls on a wake turn, through the old calls."""
    agent = _agent(display_kind_for_event(WAKE_EVENTS["wake notice"]), None)
    assert _memory_turn_start_and_prefetch(agent, "[done: abc] the report is ready") == "RECALL"
    memory = agent._memory_manager
    assert memory.ticks == ["[done: abc] the report is ready"] and memory.prefetches == memory.ticks
    assert memory.bounded == []


def test_memory_budget_zero_restores_the_legacy_path():
    agent = _agent(display_kind_for_event(WAKE_EVENTS["wake notice"]), _latency(budget=0.0))
    assert _memory_turn_start_and_prefetch(agent, "[done: abc] the report is ready") == "RECALL"
    assert agent._memory_manager.bounded == []


def _sync(agent, user_text):
    AIAgent._sync_external_memory_for_turn(
        agent, original_user_message=user_text, final_response="ok", interrupted=False, messages=[],
    )


def test_next_turns_queued_recall_is_not_keyed_on_a_wake_notice():
    """The second prefetch site: ``sync_all`` still mirrors the turn, ``queue_prefetch_all`` does not run."""
    agent = _agent(display_kind_for_event(WAKE_EVENTS["wake notice"]), _latency())
    _sync(agent, "[done: abc] the report is ready")
    assert agent._memory_manager.synced == ["[done: abc] the report is ready"]
    assert agent._memory_manager.queued == []


def test_next_turns_queued_recall_still_runs_for_a_human_turn_and_without_a_latency():
    human = _agent(None, _latency())
    _sync(human, "what did we decide about the launch")
    assert human._memory_manager.queued == ["what did we decide about the launch"]
    legacy = _agent(display_kind_for_event(WAKE_EVENTS["wake notice"]), None)
    _sync(legacy, "[done: abc] the report is ready")
    assert legacy._memory_manager.queued == ["[done: abc] the report is ready"]
