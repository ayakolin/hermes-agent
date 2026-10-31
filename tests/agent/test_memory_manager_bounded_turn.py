"""``MemoryManager.begin_turn_bounded``: the memory half of a dispatcher turn's first-token path.

Contracts, with real managers and real threads (nothing is slept on; blocked providers wait on events the
test releases):

* the caller is back inside the budget even when a provider's tick or recall blocks, and a blocked tick
  is never overlapped by the next turn's (one in flight per provider, the new tick is skipped);
* a recall the caller stopped waiting for is dropped, never handed to a later turn as its own;
* the stuck-call gate is driven with the provider's real timeout, so rapid messages against a slow but
  healthy provider do not close it;
* a provider that does not override ``on_turn_start`` costs no thread, and a legacy two-argument tick
  still gets its call.
"""
from __future__ import annotations

import threading
import time
import uuid

from agent.memory_manager import MemoryManager
from agent.memory_manager_bounded import overrides_hook
from agent.memory_provider import MemoryProvider
from tests.waiting import wait_until

# A hard bound for "returned inside the budget"; the provider would otherwise block for 10s.
RETURNED_QUICKLY = 5.0


class _Provider(MemoryProvider):
    """External provider. Neither ``on_turn_start`` nor ``prefetch`` is overridden unless a subclass says so."""

    def __init__(self):
        self._name = f"ext-{uuid.uuid4().hex[:8]}"  # the stuck-call gate is process-wide, keyed by name
        self.release = threading.Event()

    @property
    def name(self):
        return self._name

    def is_available(self):
        return True

    def initialize(self, session_id, **kwargs):
        pass

    def get_tool_schemas(self):
        return []


class _BlockingTick(_Provider):
    def __init__(self):
        super().__init__()
        self.ticks, self.queries = [], []

    def on_turn_start(self, turn_number, message, **kwargs):
        self.ticks.append(turn_number)
        self.release.wait(timeout=10)

    def prefetch(self, query, *, session_id=""):
        self.queries.append(query)
        return f"recall:{query}"


class _BlockingRecall(_Provider):
    """The first ``blocked_calls`` recalls block until released; later ones answer at once."""

    blocked_calls = 1

    def __init__(self):
        super().__init__()
        self.queries = []

    def prefetch(self, query, *, session_id=""):
        self.queries.append(query)
        if len(self.queries) <= self.blocked_calls:
            self.release.wait(timeout=10)
        return f"recall:{query}"


class _SlowButHealthy(_BlockingRecall):
    blocked_calls = 5


def _manager(provider):
    manager = MemoryManager()
    manager.add_provider(provider)
    return manager


def _begin(manager, turn, query, *, budget_s=0.2, recall=True, **kw):
    started = time.monotonic()
    out = manager.begin_turn_bounded(turn, query, budget_s=budget_s, recall=recall, session_id="s1", **kw)
    return out, time.monotonic() - started


def _gate_idle(manager, provider):
    return manager._registry_key(provider) not in MemoryManager._PREFETCH_GATE.calls


def test_blocked_tick_returns_inside_the_budget_and_is_never_overlapped():
    provider = _BlockingTick()
    manager = _manager(provider)
    try:
        out, elapsed = _begin(manager, 1, "what is on my calendar today")
        assert out == "" and elapsed < RETURNED_QUICKLY  # the tick is still blocked; recall is skipped
        assert provider.queries == []

        out, elapsed = _begin(manager, 2, "and tomorrow as well please")
        assert out == "" and elapsed < RETURNED_QUICKLY
        assert provider.ticks == [1]  # the previous tick is alive: this turn's tick was skipped, not overlapped
    finally:
        provider.release.set()
    wait_until(lambda: not manager._tick_threads[provider.name].is_alive(), what="the first tick to finish")

    out, _ = _begin(manager, 3, "now the weekend plans please", budget_s=2.0)
    assert provider.ticks == [1, 3]  # a settled tick lets the next turn tick and recall again
    assert out == "recall:now the weekend plans please"


def test_recall_false_still_ticks_and_does_not_query():
    provider = _BlockingTick()
    provider.release.set()
    manager = _manager(provider)
    out, _ = _begin(manager, 1, "[done: abc] the report is ready", budget_s=2.0, recall=False)
    assert out == "" and provider.ticks == [1] and provider.queries == []


def test_a_recall_the_caller_stopped_waiting_for_is_dropped_not_carried_over():
    provider = _BlockingRecall()
    manager = _manager(provider)
    try:
        out, elapsed = _begin(manager, 1, "first question about the launch")
        assert out == "" and elapsed < RETURNED_QUICKLY
    finally:
        provider.release.set()
    wait_until(lambda: _gate_idle(manager, provider), what="the abandoned recall to finish")

    out, _ = _begin(manager, 2, "second question about travel", budget_s=2.0)
    assert out == "recall:second question about travel"  # not the first question's late answer
    assert provider.queries == ["first question about the launch", "second question about travel"]


def test_rapid_messages_do_not_close_the_stuck_call_gate_for_a_slow_healthy_provider():
    """Five turns in a row each abandon an in-flight recall after a 0.1s budget. Judged by that budget
    the first four would all be 'stuck' by the fifth turn and the gate would skip it; judged by the
    provider's real timeout (8s) none is, so every turn still reaches the provider."""
    provider = _SlowButHealthy()
    manager = _manager(provider)
    try:
        for turn in range(5):
            out, _ = _begin(manager, turn, f"question number {turn} about the project", budget_s=0.1)
            assert out == ""
        assert len(provider.queries) == 5
    finally:
        provider.release.set()
    wait_until(lambda: _gate_idle(manager, provider), what="the abandoned recalls to finish")


def test_no_tick_override_means_no_thread_and_a_legacy_tick_still_runs():
    plain = _BlockingRecall()
    plain.release.set()
    manager = _manager(plain)
    out, _ = _begin(manager, 1, "a plain provider question", budget_s=2.0)
    assert out == "recall:a plain provider question" and manager._tick_threads == {}

    seen = []

    class _Legacy(_BlockingRecall):
        def on_turn_start(self, turn_number, message):  # written before the author kwargs existed
            seen.append((turn_number, message))

    legacy = _Legacy()
    legacy.release.set()
    manager = _manager(legacy)
    _begin(manager, 4, "a legacy provider question", budget_s=2.0, author_id="a1", author_name="Ann",
           author_is_bot=False)
    assert seen == [(4, "a legacy provider question")]


def test_overrides_hook_tells_an_override_from_the_inherited_default():
    class _Override(_Provider):
        def on_pre_compress(self, messages):
            return "insight"

    class _Duck:
        def on_pre_compress(self, messages):
            return ""

    assert overrides_hook(_Override(), "on_pre_compress")
    assert overrides_hook(_Duck(), "on_pre_compress")
    assert not overrides_hook(_Provider(), "on_pre_compress")
    assert not overrides_hook(object(), "on_pre_compress")
    assert _manager(_Override()).overrides_pre_compress() and not _manager(_Provider()).overrides_pre_compress()
