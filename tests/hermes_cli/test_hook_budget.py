"""``hook_budget``: one wall-clock budget over a hook's whole dispatch (hermes_cli/plugins_dispatch.py).

A dispatcher turn must not sit behind its ``pre_llm_call`` plugins. Contracts, with a real ``PluginManager``
(blocked callbacks wait on events the test releases; nothing is slept on):

* callbacks run in order inside the budget; once it is spent the rest are skipped, not started;
* a callback cut off by the BUDGET is only slower than this turn could afford: it is abandoned but not
  put into the 60s suppression, so the next call runs it again; one cut off by the configured timeout is
  still suppressed, as before;
* the budget bounds the hook even when ``plugins.hook_callback_timeout`` is 0 (unbounded);
* the first-party observer that runs before the plugins spends from the same budget;
* only the named hook is budgeted, and ``seconds <= 0`` budgets nothing.
"""
from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from hermes_cli.plugins import PluginManager
from hermes_cli.plugins_dispatch import hook_budget
from tests.waiting import wait_until

HOOK = "pre_llm_call"
BUDGET = 0.3


@pytest.fixture
def manager(monkeypatch):
    monkeypatch.setattr("hermes_cli.plugins._resolve_hook_callback_timeout", lambda: 30.0)
    return PluginManager()


def _slow(hold, calls):
    def slow(**_kwargs):
        calls.append("slow")
        hold.wait(timeout=10)
        return {"context": "slow"}

    return slow


def _fast(calls):
    def fast(**_kwargs):
        calls.append("fast")
        return {"context": "fast"}

    return fast


def _invoke(manager, turn):
    return manager.invoke_hook(HOOK, turn_id=f"turn-{turn}")


def _idle(manager):
    return not manager._hook_running_callbacks


def test_budget_spent_skips_the_remaining_callbacks():
    hold, calls = threading.Event(), []
    manager = PluginManager()
    manager._hooks[HOOK] = [_slow(hold, calls), _fast(calls)]
    try:
        with hook_budget(HOOK, BUDGET):
            assert _invoke(manager, 1) == []
        assert calls == ["slow"]  # the second callback found the budget spent and never started
    finally:
        hold.set()


def test_fast_callbacks_inside_the_budget_all_run(manager):
    calls = []
    manager._hooks[HOOK] = [_fast(calls), _fast(calls)]
    with hook_budget(HOOK, 5.0):
        assert _invoke(manager, 1) == [{"context": "fast"}, {"context": "fast"}]
    assert calls == ["fast", "fast"]


def test_a_budget_cut_is_not_a_60s_suppression(manager):
    """Slower than the budget, faster than ``hook_callback_timeout``: abandoned this turn, run again next turn."""
    hold, calls = threading.Event(), []
    slow = _slow(hold, calls)
    manager._hooks[HOOK] = [slow]
    with hook_budget(HOOK, BUDGET):
        assert _invoke(manager, 1) == []
    hold.set()  # the callback finishes after the budget cut it off
    wait_until(lambda: _idle(manager), what="the abandoned callback to finish")
    assert manager._hook_timeout_suppressed_until == {}

    hold.set()
    assert _invoke(manager, 2) == [{"context": "slow"}]  # no budget, next turn: it runs and answers
    assert calls == ["slow", "slow"]


def test_a_real_timeout_is_still_suppressed(manager, monkeypatch):
    """The configured limit, not the budget, set this timeout: the callback is hung, so back off as before."""
    monkeypatch.setattr("hermes_cli.plugins._resolve_hook_callback_timeout", lambda: BUDGET)
    hold, calls = threading.Event(), []
    manager._hooks[HOOK] = [_slow(hold, calls)]
    try:
        with hook_budget(HOOK, 60.0):  # a budget far larger than the configured limit does not clamp it
            assert _invoke(manager, 1) == []
        assert len(manager._hook_timeout_suppressed_until) == 1
        assert _invoke(manager, 2) == [] and calls == ["slow"]  # suppressed: not started again
    finally:
        hold.set()


def test_budget_bounds_the_hook_when_the_configured_timeout_is_unbounded(monkeypatch):
    monkeypatch.setattr("hermes_cli.plugins._resolve_hook_callback_timeout", lambda: 0.0)
    hold, calls = threading.Event(), []
    manager = PluginManager()
    manager._hooks[HOOK] = [_slow(hold, calls)]
    try:
        with hook_budget(HOOK, BUDGET):
            assert _invoke(manager, 1) == []  # without the budget this would run inline and block
    finally:
        hold.set()


def test_the_first_party_observer_spends_from_the_same_budget(manager, monkeypatch):
    """lifecycle.invoke_hook runs the observer first; its time leaves the plugins less (or none)."""
    clock = SimpleNamespace(now=1000.0)
    # Only the dispatch module's clock: it reads nothing but ``time.monotonic``.
    monkeypatch.setattr("hermes_cli.plugins_dispatch.time", SimpleNamespace(monotonic=lambda: clock.now))
    calls = []
    manager._hooks[HOOK] = [_fast(calls)]
    monkeypatch.setattr("hermes_cli.plugins._delivery_manager", lambda: manager)

    def observer_that_takes_longer_than_the_budget(hook_name, **_kwargs):
        clock.now += BUDGET + 1.0

    monkeypatch.setattr("hermes_cli.lifecycle._observe", observer_that_takes_longer_than_the_budget)
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda hook_name, **kw: manager.invoke_hook(hook_name, **kw))
    from hermes_cli.lifecycle import invoke_hook

    with hook_budget(HOOK, BUDGET):
        assert invoke_hook(HOOK, turn_id="t1") == []
    assert calls == []  # the observer used the whole budget: the plugin was skipped, not started late


def test_other_hooks_and_a_zero_budget_are_not_budgeted(manager):
    calls = []
    manager._hooks["post_llm_call"] = [_fast(calls)]
    manager._hooks[HOOK] = [_fast(calls)]
    with hook_budget(HOOK, 0.0):
        assert manager.invoke_hook(HOOK, turn_id="t1") == [{"context": "fast"}]
    with hook_budget(HOOK, 0.000001):  # spent at once, but only for pre_llm_call
        assert manager.invoke_hook("post_llm_call", tool_call_id="c1") == [{"context": "fast"}]
    assert calls == ["fast", "fast"]
