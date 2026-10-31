"""Mode resolution and the ladder runner the local busy routers share (agent/busy_input_local.py)."""
from types import SimpleNamespace

import pytest

from agent.busy_input import BusyFacts
from agent.busy_input_local import (
    DECLINED, MAYBE, TAKEN, dispatch_policy_applies, frozen_dispatcher, native_steer, plan_for_mode,
    resolve_busy_mode, run_ladder, supports_redirect)


@pytest.mark.parametrize(('override', 'configured', 'dispatch', 'expected'), [
    (None, 'queue', True, 'interrupt'),      # a dispatcher ignores display.busy_input_mode and the TUI queue default
    (None, 'steer', True, 'interrupt'),
    ('queue', 'interrupt', True, 'queue'),   # a session /busy override still wins
    ('steer', 'queue', True, 'steer'),
    (None, 'queue', False, 'queue'),         # non-dispatcher sessions keep their own mode
    (None, 'steer', False, 'steer'),
    ('queue', 'interrupt', False, 'queue'),
    (None, 'bogus', False, 'interrupt'),
    ('bogus', 'steer', False, 'steer'),
])
def test_resolve_busy_mode(override, configured, dispatch, expected):
    assert resolve_busy_mode(override, configured, dispatch=dispatch) == expected


@pytest.mark.parametrize(('config', 'dispatcher', 'expected'), [
    ({}, True, True),
    ({'dispatcher': {'busy': {'policy': 'inherit'}}}, True, False),
    ({'dispatcher': {'busy': {'policy': 'dispatch'}}}, True, True),
    ({}, False, False),
    (None, True, True),
])
def test_dispatch_policy_applies(config, dispatcher, expected):
    assert dispatch_policy_applies(config, dispatcher) is expected


def test_the_frozen_flag_is_read_from_policy_session_dict_or_agent():
    assert frozen_dispatcher(SimpleNamespace(dispatcher=True))
    assert frozen_dispatcher(None, {'dispatcher': True})
    assert frozen_dispatcher(SimpleNamespace(_dispatcher_session=True))
    assert not frozen_dispatcher(None, {}, SimpleNamespace(), SimpleNamespace(dispatcher=False))
    # A truthy non-bool (a MagicMock attribute, a stray string) is never a dispatcher.
    assert not frozen_dispatcher(SimpleNamespace(dispatcher='yes'), SimpleNamespace(_dispatcher_session=1))


@pytest.mark.parametrize(('mode', 'facts', 'ladder'), [
    ('interrupt', BusyFacts(has_text=True), ('redirect', 'steer', 'queue')),
    ('interrupt', BusyFacts(has_text=True, redirect_supported=False), ('steer', 'queue')),
    ('steer', BusyFacts(has_text=True), ('steer', 'queue')),
    ('steer', BusyFacts(has_text=True, agent_building=True), ('latch', 'queue')),
    ('queue', BusyFacts(has_text=True), ('queue',)),
    ('interrupt', BusyFacts(has_text=True, compressing=True), ('queue',)),
    ('steer', BusyFacts(has_text=True, compressing=True), ('queue',)),
    ('interrupt', BusyFacts(has_text=False), ('queue',)),
])
def test_plan_for_mode_never_reaches_a_hard_interrupt(mode, facts, ladder):
    plan = plan_for_mode(mode, facts)
    assert plan.ladder == ladder
    assert 'interrupt' not in plan.ladder


def _attempts(**answers):
    calls = []

    def make(verb):
        def attempt():
            calls.append(verb)
            return answers[verb]
        return attempt
    return {verb: make(verb) for verb in answers}, calls


def test_the_ladder_stops_at_the_first_rung_that_takes_the_text():
    attempts, calls = _attempts(redirect=TAKEN, steer=TAKEN)
    outcome = run_ladder(plan_for_mode('interrupt', BusyFacts(has_text=True)), attempts)
    assert (outcome.verb, outcome.disposition, calls) == ('redirect', 'redirected', ['redirect'])


def test_a_declined_redirect_falls_to_steer_not_to_the_queue():
    """redirect() answers False whenever no model request is in flight (a burst's second message): the next
    rung is steer, which every running turn accepts, so rapid-fire input does not wait behind the turn."""
    attempts, calls = _attempts(redirect=DECLINED, steer=TAKEN)
    outcome = run_ladder(plan_for_mode('interrupt', BusyFacts(has_text=True)), attempts)
    assert (outcome.verb, outcome.disposition, calls) == ('steer', 'steered', ['redirect', 'steer'])


def test_the_queue_is_reached_only_when_every_rung_declined():
    attempts, calls = _attempts(redirect=DECLINED, steer=DECLINED)
    outcome = run_ladder(plan_for_mode('interrupt', BusyFacts(has_text=True)), attempts)
    assert (outcome.verb, outcome.disposition, outcome.reason, outcome.maybe_delivered) == (
        'queue', 'queued_next_turn', 'steer_declined', False)
    assert calls == ['redirect', 'steer']


def test_a_declined_latch_names_the_build_window():
    attempts, _ = _attempts(latch=DECLINED)
    outcome = run_ladder(plan_for_mode('interrupt', BusyFacts(has_text=True, agent_building=True)), attempts)
    assert (outcome.verb, outcome.reason) == ('queue', 'agent_building')


def test_an_unconfirmed_rung_ends_the_walk_without_a_second_delivery():
    """A Codex native steer that timed out may already hold the text: trying steer next could deliver it twice."""
    attempts, calls = _attempts(redirect=MAYBE, steer=TAKEN)
    outcome = run_ladder(plan_for_mode('interrupt', BusyFacts(has_text=True)), attempts)
    assert (outcome.verb, outcome.maybe_delivered, calls) == ('queue', True, ['redirect'])


def test_a_plan_that_only_queues_asks_nobody():
    attempts, calls = _attempts(redirect=TAKEN, steer=TAKEN)
    outcome = run_ladder(plan_for_mode('interrupt', BusyFacts(has_text=True, compressing=True)), attempts)
    assert (outcome.verb, outcome.reason, calls) == ('queue', 'compression_in_flight', [])


def test_the_queue_reason_names_the_fact_that_forced_it():
    outcome = run_ladder(plan_for_mode('interrupt', BusyFacts(has_text=False)), {})
    assert outcome.reason == 'non_text'


def test_redirect_support_needs_the_flag_and_the_method():
    assert supports_redirect(SimpleNamespace(_supports_active_turn_redirect=True, redirect=lambda text: True))
    assert not supports_redirect(SimpleNamespace(_supports_active_turn_redirect=True))
    assert not supports_redirect(SimpleNamespace(redirect=lambda text: True))


def test_native_steer_is_the_codex_app_server_only():
    codex = SimpleNamespace(api_mode='codex_app_server', _codex_session=SimpleNamespace(request_steer=lambda text: True))
    assert native_steer(codex)
    assert not native_steer(SimpleNamespace(api_mode='chat_completions'))
    assert not native_steer(SimpleNamespace(api_mode='codex_app_server', _codex_session=SimpleNamespace()))
