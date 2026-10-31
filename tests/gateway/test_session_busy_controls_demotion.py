"""A dispatcher session's busy input never waits behind its turn and never hard-interrupts it.

Canonical ``session.redirect`` / ``session.steer`` run ``agent.busy_input``'s ladder for a session whose frozen
policy is a dispatcher: redirect, then steer when redirect() declines (no model request in flight, a prior redirect
not drained yet), and only then a refusal the client queues. Compression is refused for every session, and the
SQLite read behind that check never runs under the event-stream lock.
"""
import asyncio
import threading
from types import SimpleNamespace

import pytest

from gateway.session_busy_controls import busy_config, correct
from gateway.session_contract import SessionRef
from tests.gateway.test_session_authority_cancel_settlement import ACTOR, REF, _authority, _wire_turn_agent
from tests.gateway.test_session_steer_stale_busy_refusal import _claim_without_agent


class Agent:
    """Records which busy verbs reached it; ``redirect``/``steer`` answer what the test says."""

    def __init__(self, redirect=True, steer=True, supports_redirect=True, **extra):
        self.calls = []
        self.answers = {'redirect': redirect, 'steer': steer}
        self._supports_active_turn_redirect = supports_redirect
        vars(self).update(extra)

    def redirect(self, text):
        self.calls.append(('redirect', text))
        return self.answers['redirect']

    def steer(self, text):
        self.calls.append(('steer', text))
        return self.answers['steer']


def _policy(dispatcher=True, display=None, busy_policy=None, source='tui'):
    config = {}
    if display is not None:
        config['display'] = display
    if busy_policy is not None:
        config['dispatcher'] = {'busy': {'policy': busy_policy}}
    return SimpleNamespace(dispatcher=dispatcher, source=source, platform=source, ignore_user_config=False,
                           kanban_json=None, request_json='{}', config=lambda *a, **k: config)


async def _correction(tmp_path, monkeypatch, agent, *, verb='redirect', policy=None, before=None):
    """One correction for a claimed turn whose agent is wired (or not wired, when ``agent`` is None)."""
    from gateway import session_policy
    db, authority = _authority(tmp_path, monkeypatch)
    monkeypatch.setattr(session_policy, 'policy_for_source', lambda runner, source: policy)
    connection = SimpleNamespace(authority=authority, actor=ACTOR)
    with db:
        release = await _claim_without_agent(authority, monkeypatch)
        generation = authority._handle(REF).execution_generation
        if agent is not None:
            _wire_turn_agent(authority, generation, agent)
        if before is not None:
            before(db, authority)
        reply = await correct(connection, REF, {'session_id': 's', 'text': 'and also this',
                                               'execution_generation': generation}, verb=verb)
        release.set()
        await asyncio.wait_for(authority.sessions['s'].task, 5)
    return reply, authority


@pytest.mark.asyncio
async def test_a_declined_redirect_steers_instead_of_queueing_behind_the_turn(tmp_path, monkeypatch):
    """redirect() answers False in ordinary windows, and every message of a burst after the first hits one:
    a dispatcher falls to steer(), which a running turn accepts, so rapid-fire input is delivered, not queued."""
    agent = Agent(redirect=False, steer=True)
    reply, authority = await _correction(tmp_path, monkeypatch, agent, policy=_policy())
    assert reply == {'status': 'queued', 'disposition': 'steered', 'text': 'and also this',
                     'execution_generation': reply['execution_generation'], 'authority_epoch': authority.epoch}
    assert agent.calls == [('redirect', 'and also this'), ('steer', 'and also this')]


@pytest.mark.asyncio
async def test_an_accepted_redirect_is_the_only_verb_tried(tmp_path, monkeypatch):
    agent = Agent(redirect=True)
    reply, _ = await _correction(tmp_path, monkeypatch, agent, policy=_policy())
    assert (reply['status'], reply['disposition']) == ('redirected', 'redirected')
    assert agent.calls == [('redirect', 'and also this')]


@pytest.mark.asyncio
async def test_an_agent_without_redirect_support_is_steered(tmp_path, monkeypatch):
    agent = Agent(supports_redirect=False)
    reply, _ = await _correction(tmp_path, monkeypatch, agent, policy=_policy())
    assert (reply['status'], reply['disposition']) == ('queued', 'steered')
    assert agent.calls == [('steer', 'and also this')]


@pytest.mark.asyncio
async def test_a_steer_client_never_redirects_a_dispatcher_turn(tmp_path, monkeypatch):
    agent = Agent()
    reply, _ = await _correction(tmp_path, monkeypatch, agent, verb='steer', policy=_policy())
    assert (reply['status'], reply['disposition']) == ('queued', 'steered')
    assert agent.calls == [('steer', 'and also this')]


@pytest.mark.asyncio
async def test_a_text_no_rung_takes_is_refused_as_undelivered_so_the_client_queues_it(tmp_path, monkeypatch):
    agent = Agent(redirect=False, steer=False)
    reply, _ = await _correction(tmp_path, monkeypatch, agent, policy=_policy())
    assert (reply['status'], reply['delivered'], reply['reason']) == ('rejected', False, 'steer_declined')


@pytest.mark.asyncio
async def test_a_codex_native_redirect_that_timed_out_is_not_retried_or_resent(tmp_path, monkeypatch):
    """The app-server's redirect answers False on an ACK timeout the text may already have crossed: no steer
    after it (it could deliver twice), and no ``delivered: False`` for the client to re-admit on."""
    agent = Agent(redirect=False, api_mode='codex_app_server', _codex_session=SimpleNamespace(request_steer=lambda t: True))
    reply, _ = await _correction(tmp_path, monkeypatch, agent, policy=_policy())
    assert reply['status'] == 'rejected' and 'delivered' not in reply, reply
    assert reply['reason'] == 'redirect_unconfirmed'
    assert agent.calls == [('redirect', 'and also this')]


@pytest.mark.asyncio
async def test_a_redirect_in_the_build_window_is_latched_for_the_agent_the_turn_adopts(tmp_path, monkeypatch):
    """No agent yet: the text is latched (not refused), and the agent this generation adopts steers it in."""
    from gateway import session_policy
    db, authority = _authority(tmp_path, monkeypatch)
    monkeypatch.setattr(session_policy, 'policy_for_source', lambda runner, source: _policy())
    connection = SimpleNamespace(authority=authority, actor=ACTOR)
    with db:
        release = await _claim_without_agent(authority, monkeypatch)
        generation = authority._handle(REF).execution_generation
        reply = await correct(connection, REF, {'session_id': 's', 'text': 'early word',
                                               'execution_generation': generation}, verb='redirect')
        assert (reply['status'], reply['disposition']) == ('queued', 'steered'), reply
        agent = Agent()
        _wire_turn_agent(authority, generation, agent)
        assert agent.calls == [('steer', 'early word')]
        release.set()
        await asyncio.wait_for(authority.sessions['s'].task, 5)


@pytest.mark.asyncio
@pytest.mark.parametrize('policy', [_policy(dispatcher=True), _policy(dispatcher=False)])
async def test_compression_in_flight_refuses_a_correction_for_every_session(tmp_path, monkeypatch, policy):
    """A redirect delivered mid-compression aborts it (#61042): the text is refused as undelivered, the client
    queues it, and no agent verb is called. Dispatcher or not."""
    agent = Agent()

    def hold(db, authority):
        assert db.try_acquire_compression_lock(authority.physical_target(REF), 'compressor')

    reply, _ = await _correction(tmp_path, monkeypatch, agent, policy=policy, before=hold)
    assert (reply['status'], reply['delivered'], reply['reason']) == ('rejected', False, 'compression_in_flight')
    assert agent.calls == []


@pytest.mark.asyncio
async def test_a_manual_compress_window_refuses_the_correction(tmp_path, monkeypatch):
    """``/compress`` holds the session's claim gate across its summarization (no DB lock row yet)."""
    from gateway import session_policy
    db, authority = _authority(tmp_path, monkeypatch)
    monkeypatch.setattr(session_policy, 'policy_for_source', lambda runner, source: _policy())
    connection = SimpleNamespace(authority=authority, actor=ACTOR)
    agent = Agent()
    with db:
        release = await _claim_without_agent(authority, monkeypatch)
        generation = authority._handle(REF).execution_generation
        _wire_turn_agent(authority, generation, agent)
        async with authority.sessions['s'].claim_gate:
            reply = await correct(connection, REF, {'session_id': 's', 'text': 'x',
                                                   'execution_generation': generation}, verb='redirect')
        release.set()
        await asyncio.wait_for(authority.sessions['s'].task, 5)
    assert (reply['status'], reply['delivered'], reply['reason']) == ('rejected', False, 'compression_in_flight')
    assert agent.calls == []


@pytest.mark.asyncio
async def test_the_compression_read_runs_off_the_loop_and_outside_the_stream_lock(tmp_path, monkeypatch):
    """Event publication and agent adoption take ``event_stream.lock``: a slow state.db read must not hold it."""
    seen = []

    def holder_probe(db, authority):
        live = authority.sessions['s']
        original = db.get_compression_lock_holder

        def probe(session_id):
            got = live.event_stream.lock.acquire(blocking=False)
            if got:
                live.event_stream.lock.release()
            seen.append((got, threading.current_thread() is threading.main_thread()))
            return original(session_id)
        monkeypatch.setattr(db, 'get_compression_lock_holder', probe)

    await _correction(tmp_path, monkeypatch, Agent(), policy=_policy(), before=holder_probe)
    assert seen == [(True, False)], seen


@pytest.mark.asyncio
async def test_a_managed_workers_dispatcher_correction_is_a_refusal_not_an_error(tmp_path, monkeypatch):
    from gateway import session_policy
    from hermes_state_runtime import RuntimeStoreError
    db, authority = _authority(tmp_path, monkeypatch)
    authority._managed_workers = {'s': object()}
    connection = SimpleNamespace(authority=authority, actor=ACTOR)
    params = {'session_id': 's', 'text': 'x', 'execution_generation': 0}
    with db:
        release = await _claim_without_agent(authority, monkeypatch)
        params['execution_generation'] = authority._handle(REF).execution_generation
        monkeypatch.setattr(session_policy, 'policy_for_source', lambda runner, source: _policy())
        reply = await correct(connection, REF, params, verb='redirect')
        assert (reply['status'], reply['delivered'], reply['reason']) == ('rejected', False, 'unsupported_control')
        monkeypatch.setattr(session_policy, 'policy_for_source', lambda runner, source: None)
        with pytest.raises(RuntimeStoreError) as refused:  # a classic session keeps its error
            await correct(connection, REF, params, verb='redirect')
        assert refused.value.reason == 'unsupported_control'
        release.set()
        await asyncio.wait_for(authority.sessions['s'].task, 5)


@pytest.mark.asyncio
async def test_policy_inherit_leaves_the_classic_verbs_alone(tmp_path, monkeypatch):
    """``dispatcher.busy.policy: inherit`` opts a dispatcher session out of the ladder: a declined redirect is
    the live agent's plain 'rejected', which a client may not re-send."""
    agent = Agent(redirect=False)
    reply, _ = await _correction(tmp_path, monkeypatch, agent, policy=_policy(busy_policy='inherit'))
    assert reply['status'] == 'rejected' and 'delivered' not in reply and 'disposition' not in reply, reply
    assert agent.calls == [('redirect', 'and also this')]


def _config_connection(policy, override=None):
    live = SimpleNamespace(source=SimpleNamespace(platform='local'), busy_input_mode=override)
    authority = SimpleNamespace(sessions={'s': live}, authorize=lambda actor, ref, capability: None,
                                runner=SimpleNamespace(), profile_id='/nonexistent')
    return SimpleNamespace(authority=authority, actor=None)


@pytest.mark.asyncio
@pytest.mark.parametrize(('policy', 'override', 'expected'), [
    (_policy(display={'busy_input_mode': 'queue'}), None, {'value': 'interrupt', 'dispatch': True}),
    (_policy(display=None), None, {'value': 'interrupt', 'dispatch': True}),  # not the TUI's queue default either
    (_policy(display={'busy_input_mode': 'interrupt'}), 'queue', {'value': 'queue', 'dispatch': True}),
    (_policy(display={'busy_input_mode': 'queue'}, busy_policy='inherit'), None, {'value': 'queue'}),
    (_policy(dispatcher=False, display={'busy_input_mode': 'steer'}), None, {'value': 'steer'}),
    (_policy(dispatcher=False, display=None, source='tui'), None, {'value': 'queue'}),
])
async def test_config_get_busy_reports_the_server_resolved_mode(monkeypatch, policy, override, expected):
    """The classic CLI mirrors this value instead of its own startup config, so a dispatcher session ignores
    ``display.busy_input_mode`` there too; a session ``/busy`` override still wins."""
    from gateway import session_policy
    monkeypatch.setattr(session_policy, 'policy_for_source', lambda runner, source: policy)
    result = await busy_config(_config_connection(policy, override), SessionRef('p', 's'),
                               {'session_id': 's', 'key': 'busy'})
    assert result == {'key': 'busy', 'scope': 'session', **expected}
