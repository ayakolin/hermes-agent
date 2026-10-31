"""The quick lane answers a message the dispatcher's turn cannot take yet, from a cache-parity fork, and only
while that answer is still wanted.

Real GatewayRunner, WhatsAppAdapter, SessionAuthority and session store in a temp HERMES_HOME. The fork is
``agent.side_question.answer_side_question`` (patched where the lane imports it); the rest is production code.
"""
import threading

import pytest

from gateway.dispatcher_quick_lane import FRAMING
from tests.gateway._dispatcher_busy_rig import StandInAgent, rig
from tests.waiting import async_wait_until

LANE = {'display': {'busy_input_mode': 'queue'}, 'dispatcher': {'busy': {'quick_lane': True, 'stall_seconds': 0}}}


class FakeFork:
    """Stands in for ``answer_side_question``: records what the lane asked, and can hold until released."""

    def __init__(self, answer='Still compressing; I will get to it next.', hold=False):
        self.answer, self.hold = answer, hold
        self.calls, self.entered, self.release, self.interrupted = [], threading.Event(), threading.Event(), threading.Event()
        self.fork_handle = type('Fork', (), {'hard_interrupt': lambda s, message=None: (
            self.interrupted.set(), self.release.set())})()

    def __call__(self, question, history, **kwargs):
        self.calls.append((question, history, kwargs))
        kwargs['on_fork'](self.fork_handle)
        self.entered.set()
        if self.hold:
            self.release.wait(10)
        kwargs['on_fork'](None)
        return self.answer


async def _prepare(r, monkeypatch, fork, compressing=True):
    async def in_flight(session_key):
        return compressing
    monkeypatch.setattr(r.runner, '_session_has_compression_in_flight', in_flight)
    monkeypatch.setattr(r.runner, '_resolve_session_agent_runtime',
                        lambda **_: ('m', {'api_key': 'k', 'provider': 'p', 'base_url': 'u', 'api_mode': 'chat_completions'}))
    monkeypatch.setattr('agent.side_question.answer_side_question', fork)


def _interim(r):
    return [m for m in r.sent if (m.metadata or {}).get('_interim_send') is True]


async def _lane_idle(r):
    lane = r.runner._session_state(r.key).conversation.quick_lane
    await async_wait_until(lambda: lane is None or not lane.tasks, timeout=10, what='the lane watchers to finish')


@pytest.mark.asyncio
async def test_compression_in_flight_is_answered_from_the_transcript_snapshot_and_leaves_it_untouched(tmp_path, monkeypatch):
    fork = FakeFork()
    async with rig(tmp_path, monkeypatch, LANE, StandInAgent()) as r:
        await _prepare(r, monkeypatch, fork)
        entry = r.runner.session_store.get_or_create_session(r.source)
        for message in ({'role': 'user', 'content': 'first'}, {'role': 'assistant', 'content': 'reply one'}):
            r.runner.session_store.append_to_transcript(entry.session_id, message)
        before = r.runner.session_store.load_transcript(entry.session_id)
        assert await r.handle(r.event('are you stuck?')) is False  # queued behind the compression
        await async_wait_until(lambda: _interim(r), timeout=10, what='the quick reply')
        question, history, kwargs = fork.calls[0]
        assert len(fork.calls) == 1 and kwargs['framing'] == FRAMING
        assert [m['content'] for m in history] == ['first', 'reply one']
        assert 'are you stuck?' in question and 'earlier question' in question, 'the in-flight turn travels in the question'
        assert callable(kwargs['should_stop']) and callable(kwargs['on_fork'])
        assert _interim(r)[0].content == fork.answer
        assert r.runner.session_store.load_transcript(entry.session_id) == before, 'the fork writes no transcript'
        notes = r.runner._session_state(r.key).conversation.lane_notes
        assert len(notes) == 1 and notes[0].startswith('[Quick reply already sent]') and 'are you stuck?' in notes[0]
        await _lane_idle(r)


@pytest.mark.asyncio
async def test_a_message_the_turn_already_took_gets_no_late_reply(tmp_path, monkeypatch):
    fork = FakeFork(hold=True)
    async with rig(tmp_path, monkeypatch, LANE, StandInAgent()) as r:
        await _prepare(r, monkeypatch, fork)
        assert await r.handle(r.event('status?')) is False
        await async_wait_until(fork.entered.is_set, timeout=10, what='the fork to start')
        r.turn.started_ts += 1.0  # that turn ended and another began: the queued message was taken
        fork.release.set()
        await _lane_idle(r)
        assert _interim(r) == [] and r.runner._session_state(r.key).conversation.lane_notes == []


@pytest.mark.asyncio
async def test_stop_interrupts_the_fork_and_no_reply_follows_it(tmp_path, monkeypatch):
    fork = FakeFork(hold=True)
    async with rig(tmp_path, monkeypatch, LANE, StandInAgent()) as r:
        await _prepare(r, monkeypatch, fork)
        assert await r.handle(r.event('status?')) is False
        await async_wait_until(fork.entered.is_set, timeout=10, what='the fork to start')
        await r.runner._hm_stop_owner_turn(r.key)  # /stop
        assert fork.interrupted.wait(10), 'the fork kept running after /stop'
        assert r.runner._session_state(r.key).conversation.quick_lane is None
        await async_wait_until(lambda: not r.runner._background_tasks or all(
            t.done() for t in r.runner._background_tasks), timeout=10, what='the lane to end')
        assert _interim(r) == []


@pytest.mark.asyncio
async def test_a_new_conversation_cancels_the_lane_before_it_drops_the_reference(tmp_path, monkeypatch):
    fork = FakeFork(hold=True)
    async with rig(tmp_path, monkeypatch, LANE, StandInAgent()) as r:
        await _prepare(r, monkeypatch, fork)
        assert await r.handle(r.event('status?')) is False
        await async_wait_until(fork.entered.is_set, timeout=10, what='the fork to start')
        r.runner._clear_conversation_scope(r.key, reason='session_reset')  # /new, auto-reset, /resume
        assert fork.interrupted.wait(10), 'the lane outlived the conversation it answered'
        assert r.runner._session_state(r.key).conversation.quick_lane is None
        await async_wait_until(lambda: all(t.done() for t in r.runner._background_tasks), timeout=10, what='the lane to end')
        assert _interim(r) == [] and r.runner._session_state(r.key).conversation.lane_notes == []


@pytest.mark.asyncio
async def test_a_steer_the_turn_has_not_taken_is_answered_and_the_turn_is_told_in_its_own_channel(tmp_path, monkeypatch):
    fork, agent = FakeFork(answer='Working on it; two minutes.'), StandInAgent(redirect=False)
    agent.steer_pending = True  # accepted, but the turn is inside a long tool call
    async with rig(tmp_path, monkeypatch, LANE, agent) as r:
        await _prepare(r, monkeypatch, fork, compressing=False)
        assert await r.handle(r.event('how long?')) is True  # landed in the running turn
        await async_wait_until(lambda: _interim(r), timeout=10, what='the quick reply')
        assert agent.steers and len(fork.calls) == 1
        notice = agent._plugin_live_notes  # queued for the turn's next tool result, never as the user's words
        assert len(notice) == 1 and next(iter(notice.values())).startswith('[Quick reply already sent]')
        assert r.runner._session_state(r.key).conversation.lane_notes == [], 'a live turn is told live, once'
        await _lane_idle(r)


@pytest.mark.asyncio
async def test_a_steer_the_turn_took_in_time_is_not_answered_twice(tmp_path, monkeypatch):
    fork, agent = FakeFork(), StandInAgent(redirect=False)  # nothing pending: the turn consumed it
    async with rig(tmp_path, monkeypatch, LANE, agent) as r:
        await _prepare(r, monkeypatch, fork, compressing=False)
        assert await r.handle(r.event('how long?')) is True
        await _lane_idle(r)
        assert fork.calls == [] and _interim(r) == []


@pytest.mark.asyncio
async def test_unaddressed_group_chatter_never_starts_a_lane(tmp_path, monkeypatch):
    fork = FakeFork()
    async with rig(tmp_path, monkeypatch, LANE, StandInAgent()) as r:
        await _prepare(r, monkeypatch, fork)
        assert await r.handle(r.event('lunch?', reply_expected=False)) is False
        assert r.runner._session_state(r.key).conversation.quick_lane is None
        assert fork.calls == [] and _interim(r) == []


@pytest.mark.asyncio
async def test_one_message_is_answered_at_a_time(tmp_path, monkeypatch):
    fork = FakeFork(hold=True)
    async with rig(tmp_path, monkeypatch, LANE, StandInAgent()) as r:
        await _prepare(r, monkeypatch, fork)
        assert await r.handle(r.event('one', message_id='m1')) is False
        await async_wait_until(fork.entered.is_set, timeout=10, what='the first fork to start')
        assert await r.handle(r.event('two', message_id='m2')) is False
        fork.release.set()
        await _lane_idle(r)
        assert len(fork.calls) == 1, 'a second fork ran beside the first'


@pytest.mark.asyncio
async def test_the_lane_is_off_when_configured_off(tmp_path, monkeypatch):
    fork = FakeFork()
    config = {**LANE, 'dispatcher': {'busy': {'quick_lane': False}}}
    async with rig(tmp_path, monkeypatch, config, StandInAgent()) as r:
        await _prepare(r, monkeypatch, fork)
        assert await r.handle(r.event('are you stuck?')) is False
        assert r.runner._session_state(r.key).conversation.quick_lane is None and fork.calls == []


@pytest.mark.asyncio
async def test_lane_notes_are_delivered_once(tmp_path, monkeypatch):
    async with rig(tmp_path, monkeypatch, LANE, StandInAgent()) as r:
        r.runner._session_state(r.key).conversation.lane_notes = ['[Quick reply already sent] note']
        assert r.runner._drain_lane_notes(r.key) == ['[Quick reply already sent] note']
        assert r.runner._drain_lane_notes(r.key) == []
