"""A busy message of a dispatcher session gets the same start/complete bracket as an idle one, from arrival to
the end of its own handling, and a path that ends it early closes it exactly once.

Real GatewayRunner, SessionAuthority and burst code; the adapter is a WhatsApp adapter whose processing hooks
record into ``calls`` and whose ``acknowledges_event`` says whether it reacts. The running turn is a stand-in.
"""
import asyncio

import pytest

from gateway.platforms.event import MessageEvent, MessageType, ProcessingOutcome
from gateway.session_ingress_burst import discard_busy_burst, dispatch_busy_burst, holds_busy_burst
from gateway.session_receipt import open_receipt
from tests.gateway._dispatcher_busy_rig import StandInAgent, reacting_adapter_class, rig
from tests.waiting import async_wait_until

CONFIG = {'display': {'busy_input_mode': 'queue'}, 'dispatcher': {'busy': {'quick_lane': False}}}
HELD = {**CONFIG, 'display': {'busy_input_mode': 'queue', 'busy_text_debounce_seconds': 3600,
                              'busy_text_hard_cap_seconds': 3600}}
PNG = b'\x89PNG\r\n\x1a\n' + b'\x00' * 24


async def _compressing(r, monkeypatch):
    async def in_flight(session_key):
        return True
    monkeypatch.setattr(r.runner, '_session_has_compression_in_flight', in_flight)


@pytest.mark.asyncio
async def test_a_redirected_busy_message_gets_one_bracket_and_no_text(tmp_path, monkeypatch):
    agent = StandInAgent()
    async with rig(tmp_path, monkeypatch, CONFIG, agent, reacting_adapter_class()) as r:
        await dispatch_busy_burst(r.adapter, r.event('fix the typo'), r.key)
        await async_wait_until(lambda: len(r.adapter.calls) == 2, timeout=10, what='the bracket to close')
        assert r.adapter.calls == [('start', 'm1'), ('complete', 'm1', ProcessingOutcome.SUCCESS)]
        assert len(agent.redirects) == 1 and r.sent == []


@pytest.mark.asyncio
async def test_a_queued_message_keeps_its_mark_until_its_own_turn_ends(tmp_path, monkeypatch):
    from gateway.session_results import execution_result
    async with rig(tmp_path, monkeypatch, CONFIG, adapter_cls=reacting_adapter_class(), drain=True) as r:
        running, finish, executed = asyncio.Event(), asyncio.Event(), []

        async def turn(event, source, quick_key, run_generation):
            executed.append(event.text)
            if event.text == 'A':
                running.set()
                await finish.wait()
            if (captured := execution_result.get()) is not None:
                captured['result'] = {'final_response': 'ok', 'messages': []}
            return 'ok'
        monkeypatch.setattr(r.runner, '_handle_message_with_agent', turn)
        await r.adapter.handle_message(r.event('A', message_id='A'))
        await asyncio.wait_for(running.wait(), 10)
        r.runner._session_state(r.key).persistent.dispatcher_seen = True  # the running turn is a dispatcher's
        await r.adapter.handle_message(r.event('B', message_id='B'))  # no agent to take it yet: queued
        def marks():
            return [c for c in r.adapter.calls if c[1] == 'B']
        await async_wait_until(marks, timeout=10, what="B's mark")
        assert marks() == [('start', 'B')], 'closed before its own turn ran'
        finish.set()
        await async_wait_until(lambda: len(marks()) == 2, timeout=15, what="B's mark to close")
        assert marks() == [('start', 'B'), ('complete', 'B', ProcessingOutcome.SUCCESS)] and executed == ['A', 'B']
        assert not any('Queued' in m.content for m in r.sent), 'the reaction is the receipt; no queued text'


@pytest.mark.asyncio
async def test_stop_closes_a_held_burst_and_its_absorbed_follower_as_cancelled(tmp_path, monkeypatch):
    photo = tmp_path / 'p.png'
    photo.write_bytes(PNG)
    async with rig(tmp_path, monkeypatch, HELD, StandInAgent(), reacting_adapter_class()) as r:
        def shot(n):
            return r.event('', message_id=f'p{n}', message_type=MessageType.PHOTO,
                           media_urls=[str(photo)], media_types=['image/png'])
        assert r.runner._session_key_for_source(r.source) == r.key == r.adapter._event_session_key(shot(1))
        head = asyncio.create_task(dispatch_busy_burst(r.adapter, shot(1), r.key))
        await async_wait_until(lambda: holds_busy_burst(r.adapter, r.key), timeout=10, what='the burst to be held')
        follower = asyncio.create_task(dispatch_busy_burst(r.adapter, shot(2), r.key))
        await async_wait_until(lambda: len(r.adapter.calls) == 2, timeout=10, what='both photos to be marked')
        await r.runner._hm_stop_owner_turn(r.key)  # /stop
        await async_wait_until(lambda: len(r.adapter.calls) == 4, timeout=10, what='both marks to close')
        done = [c for c in r.adapter.calls if c[0] == 'complete']
        assert sorted(done) == [('complete', 'p1', ProcessingOutcome.CANCELLED), ('complete', 'p2', ProcessingOutcome.CANCELLED)]
        discard_busy_burst(r.adapter, r.key)  # the fenced burst the stop leaves held; closing again is a no-op
        await asyncio.wait_for(asyncio.gather(head, follower), 10)
        await asyncio.sleep(0)
        assert len(r.adapter.calls) == 4, 'a mark was closed twice'


@pytest.mark.asyncio
async def test_a_committed_burst_keeps_every_mark_open_until_it_is_stopped(tmp_path, monkeypatch):
    photo = tmp_path / 'p.png'
    photo.write_bytes(PNG)
    config = {**CONFIG, 'display': {'busy_input_mode': 'queue', 'busy_text_debounce_seconds': 0.05,
                                    'busy_text_hard_cap_seconds': 0.05}}
    agent = StandInAgent()
    async with rig(tmp_path, monkeypatch, config, agent, reacting_adapter_class()) as r:
        await _compressing(r, monkeypatch)
        shots = [r.event('', message_id=f'p{n}', message_type=MessageType.PHOTO,
                         media_urls=[str(photo)], media_types=['image/png']) for n in (1, 2)]
        await asyncio.gather(*(dispatch_busy_burst(r.adapter, s, r.key) for s in shots))
        assert sorted(c for c in r.adapter.calls if c[0] == 'start') == [('start', 'p1'), ('start', 'p2')]
        assert [c for c in r.adapter.calls if c[0] == 'complete'] == [], 'the queued turn has not run'
        await r.runner._hm_stop_owner_turn(r.key)
        await async_wait_until(lambda: len([c for c in r.adapter.calls if c[0] == 'complete']) == 2, timeout=10,
                               what='both marks to close')


def _photos(r, tmp_path, count=2):
    photo = tmp_path / 'p.png'
    photo.write_bytes(PNG)
    return [r.event('', message_id=f'p{n}', message_type=MessageType.PHOTO,
                    media_urls=[str(photo)], media_types=['image/png']) for n in range(1, count + 1)]


@pytest.mark.asyncio
async def test_discarding_a_held_burst_closes_its_marks_as_cancelled(tmp_path, monkeypatch):
    """What a serial reader's /stop does (``burst_gate`` -> ``discard_busy_burst``): nothing was ACKed."""
    async with rig(tmp_path, monkeypatch, HELD, StandInAgent(), reacting_adapter_class()) as r:
        head_shot, follower_shot = _photos(r, tmp_path)
        head = asyncio.create_task(dispatch_busy_burst(r.adapter, head_shot, r.key))
        await async_wait_until(lambda: holds_busy_burst(r.adapter, r.key), timeout=10, what='the burst to be held')
        follower = asyncio.create_task(dispatch_busy_burst(r.adapter, follower_shot, r.key))
        await async_wait_until(lambda: len(r.adapter.calls) == 2, timeout=10, what='both photos to be marked')
        discard_busy_burst(r.adapter, r.key)
        await async_wait_until(lambda: len(r.adapter.calls) == 4, timeout=10, what='both marks to close')
        assert sorted(c for c in r.adapter.calls if c[0] == 'complete') == [
            ('complete', 'p1', ProcessingOutcome.CANCELLED), ('complete', 'p2', ProcessingOutcome.CANCELLED)]
        await asyncio.wait_for(asyncio.gather(head, follower), 10)


@pytest.mark.asyncio
@pytest.mark.parametrize('serial', [True, False], ids=['serial reader logs it', 'other adapters raise it'])
async def test_a_burst_whose_commit_fails_closes_its_mark_as_failed(tmp_path, monkeypatch, serial):
    config = {**CONFIG, 'display': {'busy_input_mode': 'queue', 'busy_text_debounce_seconds': 0.05,
                                    'busy_text_hard_cap_seconds': 0.05}}

    async def refused(*args, **kwargs):
        raise RuntimeError('admission refused')
    async with rig(tmp_path, monkeypatch, config, StandInAgent(), reacting_adapter_class()) as r:
        monkeypatch.setattr('gateway.session_ingress.dispatch_shared_busy', refused)
        r.adapter.serial_receive_loop = serial
        shot = _photos(r, tmp_path, 1)[0]
        if serial:
            await dispatch_busy_burst(r.adapter, shot, r.key)  # returns once held; the commit fails later
        else:
            with pytest.raises(RuntimeError, match='admission refused'):
                await dispatch_busy_burst(r.adapter, shot, r.key)
        await async_wait_until(lambda: len(r.adapter.calls) == 2, timeout=10, what='the mark to close')
        assert r.adapter.calls == [('start', 'p1'), ('complete', 'p1', ProcessingOutcome.FAILURE)]


def _unauthorized(r):
    return r.adapter.build_source(chat_id='99999@s.whatsapp.net', user_id='99999@s.whatsapp.net', chat_type='dm')


NEGATIVE = {
    'internal': lambda r: r.event('wake', internal=True),
    'command': lambda r: r.event('/status'),
    'unaddressed': lambda r: r.event('lunch?', reply_expected=False),
    'no message id': lambda r: r.event('hi', message_id=None),
    'unauthorized sender': lambda r: MessageEvent(text='hi', source=_unauthorized(r), message_id='m1', reply_expected=True),
}


@pytest.mark.asyncio
@pytest.mark.parametrize('case', list(NEGATIVE))
async def test_no_mark_for_input_that_will_not_be_answered(tmp_path, monkeypatch, case):
    async with rig(tmp_path, monkeypatch, CONFIG, StandInAgent(), reacting_adapter_class()) as r:
        assert open_receipt(r.adapter, NEGATIVE[case](r), r.key) is False
        await asyncio.sleep(0)
        assert r.adapter.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize('case,adapter_cls,agent,config', [
    ('adapter does not react', reacting_adapter_class(False), StandInAgent(), CONFIG),
    ('not a dispatcher session', reacting_adapter_class(), StandInAgent(dispatcher=False), CONFIG),
    ('receipt off (unquoted YAML off is false)', reacting_adapter_class(), StandInAgent(),
     {**CONFIG, 'display': {'busy_input_mode': 'queue', 'dispatcher_receipt': False}}),
    ('receipt pinned to text', reacting_adapter_class(), StandInAgent(),
     {**CONFIG, 'display': {'busy_input_mode': 'queue', 'dispatcher_receipt': 'text'}}),
])
async def test_no_mark_when_the_session_or_the_setting_says_none(tmp_path, monkeypatch, case, adapter_cls, agent, config):
    async with rig(tmp_path, monkeypatch, config, agent, adapter_cls) as r:
        assert open_receipt(r.adapter, r.event('hi'), r.key) is False
        await asyncio.sleep(0)
        assert r.adapter.calls == []


# (display.dispatcher_receipt, adapter reacts) -> (a mark is opened, a queued text is sent)
TEXT_OR_MARK = [
    ('auto', True, True, False), ('auto', False, False, True),
    ('reaction', True, True, False), ('reaction', False, False, False),
    ('text', True, False, True), ('off', True, False, False),
]


@pytest.mark.asyncio
@pytest.mark.parametrize('mode,reacts,marked,texted', TEXT_OR_MARK)
async def test_a_queued_text_message_is_acknowledged_by_a_mark_or_by_words_never_both(
        tmp_path, monkeypatch, mode, reacts, marked, texted):
    config = {**CONFIG, 'display': {'busy_input_mode': 'queue', 'dispatcher_receipt': mode}}
    async with rig(tmp_path, monkeypatch, config, StandInAgent(), reacting_adapter_class(reacts)) as r:
        await _compressing(r, monkeypatch)
        await dispatch_busy_burst(r.adapter, r.event('are you there?'), r.key)
        await async_wait_until(lambda: len(r.admitted()) == 1, timeout=10, what='the message to queue')
        assert bool(r.adapter.calls) is marked and bool(r.sent) is texted, (r.adapter.calls, r.sent)


@pytest.mark.asyncio
async def test_rapid_queued_text_gets_one_text_ack_not_one_each(tmp_path, monkeypatch):
    async with rig(tmp_path, monkeypatch, CONFIG, StandInAgent(), reacting_adapter_class(False)) as r:
        await _compressing(r, monkeypatch)
        for n in (1, 2, 3):
            await dispatch_busy_burst(r.adapter, r.event(f'ping {n}', message_id=f'm{n}'), r.key)
        await async_wait_until(lambda: len(r.admitted()) == 3, timeout=10, what='all three to queue')
        assert len(r.sent) == 1
