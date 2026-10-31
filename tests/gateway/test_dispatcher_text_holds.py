"""A busy dispatcher session's text is not held for a merge window, at the runner's burst debounce or at the
adapter's text batch; every other session keeps both holds.

Real GatewayRunner, WhatsAppAdapter and SessionAuthority in a temp HERMES_HOME; the running turn is a stand-in.
"""
import asyncio

import pytest

from gateway.platforms.event import MessageEvent, MessageType
from gateway.session_ingress_burst import discard_busy_burst, dispatch_busy_burst, holds_busy_burst
from tests.gateway._dispatcher_busy_rig import StandInAgent, rig
from tests.waiting import async_wait_until

HELD = {'display': {'busy_input_mode': 'queue', 'busy_text_debounce_seconds': 3600,
                    'busy_text_hard_cap_seconds': 3600},
        'dispatcher': {'busy': {'quick_lane': False}}}


@pytest.mark.asyncio
async def test_a_dispatcher_sessions_busy_text_is_not_held_for_the_debounce_window(tmp_path, monkeypatch):
    agent = StandInAgent()
    async with rig(tmp_path, monkeypatch, HELD, agent) as r:
        await asyncio.wait_for(dispatch_busy_burst(r.adapter, r.event('quick fix'), r.key), 10)
        assert not holds_busy_burst(r.adapter, r.key) and len(agent.redirects) == 1


@pytest.mark.asyncio
async def test_another_session_of_the_same_profile_keeps_the_queue_mode_debounce(tmp_path, monkeypatch):
    async with rig(tmp_path, monkeypatch, HELD, StandInAgent(dispatcher=False)) as r:
        held = asyncio.create_task(dispatch_busy_burst(r.adapter, r.event('first half'), r.key))
        await async_wait_until(lambda: holds_busy_burst(r.adapter, r.key), timeout=10, what='the text to be held')
        discard_busy_burst(r.adapter, r.key)
        await asyncio.wait_for(held, 10)


def _pending(r, text, **fields):
    event = MessageEvent(text=text, source=r.source, message_id='b1', reply_expected=True, **fields)
    event._last_chunk_len = len(text)
    r.adapter._pending_text_batches['batch'] = event
    return event


@pytest.fixture
def batching(monkeypatch):
    def configure(adapter):
        adapter._text_batch_delay_seconds, adapter._text_batch_split_delay_seconds = 0.6, 2.0
        adapter._SPLIT_THRESHOLD = 100
    return configure


@pytest.mark.asyncio
async def test_the_adapters_text_batch_wait_is_skipped_only_for_a_busy_dispatcher(tmp_path, monkeypatch, batching):
    async with rig(tmp_path, monkeypatch, HELD, StandInAgent()) as r:
        batching(r.adapter)
        _pending(r, 'short')
        assert r.adapter._text_batch_wait('batch') == 0.0, 'a busy dispatcher waited out the merge window'
        r.turn.agent = None  # idle: the first message of a turn still merges a client-side split
        assert r.adapter._text_batch_wait('batch') == 0.6
        r.turn.agent = StandInAgent(dispatcher=False)  # busy, but not a dispatcher
        assert r.adapter._text_batch_wait('batch') == 0.6


@pytest.mark.asyncio
async def test_a_chunk_that_may_continue_keeps_its_wait(tmp_path, monkeypatch, batching):
    async with rig(tmp_path, monkeypatch, HELD, StandInAgent()) as r:
        batching(r.adapter)
        _pending(r, 'x' * 100)  # at the platform's split point: the next chunk is almost certain
        assert r.adapter._text_batch_wait('batch') == 2.0
        _pending(r, 'photo caption', media_urls=['/tmp/p.png'], media_types=['image/png'], message_type=MessageType.PHOTO)
        assert r.adapter._text_batch_wait('batch') == 0.6, 'a batch carrying media keeps its merge window'
        monkeypatch.setattr(r.adapter, '_text_batch_delay_for', lambda pending: 1.5)  # e.g. a bot handoff hold
        _pending(r, 'short')
        assert r.adapter._text_batch_wait('batch') == 1.5
