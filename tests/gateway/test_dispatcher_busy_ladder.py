"""A busy message of a dispatcher session redirects, steers or queues; it never interrupts, and it follows
the session's own frozen flag, not the profile's ``display.busy_input_mode``.

Real GatewayRunner, WhatsAppAdapter and SessionAuthority over SQLite in a temp HERMES_HOME whose config.yaml
carries ``dispatcher.busy.policy`` and ``display.busy_input_mode``; the running turn is a stand-in agent.
"""
import pytest

from gateway.platforms.event import MessageType
from tests.gateway._dispatcher_busy_rig import StandInAgent, rig

QUEUE_MODE = {'display': {'busy_input_mode': 'queue'}, 'dispatcher': {'busy': {'quick_lane': False}}}


def _inherit(config):
    return {**config, 'dispatcher': {'busy': {'policy': 'inherit', 'quick_lane': False}}}


@pytest.mark.asyncio
async def test_text_redirects_a_dispatcher_turn_that_has_children_and_never_interrupts(tmp_path, monkeypatch):
    agent = StandInAgent(children=['executor'])
    async with rig(tmp_path, monkeypatch, QUEUE_MODE, agent) as r:
        landed = await r.handle(r.event('actually, make it two paragraphs'))
        assert landed is True
        assert len(agent.redirects) == 1 and agent.redirects[0].endswith('actually, make it two paragraphs')
        assert agent.interrupts == [] and agent.steers == []
        assert r.admitted() == [], 'a redirected message must not also be admitted as its own turn'
        assert r.sent == [], 'the model reply is the acceptance; no text ack'


@pytest.mark.asyncio
async def test_a_redirect_the_agent_declines_falls_back_to_a_steer(tmp_path, monkeypatch):
    agent = StandInAgent()
    agent.redirect_result = False  # no model request in flight to cancel
    async with rig(tmp_path, monkeypatch, QUEUE_MODE, agent) as r:
        assert await r.handle(r.event('and cite the source')) is True
        assert len(agent.redirects) == 1 and len(agent.steers) == 1
        assert agent.steers[0].endswith('and cite the source')
        assert agent.interrupts == [] and r.admitted() == [] and r.sent == []


@pytest.mark.asyncio
async def test_an_agent_without_redirect_support_steers_straight_away(tmp_path, monkeypatch):
    agent = StandInAgent(redirect=False)
    async with rig(tmp_path, monkeypatch, QUEUE_MODE, agent) as r:
        assert await r.handle(r.event('also check the tests')) is True
        assert agent.redirects == [] and len(agent.steers) == 1 and agent.interrupts == []


@pytest.mark.asyncio
async def test_a_session_that_is_not_a_dispatcher_keeps_the_profile_busy_mode(tmp_path, monkeypatch):
    agent = StandInAgent(dispatcher=False)
    async with rig(tmp_path, monkeypatch, QUEUE_MODE, agent) as r:
        assert await r.handle(r.event('follow-up')) is False  # queue mode: parked, as before dispatcher mode
        assert agent.redirects == [] and agent.steers == [] and agent.interrupts == []
        assert len(r.admitted()) == 1


@pytest.mark.asyncio
async def test_policy_inherit_restores_the_profile_busy_mode_for_a_dispatcher(tmp_path, monkeypatch):
    agent = StandInAgent()
    async with rig(tmp_path, monkeypatch, _inherit(QUEUE_MODE), agent) as r:
        assert await r.handle(r.event('follow-up')) is False
        assert agent.redirects == [] and agent.steers == [] and len(r.admitted()) == 1


@pytest.mark.asyncio
async def test_compression_in_flight_queues_the_message_and_says_so(tmp_path, monkeypatch):
    agent = StandInAgent()
    async with rig(tmp_path, monkeypatch, QUEUE_MODE, agent) as r:
        async def compressing(session_key):
            return True
        monkeypatch.setattr(r.runner, '_session_has_compression_in_flight', compressing)
        assert await r.handle(r.event('are you still there?')) is False
        assert agent.redirects == [] and agent.steers == [] and agent.interrupts == []
        assert len(r.admitted()) == 1
        assert [m.content for m in r.sent] and 'Compressing context' in r.sent[0].content


@pytest.mark.asyncio
async def test_a_turn_still_building_its_agent_queues_instead_of_latching_a_steer(tmp_path, monkeypatch):
    from gateway.run import _AGENT_PENDING_SENTINEL
    async with rig(tmp_path, monkeypatch, QUEUE_MODE) as r:
        r.runner._session_state(r.key).persistent.dispatcher_seen = True  # the chat's agent was a dispatcher
        r.turn.agent = _AGENT_PENDING_SENTINEL
        assert await r.handle(r.event('hello?')) is False
        assert len(r.admitted()) == 1, 'a queued row is durable; a latched steer would be lost if the build fails'
        assert r.sent and 'Queued for the next turn' in r.sent[0].content


@pytest.mark.asyncio
async def test_a_voice_note_redirects_with_its_transcript(tmp_path, monkeypatch):
    agent = StandInAgent()
    voice = tmp_path / 'note.ogg'
    voice.write_bytes(b'OggS')
    async with rig(tmp_path, monkeypatch, QUEUE_MODE, agent) as r:
        async def transcribe(event, adapter, source, text, **_):
            return 'TRANSCRIBED correction', ['TRANSCRIBED correction']
        monkeypatch.setattr(r.runner, '_transcribe_and_echo_pending_voice', transcribe)
        monkeypatch.setattr(r.runner, '_pending_event_audio_paths', lambda event: list(event.media_urls))
        event = r.event('', message_type=MessageType.VOICE, media_urls=[str(voice)], media_types=['audio/ogg'])
        assert await r.handle(event) is True
        assert len(agent.redirects) == 1 and agent.redirects[0].endswith('TRANSCRIBED correction')
        assert r.admitted() == []


@pytest.mark.asyncio
async def test_a_photo_is_queued_without_redirecting_or_a_text_ack(tmp_path, monkeypatch):
    agent = StandInAgent()
    photo = tmp_path / 'p.png'
    photo.write_bytes(b'\x89PNG\r\n\x1a\n' + b'\x00' * 24)
    async with rig(tmp_path, monkeypatch, QUEUE_MODE, agent) as r:
        event = r.event('look', message_type=MessageType.PHOTO, media_urls=[str(photo)], media_types=['image/png'])
        assert await r.handle(event) is False
        assert agent.redirects == [] and agent.steers == [] and agent.interrupts == []
        assert len(r.admitted()) == 1 and r.sent == []


@pytest.mark.asyncio
async def test_the_in_process_busy_path_runs_the_same_ladder(tmp_path, monkeypatch):
    agent = StandInAgent()
    async with rig(tmp_path, monkeypatch, QUEUE_MODE, agent) as r:
        assert await r.runner._hm_handle_running_session_message(r.event('quick fix'), r.source, r.key) is None
        assert len(agent.redirects) == 1 and agent.interrupts == []
        assert r.adapter._pending_messages == {}, 'a redirected message is not also queued'
        async def compressing(session_key):
            return True
        monkeypatch.setattr(r.runner, '_session_has_compression_in_flight', compressing)
        await r.runner._hm_handle_running_session_message(r.event('and this', message_id='m2'), r.source, r.key)
        assert r.adapter._pending_messages[r.key].text == 'and this'
        assert len(agent.redirects) == 1 and agent.interrupts == []


def test_a_busy_command_resnapshot_never_touches_the_dispatcher_policy():
    from gateway.run import GatewayRunner
    runner = object.__new__(GatewayRunner)
    runner.config = type('Cfg', (), {'multiplex_profiles': True})()
    source = type('Src', (), {'profile': 'work', 'platform': None})()
    inherit = {'dispatcher': {'busy': {'policy': 'inherit'}}, 'display': {'busy_input_mode': 'queue'}}
    runner._snapshot_profile_busy_modes('work', inherit)  # what /busy re-runs
    assert runner._dispatcher_busy_cfg(source).settings.busy_policy == 'dispatch', 'a command changed the policy'
    runner._snapshot_profile_dispatcher_busy('work', inherit)  # gateway startup, once
    assert runner._dispatcher_busy_cfg(source).settings.busy_policy == 'inherit'


def test_a_runner_built_without_init_has_a_default_policy_and_no_dispatcher_sessions():
    from gateway.run import GatewayRunner
    runner = object.__new__(GatewayRunner)
    source = type('Src', (), {'profile': None, 'platform': None})()
    assert runner._dispatcher_busy_enabled(source, 'any-key') is False
    assert runner._drain_lane_notes('any-key') == []


@pytest.mark.asyncio
async def test_the_dispatcher_decision_survives_a_new_conversation(tmp_path, monkeypatch):
    async with rig(tmp_path, monkeypatch, QUEUE_MODE) as r:
        assert r.runner._dispatcher_busy_enabled(r.source, r.key) is True  # stamps the chat's decision
        r.turn.clear()  # no agent visible any more: the turn ended and the cache is empty
        r.runner._agent_cache.clear()
        r.runner._clear_conversation_scope(r.key, reason='session_reset')  # /new
        assert r.runner._dispatcher_busy_enabled(r.source, r.key) is True
        assert r.runner._dispatcher_busy_enabled(r.source, 'a-chat-that-never-ran') is False
