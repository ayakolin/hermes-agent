"""``acknowledges_event``: the one capability that says an adapter's processing hooks put a visible mark on the
user's message. False unless the adapter implements it; each reaction-capable adapter answers with its own
enabled flag and tracking, and the hook it guards sends nothing when it is False."""
from types import SimpleNamespace

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource


def _event(platform, chat_id='chat-1', message_id='msg-1', raw=None, scope_id=None):
    source = SessionSource(platform=platform, chat_id=chat_id, chat_type='dm', user_id='u1', scope_id=scope_id)
    return MessageEvent(text='hi', message_type=MessageType.TEXT, source=source, message_id=message_id,
                        raw_message=raw)


def _adapter(cls, platform, **attrs):
    adapter = object.__new__(cls)
    adapter.platform = platform
    adapter.config = PlatformConfig(enabled=True, token='t')
    for name, value in attrs.items():
        setattr(adapter, name, value)
    return adapter


def test_an_adapter_that_does_not_implement_it_says_no():
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter
    assert _adapter(WhatsAppAdapter, Platform.WHATSAPP).acknowledges_event(_event(Platform.WHATSAPP)) is False


def test_discord_says_no_even_with_reactions_on_because_its_hooks_write_durable_state():
    from plugins.platforms.discord.adapter import DiscordAdapter
    adapter = _adapter(DiscordAdapter, Platform.DISCORD)
    adapter._reactions_enabled = lambda: True
    assert adapter.acknowledges_event(_event(Platform.DISCORD)) is False


@pytest.mark.parametrize('enabled,chat_id,message_id,expected', [
    (True, 'c', 'm', True), (False, 'c', 'm', False), (True, None, 'm', False), (True, 'c', None, False)])
def test_telegram(monkeypatch, enabled, chat_id, message_id, expected):
    from plugins.platforms.telegram.adapter import TelegramAdapter
    monkeypatch.delenv('TELEGRAM_REACTIONS', raising=False)
    adapter = _adapter(TelegramAdapter, Platform.TELEGRAM)
    adapter.config.extra['reactions'] = enabled
    assert adapter.acknowledges_event(_event(Platform.TELEGRAM, chat_id, message_id)) is expected


def test_slack_only_for_a_message_it_is_tracking(monkeypatch):
    from plugins.platforms.slack.adapter import SlackAdapter
    monkeypatch.delenv('SLACK_REACTIONS', raising=False)
    adapter = _adapter(SlackAdapter, Platform.SLACK, _reacting_message_ids=set())
    event = _event(Platform.SLACK, message_id='1700.1', scope_id='T1')
    assert adapter.acknowledges_event(event) is False
    adapter._reacting_message_ids.add(adapter._workspace_message_marker('T1', '1700.1'))
    assert adapter.acknowledges_event(event) is True
    adapter.config.extra['reactions'] = False
    assert adapter.acknowledges_event(event) is False


def test_signal_needs_reactions_on_an_allowed_sender_and_a_target_message(monkeypatch):
    from plugins.platforms.signal.adapter import SignalAdapter
    monkeypatch.delenv('SIGNAL_REACTIONS', raising=False)
    adapter = _adapter(SignalAdapter, Platform.SIGNAL, dm_allow_from={'u1'})
    raw = {'sender': '+1555', 'timestamp_ms': 1700}
    assert adapter.acknowledges_event(_event(Platform.SIGNAL, raw=raw)) is True
    assert adapter.acknowledges_event(_event(Platform.SIGNAL, raw=None)) is False, 'no target message to react to'
    adapter.dm_allow_from = {'someone-else'}
    assert adapter.acknowledges_event(_event(Platform.SIGNAL, raw=raw)) is False, 'an unlisted contact is not told'


def test_photon_is_off_until_enabled(monkeypatch):
    from plugins.platforms.photon.adapter import PhotonAdapter
    # Photon is a plugin platform with no Platform member; the check reads only chat_id and message_id.
    adapter = _adapter(PhotonAdapter, Platform.TELEGRAM)
    monkeypatch.delenv('PHOTON_REACTIONS', raising=False)
    assert adapter.acknowledges_event(_event(Platform.TELEGRAM)) is False
    monkeypatch.setenv('PHOTON_REACTIONS', 'true')
    assert adapter.acknowledges_event(_event(Platform.TELEGRAM)) is True


def test_feishu_and_matrix_follow_their_own_flags(monkeypatch):
    from plugins.platforms.feishu.adapter import FeishuAdapter
    from plugins.platforms.matrix.adapter import MatrixAdapter
    monkeypatch.delenv('FEISHU_REACTIONS', raising=False)
    feishu = _adapter(FeishuAdapter, Platform.FEISHU)
    assert feishu.acknowledges_event(_event(Platform.FEISHU)) is True
    monkeypatch.setenv('FEISHU_REACTIONS', 'false')
    assert feishu.acknowledges_event(_event(Platform.FEISHU)) is False
    matrix = _adapter(MatrixAdapter, Platform.MATRIX, _reactions_enabled=True)
    assert matrix.acknowledges_event(_event(Platform.MATRIX)) is True
    matrix._reactions_enabled = False
    assert matrix.acknowledges_event(_event(Platform.MATRIX)) is False
    assert matrix.acknowledges_event(SimpleNamespace(message_id=None, source=SimpleNamespace(chat_id='r'))) is False
