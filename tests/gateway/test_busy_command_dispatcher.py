"""``/busy <mode>`` on a dispatcher session sets only that session's override.

The ladder's default is not a user preference, so the command must not rewrite ``display.busy_input_mode``,
which is every other session's mode (and the mode the dispatcher falls back to under ``policy: inherit``).
"""
from types import SimpleNamespace

import pytest

import gateway.run as gateway_run
from tests.gateway.test_busy_command import _make_event, _make_runner


def _policy(dispatcher, busy_policy=None):
    config = {'dispatcher': {'busy': {'policy': busy_policy}}} if busy_policy else {}
    return SimpleNamespace(dispatcher=dispatcher, source='tui', config=lambda *a, **k: config)


def _wire(monkeypatch, policy):
    from gateway import session_policy, session_setting_commands
    live = SimpleNamespace(busy_input_mode=None, source=object())
    monkeypatch.setattr(session_setting_commands, 'local_live', lambda runner, source: live)
    monkeypatch.setattr(session_policy, 'policy_for_source', lambda runner, source: policy)
    writes = []
    monkeypatch.setattr('cli.save_config_value', lambda key, value: writes.append((key, value)) or True)
    monkeypatch.setattr(gateway_run, '_load_gateway_config', dict)
    return live, writes


@pytest.mark.asyncio
async def test_a_dispatcher_session_busy_command_is_session_scoped(monkeypatch):
    live, writes = _wire(monkeypatch, _policy(dispatcher=True))
    runner = _make_runner('interrupt')
    reply = await runner._handle_busy_command(_make_event('/busy queue'))
    assert live.busy_input_mode == 'queue'
    assert writes == [], 'display.busy_input_mode was rewritten for every other session'
    assert runner._busy_input_mode == 'interrupt'
    assert 'saved' not in str(reply).lower(), 'the reply must not claim a config write'


@pytest.mark.asyncio
@pytest.mark.parametrize('policy', [_policy(dispatcher=False), _policy(dispatcher=True, busy_policy='inherit')])
async def test_other_sessions_keep_the_persisted_busy_command(monkeypatch, policy):
    live, writes = _wire(monkeypatch, policy)
    monkeypatch.setattr(gateway_run, '_load_gateway_config', lambda: {'display': {'busy_input_mode': 'queue'}})
    monkeypatch.delenv('HERMES_GATEWAY_BUSY_TEXT_MODE', raising=False)
    monkeypatch.delenv('HERMES_GATEWAY_BUSY_INPUT_MODE', raising=False)
    runner = _make_runner('interrupt')
    runner._busy_text_mode = 'interrupt'
    await runner._handle_busy_command(_make_event('/busy queue'))
    assert writes == [('display.busy_input_mode', 'queue')]
    assert live.busy_input_mode == 'queue'
