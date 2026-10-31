"""A dispatcher session always shows pre-tool text, and delegates in the background because of it.

``display.interim_assistant_messages`` is resolved per platform (tier defaults, and Mattermost needs an
explicit platform opt-in), and ``delegation.gateway_mode: auto`` reads that same value to decide between
joining children inside the turn and the background path. A dispatcher session forces the value on in ONE
place (``TurnRunner._run_sync_scoped``, after the toolsets are final), so the commentary consumer and the
delegation decision cannot disagree. ``dispatcher.ack.force_interim: false`` hands the decision back to
the ordinary display settings.

Real ``GatewayRunner._run_agent`` + real ``AIAgent`` against a loopback model (the harness of
``test_gateway_delegation_mode``); the dispatcher posture is read from the turn's enabled toolsets.
"""

from __future__ import annotations

import pytest

from gateway.config import Platform
from tests.gateway import test_gateway_delegation_mode as delegation_mode
from tests.gateway.test_gateway_delegation_mode import END_TURN, _delegate_def, _interim, _turn


@pytest.fixture
def dispatched(monkeypatch):
    """The delegation-mode suite's recording of every hand-off to the async registry (the background path)."""
    return delegation_mode.dispatches.__wrapped__(monkeypatch)

DISPATCHER_OFF = {"dispatcher": {"enabled": False}}


def _toolsets(platform, *names):
    return {"platform_toolsets": {platform.value: ["delegation", "todo", *names]}}


@pytest.mark.asyncio
@pytest.mark.parametrize(("platform", "config", "background"), [
    # interim resolved OFF by the operator's own setting: a dispatcher shows pre-tool text anyway
    (Platform.WHATSAPP, {**_interim(False), **_toolsets(Platform.WHATSAPP, "dispatcher")}, True),
    (Platform.WHATSAPP, {**_interim(False), **_toolsets(Platform.WHATSAPP), **DISPATCHER_OFF}, False),
    # the escape hatch hands the decision back to the display settings
    (Platform.WHATSAPP, {**_interim(False), **_toolsets(Platform.WHATSAPP, "dispatcher"),
                         "dispatcher": {"ack": {"force_interim": False}}}, False),
    # Mattermost needs a platform-level opt-in to show scratch text in a public thread; a dispatcher overrides it
    (Platform.MATTERMOST, _toolsets(Platform.MATTERMOST, "dispatcher"), True),
    (Platform.MATTERMOST, {**_toolsets(Platform.MATTERMOST), **DISPATCHER_OFF}, False),
], ids=["interim-off-dispatcher", "interim-off-classic", "force-interim-false", "mattermost-dispatcher",
        "mattermost-classic"])
async def test_a_dispatcher_session_resolves_interim_on_for_both_consumers(
        monkeypatch, dispatched, platform, config, background):
    _chat, requests = await _turn(monkeypatch, config, platform=platform)

    assert (END_TURN in _delegate_def(requests[0])["description"]) is background
    assert len(dispatched) == (1 if background else 0)


@pytest.mark.parametrize(("toolsets", "platform_key", "config", "forced"), [
    (["dispatcher", "todo"], "telegram", {}, True),
    (["dispatcher"], "cli", {}, True),  # a local session is a dispatcher surface too
    (["delegation", "todo"], "telegram", {}, False),  # no dispatcher posture
    (None, "telegram", {}, False),
    (["dispatcher"], "webhook", {}, False),  # a surface that cannot be a dispatcher
    (["dispatcher"], "api_server", {}, False),
    (["dispatcher"], "telegram", {"dispatcher": {"ack": {"force_interim": False}}}, False),
    (["dispatcher"], "telegram", {"dispatcher": {"enabled": False}}, True),  # frozen at creation, not live config
])
def test_interim_forced_follows_the_frozen_posture_not_live_config(toolsets, platform_key, config, forced):
    from agent.dispatcher_ack import interim_forced

    assert interim_forced(toolsets, platform_key, config) is forced
