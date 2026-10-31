"""The legacy tui_gateway busy path on a dispatcher session: redirect, then steer, then the FIFO queue.

A dispatcher session never hard-interrupts its turn, and ``display.busy_input_mode`` does not decide its mode
(the ladder does); a session ``/busy`` override still wins. Other sessions keep the existing behaviour.
"""
from __future__ import annotations

import threading
import types

import pytest

from tui_gateway import server


class Agent:
    def __init__(self, redirect=True, steer=True, supports_redirect=True, **extra):
        self.calls, self.interrupts = [], 0
        self.answers = {"redirect": redirect, "steer": steer}
        self._supports_active_turn_redirect = supports_redirect
        vars(self).update(extra)

    def redirect(self, text):
        self.calls.append(("redirect", text))
        return self.answers["redirect"]

    def steer(self, text):
        self.calls.append(("steer", text))
        return self.answers["steer"]

    def interrupt(self, *a, **k):
        self.interrupts += 1


def _session(agent, dispatcher=True, **extra):
    return {"agent": agent, "session_key": "session-key", "history": [], "history_lock": threading.Lock(),
            "history_version": 0, "running": True, "transport": None, "attached_images": [],
            "dispatcher": dispatcher, **extra}


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "queue")  # display.busy_input_mode says queue
    monkeypatch.setattr(server, "_session_compression_in_flight", lambda session: False)
    monkeypatch.setattr(server, "_load_cfg", lambda: {})
    interrupted = []
    monkeypatch.setattr(server, "_interrupt_busy_session", lambda *a: interrupted.append(a))
    return interrupted


def _submit(session, text="follow-up", **kw):
    return server._handle_busy_submit("r1", "sid", session, text, "ws-1", **kw)["result"]


def test_the_ladder_ignores_display_busy_input_mode(_env):
    agent = Agent(redirect=True)
    result = _submit(_session(agent))
    assert (result["status"], result["disposition"]) == ("redirected", "redirected")
    assert agent.calls == [("redirect", "follow-up")]
    assert _env == [] and agent.interrupts == 0


def test_a_declined_redirect_steers_not_queues():
    agent = Agent(redirect=False, steer=True)
    result = _submit(_session(agent))
    assert (result["status"], result["disposition"]) == ("steered", "steered")
    assert agent.calls == [("redirect", "follow-up"), ("steer", "follow-up")]


def test_a_text_no_rung_takes_queues_and_never_interrupts(_env):
    agent = Agent(redirect=False, steer=False)
    session = _session(agent)
    result = _submit(session)
    assert (result["status"], result["disposition"], result["reason"]) == ("queued", "queued_next_turn", "steer_declined")
    assert session["queued_prompt"]["text"] == "follow-up"
    assert _env == [], "a dispatcher session must never hard-interrupt (it drops AIAgent._pending_steer, #86134)"


def test_redirect_unavailable_only_queues_when_steer_also_declines(_env):
    agent = Agent(supports_redirect=False, steer=False)
    result = _submit(_session(agent))
    assert result["status"] == "queued" and agent.calls == [("steer", "follow-up")] and _env == []


def test_compression_queues_with_its_reason_and_calls_no_verb(_env):
    server._session_compression_in_flight = lambda session: True
    agent = Agent()
    result = _submit(_session(agent, busy_input_mode="interrupt"))
    assert (result["status"], result["reason"]) == ("queued", "compression_in_flight")
    assert agent.calls == [] and _env == []


def test_the_build_window_queues_without_an_interrupt(_env):
    result = _submit(_session(None, busy_input_mode="interrupt"))
    assert (result["status"], result["reason"]) == ("queued", "agent_building")
    assert _env == []


def test_a_media_payload_queues_as_non_text(_env):
    session = _session(Agent(), attached_images=["/tmp/a.png"])
    result = _submit(session, text="look")
    assert (result["status"], result["reason"]) == ("queued", "non_text")
    assert _env == []


def test_a_session_busy_override_still_wins():
    agent = Agent()
    result = _submit(_session(agent, busy_input_mode="queue"))
    assert result["status"] == "queued" and result["reason"] == "queue_mode"
    assert agent.calls == []
    steer_agent = Agent()
    result = _submit(_session(steer_agent, busy_input_mode="steer"))
    assert result["status"] == "steered" and steer_agent.calls == [("steer", "follow-up")]


def test_a_queue_drain_stays_a_plain_queue():
    """``queued=True`` (the client's own queue drain) is never a correction, dispatcher or not."""
    agent = Agent()
    result = _submit(_session(agent), queued=True)
    assert result == {"status": "queued"} and agent.calls == []


def test_policy_inherit_leaves_the_classic_mode(monkeypatch, _env):
    monkeypatch.setattr(server, "_load_cfg", lambda: {"dispatcher": {"busy": {"policy": "inherit"}}})
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "interrupt")
    agent = Agent(redirect=False)
    result = _submit(_session(agent))
    assert result == {"status": "queued"} and "disposition" not in result
    assert len(_env) == 1, "the classic interrupt mode keeps its hard interrupt"


def test_a_non_dispatcher_session_is_unchanged(monkeypatch, _env):
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "interrupt")
    agent = Agent(redirect=False)
    result = _submit(_session(agent, dispatcher=False))
    assert result == {"status": "queued"} and len(_env) == 1
    assert agent.calls == [("redirect", "follow-up")]


def test_a_codex_native_redirect_that_timed_out_queues_like_main(_env):
    agent = Agent(redirect=False, api_mode="codex_app_server",
                  _codex_session=types.SimpleNamespace(request_steer=lambda t: True))
    result = _submit(_session(agent))
    assert result["status"] == "queued" and agent.calls == [("redirect", "follow-up")]


# ── session.redirect / session.steer RPCs ────────────────────────────────────────────

def _rpc(name, params):
    return server._methods[name]("rid", params)["result"]


def test_session_redirect_falls_to_steer_on_a_dispatcher_session(monkeypatch):
    agent = Agent(redirect=False, steer=True)
    session = _session(agent)
    monkeypatch.setitem(server._sessions, "sid", session)
    try:
        result = _rpc("session.redirect", {"session_id": "sid", "text": "now this"})
    finally:
        server._sessions.pop("sid", None)
    assert (result["status"], result["disposition"]) == ("queued", "steered")
    assert [c[0] for c in agent.calls] == ["redirect", "steer"]


def test_session_redirect_on_a_classic_session_keeps_its_refusal(monkeypatch):
    agent = Agent(redirect=False)
    monkeypatch.setitem(server._sessions, "sid", _session(agent, dispatcher=False))
    try:
        result = _rpc("session.redirect", {"session_id": "sid", "text": "now this"})
    finally:
        server._sessions.pop("sid", None)
    assert result["status"] == "rejected" and [c[0] for c in agent.calls] == ["redirect"]


def test_session_redirect_during_compression_says_why(monkeypatch):
    monkeypatch.setattr(server, "_session_compression_in_flight", lambda session: True)
    monkeypatch.setitem(server._sessions, "sid", _session(Agent()))
    try:
        result = _rpc("session.redirect", {"session_id": "sid", "text": "now this"})
    finally:
        server._sessions.pop("sid", None)
    assert (result["status"], result["disposition"], result["reason"]) == (
        "queued", "queued_next_turn", "compression_in_flight")


# ── config.get / config.set busy ───────────────────────────────────────────────────

def _config(method, **params):
    return server._methods[method]("rid", params)


def test_config_get_busy_reports_the_ladder_for_a_dispatcher_session(monkeypatch):
    monkeypatch.setitem(server._sessions, "sid", _session(Agent()))
    try:
        assert _config("config.get", key="busy", session_id="sid")["result"] == {"value": "interrupt"}
        assert _config("config.get", key="busy")["result"] == {"value": "queue"}  # sessionless: the profile's mode
    finally:
        server._sessions.pop("sid", None)


def test_config_set_busy_on_a_dispatcher_session_is_session_scoped(monkeypatch):
    written = []
    monkeypatch.setattr(server, "_write_config_key", lambda key, value: written.append((key, value)))
    session = _session(Agent())
    monkeypatch.setitem(server._sessions, "sid", session)
    try:
        assert _config("config.set", key="busy", value="queue", session_id="sid")["result"]["value"] == "queue"
        assert written == [], "display.busy_input_mode was rewritten for every other session"
        assert _config("config.get", key="busy", session_id="sid")["result"] == {"value": "queue"}
    finally:
        server._sessions.pop("sid", None)


def test_config_set_busy_on_a_classic_session_still_writes_the_profile(monkeypatch):
    written = []
    monkeypatch.setattr(server, "_write_config_key", lambda key, value: written.append((key, value)))
    monkeypatch.setitem(server._sessions, "sid", _session(Agent(), dispatcher=False))
    try:
        _config("config.set", key="busy", value="steer", session_id="sid")
    finally:
        server._sessions.pop("sid", None)
    assert written == [("display.busy_input_mode", "steer")]
