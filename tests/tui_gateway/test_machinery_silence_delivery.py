"""A machinery turn's bare silence marker never reaches an Ink/Desktop transcript.

The gateway has always dropped it for a background completion / watch notice turn; the legacy TUI path
(which the Ink client and the Desktop app drive) does not go through the gateway's delivery, and it used
to strip the marker only for Bot Chat sessions. The notice text now asks the model for the marker, so
every consumer of that text must drop it, and only a successful turn may.
"""

import contextlib
from types import SimpleNamespace

import tui_gateway.server as srv


def _turn(result):
    return SimpleNamespace(
        result=result, agent=SimpleNamespace(_session_title_hint="Scratch"), terminal_callback=None,
        receipt_committed=True, receipt_attempted=False, marker_key="", error_retained=False,
        error_detail="", prompt_text="ping",
    )


def _session(monkeypatch):
    monkeypatch.setattr(srv, "_get_usage", lambda _agent: {})
    monkeypatch.setattr(srv, "render_message", lambda _text, _cols: None)
    monkeypatch.setattr(srv, "_clear_inflight_turn", lambda _session: None)
    monkeypatch.setattr(srv, "_session_live_title", lambda _s, _k: "Scratch")
    return {"pending_title": None, "session_key": "k", "history_lock": contextlib.nullcontext(),
            "agent": SimpleNamespace(_session_title_hint="Scratch")}


def test_a_silenced_completion_turn_delivers_no_text(monkeypatch):
    session = _session(monkeypatch)
    payload, _, status = srv._complete_turn_payload(
        session, _turn({"final_response": " [SILENT] "}), None, 80, "async_delegation_complete")
    assert (status, payload["text"]) == ("complete", "")


def test_a_human_turn_on_the_same_surface_keeps_its_answer(monkeypatch):
    """Only machinery kinds are stripped here: a human turn's marker is the gateway's decision."""
    session = _session(monkeypatch)
    payload, _, _ = srv._complete_turn_payload(
        session, _turn({"final_response": "NO_REPLY"}), None, 80, None)
    assert payload["text"] == "NO_REPLY"


def test_prose_that_mentions_a_marker_is_delivered(monkeypatch):
    session = _session(monkeypatch)
    prose = "[SILENT] is how you stay quiet; here is the actual result."
    payload, _, _ = srv._complete_turn_payload(
        session, _turn({"final_response": prose}), None, 80, "async_delegation_complete")
    assert payload["text"] == prose


def test_a_failed_completion_turn_is_never_silenced(monkeypatch):
    """A failure must be visible even if the model wrote a marker over it."""
    session = _session(monkeypatch)
    payload, _, status = srv._complete_turn_payload(
        session, _turn({"final_response": "NO_REPLY", "error": "provider failed", "failed": True}),
        None, 80, "async_delegation_complete")
    assert status == "error" and payload["text"] != ""
