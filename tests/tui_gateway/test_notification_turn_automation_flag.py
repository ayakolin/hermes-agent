"""``session["_turn_automation"]`` is True exactly while a queue-driven notification turn holds ``running``.

A busy submit reads it to queue behind a completion turn instead of hard-interrupting it. It is set under
``history_lock`` together with ``running`` (a flag set after the submit returned would be set after a fast
turn had already ended, leaving it True on an idle session so the next human busy submit degraded to queue),
and cleared wherever ``running`` is released.
"""

from __future__ import annotations

import threading
import types

import pytest

from tui_gateway import server


class _InlineThread:
    """Run the turn synchronously: it has ended before ``_run_prompt_submit`` returns."""

    def __init__(self, target=None, daemon=None, args=(), kwargs=None, name=None):
        self._target, self._args, self._kwargs = target, args, kwargs or {}

    def start(self):
        if self._target is not None:
            self._target(*self._args, **self._kwargs)

    def is_alive(self):
        return False

    def join(self, timeout=None):
        return None


@pytest.fixture()
def turn_env(monkeypatch, tmp_path):
    monkeypatch.setattr(server.threading, "Thread", _InlineThread)
    monkeypatch.setattr(server, "_emit", lambda *a, **k: None)
    monkeypatch.setattr(server, "_wire_callbacks", lambda sid: None)
    monkeypatch.setattr(server, "_sync_agent_model_with_config", lambda sid, session: None)
    monkeypatch.setattr(server, "_session_cwd", lambda session: str(tmp_path))
    monkeypatch.setattr(server, "_register_session_cwd", lambda session: None)
    monkeypatch.setattr(server, "_tts_stream_begin", lambda: None)
    monkeypatch.setattr(server, "_sync_session_key_after_compress", lambda *a, **k: None)
    monkeypatch.setattr(server, "_get_usage", lambda agent: {})


def _idle_session(seen: list):
    def run_conversation(*_a, **_k):
        seen.append((session["running"], session.get("_turn_automation")))
        return {"final_response": "ok"}
    agent = types.SimpleNamespace(
        session_id="agent-sid", run_conversation=run_conversation, clear_interrupt=lambda: None)
    session = {
        "agent": agent, "session_key": "gw-session-key", "history": [], "history_lock": threading.RLock(),
        "history_version": 0, "running": False, "attached_images": [], "image_counter": 0, "cols": 80,
        "slash_worker": None, "show_reasoning": False, "tool_progress_mode": "all", "inflight_turn": None,
    }
    return session


def test_a_queue_driven_turn_that_ends_before_the_submit_returns_leaves_no_flag(turn_env):
    seen: list = []
    session = _idle_session(seen)
    assert server._notif_claim_turn(session, automation=True)
    assert session["_turn_automation"] is True, "set with ``running``, before any submit can run the turn"

    assert server._notif_submit("rid", "ui", session, "a completion", "dispatch failed") is True

    assert seen == [(True, True)], "the turn itself saw the flag"
    assert session["running"] is False
    assert "_turn_automation" not in session, "a stale flag on an idle session degrades the next human submit"


def test_a_human_turn_after_a_queue_driven_one_does_not_inherit_the_flag(turn_env):
    seen: list = []
    session = _idle_session(seen)
    assert server._notif_claim_turn(session, automation=True)
    server._notif_submit("rid", "ui", session, "a completion", "dispatch failed")

    session["running"] = True  # prompt.submit's claim
    assert server._run_prompt_submit("rid2", "ui", session, "hello") is True

    assert seen[-1] == (True, None)


def test_a_claim_that_is_not_queue_driven_clears_a_leftover_flag(turn_env):
    """The agent's own scheduled work (a /loop or /heartbeat tick) stays interruptible."""
    session = _idle_session([])
    session["_turn_automation"] = True  # a leftover from a release path that never ran
    assert server._notif_claim_turn(session)
    assert "_turn_automation" not in session


def test_a_refused_submit_releases_the_flag_with_running(monkeypatch):
    monkeypatch.setattr(server, "_run_prompt_submit", lambda *a, **k: False)
    session = {"history_lock": threading.RLock(), "running": False}
    assert server._notif_claim_turn(session, automation=True)

    assert server._notif_submit("rid", "ui", session, "tick", "failed") is False

    assert session["running"] is False and "_turn_automation" not in session
