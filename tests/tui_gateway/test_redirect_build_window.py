"""Legacy tui_gateway redirects during the async agent build window.

Split out of ``test_tui_gateway_server.py`` (which may only shrink): a correction that lands while a fresh turn is
still building its agent must queue losslessly for the next turn, carrying the three-state reply contract —
``status`` plus ``reason`` and ``disposition`` — instead of the bare queued line it used to answer with.
"""

from __future__ import annotations

import threading
import types

from tui_gateway import server


def _session(agent=None, **extra):
    return {
        "agent": agent if agent is not None else types.SimpleNamespace(),
        "session_key": "session-key",
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": False,
        "attached_images": [],
        "image_counter": 0,
        "cols": 80,
        "slash_worker": None,
        "show_reasoning": False,
        "tool_progress_mode": "all",
        **extra,
    }



def test_session_redirect_build_window_scrubs_stale_p_when_queuing_q():
    """#84417: build-window queue of Q must not leave P ahead of Q."""
    original = "live original P"
    session = _session(running=True)
    session["agent"] = None  # async agent build window
    session["inflight_turn"] = {
        "user": original,
        "assistant": "",
        "streaming": True,
        "error": "",
    }
    session["queued_prompt"] = {"text": original, "transport": "ws-1"}
    server._sessions["sid"] = session
    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "session.redirect",
                "params": {"session_id": "sid", "text": "correction Q"},
            }
        )
    finally:
        server._sessions.pop("sid", None)

    assert resp["result"] == {"status": "queued", "text": "correction Q", "reason": "agent_building",
                                   "disposition": "queued_next_turn"}
    assert session["queued_prompt"]["text"] == "correction Q"
    assert not session.get("queued_prompts")




def test_session_redirect_queues_during_agent_build_window(monkeypatch):
    # A fresh turn flips running=True and builds the agent asynchronously, so
    # session["agent"] is briefly None. A correction landing here must queue
    # (lossless, reaches the model next turn), not hard-reject as unsupported.
    session = _session(running=True)
    session["agent"] = None
    server._sessions["sid"] = session
    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "session.redirect",
                "params": {"session_id": "sid", "text": "wait, use SQLite"},
            }
        )
    finally:
        server._sessions.pop("sid", None)

    assert resp["result"] == {"status": "queued", "text": "wait, use SQLite", "reason": "agent_building",
                                   "disposition": "queued_next_turn"}
    assert session["queued_prompt"]["text"] == "wait, use SQLite"


