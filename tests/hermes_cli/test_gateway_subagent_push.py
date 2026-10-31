"""A pushed ``subagent.*`` event makes the classic CLI re-read the roster at once.

The dock's own poll is gated on a running turn, a listed child, a non-zero active count or a watch window;
just before a spawn none of those hold. The push therefore calls ``GatewayProcessView.watch_subagents()`` —
which sets the watch window and zeroes the 1 s throttle — through ``process_view()`` resolved at call time,
because ``bind`` replaces the view on every rebind.
"""
from types import SimpleNamespace

from hermes_cli import gateway_tools
from hermes_cli.gateway_connection import GatewayConnection
from tests.hermes_cli.test_gateway_agent import SID, FakeClient, connect_with


def _view(rows=None):
    """A stand-in view whose ``subagent.list`` answers ``rows`` and counts the reads."""
    view = SimpleNamespace(rows=list(rows or ()), reads=0, watched=0,
                           connection=SimpleNamespace(final=False))
    view.watch_subagents = lambda seconds=60: setattr(view, "watched", view.watched + 1)
    view.roster_bind_session = lambda sid, previous: view.rows.clear()
    return view


def _connection():
    conn = GatewayConnection(connect=connect_with(FakeClient([])))
    conn.start(5)
    conn.state(SID)
    return conn


def _register(conn, view, monkeypatch):
    """Register the CLI's listener for ``SID`` on a stub view (what ``bind_session`` wires in production)."""
    monkeypatch.setattr(gateway_tools, "_view", view)
    conn.state(SID).roster_listeners.append(lambda: gateway_tools.register_roster_push(SID, SID))


def test_a_pushed_subagent_event_re_arms_the_watch(monkeypatch):
    conn = _connection()
    view = _view()
    try:
        _register(conn, view, monkeypatch)
        gateway_tools.register_roster_push(SID, SID)
        assert view.watched == 1
    finally:
        conn.close()


def test_the_listener_is_registered_once_per_session_state(monkeypatch):
    conn = _connection()
    view = _view()
    monkeypatch.setattr(gateway_tools, "_view", view)
    try:
        gateway_tools.bind_session(SimpleNamespace(session_id=SID), conn)
        gateway_tools.bind_session(SimpleNamespace(session_id=SID), conn)  # a rebind
        state = conn.state(SID)
        assert len(state.roster_listeners) == 1
    finally:
        conn.close()


def test_an_event_on_the_pump_path_fires_the_listener(monkeypatch):
    """``apply_event`` runs on the loop thread as frames are pumped: the listener must only mark state."""
    conn = _connection()
    view = _view()
    try:
        _register(conn, view, monkeypatch)
        state = conn.state(SID)
        state.apply_event("subagent.start", {"subagent_id": "sa-1"})
        state.apply_event("subagent.complete", {"subagent_id": "sa-1", "status": "completed"})
        assert view.watched == 2
        state.apply_event("message.delta", {"text": "x"})  # not a roster event
        assert view.watched == 2
    finally:
        conn.close()


def test_a_moved_active_subagent_count_also_re_arms(monkeypatch):
    """``session.usage.active_subagents`` moves between turns too: a completion that carries only the
    count still refreshes the dock."""
    conn = _connection()
    view = _view()
    try:
        _register(conn, view, monkeypatch)
        state = conn.state(SID)
        state.apply_event("session.usage", {"active_subagents": 1, "api_calls": 2})
        assert view.watched == 1
        state.apply_event("session.usage", {"active_subagents": 1, "api_calls": 3})  # unchanged
        assert view.watched == 1
    finally:
        conn.close()


def test_without_a_view_the_push_is_a_no_op():
    gateway_tools.register_roster_push(SID, SID)  # process_view() is None off the gateway
