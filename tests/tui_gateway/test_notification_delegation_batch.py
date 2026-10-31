"""The legacy TUI poller (Ink, Desktop) batches async-delegation completions into one turn.

Two subagents finishing together must not cost two turns on a surface whose every other completion
run already coalesces one turn per ready snapshot. Each event's durable delivery row is claimed on its
own (a row owned elsewhere stays out of the text), and an accepted submit settles all of them.
"""

from __future__ import annotations

import contextlib
import queue
import threading
import time

import pytest

from tui_gateway import server, session_notifications
from tools.process_registry_awaiting import release
from tui_gateway.method_ctx import rebind


def _delegation(delegation_id: str, *, summary: str = "done", owner_task_id: str = "") -> dict:
    event = {"type": "async_delegation", "delegation_id": delegation_id, "session_key": "chat",
             "status": "completed", "summary": summary, "api_calls": 1, "duration_seconds": 1.0,
             "dispatched_at": 1.0, "completed_at": 2.0}
    if owner_task_id:
        event["owner_task_id"] = owner_task_id  # the reapers key the in-flight hold on it
    return event


def _namespace(monkeypatch, *, owned_elsewhere=(), submit=True):
    """A rebound module namespace where the durable ledger and the submit are observable."""
    journal: list = []
    state = {"submitted": [], "settled": []}

    def claim(event, consumer):
        # ``owned_elsewhere``: rows another consumer already holds (a gateway sharing this home)
        if event["delegation_id"] in owned_elsewhere:
            return None
        return f"claim-{event['delegation_id']}"

    def submit_prompt(rid, sid, session, text, what, **kwargs):
        state["submitted"].append((text, kwargs))
        session["running"] = False
        return submit

    def claim_turn(session, **_kwargs):
        session["running"] = True
        return True

    namespace = dict(time=time, threading=threading,
                     _notif_release_turn=session_notifications._notif_release_turn,
                     _notif_refusal_backoff=lambda session: None,
                     _notif_log_failure=lambda *a: None,
                     _notif_submit=submit_prompt,
                     _notif_defer_event=lambda evt, claim, put=None: journal.append(("deferred", evt)),
                     _notif_claim_turn=claim_turn,
                     # module-level helpers the rebound body calls (`rebind` copies only explicit imports)
                     _delegation_batch_text=session_notifications._delegation_batch_text,
                     _notif_dispatch_delegations=session_notifications._notif_dispatch_delegations)
    import tools.async_delegation as ledger
    monkeypatch.setattr(ledger, "claim_event_delivery", claim)  # the function import re-reads the module
    monkeypatch.setattr(ledger, "complete_event_delivery",
                        lambda evt, c: state["settled"].append(evt["delegation_id"]))
    monkeypatch.setattr(ledger, "release_event_delivery", lambda evt, c: journal.append(("released", evt)))
    monkeypatch.setattr(server, "_emit", lambda *a, **k: None)
    monkeypatch.setattr(server, "_session_profile_runtime_scope", lambda s: contextlib.nullcontext())
    return namespace, state, journal


def _session() -> dict:
    return {"history_lock": threading.RLock(), "running": False, "session_key": "chat"}


def _dispatch(namespace):
    # A fresh ``_seen`` per test: ``rebind`` memoises by function id, so a shared one would reuse the
    # first test's globals.
    return rebind(session_notifications._notif_dispatch_delegations, namespace, _seen={})


def test_two_delegations_in_one_ready_run_make_one_turn(monkeypatch):
    namespace, state, _journal = _namespace(monkeypatch)
    dispatch = _dispatch(namespace)

    dispatch("sid", _session(), [(_delegation("deleg-a"), "text A"), (_delegation("deleg-b"), "text B")],
             None, None, complete=False)

    assert len(state["submitted"]) == 1, "one ready run is one turn"
    text = state["submitted"][0][0]
    assert "text A" in text and "text B" in text
    assert state["settled"] == ["deleg-a", "deleg-b"], "every claimed row settles on acceptance"


def test_a_row_owned_elsewhere_stays_out_of_the_turn(monkeypatch):
    namespace, state, _journal = _namespace(monkeypatch, owned_elsewhere={"deleg-a"})
    dispatch = _dispatch(namespace)

    dispatch("sid", _session(), [(_delegation("deleg-a"), "text A"), (_delegation("deleg-b"), "text B")],
             None, None, complete=False)

    text = state["submitted"][0][0]
    assert "text B" in text and "text A" not in text
    assert state["settled"] == ["deleg-b"]


def test_a_refused_submit_defers_every_claimed_row(monkeypatch):
    namespace, state, journal = _namespace(monkeypatch, submit=False)
    dispatch = _dispatch(namespace)

    dispatch("sid", _session(), [(_delegation("deleg-a"), "A"), (_delegation("deleg-b"), "B")],
             _Registry(), None, complete=False)

    assert [evt["delegation_id"] for kind, evt in journal if kind == "deferred"] == ["deleg-a", "deleg-b"]
    assert state["settled"] == [], "a refused submit settles nothing"


def test_a_mixed_run_does_not_go_through_the_process_completion_renderer(monkeypatch):
    """``_notif_dispatch_completions`` routes any run holding a delegation to the delegation path."""
    namespace, _state, _journal = _namespace(monkeypatch)
    routed = []
    namespace["_notif_dispatch_delegations"] = lambda *a, **k: routed.append((a, k))
    dispatch = rebind(session_notifications._notif_dispatch_completions, namespace, _seen={})

    dispatch("sid", _session(), [({"type": "completion", "session_id": "p"}, "P"),
                                 (_delegation("deleg-a"), "A")], None, None)

    assert routed and routed[0][1]["complete"] is False



class _Clock:
    """``time`` for the poller: opening the batch window lets the second producer's event land."""

    def __init__(self):
        self.sleeps: list[float] = []
        self.on_sleep = lambda: None

    def monotonic(self) -> float:
        return 0.0

    def time(self) -> float:
        return 0.0

    def sleep(self, delay: float) -> None:
        self.sleeps.append(delay)
        callback, self.on_sleep = self.on_sleep, lambda: None
        callback()


def test_the_poller_takes_a_second_snapshot_after_the_batch_window(monkeypatch):
    """A completion that lands while the poller holds the first one is taken with it after the window,
    so one window yields ONE ready run (and one turn) instead of two."""
    from tools.process_registry import process_registry
    stop = threading.Event()
    events = process_registry.completion_queue  # the queue the loop really reads
    while not events.empty():
        events.get_nowait()
    events.put(_delegation("deleg-a", owner_task_id="owner-a"))
    handled: list = []
    clock = _Clock()

    def window_for(_session):
        clock.on_sleep = lambda: events.put(_delegation("deleg-b", owner_task_id="owner-b"))
        return 0.3

    reads = []

    def take(queue, _timeout, accept=None):
        # The production read and in-flight hold. Its `timeout > 0` read waits for a first event, so a
        # drained queue would park here: after the window (the second read) this ends the test instead.
        from tools.process_registry_awaiting import take_ready
        reads.append(_timeout)
        if len(reads) > 2:  # a third pass would park on the drained queue; the test ends here
            stop.set()
            return []
        return take_ready(queue, 0.0, accept=accept)

    # ``rebind`` copies only the names this loop imports itself, so the seam it really consults is the
    # module function; make it ours for the duration of the test.
    monkeypatch.setattr(session_notifications, "_take_and_hold", lambda: (take, release), raising=False)

    def route(_sid, _session, evt, *_a, **_k):
        handled.append(evt.get("owner_task_id"))  # one routed event = one would-be turn
        return True

    real_handle_ready = session_notifications._notif_handle_ready

    def handle_ready(_sid, session, ready, *a, **k):
        real_handle_ready(_sid, session, ready, *a, **k)  # real grouping/routing over the run
        stop.set()  # this test is about the snapshot, not a second pass

    module = session_notifications
    for name, value in {
        "time": clock, "_LOOP_POLL_SECONDS": 0, "_KANBAN_POLL_SECONDS": 0,
        "_maybe_fire_tui_loop_tick": lambda *a: None, "_maybe_fire_tui_heartbeat_tick": lambda *a: None,
        "_notif_poll_kanban": lambda *a: None,
        "_session_profile_runtime_scope": lambda s: contextlib.nullcontext(),
        "_notif_log_failure": lambda what, exc: handled.append(("error", what, repr(exc))),
        "_completion_batch_window_seconds": window_for, "_completion_window_for": window_for,
        "_notif_handle_event": route, "_notif_handle_ready": handle_ready,
        "_take_and_hold": lambda: (take, release),  # module attr patched above; kept for clarity
        "process_registry": _Namespace(completion_queue=events, restore_completions=lambda: 0),
        "format_process_notification": lambda _evt: "text",
    }.items():
        monkeypatch.setattr(module, name, value, raising=False)  # injected globals may be absent here
    monkeypatch.setattr("tools.async_delegation.maybe_sweep_orphaned_completions", lambda _q: None)

    module._notification_poller_scoped_loop(
        stop, "sid", {"agent": object(), "history_lock": threading.RLock(), "session_key": "chat"})

    assert handled == ["owner-a", "owner-b"], "both snapshots were routed in one ready run"
    assert clock.sleeps == [0.3], "the poller waited the configured window before the second snapshot"

class _Namespace:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _Registry:
    """Stands in for the process registry: a refusal hands the event back to its queue."""

    def __init__(self):
        self.completion_queue: queue.Queue = queue.Queue()
