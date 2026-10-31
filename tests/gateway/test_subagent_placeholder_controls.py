"""Desktop's sidecar Steer / Stop on a dispatcher task that has no child yet (a queued or still-building unit)."""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import pytest

from gateway import session_ancillary
from tools.delegate_tool_registry import _active_subagents, _active_subagents_lock, register_placeholder


class _Authority:
    def __init__(self, agent) -> None:
        self.sessions = {"ui-1": SimpleNamespace(route="route-1", event_stream=SimpleNamespace(lock=threading.Lock()))}
        self._agent = agent
        self.db = None

    def authorize(self, actor, ref, capability) -> None:
        return None

    def agent(self, ref):
        return self._agent


@pytest.fixture
def desktop(monkeypatch):
    with _active_subagents_lock:
        _active_subagents.clear()
    monkeypatch.setattr(session_ancillary, "_require_current_route", lambda *a, **k: None)
    monkeypatch.setattr(session_ancillary, "physical_target", lambda authority, ref: "ui-1")
    monkeypatch.setattr("gateway.session_managed_worker.managed_policy", lambda authority, ref: None)
    # The roster rule is exercised directly (a real authority store is not needed to prove control reaches the task).
    monkeypatch.setattr(session_ancillary, "_owner_objects",
                        lambda authority, ref, **_k: (authority.agent(ref), session_ancillary.subagent_records(authority.agent(ref))))
    connection = SimpleNamespace(authority=_Authority(agent=None), actor=SimpleNamespace(roles=()))
    yield connection
    with _active_subagents_lock:
        _active_subagents.clear()


def _queued(cancelled: list):
    """A task whose record is reachable by lineage (the roster rule), exactly as a queued dispatch publishes it."""
    register_placeholder(
        subagent_id="sa-q", parent_id=None, depth=0, goal="waiting", delegation_id="deleg_q", model="m",
        started_at=1000.0, status="queued", phase="queued", queue_position=1, tool_count=0, last_tool=None,
        accepting_steer=True, owner_agent_session_id="ui-1", cancel_fn=lambda: cancelled.append(True) or True)
    record = _active_subagents["sa-q"]
    record["owner_db_path"] = "same.db"  # the lineage rule compares the child's SessionDB path
    return record


def _parent_agent():
    """The session's live agent: same durable id and db file as the queued record (the lineage rule)."""
    return SimpleNamespace(session_id="ui-1", _session_db=SimpleNamespace(db_path="same.db"),
                           _delegate_parent_ref=None, gateway_session_key=None)


def test_desktop_stop_cancels_a_task_that_has_no_child(desktop):
    cancelled: list = []
    _queued(cancelled)
    desktop.authority._agent = SimpleNamespace(
        session_id="ui-1", _session_db=SimpleNamespace(db_path="same.db"), _delegate_parent_ref=None,
        gateway_session_key=None)
    ref = SimpleNamespace(session_id="ui-1")
    stop = asyncio.run(session_ancillary.subagent_control(
        desktop, ref, {"subagent_id": "sa-q"}, steer=False))
    assert stop == {"found": True, "subagent_id": "sa-q"}
    assert cancelled == [True]


def test_desktop_steer_parks_text_for_a_task_that_has_no_child(desktop):
    from tools.delegate_tool_registry import take_early_steers

    cancelled: list = []
    _queued(cancelled)
    desktop.authority._agent = SimpleNamespace(
        session_id="ui-1", _session_db=SimpleNamespace(db_path="same.db"), _delegate_parent_ref=None,
        gateway_session_key=None)
    ref = SimpleNamespace(session_id="ui-1")
    steer = asyncio.run(session_ancillary.subagent_control(
        desktop, ref, {"subagent_id": "sa-q", "text": "keep it small"}, steer=True))
    assert steer == {"status": "queued", "subagent_id": "sa-q", "text": "keep it small"}
    assert take_early_steers("sa-q") == ["keep it small"]
