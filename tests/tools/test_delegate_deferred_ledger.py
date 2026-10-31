"""Ledger ordering for a deferred (dispatcher) dispatch.

The dispatch row is written by the unit's worker, after the handle has already returned; its completion can then be
written before a slow dispatch write lands. The dispatch INSERT is conditional (once per unit), so that late write
can never reset a settled row to 'running' and have restart replay a finished unit.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from tools import async_delegation as ad


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    ad._reset_for_tests()
    yield tmp_path
    ad._reset_for_tests()


def _record(delegation_id: str) -> dict:
    return ad._new_record(
        delegation_id=delegation_id, goal="g", goals=["g"], context=None, toolsets=None, role="leaf", model="m",
        session_key="s", origin_ui_session_id="", origin_session_id="", parent_session_id="p", interrupt_fn=None,
        progress_fn=None, slot_key=delegation_id, task_transcripts={}, task_indexes=[0])


def _row(tmp_path, delegation_id: str):
    with sqlite3.connect(tmp_path / "state.db") as conn:
        return conn.execute(
            "SELECT state, delivery_state, event_json FROM async_delegations WHERE delegation_id=?",
            (delegation_id,)).fetchone()


def test_a_late_dispatch_write_does_not_regress_a_settled_row(ledger):
    record = _record("deleg_ordered")
    event = {"delegation_id": "deleg_ordered", "status": "completed", "completed_at": 2.0}
    ad.persist_dispatch(record)  # what the worker writes first in the normal order
    ad._persist_completion(event, {"status": "completed", "summary": "done"})
    assert _row(ledger, "deleg_ordered")[0] == "completed"

    ad.persist_dispatch(record)  # the same unit's write arriving again, after the completion
    state, delivery_state, event_json = _row(ledger, "deleg_ordered")
    assert state == "completed" and delivery_state == "pending", (state, delivery_state)
    assert json.loads(event_json)["status"] == "completed"


def test_dispatch_first_then_completion_is_the_normal_order(ledger):
    record = _record("deleg_normal")
    ad.persist_dispatch(record)
    assert _row(ledger, "deleg_normal")[:2] == ("running", "pending")
    ad._persist_completion({"delegation_id": "deleg_normal", "status": "completed"}, {"status": "completed"})
    assert _row(ledger, "deleg_normal")[0] == "completed"
