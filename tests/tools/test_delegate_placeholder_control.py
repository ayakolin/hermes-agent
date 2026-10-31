"""A dispatcher task has no child while it waits: it must still be listed, steered and stopped from every surface
(delegate_task control actions, the TUI RPC projection, the Desktop sidecar)."""

from __future__ import annotations

import pytest

from tools.delegate_tool_registry import (
    _active_subagents, _active_subagents_lock, _handle_control_action, register_placeholder,
    steer_authorized_record, take_early_steers,
)


class _Parent:
    def __init__(self, session_id: str = "sess-1") -> None:
        self.session_id = session_id
        self._session_db = None


def _placeholder(**overrides):
    fields = dict(
        subagent_id="sa-queued", parent_id=None, depth=0, goal="wait for a slot", delegation_id="deleg_q",
        model="m", started_at=1000.0, status="queued", phase="queued", queue_position=1, tool_count=0,
        last_tool=None, accepting_steer=True, owner_agent_session_id="sess-1",
    )
    fields.update(overrides)
    register_placeholder(**fields)
    with _active_subagents_lock:
        return _active_subagents[fields["subagent_id"]]


@pytest.fixture(autouse=True)
def clean_registry():
    with _active_subagents_lock:
        _active_subagents.clear()
    yield
    with _active_subagents_lock:
        _active_subagents.clear()


def test_list_reports_the_place_in_line_not_a_silent_clock():
    from tools.delegate_tool_registry import _list_payload
    _placeholder()
    payload = _list_payload(_Parent())
    entry = payload["subagents"][0]
    assert entry["status"] == "queued" and entry["phase"] == "queued" and entry["queue_position"] == 1
    assert "running_seconds" not in entry and "waiting_seconds" in entry
    assert payload["queued"] == 1 and payload["running"] == 0


def test_steer_is_parked_until_the_child_exists_and_can_be_ignored_by_a_stop():
    _placeholder()
    out = _handle_control_action("steer", "sa-queued", "prefer the smaller fix", _Parent())
    assert '"status": "queued"' in out, out
    assert take_early_steers("sa-queued") == ["prefer the smaller fix"]
    assert take_early_steers("sa-queued") == []  # consumed once


def test_stop_goes_through_the_placeholder_cancel_and_reports_missed_steer():
    stopped: list = []
    record = _placeholder(cancel_fn=lambda: stopped.append(True) or True)
    _handle_control_action("steer", "sa-queued", "land it as one commit", _Parent())
    out = _handle_control_action("stop", "sa-queued", None, _Parent())
    assert stopped == [True]
    assert '"status": "interrupt_requested"' in out
    assert "will not start" in out
    text, missed = record.get("pending_steers"), None
    assert text == ["land it as one commit"] and missed is None


def test_desktop_steer_uses_the_authorized_snapshot():
    _placeholder()
    with _active_subagents_lock:
        snapshot = dict(_active_subagents["sa-queued"])
    assert steer_authorized_record("sa-queued", snapshot, "keep it small") is True
    assert take_early_steers("sa-queued") == ["keep it small"]
    with _active_subagents_lock:
        _active_subagents["sa-queued"]["agent"] = object()  # the record now belongs to a live child
    assert steer_authorized_record("sa-queued", snapshot, "too late") is False


def test_a_placeholder_never_appears_in_the_public_snapshot_with_private_state():
    from tools.delegate_tool_registry import list_active_subagents
    _placeholder(cancel_fn=lambda: True)
    row = list_active_subagents()[0]
    assert "cancel_fn" not in row and "pending_steers" not in row and "owner_session_id" not in row
