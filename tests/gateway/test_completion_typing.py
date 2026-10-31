"""A completion turn shows typing on a chat platform; a heartbeat/watch wake does not.

The gateway's completion lane answers a finished job through the same adapter path every other turn
takes, so a chat that shows typing for a human turn must show it for this one too (before, the
completion turn was the only turn that went silent-typing). The lane is named by the producer's
notice (``automation_lane``), carried on the stored admission, restored onto the executing event, and
turned into a typing task by the executor's bracket when the row starts running.
"""

from __future__ import annotations


import pytest

from gateway.config import Platform
from gateway.platforms.event import MessageEvent
from gateway.session_automation import admit_process_notice
from gateway.session_producer_lifecycle import mark_producer_admission
from tests.gateway.test_automation_notification_category import _native_owner


class _TypingAdapter:
    """A chat adapter that DOES override ``send_typing`` (the bracket's type check)."""

    def __init__(self):
        self.started = []
        self.stopped = []
        self.hooks = []

    async def send_typing(self, chat_id, *args, **kwargs):
        return None

    def _start_typing_refresh(self, event, stop, metadata):
        self.started.append(event)

    async def _stop_typing_refresh(self, chat_id, task, metadata=None):
        self.stopped.append(chat_id)

    async def _run_processing_hook(self, name, event, *args):
        self.hooks.append(name)


def _completion(route, event_type="async_delegation"):
    return {"type": event_type, "delegation_id": "deleg-1", "session_key": route,
            "summary": "done", "status": "completed", "dispatched_at": 1.0, "completed_at": 2.0}


async def _rows(authority, owner):
    from hermes_state_runtime import list_session_admissions
    return list_session_admissions(authority.db, session_id=owner, pending_only=False)


def _automation_descriptor(payload):
    return payload.get("local_automation_v1") or (payload.get("native_text_v1") or {}).get("automation") or {}


@pytest.mark.asyncio
@pytest.mark.parametrize("event_type,expect_lane", [("async_delegation", "completion"),
                                                    ("watch_match", None),
                                                    ("heartbeat", None)])
async def test_the_notice_lane_is_stored_with_its_admission(tmp_path, monkeypatch, event_type, expect_lane):
    """A completion's admission carries ``lane: completion`` so the executor bracket (on the row, at
    execution, possibly after a restart) knows to type; a watch/heartbeat notice carries none."""
    runner, authority, adapter, source, metadata = await _native_owner(tmp_path, monkeypatch, "human-1")
    try:
        evt = _completion(metadata["gateway_session_key"], event_type)
        evt.update({**metadata, "session_id": "proc-1", "event_id": "evt-1",
                    "started_at": 1.0, "output": "READY", "pattern": "READY", "seq": 1})
        metadata_for_notice = {
            "gateway_session_key": metadata["gateway_session_key"],
            "gateway_session_id": metadata["gateway_session_id"],
        }
        if expect_lane:
            metadata_for_notice["automation_lane"] = expect_lane
        event = MessageEvent(text="[IMPORTANT: notice]", source=source, internal=True,
                             metadata=metadata_for_notice)

        await admit_process_notice(runner, authority, adapter, event, evt)

        owner = authority.logical_owner(metadata["gateway_session_id"])
        descriptor = _automation_descriptor((await _rows(authority, owner))[-1]["payload"])
        assert descriptor.get("lane") == expect_lane
    finally:
        authority.db.close()


@pytest.mark.asyncio
async def test_the_stored_lane_comes_back_on_the_executing_event(tmp_path, monkeypatch):
    """Admission stores the lane; execution may be a later task (or a restart), so the restored event
    must name it again or the bracket would type nothing."""
    from gateway.session_envelope import restore_native
    runner, authority, adapter, source, metadata = await _native_owner(tmp_path, monkeypatch, "human-1")
    try:
        evt = _completion(metadata["gateway_session_key"])
        evt.update({**metadata, "started_at": 1.0, "output": "READY"})
        event = MessageEvent(text="[IMPORTANT: notice]", source=source, internal=True, metadata={
            "gateway_session_key": metadata["gateway_session_key"],
            "gateway_session_id": metadata["gateway_session_id"], "automation_lane": "completion"})
        await admit_process_notice(runner, authority, adapter, event, evt)

        owner = authority.logical_owner(metadata["gateway_session_id"])
        row = (await _rows(authority, owner))[-1]
        assert _automation_descriptor(row["payload"])["lane"] == "completion"
        assert restore_native(row["payload"], runner).metadata["automation_lane"] == "completion"
        assert restore_native(row["payload"], runner).metadata["automation_lane"] == "completion"
    finally:
        authority.db.close()


@pytest.mark.asyncio
async def test_a_completion_report_is_written_with_the_completion_lane():
    """The producer side: an async-delegation notice names the lane, a process/watch notice does not."""
    from gateway.run_notifications import _watch_notification_metadata
    from gateway.session import SessionSource

    synth_source = SessionSource(platform=Platform.DISCORD, chat_id="42", chat_type="dm")
    assert _watch_notification_metadata(_completion("route"), synth_source)["automation_lane"] == "completion"
    for event_type in ("completion", "watch_match", "heartbeat"):
        assert "automation_lane" not in _watch_notification_metadata(_completion("route", event_type), synth_source)


@pytest.mark.asyncio
async def test_only_a_completion_lane_row_gets_a_typing_bracket(monkeypatch):
    """``_open`` types for a completion-lane automation row and never runs its processing hooks; a
    producer delivery keeps its hooks and typing as before."""
    from gateway import session_producer_lifecycle as lifecycle
    from gateway.platforms.event import MessageEvent

    adapter = _TypingAdapter()
    adapter._start_typing_refresh = lambda event, stop, metadata: adapter.started.append(event)
    runner = type("Runner", (), {"_adapter_for_source": lambda self, source: adapter})()
    authority = type("Authority", (), {"runner": runner, "producer_turns": {}})()

    def _event(lane):
        event = MessageEvent(text="t", source=type("S", (), {"chat_id": "42"})(), internal=True,
                             metadata={"automation_lane": lane} if lane else {})
        return event

    completion = _event("completion")
    mark_producer_admission(authority, completion, {"admission_id": "a1", "status": "queued"})
    await lifecycle._open(authority, {"admission_id": "a1"}, completion)
    assert adapter.started == [completion]
    assert adapter.hooks == [], "a completion turn is not a producer delivery: no processing hooks"

    # A producer delivery keeps typing AND its hooks, exactly as before.
    delivery = _event(None)
    delivery._producer_delivery = True
    monkeypatch.setattr("gateway.run_turn_followup_ack._followup_processing_hooks_apply", lambda a, e: True)
    mark_producer_admission(authority, delivery, {"admission_id": "a2", "status": "queued"})
    await lifecycle._open(authority, {"admission_id": "a2"}, delivery)
    assert adapter.started == [completion, delivery]
    assert adapter.hooks == ["on_processing_start"]

    # Any other automation row (a heartbeat/loop/kanban wake) is not bracketed at all.
    heartbeat = _event(None)
    mark_producer_admission(authority, heartbeat, {"admission_id": "a3", "status": "queued"})
    assert await lifecycle._open(authority, {"admission_id": "a3"}, heartbeat) is None
    assert adapter.started == [completion, delivery]
