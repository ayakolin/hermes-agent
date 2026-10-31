"""Deferred dispatch for dispatcher sessions: ``delegate_task(background=true)`` returns its handle before any child
exists, and the pool builds each child just before running it.

Phase A stays on the turn thread and touches no disk: every check that can fail (control action, spawn pause, depth,
task list, schemas, images, routes, one-shot budget, async delivery, queue capacity) returns the same immediate tool
error it always did, then the origin identity and each task's placeholder are published and the call is submitted.

Phase B (``_DeferredCall.start``, on the pool) writes the ledger rows, creates the live transcripts, builds children
one at a time and runs each unit through the ordinary ``_execute_and_aggregate`` finalizer — a construction failure is
one task's error entry and the dispatcher still receives a completion event.

Everything the worker needs from the turn — session ids, the transcript home, the captured Context, and the approval
and sudo callbacks — is captured at admission. By the time it runs the parent's turn is over and the session may have
rotated; and a unit promoted later by another thread must never supply that thread's callbacks.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, replace
from typing import Any, Optional

from hermes_cli.observability.shared_metrics_loop import begin_delegation_run, finish_delegation_unit
from tools.async_delegation import BEFORE_START
from tools.delegate_tool_child_run import _result_entry
from tools.delegate_tool_dispatch import _Batch, _batch_progress_token, _execute_and_aggregate
from tools.delegate_tool_progress import _quiet
from tools.delegate_tool_results import ChildBuildCancelled
from tools.thread_context import propagate_context_to_thread

logger = logging.getLogger("tools.delegate_tool")

# Why a task ends without ever having a live child; the completion vocabulary's failure reasons.
CONSTRUCTION_FAILED = "construction_failed"


@dataclass
class UnitSpec:
    """One async unit of a deferred call: the call-wide task indexes it runs and the ids already handed to the model."""

    tasks: list[int]
    delegation_id: str
    subagent_ids: list[str]
    group: Optional[str]


def _new_call_id() -> str:
    """The call's live-transcript directory id, fixed before admission so the handle can name the transcript paths."""
    return f"deleg_{uuid.uuid4().hex[:8]}"


def split_units(task_list: list[dict[str, Any]], task_indexes: list[int], call_id: str,
                subagent_ids: list[str]) -> list[UnitSpec]:
    """Partition a call exactly like ``delegate_tool_dispatch._units_of`` (one unit per task ``group``, ungrouped
    tasks alone; the whole call when ``delegation.independent_completions`` is off), from task indexes alone."""
    from tools.delegate_tool_config import _get_independent_completions
    if not _get_independent_completions():
        groups: list[tuple[list[int], Optional[str]]] = [(task_indexes, None)]
    else:
        members: dict[Any, list[int]] = {}
        for i in task_indexes:
            g = task_list[i].get("group")
            members.setdefault(("g", str(g)) if g not in (None, "") else ("i", i), []).append(i)
        groups = [(indexes, key[1] if key[0] == "g" else None) for key, indexes in members.items()]
    return [
        UnitSpec(tasks=indexes, delegation_id=call_id if len(groups) == 1 else f"{call_id}-{k + 1}",
                 subagent_ids=[subagent_ids[i] for i in indexes], group=group)
        for k, (indexes, group) in enumerate(groups)
    ]


class _UnitControl:
    """Construction control of one unit: stop what has not started and find its children once they exist. ``cancel``
    is set before anything is signalled, so a build waiting for the construction lock leaves instead of starting."""

    def __init__(self, unit: UnitSpec, call: _DeferredCall) -> None:
        self.unit, self.call = unit, call
        self.cancel = threading.Event()
        self.lock = threading.Lock()
        self.children: dict[int, Any] = {}

    def stop(self) -> bool:
        """The unit record's ``interrupt_fn`` (TUI/Desktop stop, ``/stop``): stop every task that has not started and
        signal the children that have. A unit that never started ends right here (its completion says why)."""
        self.cancel.set()
        with self.lock:
            children = list(self.children.values())
        for child in children:
            with _quiet("Failed to stop a deferred child"):
                from tools.delegate_tool_child_run import _signal_child_stop
                _signal_child_stop(child, "The delegated task was stopped")
        return self.call.cancel_unit(self.unit)

    def register(self, task_index: int, child: Any) -> None:
        with self.lock:
            self.children[task_index] = child

    def progress(self) -> tuple:
        with self.lock:
            children = list(self.children.values())
        if not children:
            return (0, ()), False
        token, in_tool = _batch_progress_token(children)
        return (len(children), token), in_tool


class _DeferredCall:
    """One deferred call: what phase A captured and what its units did."""

    def __init__(self, batch: _Batch, routes: list[Any], task_schemas: list, task_images: list, *,
                 wake_sid: str, session_key: str, origin_ui_session_id: str, call_id: str, max_slots: int) -> None:
        self.batch, self.routes, self.task_schemas, self.task_images = batch, routes, task_schemas, task_images
        self.wake_sid, self.session_key, self.origin_ui_session_id = wake_sid, session_key, origin_ui_session_id
        self.call_id, self.max_slots, self.live_home = call_id, max_slots, batch.live_home
        self.parent_agent = batch.parent_agent
        self.parent_session_id = str(getattr(batch.parent_agent, "session_id", "") or "") or None
        self.controls: dict[str, _UnitControl] = {}
        self.units: list[UnitSpec] = []
        self.live_writers: list = []
        self.live_paths: list = []
        self.transcripts = threading.Event()
        self.finished = threading.Event()
        self.lock = threading.Lock()
        self.settled: dict[str, list[dict[str, Any]]] = {}  # per unit: tasks that ended without running
        # Ledger ordering: one row per unit, written on the unit's own thread before anything can complete it.
        self.persisted: dict[str, threading.Event] = {}

    # ── phase A: the turn thread ───────────────────────────────────────────────
    def plan(self) -> None:
        """Name the call's tasks and units (published placeholders use these ids) and open the call's telemetry. The
        call id is the live-transcript directory's id, fixed before admission so the handle can name the transcripts."""
        ids = [f"sa-{i}-{uuid.uuid4().hex[:8]}" for i in range(len(self.batch.task_list))]
        self.batch.subagent_ids = ids
        self.batch.live_deleg_id = self.call_id
        self.units = split_units(self.batch.task_list, [i for (i, _, _) in self.batch.children], self.call_id, ids)
        begin_delegation_run(self, subagents=len(self.batch.children),
                             depth=getattr(self.parent_agent, "_delegate_depth", 0) + 1)

    def admit(self) -> str:
        """Publish placeholders + records, then submit the call. Returns '' or the tool error to hand back."""
        from tools import async_delegation_queue as queue
        from tools.async_delegation import _get_executor
        from tools.delegate_tool_registry import discard_placeholder
        records = [self._record(unit) for unit in self.units]
        self._publish_placeholders(records)
        verdict, error = queue.admit(records, self.max_slots)
        if verdict == "rejected":
            for unit in self.units:
                for sid in unit.subagent_ids:
                    discard_placeholder(sid)
            self._settle_all(BEFORE_START)
            self._close_telemetry()
            return error
        for unit in self.units:
            self.persisted[unit.delegation_id] = threading.Event()
        _get_executor(max(self.max_slots, len(self.units))).submit(propagate_context_to_thread(self.start))
        queue.publish_state()
        return ""

    def payload(self, refused: list, dispatched: list[UnitSpec]) -> dict:
        """The model-facing handle, from the pre-issued ids (no child exists yet). The transcript paths are the ones
        the worker is about to create under this call's directory, so ``tail -f`` works as soon as it does."""
        from tools.delegate_tool_dispatch import _BACKGROUND_NOTES, _dispatched_body
        from tools.delegation_live_log import task_log_path
        paths = [task_log_path(self.call_id, i, self.live_home) for i in range(len(self.batch.task_list))]
        payload = _dispatched_body(
            replace(self.batch, refused=refused), [(None, unit.delegation_id) for unit in dispatched],
            n=len(self.batch.children), sids=self.batch.subagent_ids,
            units_of=lambda unit: unit.tasks, live_paths=paths)
        payload["live_transcripts_hint"] = _BACKGROUND_NOTES["live_transcripts_hint"]
        return payload

    def queued_position(self) -> Optional[int]:
        """Place in line of a single-unit call that had to wait (None when it did not)."""
        from tools import async_delegation_queue as queue
        return queue.position_of(self.units[0].delegation_id) if len(self.units) == 1 else None

    def _record(self, unit: UnitSpec) -> dict[str, Any]:
        from tools.async_delegation import _new_record
        b = self.batch
        control = self.controls.setdefault(unit.delegation_id, _UnitControl(unit, self))
        return _new_record(
            delegation_id=unit.delegation_id, goal=_unit_goal(b.task_list, unit.tasks),
            goals=[t["goal"] for t in b.task_list], context=b.context, toolsets=None, role=b.top_role,
            model=b.creds["model"], session_key=self.session_key, origin_ui_session_id=self.origin_ui_session_id,
            origin_session_id=self.wake_sid, parent_session_id=self.parent_session_id,
            interrupt_fn=control.stop, progress_fn=control.progress, slot_key=self.units[0].delegation_id,
            task_transcripts={}, task_indexes=unit.tasks if len(unit.tasks) < len(b.task_list) else None,
            call_key=self, call_settle=self.settle)

    def _publish_placeholders(self, records: list[dict[str, Any]]) -> None:
        from tools.delegate_tool_registry import register_placeholder
        owner = self.batch.owner_kwargs()
        owner_db_path = getattr(getattr(self.parent_agent, "_session_db", None), "db_path", None)
        for unit, record in zip(self.units, records):
            queued = record.get("phase") == "queued"
            for task_index, sid in zip(unit.tasks, unit.subagent_ids):
                register_placeholder(
                    subagent_id=sid, parent_id=None, depth=0, goal=self.batch.task_list[task_index]["goal"],
                    delegation_id=unit.delegation_id, model=record.get("model"), started_at=time.time(),
                    status="queued" if queued else "running", phase=record.get("phase"), queue_position=None,
                    tool_count=0, last_tool=None, accepting_steer=True,
                    owner_session_id=owner["owner_session_id"], owner_transport=owner["owner_transport"],
                    owner_session_record=owner["owner_session_record"],
                    owner_agent_session_id=self.parent_session_id,
                    owner_db_path=str(owner_db_path) if owner_db_path is not None else None,
                    cancel_fn=self.controls[unit.delegation_id].stop)

    # ── phase B: the pool ──────────────────────────────────────────────────────
    def start(self) -> None:
        """Worker body of the call (it already carries the admitting thread's context and prompt callbacks)."""
        try:
            self._persist()
            self._create_transcripts()
            for unit in self.units:
                self._run_unit(unit)
        finally:
            self.finished.set()

    def _persist(self) -> None:
        """Write every unit's row, then let the queue start it. A failed write leaves the unit running in memory only:
        its completion still arrives, and a late row is repaired by the guarded write."""
        from tools import async_delegation_queue as queue
        from tools.async_delegation import _records, _records_lock, persist_dispatch, prune_durable_records
        for unit in self.units:
            try:
                with _records_lock:
                    row = dict(_records[unit.delegation_id])
                persist_dispatch(row)
            except Exception:
                # Boundary: a failed ledger write must not cost the run, and the unit still delivers in memory.
                logger.error("deferred dispatch %s: ledger write failed; it runs in memory only", unit.delegation_id,
                             exc_info=True)
            finally:
                self.persisted[unit.delegation_id].set()
        with _quiet("Durable record pruning failed"):
            prune_durable_records()
        queue.mark_ready([unit.delegation_id for unit in self.units])

    def _create_transcripts(self) -> None:
        """Live transcripts for the whole call (every unit writes into the call's directory), created once."""
        from tools.delegation_live_log import create_live_transcripts
        with _quiet("Live transcript creation failed", exc_info=True):
            _, writers, paths = create_live_transcripts(
                self.batch.task_list, self.batch.context, delegation_id=self.call_id,
                model=self.batch.creds.get("model"), provider=self.batch.creds.get("provider"), home=self.live_home)
            self.live_writers, self.live_paths = writers, paths
        self.transcripts.set()

    def _run_unit(self, unit: UnitSpec) -> None:
        """Run one unit whose record admission already published. A unit that already ended (cancelled) is skipped."""
        from tools.async_delegation import run_admitted_unit
        run_admitted_unit(unit.delegation_id, lambda: self._execute(unit))

    def _execute(self, unit: UnitSpec) -> dict[str, Any]:
        """The unit's runner: build its children one at a time (a stopped task is never built) and run them."""
        from tools.delegate_tool_registry import update_placeholders
        control = self.controls[unit.delegation_id]
        children: list[tuple] = []
        for task_index in unit.tasks:
            if control.cancel.is_set() or self.finished.is_set():
                self.settle_unstarted(unit, BEFORE_START)
                break
            built, failure_exc = self._build(unit, task_index, control)
            if built is None:
                self.settle_unstarted(unit, CONSTRUCTION_FAILED, only=task_index, exc=failure_exc)
                continue
            children.append(built)
        self.transcripts.wait(timeout=30)
        update_placeholders({sid: {"phase": "running"} for sid in unit.subagent_ids})
        combined = _execute_and_aggregate(replace(self.batch, children=children, refused=[]),
                                          honor_parent_interrupt=False)
        # Tasks that never got a child (stopped, or not buildable) settle onto this unit's own result.
        with self.lock:
            extra = list(self.settled.get(unit.delegation_id, ()))
        if extra:
            combined["results"] = sorted(list(combined.get("results") or []) + extra,
                                         key=lambda entry: entry.get("task_index", -1))
        return combined

    def _build(self, unit: UnitSpec, task_index: int,
               control: _UnitControl) -> tuple[Optional[tuple], Optional[BaseException]]:
        """Build one child on this thread, behind the construction lock, cancellable while it waits for it."""
        from tools.delegate_tool import _build_children, _max_iterations
        subagent_id = unit.subagent_ids[unit.tasks.index(task_index)]
        extra = {"subagent_id": subagent_id, "attach": False, "parent_session_id": self.parent_session_id}
        try:
            children, error = _build_children(
                self.batch.task_list, self.task_schemas, self.routes, top_role=self.batch.top_role,
                max_iterations=_max_iterations(), parent_agent=self.parent_agent,
                live_deleg_id=self.call_id if self.live_writers else None, live_writers=self.live_writers,
                task_images=self.task_images, only=task_index, child_extra=extra, cancel=control.cancel)
        except ChildBuildCancelled:
            return None, None
        except Exception as exc:
            logger.error("deferred dispatch %s task %d: child construction failed: %s", unit.delegation_id,
                         task_index, exc, exc_info=True)
            return None, exc
        if error or not children:
            logger.warning("deferred dispatch %s task %d: not built: %s", unit.delegation_id, task_index, error)
            return None, RuntimeError(error or "no child was built")
        control.register(task_index, children[0][2])
        return children[0], None

    # ── task settlement: one path for every way a task can end without a run ────
    def cancel_unit(self, unit: UnitSpec) -> bool:
        """Stop a unit before anything of it ran: settle its tasks as interrupted and end the unit now, so the
        dispatcher is not left waiting for a slot that will never come."""
        from tools.async_delegation import cancel_unstarted_unit
        entries = self.settle_unstarted(unit, BEFORE_START)
        return bool(entries) and cancel_unstarted_unit(unit.delegation_id, entries) is not None

    def _settle_all(self, failure: str) -> None:
        for unit in self.units:
            self.settle_unstarted(unit, failure)

    def settle_unstarted(self, unit: UnitSpec, failure: str, only: Optional[int] = None,
                         exc: Optional[BaseException] = None) -> list[dict[str, Any]]:
        """Settle each of the unit's tasks that has not produced a child yet as a fabricated entry on the unit's
        result, so a stop and a construction failure share one path (and one telemetry status). Returns the entries."""
        settled: list[dict[str, Any]] = []
        from tools.delegate_tool_registry import discard_placeholder
        for task_index in unit.tasks:
            if only is not None and task_index != only:
                continue
            sid = unit.subagent_ids[unit.tasks.index(task_index)]
            missed = discard_placeholder(sid)
            if missed is None and self._task_has_child(unit, task_index):
                continue  # already running: its own stop path owns the entry
            entry = _result_entry(task_index, "interrupted" if failure == BEFORE_START else "error",
                                  self._error_text(failure, exc),
                                  failure_reason=None if failure == BEFORE_START else CONSTRUCTION_FAILED)
            if missed:
                entry["missed_steer"] = missed
            self._append(unit.delegation_id, entry)
            settled.append(entry)
        return settled

    def _task_has_child(self, unit: UnitSpec, task_index: int) -> bool:
        with self.controls[unit.delegation_id].lock:
            return task_index in self.controls[unit.delegation_id].children

    @staticmethod
    def _error_text(failure: str, exc: Optional[BaseException] = None) -> str:
        if failure == CONSTRUCTION_FAILED:
            detail = f": {exc}" if exc is not None else ""
            return (f"The executor for this task could not be built ({CONSTRUCTION_FAILED}{detail}). "
                    "Nothing ran; dispatch it again.")
        return "The task was stopped before it started. Nothing ran."

    def _append(self, delegation_id: str, entry: dict[str, Any]) -> None:
        """Put a settled task's entry on its unit's result: kept for the live completion and recorded durably so a
        crash before the unit ends still replays it."""
        from tools.async_delegation import record_unit_child
        with self.lock:
            self.settled.setdefault(delegation_id, []).append(entry)
        record_unit_child(delegation_id, entry)

    # ── completion ─────────────────────────────────────────────────────────────
    def settle(self, statuses: list[str]) -> None:
        """Last unit of the call finished: fold every unit's statuses (one per task) into the call's single row."""
        finish_delegation_unit(self, [{"status": s} for s in statuses], background=True)

    def _close_telemetry(self) -> None:
        """Close the call's telemetry for a call that never started any unit."""
        finish_delegation_unit(self, [{"status": "error"} for _ in self.batch.task_list], background=True)


def _unit_goal(task_list: list[dict[str, Any]], tasks: list[int]) -> str:
    from tools.async_delegation import unit_goal_label
    return unit_goal_label([t["goal"] for t in task_list], tasks)
