"""Capacity queue for dispatcher sessions' background delegations (a sibling of ``tools/async_delegation.py``).

A dispatcher session never waits behind its own executors: when the pool is full, a ``delegate_task`` call is admitted
into a bounded per-session queue (``delegation.queue_limit``) and starts when a slot frees, instead of being rejected
and run inline. Every other caller keeps the old reject-at-capacity behaviour (``_dispatch_admitted``).

State lives on the registry records in ``async_delegation._records`` (all functions named ``*_locked`` expect the
caller to hold ``_records_lock``): ``phase`` (``queued`` -> ``starting`` -> ``running``; absent on non-dispatcher
records, which count as running), ``queued_at``/``_seq`` (arrival order), ``_ready`` (the dispatch row is in the
ledger, so the unit may start), ``_wrapped`` (the worker, wrapped ON THE ADMITTING THREAD so it carries that
session's approval/sudo callbacks and ContextVars, never those of whichever thread promotes it) and ``_abort``.

A queued record keeps ``status == "running"``: the gateway idle probes, ``/stop`` and goal logic all read that, so a
session with only queued work is not idle. Slots are counted over records that are not ``queued``.
"""

from __future__ import annotations

import itertools
import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

QUEUED, STARTING, RUNNING = "queued", "starting", "running"
_SESSION_FIELDS = ("session_key", "parent_session_id", "origin_ui_session_id")
_seq = itertools.count(1)  # arrival order; only advanced while ``_records_lock`` is held
_warned_bad_limit = False


def queue_limit() -> int:
    """``delegation.queue_limit``: slots a session may keep waiting (0 = never queue, a full pool is an error)."""
    global _warned_bad_limit
    from hermes_cli.config_defaults_delegation import DELEGATION_DEFAULTS
    from tools.delegate_tool_config import _cfg
    default = DELEGATION_DEFAULTS["queue_limit"]
    raw = _cfg().get("queue_limit", default)
    if isinstance(raw, int) and not isinstance(raw, bool) and raw >= 0:
        return raw
    if not _warned_bad_limit:
        _warned_bad_limit = True
        logger.warning("delegation.queue_limit=%r is not an integer >= 0; using %d", raw, default)
    return default


def _slot(record: dict[str, Any]) -> str:
    return record.get("slot_key") or record["delegation_id"]


def holds_slot(record: dict[str, Any]) -> bool:
    from tools.async_delegation import _ACTIVE_STATES
    return record.get("status") in _ACTIVE_STATES and record.get("phase") != QUEUED


def owner_key(record: dict[str, Any]) -> str:
    """The session a record's slot is charged to (fair promotion): its routing key, else its spawner's id."""
    return str(record.get("session_key") or record.get("parent_session_id") or record.get("origin_ui_session_id") or "")


def _same_session(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return any(a.get(f) and str(a[f]) == str(b.get(f) or "") for f in _SESSION_FIELDS)


def _admission_locked(selectors: dict[str, Any], slot: str, max_slots: int, limit: int) -> tuple[str, str]:
    """``(verdict, error)`` for a call whose units share *slot*: run now, queue, or reject. Does not mutate."""
    from tools.async_delegation import _FENCED_FIELDS, _records, _refused_by
    if any((f, selectors.get(f)) in _refused_by for f in _FENCED_FIELDS):
        return "rejected", "This session was closed (exit or /new): no new background work."
    held = {_slot(r) for r in _records.values() if holds_slot(r)}
    waiting = any(r.get("phase") == QUEUED for r in _records.values())
    if slot in held or (len(held) < max_slots and not waiting):
        return "run", ""
    if limit <= 0:
        return "rejected", (
            f"All {max_slots} background delegation slots are busy and delegation.queue_limit is 0, so nothing can "
            "wait. Wait for a running task to finish (its result arrives as a message), or raise "
            "delegation.max_concurrent_children / delegation.queue_limit in config.yaml.")
    queued_slots = {_slot(r) for r in _records.values() if r.get("phase") == QUEUED and _same_session(selectors, r)}
    if len(queued_slots) >= limit:
        return "rejected", (
            f"The background queue is full ({limit} waiting, {max_slots} running). Do not dispatch more now: wait "
            "for results to arrive, or raise delegation.queue_limit in config.yaml.")
    return QUEUED, ""


def preflight(selectors: dict[str, Any], max_slots: int) -> str:
    """The error a new call would be rejected with right now ('' when it would be admitted): checked before the call
    creates transcripts or telemetry, so the common rejection costs nothing. ``admit`` decides authoritatively."""
    from tools.async_delegation import _records_lock
    slot = "\0preflight"
    with _records_lock:
        verdict, error = _admission_locked(selectors, slot, max_slots, queue_limit())
    return error if verdict == "rejected" else ""


def admit(records: list[dict[str, Any]], max_slots: int) -> tuple[str, str]:
    """Insert one call's unit records (they share a ``slot_key``) as running or queued, atomically with the capacity
    decision. They are not ``_ready``: nothing starts until the caller has written their ledger rows."""
    from tools.async_delegation import _records, _records_lock
    head = records[0]
    with _records_lock:
        verdict, error = _admission_locked(head, _slot(head), max_slots, queue_limit())
        if verdict == "rejected":
            return verdict, error
        seq = next(_seq)
        for record in records:
            record.update(phase=QUEUED if verdict == QUEUED else STARTING, queued_at=record["dispatched_at"], _seq=seq,
                          _ready=False, _max_slots=max_slots)
            _records[record["delegation_id"]] = record
    return verdict, ""


def _running_by_owner_locked() -> dict[str, int]:
    from tools.async_delegation import _records
    slots: dict[str, set] = {}
    for r in _records.values():
        if holds_slot(r):
            slots.setdefault(owner_key(r), set()).add(_slot(r))
    return {owner: len(held) for owner, held in slots.items()}


def _promote_locked() -> None:
    """Move queued calls into free slots, the session with the fewest running slots first (arrival order breaks ties),
    stopping at the first call that does not fit so a large head is never starved by later small ones."""
    from tools.async_delegation import _records
    while True:
        groups: dict[str, list[dict[str, Any]]] = {}
        for r in _records.values():
            if r.get("phase") == QUEUED and r.get("_ready"):
                groups.setdefault(_slot(r), []).append(r)
        if not groups:
            return
        running = _running_by_owner_locked()
        slot, members = min(groups.items(), key=lambda kv: (running.get(owner_key(kv[1][0]), 0), kv[1][0]["_seq"]))
        held = {_slot(r) for r in _records.values() if holds_slot(r)}
        if slot not in held and len(held) >= members[0]["_max_slots"]:
            return
        for r in members:
            r["phase"] = STARTING


def collect_starts_locked() -> list[str]:
    """Promote what fits and return the ids of dispatcher units that are ready to be handed to the pool (each is
    returned once). Free for sessions that never queue: no record has a phase."""
    from tools.async_delegation import _ACTIVE_STATES, _records
    if not any("phase" in r for r in _records.values()):
        return []
    _promote_locked()
    return [r["delegation_id"] for r in _records.values()
            if r.get("phase") == STARTING and r.get("_ready") and r.get("_wrapped") is not None
            and r.get("status") in _ACTIVE_STATES]


def run_starts(delegation_ids: list[str]) -> None:
    """Hand each id's worker to the pool (outside the lock) and publish the new queue state to the placeholders."""
    if not delegation_ids:
        return
    for delegation_id in delegation_ids:
        _start(delegation_id)
    publish_state()


def _start(delegation_id: str) -> None:
    from hermes_cli.backend_retirement import retirement
    from tools.async_delegation import _get_executor, _LIVE_STATES, _ensure_stale_monitor, _records, _records_lock
    with _records_lock:
        record = _records.get(delegation_id)
        wrapped = record.pop("_wrapped", None) if record is not None else None
        if wrapped is None:
            return
        max_slots = record["_max_slots"]
        live_units = sum(1 for r in _records.values() if r.get("status") in _LIVE_STATES)
        monitored = record.get("progress_fn") is not None
    if not retirement.acquire():
        fail_unstarted(delegation_id, "backend is retiring; reconnect to continue")
        return
    try:
        future = _get_executor(max(max_slots, live_units)).submit(wrapped)
        future.add_done_callback(lambda _: retirement.release())
    except Exception as exc:  # pragma: no cover - pool submit failure is rare
        retirement.release()
        logger.error("Async delegation %s could not be scheduled: %s", delegation_id, exc, exc_info=True)
        fail_unstarted(delegation_id, f"Failed to schedule async delegation: {exc}")
        return
    if monitored:
        _ensure_stale_monitor()


def fail_unstarted(delegation_id: str, reason: str) -> None:
    """Terminal path of a unit that will never run (pool or retirement refusal): its tasks settle as errors, its
    telemetry closes and its completion event is delivered like any other."""
    from tools.async_delegation import _batch_crash, _finalize, _records, _records_lock
    with _records_lock:
        record = _records.get(delegation_id)
        abort, ctx = (record.get("_abort"), record.get("_context")) if record is not None else (None, None)
    result = abort(reason) if abort is not None else _batch_crash(reason, 0.0)
    if ctx is not None:
        ctx.run(_finalize, delegation_id, result, "error")
    else:
        _finalize(delegation_id, result, "error")


def mark_ready(delegation_ids: list[str]) -> None:
    """The units' ledger rows are written: they may start as soon as they hold a slot."""
    from tools.async_delegation import _records, _records_lock
    with _records_lock:
        for delegation_id in delegation_ids:
            if (record := _records.get(delegation_id)) is not None:
                record["_ready"] = True
        starts = collect_starts_locked()
    run_starts(starts)


def rollback(delegation_ids: list[str]) -> None:
    """Withdraw units that were admitted but never started (their ledger write failed): rows go, slots are re-offered."""
    from tools.async_delegation import _forget_dispatch, _records, _records_lock
    with _records_lock:
        for delegation_id in delegation_ids:
            _records.pop(delegation_id, None)
        starts = collect_starts_locked()
    for delegation_id in delegation_ids:
        _forget_dispatch(delegation_id)
    run_starts(starts)


def fast_track(delegation_id: str) -> None:
    """Start a queued unit now so it can settle without waiting for a slot: every task in it was stopped, so it only
    has to report them as interrupted. It cannot run anything, so it does not displace running work."""
    from tools.async_delegation import _records, _records_lock
    with _records_lock:
        record = _records.get(delegation_id)
        if record is None or record.get("phase") != QUEUED:
            return
        record["phase"] = STARTING
        starts = [delegation_id] if record.get("_ready") and record.get("_wrapped") is not None else []
    run_starts(starts)


def position_of(delegation_id: str) -> Optional[int]:
    """The unit's place in line, or None when it is not waiting."""
    from tools.async_delegation import _records_lock
    with _records_lock:
        return queue_positions_locked().get(delegation_id)


def queue_positions_locked() -> dict[str, int]:
    """1-based place in line, per delegation id, among the queued calls of the same session (arrival order)."""
    from tools.async_delegation import _records
    order: dict[str, dict[str, int]] = {}
    for r in sorted((r for r in _records.values() if r.get("phase") == QUEUED), key=lambda r: r["_seq"]):
        ranks = order.setdefault(owner_key(r), {})
        ranks.setdefault(_slot(r), len(ranks) + 1)
    return {r["delegation_id"]: order[owner_key(r)][_slot(r)] for r in _records.values() if r.get("phase") == QUEUED}


def publish_state() -> None:
    """Mirror queue state onto the units' registry placeholders (status, phase, place in line). Computed under the
    records lock and applied after it is released: the two locks are never held together."""
    from tools.async_delegation import _records, _records_lock
    with _records_lock:
        positions = queue_positions_locked()
        updates: dict[str, dict[str, Any]] = {}
        for r in _records.values():
            if "phase" not in r:
                continue
            queued = r.get("phase") == QUEUED
            fields = {"status": QUEUED if queued else RUNNING, "phase": QUEUED if queued else STARTING,
                      "queue_position": positions.get(r["delegation_id"]) if queued else None}
            for sid in r.get("_placeholder_ids") or ():
                updates[sid] = fields
    if updates:
        from tools.delegate_tool_registry import update_placeholders
        update_placeholders(updates)
