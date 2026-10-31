"""Live-subagent registry + model-facing control plane (list/steer/stop) for delegate_task."""

from __future__ import annotations

import logging
import json
import threading
import time
from typing import Any, Dict, List, Optional
from agent.interrupt_compat import request_hard_interrupt
from tools.registry import tool_error

logger = logging.getLogger("tools.delegate_tool")  # log-record parity with the origin module

_spawn_pause_lock = threading.Lock()
_spawn_paused: bool = False
_active_subagents_lock = threading.Lock()
# subagent_id -> mutable record tracking the live child agent.  Stays only
# for the lifetime of the run; _run_single_child is the owner.
_active_subagents: dict[str, dict[str, Any]] = {}
# subagent_id -> {goal, delegation_id, owner_agent_session_id} retained AFTER the child finishes (bounded FIFO).
# Child-started background processes routinely outlive the child (its npm ci with notify_on_complete=true finishes
# after the summary was delivered); their completion notifications reach the parent via the shared completion_queue
# and need delegation attribution even though the live registry entry is gone.
_RECENT_SUBAGENTS_CAP = 200
_recent_subagents: dict[str, dict[str, Any]] = {}

def get_subagent_attribution(task_id: Optional[str]) -> Optional[dict[str, Any]]:
    """``{subagent_id, goal, delegation_id}`` for a process task_id that belongs to a live or recently-finished child
    (children run their terminal sessions under ``task_id == subagent_id``), else None."""
    if not task_id or not isinstance(task_id, str):
        return None
    with _active_subagents_lock:
        record = _active_subagents.get(task_id) or _recent_subagents.get(task_id)
    if record is None:
        return None
    return {"subagent_id": task_id, "goal": record.get("goal"), "delegation_id": record.get("delegation_id")}

def set_spawn_paused(paused: bool) -> bool:
    """Globally block/unblock NEW delegate_task spawns (active children keep running). Returns the new state."""
    global _spawn_paused
    with _spawn_pause_lock:
        _spawn_paused = bool(paused)
        return _spawn_paused

def is_spawn_paused() -> bool:
    with _spawn_pause_lock:
        return _spawn_paused

def _register_subagent(record: dict[str, Any]) -> None:
    """Publish a live child. It replaces its own placeholder (same id, ``agent`` None) atomically; steering text the
    placeholder had accepted rides over as ``early_steers`` for the run to prepend (``take_early_steers``)."""
    sid = record.get("subagent_id")
    if not sid:
        return
    record.setdefault("accepting_steer", True)
    with _active_subagents_lock:
        prior = _active_subagents.get(sid)
        if prior is not None and prior.get("agent") is None and prior.get("pending_steers"):
            record["early_steers"] = list(prior["pending_steers"])
        _active_subagents[sid] = record

def _forget_locked(record: dict[str, Any]) -> None:
    """Drop *record* and keep a bounded attribution stub. Caller holds ``_active_subagents_lock``."""
    sid = record.get("subagent_id")
    _active_subagents.pop(sid, None)
    if not sid:
        return
    _recent_subagents[sid] = {k: record.get(k) for k in ("goal", "delegation_id", "owner_agent_session_id")}
    while len(_recent_subagents) > _RECENT_SUBAGENTS_CAP:
        _recent_subagents.pop(next(iter(_recent_subagents)), None)

def _unregister_subagent(subagent_id: str, *, agent: Any = None) -> None:
    """Drop the live record (exact agent identity when given) and keep a bounded attribution stub."""
    with _active_subagents_lock:
        record = _active_subagents.get(subagent_id)
        if record is None or not (agent is None or record.get("agent") is agent):
            return
        _forget_locked(record)

# ── Placeholders: a deferred (dispatcher) unit's task is listed, steerable and stoppable before its child exists ──
# A placeholder is a registry record with ``agent`` None. It carries the same owner authority fields as a live child's
# record (so the TUI, Desktop and CLI authority checks accept it), a ``cancel_fn`` that stops the task wherever it is
# (queued, waiting for construction, mid-construction) and the steering text accepted so far.
_MAX_PENDING_STEERS = 8
_MAX_STEER_CHARS = 2000

def register_placeholder(**fields: Any) -> None:
    """Publish a task that has no child yet. ``fields`` are the registry record fields (``subagent_id``, ``goal``,
    owner authority, ``cancel_fn``, ``status``/``phase``/``queue_position``...)."""
    _register_subagent({**fields, "agent": None, "pending_steers": []})

def update_placeholders(updates: dict[str, dict[str, Any]]) -> None:
    """Patch still-unstarted placeholders by id (queue position, promotion); a record that has a child is left alone."""
    with _active_subagents_lock:
        for sid, fields in updates.items():
            record = _active_subagents.get(sid)
            if record is not None and record.get("agent") is None:
                record.update(fields)

def take_early_steers(subagent_id: str) -> list[str]:
    """Steering text the task's placeholder accepted before its child registered (consumed once)."""
    with _active_subagents_lock:
        record = _active_subagents.get(subagent_id)
        if record is None:
            return []
        return record.pop("early_steers", None) or []

def discard_placeholder(subagent_id: str) -> Optional[str]:
    """Drop a task's placeholder that never got a child (cancelled, refused or failed to build). Returns the
    steering text it had accepted, which the caller reports as ``missed_steer``; None when nothing is pending or the
    task already has a live child (that record is the run's to clean up)."""
    with _active_subagents_lock:
        record = _active_subagents.get(subagent_id)
        if record is None or record.get("agent") is not None:
            return None
        record["accepting_steer"] = False
        _forget_locked(record)
        pending = record.get("pending_steers") or []
    return "\n".join(pending) or None

def _close_subagent_steering(subagent_id: str, agent: Any) -> Optional[str]:
    """Atomically close steer acceptance and drain its final durable artifact. ``steer_subagent`` holds the same
    registry lock through ``agent.steer``, so either acceptance wins and this drain sees its exact text, or closure
    wins and the caller is rejected. Exact agent identity prevents a finishing child with a recycled public id from
    closing its replacement."""
    with _active_subagents_lock:
        record = _active_subagents.get(subagent_id)
        if record is None or record.get("agent") is not agent:
            return None
        record["accepting_steer"] = False
        drain = getattr(agent, "_drain_pending_steer", None)
        if not callable(drain):
            return None
        try:
            pending = drain()
        except Exception as exc:
            logger.debug("final steer drain for %s failed: %s", subagent_id, exc)
            return None
        return pending if isinstance(pending, str) and pending.strip() else None

def interrupt_subagent(subagent_id: str) -> bool:
    """Request that one running subagent stop at its next iteration boundary
    (cooperative: the flag propagates to in-flight tools and recurses into
    grandchildren via AIAgent.interrupt()). True iff a matching subagent was found."""
    with _active_subagents_lock:
        record = _active_subagents.get(subagent_id)
    return interrupt_record(record, f"Interrupted via TUI ({subagent_id})") if record else False

def interrupt_record(record: dict[str, Any], reason: str) -> bool:
    """Stop the task *record* describes: hard-interrupt its live child, or call a placeholder's ``cancel_fn``
    (a deferred task that has no child yet, so it never starts). The caller authorized the record; this acts on that
    object and never re-resolves its recyclable public id."""
    agent, cancel_fn = record.get("agent"), record.get("cancel_fn")
    try:
        if agent is not None:
            return bool(request_hard_interrupt(agent, reason))
        if callable(cancel_fn):
            return bool(cancel_fn(reason))
    except Exception as exc:
        logger.debug("interrupt of %s failed: %s", record.get("subagent_id"), exc)
    return False

def _subagent_transport_matches(record, transport) -> bool:
    """Authority follows the owning session's LIVE transport slot, read at check time.

    ``owner_transport`` on the record is only the capture-time marker that a gateway session
    commissioned the child (``None`` = no RPC authority ever). The slot is authoritative because
    every reattach path (prompt.submit, queued drain, resume, activate, viewer failover) already
    mutates it; a per-record copy needed a matching registry sync at each of those sites and two
    were missed (#106663). Records whose owner is not a session dict keep the exact-object rule."""
    from tui_gateway.transport import FanoutTransport

    if record.get("owner_transport") is None:
        return False
    owner = record.get("owner_session_record")
    bound = owner.get("transport") if isinstance(owner, dict) else record.get("owner_transport")
    return bound is transport or (isinstance(bound, FanoutTransport) and bound.contains(transport))


def steer_subagent(
    subagent_id: str, text: str, *, owner_session_id: Optional[str] = None, owner_transport: Any = None,
    owner_session_record: Any = None,
) -> bool:
    """Queue steering text into a running subagent without stopping it.

    AIAgent.steer() appends the text to the child's last tool result at its next iteration boundary — the current tool
    call is never cut. True iff the text was QUEUED while the child still accepted work; False for unknown/closed id,
    ownership mismatch, no live agent, or empty text. ``owner_session_id=None`` keeps the in-process helper contract;
    gateway callers must pass exact authority. Acceptance and completion are linearized by the registry lock: if
    acceptance wins but no delivery boundary remains, the text lands in the entry as ``missed_steer``.
    """
    if not text or not text.strip():
        return False
    with _active_subagents_lock:
        record = _active_subagents.get(subagent_id)
        if not record or not record.get("accepting_steer", False):
            return False
        if owner_session_id is not None and (
            record.get("owner_session_id") != owner_session_id
            or owner_transport is None
            or not _subagent_transport_matches(record, owner_transport)
            or owner_session_record is None
            or record.get("owner_session_record") is not owner_session_record
        ):
            return False
        return _steer_locked(record, subagent_id, text)

def _steer_locked(record: dict[str, Any], subagent_id: str, text: str) -> bool:
    """Deliver *text* to the record's child, or park it on a placeholder (bounded) for the child to start with.
    Caller holds ``_active_subagents_lock`` and has authorized the record."""
    agent = record.get("agent")
    if agent is None:
        pending = record.get("pending_steers")
        if pending is None or len(pending) >= _MAX_PENDING_STEERS:
            return False
        pending.append(text.strip()[:_MAX_STEER_CHARS])
        return True
    try:
        return bool(agent.steer(text))
    except Exception as exc:
        logger.debug("steer_subagent(%s) failed: %s", subagent_id, exc)
        return False

def steer_authorized_record(subagent_id: str, snapshot: dict[str, Any], text: str) -> bool:
    """Steer the task a caller already authorized through a snapshot of its record (Desktop's sidecar): the live
    record must still be the same object of work, that is the same child or the same still-unstarted placeholder."""
    with _active_subagents_lock:
        record = _active_subagents.get(subagent_id)
        if (not record or not record.get("accepting_steer") or record.get("agent") is not snapshot.get("agent")
                or not text.strip()):
            return False
        return _steer_locked(record, subagent_id, text)

def _capture_gateway_steer_authority(owner_session_id: Optional[str]) -> tuple[Any, Any]:
    """Exact request transport + live session generation, if any — an in-process
    bridge, not a serializable capability. Non-gateway hosts get ``(None, None)``."""
    if not owner_session_id:
        return None, None
    try:
        from tui_gateway.server import _current_session_steer_authority
        return _current_session_steer_authority(owner_session_id)
    except Exception:
        return None, None

# Registry record fields never exposed to the TUI/RPC snapshot.
_PRIVATE_RECORD_KEYS = frozenset({
    "agent", "owner_session_id", "owner_transport", "owner_session_record", "accepting_steer",
    "cancel_fn", "pending_steers", "early_steers", "owner_db_path",
})

def list_active_subagents() -> list[dict[str, Any]]:
    """Copy of the running subagent tree ({subagent_id, parent_id, depth, goal, model,
    started_at, tool_count, status, ...}); safe from any thread."""
    with _active_subagents_lock:
        return [{k: v for k, v in r.items() if k not in _PRIVATE_RECORD_KEYS} for r in _active_subagents.values()]

def _is_descendant_of(child_agent: Any, parent_agent: Any, max_hops: int = 8) -> bool:
    """True when *child_agent* sits below *parent_agent* in the spawn tree (walks the ``_delegate_parent_ref`` weakref
    chain stamped at build time). Identity only — a parent may steer/stop its own children and grandchildren, never
    a sibling tree owned by another conversation."""
    if child_agent is None or parent_agent is None:
        return False
    cur = child_agent
    for _ in range(max_hops):
        ref = getattr(cur, "_delegate_parent_ref", None)
        ancestor = ref() if callable(ref) else None
        if ancestor is None:
            return False
        if ancestor is parent_agent:
            return True
        cur = ancestor
    return False

# Model-facing control actions accepted by delegate_task(action=...).
# "spawn" (or omitted) keeps the historical spawn semantics.
_CONTROL_ACTIONS = frozenset({"list", "steer", "stop"})

def _resolve_session_lineage(session_id: Optional[str], parent_agent: Any) -> str:
    """Tip of a session id's compression lineage via the parent's live SessionDB (best-effort; input unchanged when
    unavailable) so a delegation dispatched before a compression rotation still matches the rotated parent."""
    sid = str(session_id or "")
    db = getattr(parent_agent, "_session_db", None)
    if not sid or db is None:
        return sid
    try:
        resolved = db.resolve_resume_session_id(sid)
        return str(resolved) if resolved else sid
    except Exception:
        return sid

def _owns_subagent_record(record: dict[str, Any], parent_agent: Any) -> bool:
    """True when *parent_agent*'s conversation owns this live-child record.

    Tier 1: identity — the ``_delegate_parent_ref`` weakref chain reaches
    *parent_agent* (fast path while the parent AIAgent survives the run). Tier 2:
    durable lineage — the record's ``owner_agent_session_id`` matches the caller's
    ``session_id`` after resolving compression-rotation lineage on both sides.
    Tier 2 exists because the identity chain is BRITTLE across parent rebuilds:
    the CLI sets ``self.agent = None`` mid-session (route change, credential
    refresh, /model, MoA one-shots) and builds a NEW AIAgent while the child keeps
    a weakref to the old one. Delivery routes by durable session id; control must
    use the same spine or running children go invisible/unsteerable.
    """
    if _is_descendant_of(record.get("agent"), parent_agent):
        return True
    owner_sid = str(record.get("owner_agent_session_id") or "")
    parent_sid = str(getattr(parent_agent, "session_id", "") or "")
    if not owner_sid or not parent_sid:
        return False
    if owner_sid == parent_sid:
        return True
    # Compression rotation on either side: compare lineage tips.
    return _resolve_session_lineage(owner_sid, parent_agent) in {parent_sid, _resolve_session_lineage(parent_sid, parent_agent)}

def _list_payload(parent_agent: Any) -> dict[str, Any]:
    with _active_subagents_lock:
        records = list(_active_subagents.values())
    entries = []
    for r in records:
        if not _owns_subagent_record(r, parent_agent):
            continue
        started = r.get("started_at")
        seconds = round(time.time() - started, 1) if isinstance(started, (int, float)) else None
        waiting = r.get("agent") is None and r.get("phase") is not None  # a deferred task without a child yet
        entries.append({
            "subagent_id": r.get("subagent_id"),
            "parent_id": r.get("parent_id"),
            "goal": r.get("goal"),
            "model": r.get("model"),
            "status": r.get("status"),
            **({"phase": r["phase"], "queue_position": r.get("queue_position"), "waiting_seconds": seconds}
               if waiting else {"running_seconds": seconds}),
            "accepting_steer": bool(r.get("accepting_steer", False)),
            "live_transcript": getattr(r.get("agent"), "_live_transcript_path", None) or r.get("live_transcript"),
        })
    payload: dict[str, Any] = {"action": "list", "count": len(entries), "subagents": entries}
    if any("phase" in e for e in entries):
        payload["queued"] = sum(e.get("status") == "queued" for e in entries)
        payload["running"] = len(entries) - payload["queued"]
    if not entries:
        payload["note"] = (
            "No live subagents right now. Children that already finished "
            "have delivered (or will deliver) their results as normal "
            "completion messages — there is nothing to steer or stop."
        )
    return payload

def _handle_control_action(action: str, subagent_id: Optional[str], message: Optional[str], parent_agent: Any) -> str:
    """Synchronous control plane for delegate_task: list/steer/stop. Runs in-turn (never backgrounded) over the same
    registry the TUI overlay drives, scoped so a conversation can only control its own spawn tree."""
    if action == "list":
        return json.dumps(_list_payload(parent_agent), ensure_ascii=False)

    # steer / stop need a resolvable, owned target.
    sid = (subagent_id or "").strip()
    if not sid:
        return tool_error(f"action='{action}' requires subagent_id (from the spawn dispatch response or action='list').")
    with _active_subagents_lock:
        record = _active_subagents.get(sid)
    if record is None or not _owns_subagent_record(record, parent_agent):
        return tool_error(
            f"No live subagent '{sid}' in this conversation's spawn tree. It "
            "may have already finished (its result arrives as a normal "
            "completion message). Use action='list' to see live children."
        )
    if action == "steer" and not (message or "").strip():
        return tool_error("action='steer' requires a non-empty 'message' describing the course correction.")
    outcome = _CONTROL_OUTCOMES.get(action)
    if outcome is None:
        return tool_error(f"Unknown action '{action}'. Use spawn, list, steer, or stop.")
    status, note, failure = outcome
    if record.get("agent") is None and action in _PLACEHOLDER_NOTES:
        note = _PLACEHOLDER_NOTES[action]
    ok = interrupt_subagent(sid) if action == "stop" else steer_subagent(sid, message.strip())
    if ok:
        return json.dumps({"action": action, "subagent_id": sid, "status": status, "note": note}, ensure_ascii=False)
    return tool_error(failure.format(sid=sid))

# Success notes for a task whose child does not exist yet (queued, or waiting its turn to be built).
_PLACEHOLDER_NOTES = {
    "stop": (
        "The task had not started running: it is cancelled and will not start. Its completion message (status "
        "interrupted) still arrives — do not wait or poll."
    ),
    "steer": "Steering text queued. The subagent is given it together with its task when it starts.",
}

# action -> (success status, success note, failure error template)
_CONTROL_OUTCOMES = {
    "stop": (
        "interrupt_requested",
        "The subagent stops at its next iteration boundary (in-flight tool calls are asked to cancel). Its "
        "partial result still re-enters the conversation as a completion message — do not wait or poll.",
        "Could not interrupt '{sid}' — it likely finished in the last "
        "moment. Its result arrives as a normal completion message.",
    ),
    "steer": (
        "queued",
        "Steering text queued. The subagent sees it appended to its next tool result — the current tool call is "
        "never cut. If the child finishes before a delivery boundary remains, the text is reported back as "
        "missed_steer in its completion entry.", "Subagent '{sid}' is no longer accepting steering (finishing or "
        "already finished). Its result arrives as a normal completion "
        "message; re-delegate a follow-up task if more work is needed.",
    ),
}
