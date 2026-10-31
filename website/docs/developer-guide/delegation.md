---
title: Delegation internals
sidebar_label: Delegation
---

# Delegation internals

Long-form contract behind the `## Delegation` summary in `tools/AGENTS.md` (the model-facing tool is
`tools/delegate_tool.py`; its siblings own the pieces).

## Shapes, units and capacity

A call spawns a subagent with isolated context + terminal session; the parent waits for the summary unless
`background=true`, which returns a delegation id and re-enters the result via the async-delegation
completion queue. **Dispatcher sessions** (`agent/dispatcher_session.py::agent_is_dispatcher`) take the deferred path
(`tools/delegate_tool_deferred.py`): every validation still runs synchronously (a failure is an immediate tool
error), then the handle returns before any child exists — the pool writes the ledger row, creates the live
transcripts and builds each child just before running it. Each task is published as a registry placeholder
(`agent` None, with owner authority, `cancel_fn` and bounded pending steering) so list/steer/stop work from the
model, the TUI and the Desktop before the child exists. A full pool queues the call in
`tools/async_delegation_queue.py` (`delegation.queue_limit` slots per session, promoted by fewest running slots,
arrival order breaking ties); a full queue is a tool error, never an inline run. Other sessions keep the old
reject-at-capacity behavior, which runs the batch inline with a note. Shapes: single (`goal` + optional `context`, `toolsets`) or batch (`tasks: [...]`,
concurrency capped by `delegation.max_concurrent_children`, default 3). A background batch returns as ONE
completion by default; with `delegation.independent_completions` it is split into completion **units**
(`delegate_tool_dispatch._units_of`): tasks sharing a `group` join and report together; each ungrouped
task reports alone as it finishes. Units of one call share ONE pool slot (`slot_key` in
`async_delegation._dispatch`) — never count units against capacity; the executor is sized by live UNITS
and the stall clock arms when the runner starts, so a queued unit is never judged stalled. Roles: `leaf` (default;
no `delegate_task`, `clarify`, `memory`, `send_message`, `cronjob`; keeps `execute_code`) and
`orchestrator` (keeps `delegate_task`; gated by `delegation.orchestrator_enabled`, bounded by
`delegation.max_spawn_depth`, default 2). Config knobs under `delegation:`:
`max_concurrent_children, independent_completions, gateway_mode, max_spawn_depth, child_timeout_seconds, orchestrator_enabled,
subagent_auto_approve, inherit_mcp_toolsets, max_iterations, backend, backends`. **Routes:** `delegation.backend` picks a
delegation provider (`agent/delegation_provider.py`, `ctx.register_delegation_provider`; its settings live in
`delegation.backends.<name>`) that answers each task's route (`tools/delegate_tool_routes.py`: model/provider fields
resolved like the `delegation` section, `service_tier` checked like `/fast` in `delegate_tool_tuning.py`, or `profile` →
the child is built, leases and converses inside that profile's scope) and may add enum-checked per-task schema params; it
never owns lifecycle. The `default` strategy is the bundled `plugins/delegation/default` plugin (no core special case
beyond selecting it). Fails closed: an unusable backend is a tool error naming it (never a silent `default`), an
unroutable task is refused alone (`TaskRoute.error`), as is a task sending a parameter its backend lacks. A session routes on the backend its `delegate_task` was built from (`agent._delegation_backend`, pinned with its tools), so a save applies at `/new` or `--now` (`agent/delegation_backends.py`). Internal `credentials_cfg` callers (/review) and the lifecycle API bypass it. 
## Roles and dispatcher executors

Roles: `leaf` (default; no `delegate_task`, `clarify`, `memory`, `send_message`, `cronjob`; keeps `execute_code`)
and `orchestrator` (keeps `delegate_task`; gated by `delegation.orchestrator_enabled`, bounded by
`delegation.max_spawn_depth`).

**Dispatcher executors** (`agent/executor_baseline.py`) are the one exception to "a child never exceeds its
parent". Only `_build_children` passes a top-level dispatcher parent's frozen baseline (from
`delegation.executor.toolsets` / `.mcp`, honouring the session's own tool ceiling); the lifecycle API and a task's
`toolsets` never do. Executors stay forced leaves, `needs_input` ends their turn with a question for the
dispatcher, and `approval._pending_result` answers one with no approver `no_approver` instead of leaving it
pending. User guide: "Dispatcher executors".

## Child processes, kernels, output_schema

**Child processes:** a child's background
processes are killed at its teardown and their notices are suppressed in the parent; `process_manage(action="handoff")`
(children only) flips `ProcessSession.owner_task_id` to the parent under the registry lock
(`process_registry.transfer_ownership`) so the completion routes and reaps by the new owner; un-handed leftovers land on
the result as `orphaned_processes`, exited-but-never-read notify processes as `unread_completions` (`_ChildRun.account_background_processes`, before `cleanup` kills them). **Child kernels:** a child's `execute_code` kernels are keyed `<parent-owner>::child::<child-session-id>`, pinned against the `max_session_kernels` LRU cap while the child runs and disposed by `cleanup` (`code_kernel.shutdown_kernels_for_delegated_child`) — never let a finished child's kernel squat the cap. **output_schema:** a miss after the one retry keeps `status: completed` with the raw text in `summary` plus `schema_valid: false` / `schema_errors` / `schema_note` — never discard a child's result. **Durability:** background
delegation is process-local; work that must survive restart uses `cronjob` or
`terminal(background=True, notify_on_complete=True)`. `heartbeat=N` on the same spawn emits a `heartbeat` event (type on the completion queue; delta output only, `HEARTBEAT_MIN_SECONDS` floor, one daemon timer thread for all sessions in `process_registry.py::ProcessRegistry._heartbeat_loop`) — every surface that renders `watch_match` must render `heartbeat` (`process_registry_notifications.py`, `gateway/run.py::_drain_gateway_watch_events`, `tui_gateway/session_notifications.py::_DEDUP_EXTRA_FIELDS`). 
## Stop fan-out

**Stop fan-out:** a turn's hard interrupt reaches only
`_active_children`; background units are detached at dispatch and survive it on purpose (Teknium ruling: a
background subagent, `/review` included, is independent of the chat that spawned it). Turn stops — Ctrl+C,
Esc, the Desktop/TUI Stop button, a message that interrupts, `session.interrupt`, ACP `cancel` — never call
`async_delegation.interrupt_for_session`. Only the explicit `/stop` command (gateway busy AND idle paths, CLI
and Desktop process stop) and session close/reset end them. Depth>0 delegations are always synchronous
(`_model_background_value`), so a stop recurses through the child's own `_active_children` fan-out; an
interrupted entry's `summary` is the child's last real assistant text (`_build_result_entry`). API: `website/docs/developer-guide/subagent-lifecycle-api.md`.

## Dispatcher deferred dispatch (`tools/delegate_tool_deferred.py`)

Phase A runs on the turn thread and touches no disk. Every check that can fail — control action, spawn pause,
depth, task list, schemas, images, routes (a fail-closed route is still an immediate tool error), one-shot budget,
async delivery, queue capacity — happens there, so the model still gets an immediate tool error. Then the origin
identity and one placeholder per task are published, the units are admitted to `tools/async_delegation_queue.py`,
and the call is submitted with `propagate_context_to_thread` wrapped **on that thread**: the worker carries the
session's ContextVars and approval/sudo callbacks, and a unit promoted later by another thread can never substitute
that thread's callbacks.

Phase B runs on the pool: it writes each unit's ledger row (`persist_dispatch`, the one dispatch persistence site),
prunes, creates the live transcripts for the whole call (one directory, every unit writes into it), then builds
children one at a time behind the construction lock and runs each unit through `_execute_and_aggregate`. A child
that cannot be built is that task's `error` entry (`failure_reason: construction_failed`); the dispatcher still gets
a completion event.

Placeholders are ordinary registry records with `agent` None: they carry the same owner authority fields as a live
child plus `cancel_fn`, bounded `pending_steers`, `phase`/`queue_position`. `steer` parks text for the child to start
with (`take_early_steers`), `stop` calls `cancel_fn` — a unit that had not started ends immediately with
`interrupted`, its accepted steering reported as `missed_steer`. `delegation.queue_limit` bounds the per-session
queue; `0` means a full pool is a tool error. Promotion happens in the same lock hold that frees a slot
(`_finalize`), fewest running slots first, arrival order breaking ties, and `admit` refuses to run while any unit is
waiting, so a concurrent dispatch cannot jump the queue head.

Ledger ordering: `_persist_dispatch` is `INSERT OR IGNORE` (the row is written once), and every completion path
writes after `_finalize`'s status flip, so a slow dispatch write that lands after a completion cannot reset the row
to `running`. Telemetry (`begin_delegation_run` / `finish_delegation_unit`) opens once per call and folds one status
per task; the last unit to finish emits the call's single row.
