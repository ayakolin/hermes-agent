---
sidebar_position: 12
---

# Dispatcher mode

Dispatcher mode makes the main agent a front desk. It answers you at once and hands anything that will take
more than a few seconds to background executors, which report back as new messages. It is on by default for
interactive sessions.

## What changes in a dispatcher session

- **A narrow tool surface.** The agent runs the `dispatcher` posture toolset: `delegate_task`, `todo_list`,
  `memory`, `cronjob_manage`, `read_file` and `search_files`. Long or non-interruptible work (the terminal,
  browser, web, code execution, skills) belongs to the executors it spawns.
- **Background delegation only.** `delegation.gateway_mode` does not apply: a dispatcher's `delegate_task`
  always returns at once and the results arrive as later turns.
- **A dispatcher brief instead of "Finishing the job".** The system prompt tells the agent to acknowledge
  first, hand off long work with self-contained task descriptions, and deliver the result itself. The coding
  brief (which points at `patch`/`write_file`) is left out because the dispatcher has no such tools.

The decision is made once, when the session is created, and frozen for the life of the conversation. Changing
`dispatcher.enabled` or `dispatcher.extra_toolsets` afterwards affects new sessions only: a tool or prompt
change mid-conversation would break prompt caching.

## Turning it off

Dispatcher mode does not apply to a session when any of these is true:

- `dispatcher.enabled` is `false` (globally, or in the profile you launch: `hermes -p code`).
- `agent.coding_context` is `focus` or `on`. The coding posture needs the coding brief and its tools.
- The session's tool selection was pinned: `-t/--toolsets`, a saved `platform_toolsets.cli` (what
  `hermes tools` and `profiles.configure` write), or an adapter's per-source selection for a messaging route.
- The session cannot receive a later result as a turn: one-shot runs (`chat -q`, `-z`), cron, Kanban workers,
  ACP editors, `--source tool`, forwarded A2A conversations, the API server and webhook surfaces, and local
  sessions executed by a managed worker (`gateway.managed_workers: true`).
- The session is a side agent (`/bg`, `/btw`, `/refine`, a background review) or an onboarding chat.

```yaml
dispatcher:
  enabled: true            # the switch; read once, when a session is created
  extra_toolsets: []       # toolsets appended to the posture, e.g. [session_search, web]
```

`extra_toolsets` is validated when the session is created: unknown names are dropped with a warning, and the
resolved list is frozen with the session. Side agents keep the classic surface, whatever the session's own
mode is.

## How each surface decides

| Surface | Where the decision is frozen |
|---|---|
| Classic CLI, Ink TUI, Desktop (local sessions) | The session's launch policy (`LocalSessionPolicy.dispatcher`) |
| Messaging platforms (Telegram, Discord, Slack, …) | `sessions.model_config.dispatcher` on the conversation's compression tip |
| Legacy tui_gateway sessions | The same session-row marker |

An existing conversation that was created without a marker stays classic after you upgrade, and a
conversation that already built its prompt never flips when the configuration changes.

## Related settings

The rest of the `dispatcher:` section tunes behaviour that other parts of the runtime read:

```yaml
dispatcher:
  ack:
    reminder: true         # one-shot "the user has seen no text yet" note on the first tool result
    max_nudges: 1          # end-of-turn nudges when a human turn ended with no visible text; 0 = off
    force_interim: true    # deliver pre-tool-call text immediately on every platform
    early_interim: true    # messaging platforms: flush the preamble when the first tool call starts
  latency:
    memory_budget_s: 2.0         # external memory tick + prefetch budget before the first token; 0 = legacy
    pre_llm_call_budget_s: 3.0   # total budget for pre_llm_call plugin callbacks; 0 = per-callback only
    skills_recheck_s: 30         # min seconds between full skills-index comparisons; 0 = every turn
    defer_compaction: true       # defer compaction to a background-prepared summary below the ceiling
    defer_ceiling_ratio: 0.85
    stream_deltas: true          # platforms that opt in to streaming stream for dispatcher sessions
  busy:
    policy: dispatch             # dispatch: redirect -> steer -> queue, never a hard interrupt | inherit
    quick_lane: true             # answer from a parity fork while a redirect/steer cannot land yet
    stall_seconds: 4.0           # seconds an accepted redirect/steer may stay unconsumed before the quick lane
```

`display.dispatcher_receipt` picks the acknowledgement a messaging platform shows while a busy session takes
your message: `auto` (reaction, then typing, then text), `reaction`, `typing`, `text`, or `off`. It can be
overridden per platform with `display.platforms.<platform>.dispatcher_receipt`.

Delegation settings a dispatcher session relies on:

- `delegation.executor.toolsets` / `delegation.executor.mcp` — what the background executors may use.
- `delegation.queue_limit` — how many units may wait per session when the pool is full; a full queue is a
  tool error, never a synchronous run.

```yaml
delegation:
  max_concurrent_children: 10                     # running executors
  queue_limit: 32                                 # units allowed to wait per session when the pool is full
  completion_batch_window_seconds: 0.3            # batch completions that land together into one turn
  end_turn_on_dispatch: true                      # end a dispatcher turn right after accepted dispatches
  executor:
    toolsets: [terminal, file, web, browser, code_execution, vision, image_gen, skills, todo, session_search]
    mcp: all                                      # all | none | [server names]
```
