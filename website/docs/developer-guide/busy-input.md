---
title: Busy input on dispatcher sessions
description: How text typed while a turn runs is routed on a dispatcher session, on every local surface and on the gateway.
---

A **dispatcher session** answers the user at once and hands long work to background executor subagents. Its
tool surface and system prompt are frozen at creation, and so is its busy-input policy: **a message typed while
a turn runs never waits behind that turn and never hard-interrupts it.** The decision is made server-side, once,
by `agent/busy_input.plan_dispatcher_busy` — a pure ladder — so the classic CLI, the canonical local transport
(Ink TUI, Desktop) and the legacy `tui_gateway` cannot drift apart.

## The ladder

| Rung | When it lands | What the user sees |
|---|---|---|
| `redirect` | A model request is in flight: it is cancelled, finished messages are kept, and the correction becomes a real user message | `↪ Redirected current turn` |
| `steer` | `redirect()` declined — no model request in flight, or the agent cannot redirect: the text is injected after the next tool result | `⏩ Steered` |
| `queue` | Neither rung took the text (compression in flight, an agent still building, media payload, or both declined): it becomes the next turn | `Queued for the next turn` |

Facts that force the queue rung before any verb is tried: a non-text payload, context compression in flight
(a correction delivered mid-compression aborts it, #61042), and the turn's agent still being built (the text is
latched for the agent the turn adopts). A Codex app-server `redirect()` that answers false may already have
delivered the text (its `turn/steer` ACK can time out), so it is reported as unconfirmed and the text is
neither re-sent nor queued as a fresh prompt.

## Configuration

`dispatcher.busy.policy` decides whether a dispatcher session runs the ladder:

```yaml
dispatcher:
  busy:
    policy: dispatch   # redirect -> steer -> queue (default) | inherit
```

`inherit` opts a dispatcher session out: it falls back to the classic `display.busy_input_mode` behaviour.
A session's own `/busy <mode>` always wins, and on a dispatcher session it sets only that session — it never
rewrites `display.busy_input_mode`, which is every other session's mode.

## What each surface does

- **Canonical local transport** (`gateway/session_busy_controls.py::correct`): `session.redirect` and
  `session.steer` walk the ladder. Facts that cost a SQLite read (the compression lock row of the physical
  transcript, the manual `/compress` claim gate) are collected **before** `event_stream.lock` is taken, so a
  slow `state.db` never stalls event publication. A text no rung took is answered
  `{status: "rejected", delivered: false, reason}` — the client queues it. `config.get busy` reports the
  server-resolved mode and a `dispatch` flag.
- **Legacy `tui_gateway`** (`session_auto_continue._handle_busy_submit`): the same ladder, and
  `_interrupt_busy_session` is never called for a dispatcher session. Replies carry optional `disposition`
  and `reason`. `config.get/set busy` are session-scoped for such a session.
- **Classic CLI** (`hermes_cli/cli_busy_input_mixin.py`): `_busy_submit` is the one busy path — Enter,
  Tab-queue, the external editor and `/steer` all go through it. A dispatcher session resolves its corrections
  on a serial worker thread (the gateway answers a socket's requests one at a time, so a slow correction would
  otherwise freeze Enter), keeps submission order, and never writes `_interrupt_queue`. The CLI adopts the
  mode the owner reports (`config.get busy`) after session open, after `/resume`, and after `/busy`.
- **Desktop**: `config.get busy` stays the authority; the session's published `session.info.busy_input_mode`
  only seeds the composer so there is no window where the policy is unknown. An unknown (`null`) policy means
  queue — never a dead Send and never an unintended interrupt.
- **Ink TUI**: the session-scoped `config.get busy` wins over `display.busy_input_mode` on both transports.

## Three answers from a correction

`GatewayAgent._correct` (and the canonical ack) distinguishes three cases, because a bare "not taken" used to
lose the user's text:

- **Accepted** — the text was delivered (or latched). Includes a dispatcher redirect the owner took as a
  steer (`STEERED`, so the turn's ask is not re-anchored).
- **Refused, undelivered** (`delivered: false`, or a refusal before delivery) — safe to queue as the next
  prompt; the CLI queues it and `/steer` enqueues its refusal instead of dropping it.
- **Unknown** — the request was sent and the outcome is not known (a lost answer, a Codex steer that may
  already hold the text). Never re-queued, never turned into an interrupt; the user is told to check
  `/queue list` first.

## Related CLI behaviour

- The idle tick drains a gateway-admitted turn **before** MCP housekeeping, and skips that housekeeping while a
  turn is queued. A dispatcher session only gets the MCP reload notice: its tool surface is frozen.
- A notification turn whose `message.start` was dropped outside its turn is repainted from its receipt by
  `hermes_cli/gateway_foreign_turns.py`, once per idle tick.
- A pushed `subagent.*` event (or a moved active-subagent count) makes the CLI re-arm
  `GatewayProcessView.watch_subagents()`, so a child spawned between two polls appears at once.
- A completed gateway turn drains before `/goal` and `/loop` continuations start another turn in front of it.
