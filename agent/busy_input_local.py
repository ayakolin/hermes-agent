"""Busy-input routing for the local surfaces: classic CLI, canonical local sessions, legacy tui_gateway.

``agent.busy_input.plan_dispatcher_busy`` stays the one ladder (redirect -> steer -> queue, never a hard
interrupt). This module adds what the local surfaces need around it: which mode a session runs under, and a
runner that walks a plan one verb at a time and reports which verb took the text, so the canonical
``session.redirect`` / ``session.steer`` and the legacy ``prompt.submit`` busy branch cannot drift apart.

Only a dispatcher session whose ``dispatcher.busy.policy`` is ``dispatch`` uses any of this; every other
session keeps its existing ``display.busy_input_mode`` logic untouched.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from agent.busy_input import BusyFacts, BusyPlan, plan_dispatcher_busy
from agent.dispatcher_session import agent_is_dispatcher, dispatcher_settings

BUSY_MODES = ("interrupt", "steer", "queue")

# What one rung of the ladder answers: the agent took the text; declined it (nothing was applied, try the
# next rung); or answered False where the text may still have landed (a Codex native steer whose ACK timed out).
TAKEN, DECLINED, MAYBE = "taken", "declined", "maybe"

# The rung that took the text -> the disposition clients can tell apart without parsing ``status``.
DISPOSITION = {"redirect": "redirected", "steer": "steered", "latch": "steered", "queue": "queued_next_turn"}
# Why a rung that was asked declined. A latch is only asked in the build window, so its decline is that window.
DECLINE_REASON = {"redirect": "redirect_declined", "steer": "steer_declined", "latch": "agent_building"}


def frozen_dispatcher(*holders: Any) -> bool:
    """True when the session policy, the legacy session dict or the agent carries the creation-time
    dispatcher flag. The flag is frozen at creation; nothing here toggles it."""
    for holder in holders:
        if holder is None:
            continue
        if isinstance(holder, Mapping):
            if holder.get("dispatcher") is True:
                return True
        elif getattr(holder, "dispatcher", False) is True or agent_is_dispatcher(holder):
            return True
    return False


def dispatch_policy_applies(config: Mapping[str, Any] | None, dispatcher: bool) -> bool:
    """A dispatcher session uses the redirect-first ladder unless ``dispatcher.busy.policy`` is ``inherit``."""
    return dispatcher and dispatcher_settings(config).busy_policy == "dispatch"


def resolve_busy_mode(override: str | None, configured: str, *, dispatch: bool) -> str:
    """The mode a session's next busy input runs under.

    A session ``/busy`` override wins. Without one a dispatch-policy session runs the redirect-first ladder
    (``interrupt``) whatever ``display.busy_input_mode`` or a surface default says; every other session keeps
    ``configured``, which the caller already resolved from its own display config."""
    if override in BUSY_MODES:
        return override
    if dispatch:
        return "interrupt"
    return configured if configured in BUSY_MODES else "interrupt"


def plan_for_mode(mode: str, facts: BusyFacts) -> BusyPlan:
    """The dispatcher ladder narrowed to ``mode``: ``queue`` only queues, ``steer`` drops the redirect rung,
    ``interrupt`` is the full ladder. None of them can reach a hard interrupt."""
    if mode == "queue":
        return BusyPlan(("queue",), "queue_mode")
    plan = plan_dispatcher_busy(facts)
    if mode == "steer":
        return BusyPlan(tuple(verb for verb in plan.ladder if verb != "redirect"), plan.reason)
    return plan


def supports_redirect(agent: Any) -> bool:
    return getattr(agent, "_supports_active_turn_redirect", False) is True and callable(getattr(agent, "redirect", None))


def native_steer(agent: Any) -> bool:
    """The Codex app-server owns its loop: its ``redirect`` answers False on an ACK timeout the text may
    already have crossed, so a False there is not a refusal."""
    from agent.interrupt_control import _ic_codex_method
    return _ic_codex_method(agent, "request_steer") is not None


@dataclass(frozen=True)
class BusyOutcome:
    verb: str  # the rung that took the text, or "queue" when none did
    reason: str = ""  # why the text was not taken ("queue"), else ""
    maybe_delivered: bool = False  # the last rung answered MAYBE: never re-send, never interrupt

    @property
    def disposition(self) -> str:
        return DISPOSITION[self.verb]


def run_ladder(plan: BusyPlan, attempts: Mapping[str, Callable[[], str]]) -> BusyOutcome:
    """Walk ``plan`` until a rung takes the text. A rung without an attempt counts as declined; a rung that
    answers MAYBE ends the walk (a later steer could deliver the same text twice)."""
    reason = plan.reason
    for verb in plan.ladder:
        if verb == "queue":
            break
        attempt = attempts.get(verb)
        result = attempt() if attempt is not None else DECLINED
        if result == TAKEN:
            return BusyOutcome(verb)
        if result == MAYBE:
            return BusyOutcome("queue", f"{verb}_unconfirmed", maybe_delivered=True)
        reason = DECLINE_REASON.get(verb, "declined")
    return BusyOutcome("queue", reason or "declined")
