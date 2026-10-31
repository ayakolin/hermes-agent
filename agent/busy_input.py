"""What to do with a user message that arrives while a turn is running (dispatcher sessions).

Pure decision, no I/O, so the gateway, the canonical local-session controls and the legacy TUI backend
share one ladder. Non-dispatcher sessions keep their existing ``display.busy_input_mode`` logic and never
call this. In a dispatcher session a user message never waits behind a running turn and never hard
interrupts it: redirect (cancel the in-flight model request, keep finished work), then steer (deliver at
the next tool boundary), then queue.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

BusyVerb = Literal["redirect", "steer", "latch", "queue"]


@dataclass(frozen=True)
class BusyFacts:
    has_text: bool
    redirect_supported: bool = True
    compressing: bool = False
    agent_building: bool = False
    draining: bool = False


@dataclass(frozen=True)
class BusyPlan:
    ladder: tuple[BusyVerb, ...]
    reason: str = ""


def plan_dispatcher_busy(facts: BusyFacts) -> BusyPlan:
    """Ordered verbs to try; the caller stops at the first the agent accepts."""
    if facts.draining:
        return BusyPlan(("queue",), "draining")
    if facts.compressing:
        return BusyPlan(("queue",), "compression_in_flight")
    if not facts.has_text:
        return BusyPlan(("queue",), "non_text")
    if facts.agent_building:
        return BusyPlan(("latch", "queue"), "agent_building")
    if facts.redirect_supported:
        return BusyPlan(("redirect", "steer", "queue"))
    return BusyPlan(("steer", "queue"), "redirect_unsupported")
