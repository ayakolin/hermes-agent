"""Instant, non-model receipt for a message that arrives while a dispatcher session is busy.

The idle path already brackets a message with ``on_processing_start`` / ``on_processing_complete`` (a reaction
on the user's message). A busy-path message gets the same bracket when its adapter says it puts a visible mark
on the message (``acknowledges_event``, the one capability; Discord stays out because its hooks also write
durable per-message state). No typing indicator: the running turn already owns that task and its pause state.

Lifecycle, owned by ``session_ingress_burst`` and ``dispatch_shared_busy.observe``: ``open_receipt`` at the
top of ``dispatch_busy_burst`` (before any await, so it is instant and ahead of every debounce), one
``close_receipt`` per path out. Close is idempotent and event-keyed, so the paths may overlap: the observer's
finally, a discarded or failed burst, and the /stop and /new funnel (``close_route_receipts``). A message
absorbed into a burst is a follower of the burst head and closes with it. The first close wins; the
``start`` hook always finishes before ``complete`` starts.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional

from gateway.platforms.event import ProcessingOutcome

logger = logging.getLogger("gateway.run")  # log-record parity with gateway/run.py


@dataclass(eq=False)
class _Receipt:
    adapter: object
    route: str
    start: Optional[asyncio.Task] = None
    closed: bool = False
    followers: list = field(default_factory=list)


def receipt_mode(runner, source) -> str:
    return runner._dispatcher_busy_cfg(source).receipt_for(_platform_key(source))


def _platform_key(source) -> str:
    from gateway.run import _platform_config_key
    return _platform_config_key(source.platform)


def _wanted(adapter, event, route: str) -> bool:
    runner = getattr(adapter, "gateway_runner", None)
    if runner is None or getattr(event, "internal", False) or event.get_command():
        return False
    if event.reply_expected is False or not event.message_id or getattr(event, "_busy_receipt", None) is not None:
        return False
    if receipt_mode(runner, event.source) not in ("auto", "reaction"):
        return False
    return (runner._dispatcher_session_for(route) and adapter.acknowledges_event(event) is True
            and runner._is_user_authorized_for_source(event.source))


def open_receipt(adapter, event, route: str) -> bool:
    """Start the receipt for a busy-path ``event`` of chat ``route`` if one applies. Synchronous: it only
    schedules the hook, so the caller proceeds at once."""
    try:
        if not _wanted(adapter, event, route):
            return False
    except Exception:
        logger.warning("Busy-path receipt check failed for %s", route, exc_info=True)
        return False
    receipt = event._busy_receipt = _Receipt(adapter, route)
    receipt.start = task = asyncio.get_running_loop().create_task(
        adapter._run_processing_hook("on_processing_start", event))
    adapter._background_tasks.add(task)
    task.add_done_callback(adapter._background_tasks.discard)
    adapter.gateway_runner.__dict__.setdefault("_busy_receipts", {}).setdefault(route, []).append(event)
    return True


def follow_receipt(adapter, head, event, route: str) -> None:
    """``event`` was absorbed into the burst ``head``: it gets its own receipt and closes with the head."""
    open_receipt(adapter, event, route)
    if getattr(event, "_busy_receipt", None) is not None:
        head.__dict__.setdefault("_receipt_followers", []).append(event)


def close_receipt(event, outcome: ProcessingOutcome) -> None:
    """Close ``event``'s receipt (and its followers') with ``outcome``; a no-op once closed."""
    for follower in event.__dict__.pop("_receipt_followers", ()):
        close_receipt(follower, outcome)
    receipt = getattr(event, "_busy_receipt", None)
    if receipt is None or receipt.closed:
        return
    receipt.closed = True
    registry = getattr(receipt.adapter.gateway_runner, "_busy_receipts", {})
    if receipt.route in registry:  # events compare by value: drop this one by identity
        registry[receipt.route] = [other for other in registry[receipt.route] if other is not event]
        if not registry[receipt.route]:
            del registry[receipt.route]
    task = asyncio.get_running_loop().create_task(_finish(receipt, event, outcome))
    receipt.adapter._background_tasks.add(task)
    task.add_done_callback(receipt.adapter._background_tasks.discard)


async def _finish(receipt: _Receipt, event, outcome: ProcessingOutcome) -> None:
    if receipt.start is not None:
        await asyncio.wait({receipt.start})  # the mark is on the message before it is swapped
    await receipt.adapter._run_processing_hook("on_processing_complete", event, outcome)


def close_route_receipts(runner, route: str) -> None:
    """/stop and /new: every receipt still open on this chat ends CANCELLED, whichever path would have
    closed it (a held burst that a fence will drop, a queued row the funnel cancels)."""
    for event in list(getattr(runner, "_busy_receipts", {}).get(route, ())):
        close_receipt(event, ProcessingOutcome.CANCELLED)


def queue_text_ack_allowed(runner, event) -> bool:
    """Whether a text message that could not join the running turn is acknowledged in words: not when a
    reaction already is, and only per ``display.dispatcher_receipt`` (auto: reaction else text)."""
    mode = receipt_mode(runner, event.source)
    return mode == "text" or (mode == "auto" and getattr(event, "_busy_receipt", None) is None)
