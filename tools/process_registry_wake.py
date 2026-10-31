"""Wake an asyncio consumer of ``ProcessRegistry.completion_queue`` the moment a producer puts an event.

Every producer (process exit, watch match, heartbeat, async-delegation completion) calls plain
``completion_queue.put``, so the wake lives in the queue: ``WakingQueue.put`` notifies the registry's
``CompletionWake``, and a consumer parks on a flag instead of polling. A consumer that hands an event back
(a refusal it will retry later) uses ``requeue``, which does NOT notify: a notifying put would wake the
consumer that just requeued, and it would spin on its own backoff items. A requeue that must be seen again
at a known time passes ``wake_in`` so a timed wake is armed for it, because the only other thing that would
wake a parked consumer is the next unrelated put.

A consumer subscribes through the registry's wake (``registry.completion_wake``), never through the queue
object, so a test or embedder that swaps ``completion_queue`` keeps a consumer that is still woken by
whatever puts through the registry.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
from typing import Callable

logger = logging.getLogger(__name__)


class CompletionWake:
    """Fan-out of "something was queued" to asyncio flags living on other threads' loops."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subscribers: dict[int, tuple[asyncio.AbstractEventLoop, asyncio.Event]] = {}
        self._next = 0

    def subscribe(self, flag: asyncio.Event) -> Callable[[], None]:
        """Set ``flag`` (on its loop) on every notify until the returned callable is called. Must be
        called from the loop that awaits ``flag``. Unsubscribe in a ``finally``: a respawned consumer
        otherwise leaves a callback on a closed loop behind."""
        loop = asyncio.get_running_loop()
        with self._lock:
            token = self._next
            self._next += 1
            self._subscribers[token] = (loop, flag)

        def unsubscribe() -> None:
            with self._lock:
                self._subscribers.pop(token, None)
        return unsubscribe

    def notify(self) -> None:
        """Thread-safe; never raises into the producer."""
        for token, loop, flag in self._snapshot():
            self._schedule(token, loop, flag.set)

    def notify_in(self, delay: float) -> None:
        """Wake the subscribers once ``delay`` seconds from now (immediately when not positive)."""
        if delay <= 0:
            self.notify()
            return
        for token, loop, flag in self._snapshot():
            self._schedule(token, loop, lambda lp=loop, f=flag: lp.call_later(delay, f.set))

    def _snapshot(self) -> list:
        with self._lock:
            return [(token, loop, flag) for token, (loop, flag) in self._subscribers.items()]

    def _schedule(self, token: int, loop: asyncio.AbstractEventLoop, fn: Callable[[], object]) -> None:
        try:
            loop.call_soon_threadsafe(fn)
        except RuntimeError:  # the subscriber's loop is closed: drop it, the producer never learns
            logger.debug("Dropping a completion-wake subscriber on a closed loop")
            with self._lock:
                self._subscribers.pop(token, None)


class WakingQueue(queue.Queue):
    """``queue.Queue`` whose ``put`` also notifies ``wake``. ``get``/``take_ready`` are unchanged, so the
    in-flight convention of ``process_registry_awaiting`` holds as before."""

    def __init__(self, wake: CompletionWake, maxsize: int = 0) -> None:
        super().__init__(maxsize)
        self.wake = wake

    def put(self, item, block: bool = True, timeout: float | None = None) -> None:
        super().put(item, block, timeout)
        self.wake.notify()


def new_completion_queue() -> tuple[CompletionWake, WakingQueue]:
    wake = CompletionWake()
    return wake, WakingQueue(wake)


def requeue(target: queue.Queue, evt, *, wake_in: float | None = None) -> None:
    """Hand ``evt`` back to ``target`` without waking the consumers (``queue.Queue.put``, bypassing the
    notifying override). ``wake_in``: also wake them that many seconds from now through the process
    registry's wake, for an event the caller will refuse again until then."""
    queue.Queue.put(target, evt)
    if wake_in is None:
        return
    from tools.process_registry import process_registry
    wake = getattr(process_registry, "completion_wake", None)
    if wake is not None:
        wake.notify_in(wake_in)
