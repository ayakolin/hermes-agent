"""The gateway's idle completion consumer: woken by a producer's put, not by a poll.

``GatewayCompletionWakeMixin._async_delegation_watcher`` parks on a flag that ``ProcessRegistry.completion_wake``
sets whenever anything is put on ``completion_queue`` (tools/process_registry_wake.py), waits
``delegation.completion_batch_window_seconds`` so completions that land together become one turn, then makes one
pass over a single ``take_ready`` snapshot: watch events are routed, async-delegation completions are delivered per
route group, and anything else (a process 'completion' belongs to its per-process watcher) is dropped. The flag is
cleared BEFORE the snapshot, so a put that lands after it always sets the flag again and the next park returns at
once.

A parked consumer is woken by three things: a put, a timed wake armed by whoever requeued a refused event
(``process_registry_wake.requeue(..., wake_in=)``), and its own timeout, which is the soonest of the next orphan
sweep and the earliest retry stamp of an event still queued. The timeout also covers a swapped-in plain queue that
never notifies. Time arithmetic lives in ``gateway.run_notifications`` (``_until``) so one module clock moves
every backoff.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from gateway.run_shutdown import _log_suppressed

logger = logging.getLogger("gateway.run")

# Events the gateway routes itself; async_delegation is delivered per group; every other type is dropped here.
WATCH_EVENT_TYPES = frozenset({
    "watch_match", "watch_disabled", "watch_overflow_tripped", "watch_overflow_released", "heartbeat"})

DEFAULT_COMPLETION_BATCH_WINDOW_S = 0.3
MAX_COMPLETION_BATCH_WINDOW_S = 5.0
# A parked watcher never sleeps past this, so a lost wake costs a bounded delay rather than a stall.
_PARK_CEILING_S = 30.0
# Floor under the park timeout: a due event nothing consumed would otherwise make the loop spin.
_PARK_FLOOR_S = 0.25


def completion_batch_window(config: Any) -> float:
    """``delegation.completion_batch_window_seconds`` from raw gateway YAML: seconds to wait after a wake so
    completions that land together become one turn. Default 0.3, clamped to [0, 5]; 0 disables batching."""
    from hermes_cli.config import cfg_get
    raw = cfg_get(config, "delegation", "completion_batch_window_seconds", default=DEFAULT_COMPLETION_BATCH_WINDOW_S)
    try:
        return min(max(float(raw), 0.0), MAX_COMPLETION_BATCH_WINDOW_S)
    except (TypeError, ValueError):
        return DEFAULT_COMPLETION_BATCH_WINDOW_S


def _is_async_delegation(evt: Any) -> bool:
    return isinstance(evt, dict) and evt.get("type") == "async_delegation"


def earliest_retry_delay(completion_queue) -> Optional[float]:
    """Seconds until the soonest ``_watch_next_retry_at`` stamp among queued events (``None`` when none is
    stamped). Reads the queue under its mutex; a queue without one (a stand-in) reports ``None``."""
    mutex = getattr(completion_queue, "mutex", None)
    if mutex is None:
        return None
    from gateway.run_notifications import _until
    with mutex:
        stamps = [evt["_watch_next_retry_at"] for evt in completion_queue.queue
                  if isinstance(evt, dict) and evt.get("_watch_next_retry_at") is not None]
    return min((_until(stamp) for stamp in stamps), default=None)


class GatewayCompletionWakeMixin:
    """The event-driven idle consumer of ``completion_queue`` for ``GatewayRunner``."""

    async def _async_delegation_watcher(self) -> None:
        """Drain async completions and pattern notifications even while sessions are idle.

        Background subagents and process pattern events have no per-process notification consumer; both
        must progress without a later foreground turn."""
        from gateway.run import _load_gateway_config
        from tools.async_delegation import ORPHAN_SWEEP_INTERVAL_S
        from tools.process_registry import process_registry as _pr
        loop = asyncio.get_running_loop()
        window = completion_batch_window(await asyncio.to_thread(_load_gateway_config))
        flag = asyncio.Event()
        unsubscribe = _pr.completion_wake.subscribe(flag)
        next_sweep = loop.time()
        try:
            woken = False  # the first pass drains whatever was queued before this watcher existed
            while self._running:
                if woken and window > 0:
                    await asyncio.sleep(window)  # completions that land together become one turn
                if loop.time() >= next_sweep:
                    next_sweep = loop.time() + ORPHAN_SWEEP_INTERVAL_S
                    # Completions whose owner process died while this gateway runs (#97202).
                    with _log_suppressed(logging.DEBUG, "Orphaned async completion sweep failed: %s"):
                        await asyncio.to_thread(self._sweep_orphaned_completion_ledgers)
                flag.clear()  # before the snapshot: a put after it sets the flag again
                with _log_suppressed(logging.DEBUG, "Async delegation watcher error: %s"):
                    await self._completion_watcher_pass(_pr.completion_queue)
                retry = earliest_retry_delay(_pr.completion_queue)
                timeout = min(next_sweep - loop.time(), _PARK_CEILING_S, *([retry] if retry is not None else []))
                woken = await self._park_until_completion(flag, max(timeout, _PARK_FLOOR_S))
        finally:
            unsubscribe()

    @staticmethod
    async def _park_until_completion(flag: asyncio.Event, timeout: float) -> bool:
        """Wait for ``flag`` (a put or a timed wake) up to ``timeout``; whether it was set. The seam tests
        patch to bound the watcher loop."""
        try:
            await asyncio.wait_for(flag.wait(), timeout)
        except TimeoutError:
            return False
        return True

    async def _completion_watcher_pass(self, completion_queue) -> None:
        """One pass over a single ``take_ready`` snapshot, held in flight until routing returns."""
        from tools.process_registry_awaiting import release, take_ready
        ready = take_ready(completion_queue, 0)
        try:
            watch = [evt for evt in ready if evt.get("type", "completion") in WATCH_EVENT_TYPES]
            try:
                await self._route_watch_events(completion_queue, watch)
            finally:  # a routing failure must not strand the async completions taken in the same snapshot
                await self._deliver_async_events(completion_queue, [evt for evt in ready if _is_async_delegation(evt)])
        finally:
            release(ready)

    async def _deliver_async_events(self, completion_queue, async_events: list[dict]) -> None:
        """A fan-out finishing together yields N completions for one session; group by full route + parent
        session so each group becomes ONE consolidated turn (#70300). Events for different sessions never
        coalesce."""
        from gateway.run_notifications import _until
        from tools.process_registry_wake import requeue
        groups: dict[tuple[str, ...], list[dict]] = {}
        for evt in async_events:
            self._enrich_async_delegation_routing(evt)
            groups.setdefault(self._event_route_key(evt, self._ASYNC_GROUP_KEY_FIELDS), []).append(evt)
        for group in groups.values():
            if any(_until(evt.get("_watch_next_retry_at", 0)) > 0 for evt in group):
                for evt in group:
                    requeue(completion_queue, evt)  # held: the stamping requeue armed its wake
                continue
            try:
                delivered = await self._deliver_async_delegation_group(group)
                if delivered is not False:
                    continue
            except Exception:
                logger.exception("Async delegation injection error")
            self._back_off_refused_completions(group)
            for evt in group:
                requeue(completion_queue, evt, wake_in=_until(evt["_watch_next_retry_at"]))

    @staticmethod
    def _back_off_refused_completions(group: list[dict]) -> None:
        """Space out an undelivered async group's retries instead of re-offering it on every pass. The
        backoff only paces; the durable ledger decides when a row gives up (a permanent refusal spends a
        delivery attempt there, so the count survives restarts) (#141153)."""
        from gateway.run_notifications import _refusal_retry_at
        for evt in group:
            step = int(evt.get("_watch_retry_step") or 0) + 1
            evt["_watch_retry_step"] = step
            evt["_watch_next_retry_at"] = _refusal_retry_at(step)
