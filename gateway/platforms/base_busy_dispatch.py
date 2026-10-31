"""Adapter-side hooks of the dispatcher busy path (``gateway/run_busy_dispatcher.py``).

``BasePlatformAdapter`` inherits this mixin (base.py is over its line cap, so new adapter behaviour lives in
a sibling): the one explicit "does this adapter put a visible receipt on a message" capability, and the
inbound text-batch quiet period with the floor a busy dispatcher session gets.
"""

from __future__ import annotations

from typing import Optional

from gateway.platforms.event import MessageEvent


class BusyDispatchAdapterMixin:
    def acknowledges_event(self, event: MessageEvent) -> bool:
        """Whether ``on_processing_start(event)`` would put a visible mark on the user's message (a
        reaction), so a dispatcher session's busy-path receipt can use it instead of sending text.

        Pure and cheap: it runs on the event loop at arrival, before anything knows whether the turn
        will be answered, steered or queued. An adapter that returns True promises that start and
        complete only swap the mark; one whose hooks also write durable per-message state (Discord's
        missed-message ledger would record a steered message as responded) must stay False. Each
        reaction-capable adapter evaluates its own enabled flag and tracking set; the default is False."""
        return False

    def _text_batch_delay_for(self, pending: Optional[MessageEvent]) -> float:
        """Quiet period before ``pending`` is dispatched; near-split chunks wait longer."""
        last_len = getattr(pending, "_last_chunk_len", 0) if pending is not None else 0
        return self._text_batch_split_delay_seconds if last_len >= self._SPLIT_THRESHOLD else self._text_batch_delay_seconds

    def _text_batch_wait(self, key: str) -> float:
        """Seconds to hold the pending text batch ``key`` before dispatching it. The adapter's quiet
        period, except for a text-only chunk of a session whose dispatcher is busy: a user message there
        never waits on a merge window, so it dispatches at once. A chunk that may continue (near the
        platform's split point, or any hold longer than the normal quiet period such as a bot handoff)
        and a batch carrying media keep their wait, which is what merges a client-side split."""
        pending = self._pending_text_batches.get(key)
        delay = self._text_batch_delay_for(pending)
        if delay <= 0 or pending is None or pending.media_urls:
            return delay
        if delay > self._text_batch_delay_seconds or getattr(pending, "_last_chunk_len", 0) >= self._SPLIT_THRESHOLD:
            return delay
        floor = getattr(getattr(self, "gateway_runner", None), "_dispatcher_text_batch_floor", None)
        return 0.0 if callable(floor) and floor(pending) is True else delay
