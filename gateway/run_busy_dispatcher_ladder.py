"""Runs the dispatcher busy ladder (redirect, then steer, then queue) for a user message that arrives while a
turn is running. ``agent/busy_input.py`` orders the verbs; this applies them.

The return contract is ``_hm_busy_preempt``'s: True when the text landed inside the running turn (the caller
must not also admit it), False when the caller admits it to the durable FIFO. There is no interrupt rung. A
turn still building its agent also queues here: the planner's 'latch' rung would hold a steer for the agent
the turn adopts, but a latched steer is dropped silently if that build fails, while a queued row is durable
and ordered; the quick lane answers meanwhile.
"""

from __future__ import annotations

import logging
import time

from agent.busy_input import BusyFacts, plan_dispatcher_busy
from agent.i18n import t
from gateway.platforms.event import MessageType
from gateway.run_busy_dispatcher import DispatcherBusyMixin

logger = logging.getLogger("gateway.run")  # log-record parity with gateway/run.py

# The queued-text receipt's own clock: the shared 30 s one belongs to the legacy interrupt/steer ack.
_QUEUE_ACK_DEBOUNCE_S = 5.0


def _plain_text(event) -> bool:
    return event.message_type == MessageType.TEXT and not event.media_urls and not event.media_types


class DispatcherBusyLadderMixin(DispatcherBusyMixin):
    async def _dispatcher_busy_text(self, event, live: bool) -> str:
        """What a busy message carries into the ladder: its text, or a voice note's transcript once the turn
        is live. Other media carry none, so the ladder queues them."""
        if _plain_text(event):
            return (event.text or "").strip()
        urls = event.media_urls or []
        if live and urls and len(self._pending_event_audio_paths(event)) == len(urls):
            return await self._prepare_busy_steer_text(event)
        return ""

    async def _dispatcher_busy_preempt(self, event, source, session_key: str) -> bool:
        from gateway.run import _AGENT_PENDING_SENTINEL
        state = self._peek_session_state(session_key)
        agent = state.turn.agent if state else None
        adapter = self._delivery_adapter_for(source)
        if self._draining or adapter is None or agent is None:
            return False  # drain: refused at admission; no turn: the message is the next turn
        live = agent is not _AGENT_PENDING_SENTINEL
        text = await self._dispatcher_busy_text(event, live)
        compressing = live and bool(text) and await self._session_has_compression_in_flight(session_key)
        redirect = live and getattr(agent, "_supports_active_turn_redirect", False) is True and hasattr(agent, "redirect")
        plan = plan_dispatcher_busy(BusyFacts(
            has_text=bool(text), redirect_supported=redirect, compressing=compressing, agent_building=not live))
        rungs = {"redirect": self._dispatcher_rung_redirect, "steer": self._dispatcher_rung_steer}
        for verb in plan.ladder:
            rung = rungs.get(verb)  # no rung for 'latch' or 'queue': see the module docstring
            if rung is not None and rung(agent, text, session_key, event):
                self._dispatcher_landed(agent, text, source, session_key, event)
                return True
        await self._dispatcher_queued(event, source, session_key, adapter, text, compressing)
        return False

    def _dispatcher_rung_redirect(self, agent, text: str, session_key: str, event) -> bool:
        return self._redirect_active_turn(agent, text, session_key, event)

    def _dispatcher_rung_steer(self, agent, text: str, session_key: str, event) -> bool:
        """Straight to the agent: ``_steer_running_agent`` would also fan the text out to subagents, but a
        dispatcher's executors are detached and the user is talking to the dispatcher."""
        if not hasattr(agent, "steer"):
            return False
        try:
            accepted = bool(agent.steer(self._steer_text_with_origin(text, event)))
        except Exception:
            logger.warning("Dispatcher steer failed for session %s", session_key, exc_info=True)
            return False
        if accepted:
            self._fold_into_running_turn(agent, session_key, event)
        return accepted

    async def _dispatcher_busy_legacy(self, event, source, session_key: str) -> None:
        """The in-process busy path (the routed profile has no session authority): the same ladder, and a
        message that did not land goes to the adapter's FIFO."""
        if not await self._dispatcher_busy_preempt(event, source, session_key):
            self._queue_or_replace_pending_event(session_key, event)

    def _dispatcher_turn_token(self, session_key: str):
        """Identity of the running turn: it changes when that turn ends or another begins."""
        state = self._peek_session_state(session_key)
        turn = state.turn if state else None
        return (turn.started_ts, id(turn.event)) if turn is not None and turn.agent is not None else None

    @staticmethod
    def _dispatcher_unconsumed(agent) -> bool:
        """The accepted steer or redirect still waits for the turn to take it (a stub without the probes
        reports nothing pending)."""
        for probe in ("has_pending_steer", "_has_pending_redirect"):
            check = getattr(agent, probe, None)
            if callable(check) and check() is True:
                return True
        return False

    def _dispatcher_landed(self, agent, text: str, source, session_key: str, event) -> None:
        """A redirect or steer was accepted. A model request takes it within moments, but a running tool
        or an inline-joined subagent holds it until that returns, so watch it: if the same turn still has
        not taken it after ``stall_seconds`` (at once with subagents running), the quick lane answers."""
        cfg = self._dispatcher_busy_cfg(source).settings
        if not (cfg.quick_lane and text and event.reply_expected is not False):
            return
        token = self._dispatcher_turn_token(session_key)
        from gateway.dispatcher_quick_lane import start_quick_lane
        start_quick_lane(
            self, event, source, session_key, text, landed=True,
            delay=0.0 if self._agent_has_active_subagents(agent) else cfg.stall_seconds,
            pending=lambda: token is not None and self._dispatcher_turn_token(session_key) == token
            and self._dispatcher_unconsumed(agent))

    async def _dispatcher_queued(self, event, source, session_key: str, adapter, text: str, compressing: bool) -> None:
        """The message goes to the FIFO behind the running turn. Receipt, then the quick lane: at once
        while compression runs (nothing can take it for a while), else if the turn is still running
        ``stall_seconds`` later (a turn still building its agent, or a long tool call)."""
        await self._dispatcher_queue_ack(event, session_key, adapter, text, compressing)
        cfg = self._dispatcher_busy_cfg(source).settings
        if not (cfg.quick_lane and text and event.reply_expected is not False):
            return
        token = self._dispatcher_turn_token(session_key)
        from gateway.dispatcher_quick_lane import start_quick_lane
        start_quick_lane(
            self, event, source, session_key, text, landed=False, delay=0.0 if compressing else cfg.stall_seconds,
            pending=lambda: token is not None and self._dispatcher_turn_token(session_key) == token)

    async def _dispatcher_queue_ack(self, event, session_key: str, adapter, text: str, compressing: bool) -> None:
        """Text receipt for a text message that could not join the running turn, unless a reaction already
        acknowledged it. Its own short debounce, and none of the legacy ack's onboarding hint (that one
        points at /busy, which a dispatcher session ignores)."""
        from gateway.session_receipt import queue_text_ack_allowed
        if not (text and self._busy_acks_enabled() and queue_text_ack_allowed(self, event)):
            return
        turn = self._session_state(session_key).turn
        now = time.time()
        if now - turn.busy_ack_ts < _QUEUE_ACK_DEBOUNCE_S:
            return
        turn.busy_ack_ts = now
        head = t("gateway.busy.compressing_head" if compressing else "gateway.busy.queued_head")
        tail = self._BUSY_DEMOTED_TAIL if compressing else t("gateway.busy.queued_tail")
        await self._send_busy_ack_reply(event, adapter, t("gateway.busy.ack", head=head, detail="", tail=tail))
