"""The quick lane: a short answer from a cache-parity fork while a dispatcher's turn cannot take the user's
message yet.

Redirect and steer reach a running turn at once only while it is in a model request. During compression, a
long tool call, a turn still building its agent, or an inline-joined subagent the message waits. The lane
covers that: after a delay (none for compression) it asks a fork of the conversation to answer the one
message. The fork writes nothing to the transcript and calls no tool; its reply goes out as an interim send.

Lifecycle: one ``QuickLane`` per conversation (``ConversationState.quick_lane``), created by the first
watcher and cancelled by the /stop and /new funnel BEFORE ``ConversationState.clear()`` drops the reference.
Before every send the lane re-checks that it was not stopped, that it is still the conversation's lane, that
no /stop or /new fenced the route since, and that the message is still waiting (``pending``): a late reply
never lands in a new conversation, after a stop, or beside the turn's own answer. At most one watcher
answers at a time; the others back off and the main turn answers their message.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger("gateway.run")  # log-record parity with gateway/run.py

_MAX_WATCHERS = 8
_NOTE_CHARS = 240
_MAX_LANE_NOTES = 4

FRAMING = (
    "You are writing a quick reply on behalf of the assistant in the conversation above. The assistant's main "
    "thread is busy and cannot take the user's newest message yet.\nRules:\n- Answer ONLY that message, in a "
    "sentence or two, using the conversation and the live status below.\n- Do NOT call any tools; they are "
    "disabled.\n- Do not promise or claim anything the status does not show: if you cannot answer from what "
    "you can see, say the main thread will get to it.\n- Do not redo or continue the main task."
)

_NOTE = (
    "[Quick reply already sent] While you were busy, a stopgap reply to the user's message \"{ask}\" went "
    "out: \"{reply}\". Do not repeat that reply; handle whatever in the message still needs doing."
)


@dataclass(eq=False)
class QuickLane:
    epoch: int
    stop: threading.Event = field(default_factory=threading.Event)
    tasks: set = field(default_factory=set)
    answering: bool = False
    fork: Any = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def set_fork(self, fork) -> None:
        """``answer_side_question``'s ``on_fork``: a stop that came first interrupts the fork at once."""
        with self._lock:
            self.fork = fork
        if fork is not None and self.stop.is_set():
            self._interrupt(fork, "quick lane stopped")

    def hard_stop(self, message: str) -> None:
        self.stop.set()
        with self._lock:
            fork = self.fork
        if fork is not None:
            self._interrupt(fork, message)

    @staticmethod
    def _interrupt(fork, message: str) -> None:
        from agent.interrupt_compat import request_hard_interrupt
        request_hard_interrupt(fork, message)


def cancel_quick_lane(runner, session_key: str) -> None:
    """Stop the conversation's lane: its watchers, and the fork when one is answering. Safe from any
    thread and when there is no lane."""
    state = runner._peek_session_state(session_key)
    lane = state.conversation.quick_lane if state is not None else None
    if lane is None:
        return
    state.conversation.quick_lane = None
    lane.hard_stop("conversation stopped")
    for task in list(lane.tasks):
        try:
            task.get_loop().call_soon_threadsafe(task.cancel)
        except RuntimeError:  # its loop is already closed: nothing left to cancel
            logger.debug("Quick lane watcher for %s outlived its loop", session_key)


def start_quick_lane(
    runner, event, source, session_key: str, text: str, *, landed: bool, delay: float,
    pending: Callable[[], bool],
) -> None:
    """Watch ``event`` (whose text is ``text``): after ``delay`` seconds, if ``pending()`` still holds, answer
    it from a fork. ``landed``: the text went into the running turn (redirect or steer) rather than the
    FIFO, which decides where the 'already answered' note goes."""
    state = runner._session_state(session_key)
    lane = state.conversation.quick_lane
    if lane is None:
        lane = state.conversation.quick_lane = QuickLane(runner._route_stop_epoch(session_key))
    if lane.answering or len(lane.tasks) >= _MAX_WATCHERS:
        return
    task = asyncio.create_task(_watch(runner, lane, event, source, session_key, text, landed, delay, pending))
    for owner in (lane.tasks, runner._background_tasks):
        owner.add(task)
        task.add_done_callback(owner.discard)


async def _watch(runner, lane, event, source, session_key, text, landed, delay, pending) -> None:
    try:
        if delay > 0:
            await asyncio.sleep(delay)
        async with runner._async_profile_scope_for_source(source):
            if lane.answering or not _open(runner, lane, session_key, pending):
                return
            lane.answering = True
            try:
                await _answer_and_send(runner, lane, event, source, session_key, text, landed, pending)
            finally:
                lane.answering = False
    except Exception:
        if not lane.stop.is_set():  # a stopped lane's fork and fallback end in errors on purpose
            logger.warning("Quick lane for %s failed", session_key, exc_info=True)


def _open(runner, lane, session_key: str, pending: Callable[[], bool]) -> bool:
    """The lane may still speak: not stopped, still this conversation's lane, no /stop or /new since it
    began, and the message it answers is still waiting."""
    state = runner._peek_session_state(session_key)
    return (not lane.stop.is_set() and state is not None and state.conversation.quick_lane is lane
            and runner._route_stop_epoch(session_key) == lane.epoch and bool(pending()))


async def _answer_and_send(runner, lane, event, source, session_key, text, landed, pending) -> None:
    from agent.side_question import answer_side_question
    inputs, _refusal = await runner._side_question_inputs(source, require_history=False)
    if inputs is None:
        return  # no provider to ask: the main turn answers
    history, main_runtime = inputs
    answer = await asyncio.to_thread(
        answer_side_question, _compose_question(session_key, runner, text), history,
        parent_agent=runner._resident_agent_for(session_key), main_runtime=main_runtime,
        framing=FRAMING, on_fork=lane.set_fork, should_stop=lane.stop.is_set)
    adapter = runner._delivery_adapter_for(source)
    if not (answer and adapter is not None and _open(runner, lane, session_key, pending)):
        return
    sent = await adapter.send(source.chat_id, answer,
                              metadata={**(runner._reply_metadata(event) or {}), "_interim_send": True})
    if getattr(sent, "success", False):
        _leave_note(runner, session_key, event, text, answer, landed)


def _clip(text, limit: int = _NOTE_CHARS) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _compose_question(session_key: str, runner, text: str) -> str:
    """The fork's final user message. The transcript snapshot ends before the running turn (a mid-turn
    tail is trimmed so alternation holds), so the turn's own request, its progress and the user's newest
    message all travel here."""
    return f"{_status_block(session_key, runner)}\n\nThe user's newest message:\n{text}"


def _status_block(session_key: str, runner) -> str:
    import time
    state = runner._peek_session_state(session_key)
    turn = state.turn if state is not None else None
    lines = ["Live status of the main thread (it is running right now):"]
    if opener := getattr(getattr(turn, "event", None), "text", ""):
        lines.append(f"- It is working on the user's earlier message: \"{_clip(opener)}\"")
    if turn is not None and turn.started_ts:
        lines.append(f"- It has been running for {int(time.time() - turn.started_ts)} seconds.")
    activity = getattr(getattr(turn, "agent", None), "get_activity_summary", None)
    summary = activity() if callable(activity) else None
    if isinstance(summary, dict) and summary.get("current_tool"):
        lines.append(f"- It is running the tool: {summary['current_tool']}")
    from tools.async_delegation import list_async_delegations
    running = [item for item in list_async_delegations()
               if item.get("session_key") == session_key and item.get("status") in ("running", "stalling")]
    lines.extend(f"- A background executor is working on: {_clip(item.get('goal') or '', 80)}" for item in running[:5])
    return "\n".join(lines)


def _leave_note(runner, session_key: str, event, ask: str, reply: str, landed: bool) -> None:
    """Tell the main thread the message was answered, in a channel of its own (never as the user's words).
    Text that landed in the running turn gets a live plugin notice, which rides the turn's next tool result
    or else the next user message. Text still in the FIFO gets a sidecar note on the user turn that runs it."""
    from gateway.run import _AGENT_PENDING_SENTINEL
    note = _NOTE.format(ask=_clip(ask), reply=_clip(reply))
    state = runner._peek_session_state(session_key)
    agent = state.turn.agent if state is not None else None
    if landed and agent is not None and agent is not _AGENT_PENDING_SENTINEL:
        from agent.plugin_live_notes import queue_plugin_notice
        queue_plugin_notice(agent, f"dispatcher.quick_reply.{event.message_id or id(event)}", note)
    elif state is not None:
        state.conversation.lane_notes = [*state.conversation.lane_notes, note][-_MAX_LANE_NOTES:]
