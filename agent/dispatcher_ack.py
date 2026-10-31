"""Dispatcher mode, model side: the user sees a first line before the work starts.

A dispatcher session answers at once and hands long work to executors, so a turn that goes quiet
(tool calls with no text, or a bare silence marker) looks like a dead bot. Four bounded mechanisms keep
the model honest, all gated on the session-frozen dispatcher flag and on a per-turn *reply obligation*
the surface arms before ``run_conversation`` (``arm_ack_obligation``; a human turn that may not stay
silent, never a cron run, subagent, machinery notification or unaddressed group message):

* a one-shot reminder appended to the first tool result of a turn that has shown no text
  (``queue_ack_reminder``: a new tool row, so no cached byte moves; best effort, a row that does not
  go through ``_commit_tool_result`` carries none);
* a bounded end-of-turn nudge when the turn would end on a bare silence marker with nothing shown
  (``ack_nudge``, the fifth stop gate in ``turn_stop_gates``);
* interim assistant messages forced on (``interim_forced``, consumed once per turn by the gateway
  turn runner for both the commentary consumer and the delegation inline decision);
* the preamble flushed when the first tool call starts, on messaging platforms only
  (``flush_preamble``), instead of after the whole tool-call response is generated.

Knobs live in ``dispatcher.ack`` (``DispatcherSettings``). Nothing here runs for a non-dispatcher agent.
"""

from __future__ import annotations

import contextlib
from typing import Any, Mapping, Optional

from agent.dispatcher_session import (
    DISPATCHER_TOOLSET, NON_DISPATCHER_SURFACES, DispatcherSettings, agent_is_dispatcher, dispatcher_settings,
)

# Flag on the nudge row and on the marker row it answers: both are scaffolding, never durable.
NUDGE_FLAG = "_ack_nudge_synthetic"

ACK_REMINDER_NOTE = (
    "[System note: nothing you have written this turn has reached the user yet. Begin your next message "
    "with one short line saying what you are doing, then carry on.]"
)
ACK_NUDGE_NOTE = (
    "[System note: the user is waiting on you and has seen nothing from you this turn, so staying silent "
    "is not an option here. Write your reply now: a line or two, in the user's language.]"
)

_DEFAULTS = DispatcherSettings()


def _is_silence(text: Any) -> bool:
    from gateway.response_filters import is_intentional_silence_response

    return is_intentional_silence_response(text)


def ack_obligation(agent: Any) -> bool:
    """True while a surface has armed the reply obligation for the running turn."""
    return getattr(agent, "_ack_obligation", False) is True


def ack_settings(agent: Any) -> DispatcherSettings:
    """The ``dispatcher.ack`` knobs captured when the surface armed this turn (code defaults otherwise)."""
    stored = getattr(agent, "_ack_settings", None)
    return stored if isinstance(stored, DispatcherSettings) else _DEFAULTS


def arm_ack_obligation(
    agent: Any, *, display_kind: Optional[str] = None, reply_expected: Optional[bool] = None,
    config: Optional[Mapping[str, Any]] = None, muted: bool = False,
) -> None:
    """Arm (or clear) the reply obligation for the turn about to run.

    Armed only for a dispatcher session on a turn somebody waits on whose silence marker would become
    the "needed a reply" warning (the same ``silence_allowed`` verdict that sets
    ``_silence_marker_needs_reply``). The frozen flag already excludes sessions that cannot receive a
    later result, so this does not re-read the async-delivery context. ``config`` is the turn's own
    (profile-scoped) user config; the knobs are read here, once per turn."""
    from agent.side_question_lease import is_unattended_turn
    from gateway.response_filters import silence_allowed

    armed = (agent_is_dispatcher(agent) and not muted and not is_unattended_turn(agent)
             and not silence_allowed(display_kind, reply_expected))
    agent._ack_obligation = armed
    if armed:
        agent._ack_settings = dispatcher_settings(config)


def disarm_ack_obligation(agent: Any) -> None:
    agent._ack_obligation = False


def turn_has_visible_text(agent: Any) -> bool:
    """Whether the user has seen model text this turn. A bare silence marker never counts, whether it
    was delivered as an interim or streamed live (the streaming consumer records it in the accumulator
    before the final response resolves it as silence). Latched for the rest of the turn: the streamed
    accumulator is reset by every model request."""
    if getattr(agent, "_ack_text_seen", False) is True:
        return True
    delivered = getattr(agent, "_delivered_interim_texts", None)
    streamed = agent._strip_think_blocks(getattr(agent, "_current_streamed_assistant_text", "") or "").strip()
    seen = (any(isinstance(t, str) and t.strip() and not _is_silence(t) for t in delivered or ())
            or (bool(streamed) and not _is_silence(streamed)))
    if seen:
        agent._ack_text_seen = True
    return seen


def queue_ack_reminder(agent: Any) -> None:
    """Queue the one-shot reminder for the tool result being committed: once per turn, only while the
    user has seen nothing. The caller attaches queued notices to that very row in the same call."""
    if not ack_obligation(agent) or getattr(agent, "_ack_reminder_sent", False) is True:
        return
    if not ack_settings(agent).ack_reminder or turn_has_visible_text(agent):
        return
    from agent.plugin_live_notes import queue_plugin_notice

    queue_plugin_notice(agent, "dispatcher.ack", ACK_REMINDER_NOTE)
    agent._ack_reminder_sent = True


def ack_nudge(agent: Any, final_response: Any) -> Optional[str]:
    """The nudge text when the turn is about to end on a bare silence marker with nothing shown, at most
    ``dispatcher.ack.max_nudges`` times a turn; ``None`` lets the existing silence handling run."""
    if not ack_obligation(agent) or getattr(agent, "_interrupt_requested", False) is True:
        return None
    issued = getattr(agent, "_ack_nudges", 0)
    if not isinstance(issued, int) or issued >= ack_settings(agent).ack_max_nudges:
        return None
    if not _is_silence(final_response) or turn_has_visible_text(agent):
        return None
    agent._ack_nudges = issued + 1
    return ACK_NUDGE_NOTE


def end_marker_segment(agent: Any) -> None:
    """Close the display segment holding a streamed marker before the turn continues: the streaming
    consumer holds a bare marker unshown, and the continuation's text would otherwise be appended to it."""
    callback = getattr(agent, "stream_delta_callback", None)
    if callback is None or getattr(agent, "_hold_stream", False):
        return
    if getattr(agent, "_current_streamed_assistant_text", ""):
        with contextlib.suppress(Exception):
            callback(None)


def interim_forced(enabled_toolsets: Any, platform_key: Any, config: Optional[Mapping[str, Any]]) -> bool:
    """Whether a turn's interim assistant messages are forced on: the session carries the dispatcher
    posture toolset (frozen at creation), is not a surface that cannot be a dispatcher, and
    ``dispatcher.ack.force_interim`` is on. ``config`` is the turn's profile-scoped user config."""
    if DISPATCHER_TOOLSET not in (enabled_toolsets or ()):
        return False
    if str(platform_key or "").strip().lower() in NON_DISPATCHER_SURFACES:
        return False
    return dispatcher_settings(config).force_interim


def agent_interim_forced(agent: Any, config: Optional[Mapping[str, Any]]) -> bool:
    """``interim_forced`` for a surface that already holds the built agent (the frozen flag decides)."""
    return agent_is_dispatcher(agent) and dispatcher_settings(config).force_interim


def _messaging_platform(agent: Any) -> bool:
    from gateway.session_context import NON_MESSAGING_SESSION_SURFACES

    return str(getattr(agent, "platform", "") or "").strip().lower() not in NON_MESSAGING_SESSION_SURFACES


def flush_preamble(agent: Any, text: str) -> None:
    """Deliver the text written before the first tool call as an interim as soon as that call starts.

    Runs from the tool-call-started event, so it fires once per tool name; the latch makes it one flush
    per model response. Only where no live delta sink already showed the text, and only on messaging
    platforms: a local viewer rebuilds from the stored rows, which are written after the response.
    The ``on_interim_message`` hook and the normal delivery still run once, after persistence, in
    ``_emit_interim_assistant_message`` (``take_early_interim`` makes it skip only the second send)."""
    if (not ack_obligation(agent) or getattr(agent, "_ack_preamble_flushed", False) is True
            or not ack_settings(agent).early_interim or not isinstance(text, str) or not text.strip()):
        return
    if (getattr(agent, "interim_assistant_callback", None) is None or getattr(agent, "_hold_stream", False)
            or not _messaging_platform(agent)):
        return
    visible = agent._strip_think_blocks(text).strip()
    if not visible:
        return
    agent._ack_preamble_flushed = True
    from agent.conversation_loop import _STALE_MARKER_RE

    if (_is_silence(visible) or _STALE_MARKER_RE.fullmatch(visible)
            or (getattr(agent, "_current_streamed_assistant_text", "") or "").strip()
            or agent._interim_text_was_delivered(visible)):
        return
    agent._deliver_interim(visible, already_streamed=False, record=[visible])
    if agent._interim_text_was_delivered(visible):
        agent._ack_early_interim = agent._normalize_interim_visible_text(visible)


def take_early_interim(agent: Any, visible: str) -> bool:
    """True once for the text ``flush_preamble`` already delivered this turn."""
    early = getattr(agent, "_ack_early_interim", None)
    if early and early == agent._normalize_interim_visible_text(visible):
        agent._ack_early_interim = None
        return True
    return False
