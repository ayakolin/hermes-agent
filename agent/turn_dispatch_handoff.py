"""Dispatch-and-end: a dispatcher turn whose tool calls were all accepted background dispatches
ends without another model call.

The model already told the user what it is doing in the text beside the tool calls (``T``). The
extra call that would only restate it costs a full round trip, so ``run_tool_round`` asks
:func:`dispatch_handoff_reply` whether the round qualifies. The persisted tail never stays a bare
``tool`` row (``finalize_turn`` fails such a turn as ``pending_tool_result`` and the next user
message would land as ``tool -> user``): a hidden assistant boundary row carrying ``T`` closes the
round, and the finalizer persists it once, after ``transform_llm_output``.

Fail closed: any call that is not a plain accepted dispatch (or todo write), a dropped call, a
pending interrupt, redirect or file edit, or no visible ``T`` sends the turn down the normal path.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from agent.dispatcher_session import agent_is_dispatcher
from agent.message_metadata import append_message
from agent.message_sanitization import _tc_field, tool_call_id_variants, tool_result_id_variants

DISPATCH_HANDOFF_EXIT = "dispatch_handoff"
# display_metadata key of the boundary row; its value says whether the user already saw the text. A
# history load skips the row, so the persisted-turn receipt must not address it, and crash recovery
# must not resend a text the user already has.
HANDOFF_ROW_KEY = "dispatch_handoff"

# delegate_task control action -> the "status" of its success result. A spawn is judged separately
# (status, mode and the absence of inline results); "list" returns data the model must read.
_CONTROL_STATUS = {"steer": "queued", "stop": "interrupt_requested"}


def dispatch_handoff_enabled(agent: Any) -> bool:
    """A top-level dispatcher session that kept ``delegation.end_turn_on_dispatch`` on.

    The flag is read at first use and frozen on the agent (like ``agent_joins_delegation``): it
    changes loop control only, never the prompt or tools[] bytes, and a mid-session flip is not
    supported. ``codex_app_server`` runs whole turns outside the tool round, so it never gets here."""
    if not agent_is_dispatcher(agent) or getattr(agent, "_delegate_depth", 0) != 0:
        return False
    if getattr(agent, "api_mode", None) == "codex_app_server":
        return False
    cached = getattr(agent, "_end_turn_on_dispatch", None)
    if not isinstance(cached, bool):
        from tools.delegate_tool import _load_config
        from utils import is_truthy_value

        cached = agent._end_turn_on_dispatch = is_truthy_value(
            _load_config().get("end_turn_on_dispatch", True), default=True)
    return cached


def _trailing_round(messages: list) -> Optional[tuple[dict, list[dict]]]:
    """The latest assistant row with tool calls and the tool rows after it, when those rows answer
    its calls one to one and nothing else follows (a steer row or a compaction rewrite breaks it)."""
    idx = next((i for i in range(len(messages) - 1, -1, -1)
                if isinstance(messages[i], dict) and messages[i].get("role") != "tool"), None)
    if idx is None:
        return None
    assistant, results = messages[idx], messages[idx + 1:]
    calls = assistant.get("tool_calls") if assistant.get("role") == "assistant" else None
    if not calls or len(calls) != len(results):
        return None
    for call, row in zip(calls, results):
        if not (tool_call_id_variants(call) & tool_result_id_variants(row.get("tool_call_id"))):
            return None
    return assistant, results


def _json_dict(raw: Any) -> Optional[dict]:
    """``raw`` parsed as a JSON object; None for anything else (a plugin-appended note included)."""
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else None
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _accepted_dispatch(call: Any, row: dict) -> Optional[str]:
    """``"delegate"`` for an accepted delegate_task call, ``"todo"`` for a successful todo write,
    None for anything the model has to read the outcome of."""
    from tools.todo_tool import TODO_TOOL_NAMES

    fn = _tc_field(call, "function")
    name = _tc_field(fn, "name")
    args, result = _json_dict(_tc_field(fn, "arguments")), _json_dict(row.get("content"))
    if args is None or result is None or result.get("error"):
        return None
    if name in TODO_TOOL_NAMES:
        return "todo" if isinstance(args.get("todos"), list) and "todos" in result else None
    if name != "delegate_task":
        return None
    action = str(args.get("action") or "spawn").strip().lower()
    if action == "spawn":
        accepted = (result.get("status") == "dispatched" and result.get("mode") == "background"
                    and not result.get("inline_results") and not result.get("refused"))
    else:  # list returns data the model must read; an unknown action is a tool error
        accepted = action in _CONTROL_STATUS and result.get("status") == _CONTROL_STATUS[action]
    return "delegate" if accepted else None


def _handoff_text(agent: Any, assistant: dict) -> str:
    """The text the model wrote beside the calls, or "" when there is none worth ending on."""
    from gateway.response_filters import is_intentional_silence_response

    text = agent._interim_assistant_visible_text(assistant)
    return "" if not text or is_intentional_silence_response(text) else text


def _user_saw(agent: Any, assistant: dict, text: str) -> bool:
    """``text`` already reached the user this turn: sent as an interim, or streamed whole as live
    deltas (the classic CLI streams without an interim callback). Codex commentary is delivered per
    part, so each part is checked. Exact match on the streamed text: a prefix could leave the user
    a truncated message with no resend."""
    parts = agent._extract_codex_interim_visible_parts(assistant)
    if parts:
        return all(agent._interim_text_was_delivered(part) for part in parts)
    return agent._interim_text_was_delivered(text) or agent._interim_content_fully_streamed(text)


def dispatch_handoff_reply(agent: Any, messages: list, *, calls_dropped: bool) -> Optional[str]:
    """``T`` when the round just executed may end the turn; None keeps the normal path.

    On success, and only then, appends the hidden assistant boundary row carrying ``T`` and (when the
    user already saw ``T``) names the final a reuse so no surface paints it twice. The row is not
    flushed here: ``finalize_turn`` persists it after ``transform_llm_output`` so a rewrite reaches
    the stored transcript too."""
    if calls_dropped or not agent_is_dispatcher(agent):
        return None
    trailing = _trailing_round(messages)
    if trailing is None or not dispatch_handoff_enabled(agent):
        return None
    if (getattr(agent, "_interrupt_requested", False) or agent._has_pending_redirect()
            or getattr(agent, "_hold_stream", False) is True or getattr(agent, "_turn_file_mutation_paths", None)):
        return None
    assistant, results = trailing
    kinds = [_accepted_dispatch(call, row) for call, row in zip(assistant["tool_calls"], results)]
    if None in kinds or "delegate" not in kinds:
        return None
    text = _handoff_text(agent, assistant)
    if not text:
        return None
    seen = _user_saw(agent, assistant, text)
    append_message(messages, {
        "role": "assistant", "content": text, "display_kind": "hidden",
        "display_metadata": {HANDOFF_ROW_KEY: seen},
    })
    if seen:
        agent._response_was_previewed = True
        agent._reused_response_text = text
    return text
