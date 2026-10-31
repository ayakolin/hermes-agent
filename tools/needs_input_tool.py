"""``needs_input``: how a dispatcher session's executor hands a question back instead of guessing.

An executor cannot ask the user anything (``clarify`` is blocked for every child) and has no approver once the
dispatcher's turn ends. When it cannot continue it calls this once. The call records the request on the child and
ends the child's turn deterministically with an interrupt carrying a system reason, so no further model call runs
and the empty-response ladder never sees an empty follow-up. ``attach_needs_input`` then reports the run as
``status=completed`` (its question is the deliverable) and adds a ``needs_input`` block to the result entry, which
the completion formatters print for the dispatcher to relay to the user.

The ``executor`` toolset carrying it is platform-gated to ``subagent`` and added by code to a dispatcher
session's children only (``agent/executor_baseline.py``), so it is in no other session's tools[].
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Optional

from agent.redact import redact_sensitive_text
from tools.registry import registry, tool_error

KINDS = ("question", "approval", "blocked")
MAX_OPTIONS = 4
_MAX_QUESTION, _MAX_DETAIL, _MAX_OPTION = 600, 1500, 80

# The system reason the child's turn is interrupted with. ``interrupt_issuer`` turns it into the slug below, which
# is the tail of the turn exit reason ``interrupted_by_system(needs_input)``.
INTERRUPT_REASON = "needs input"
EXIT_REASON = "interrupted_by_system(needs_input)"


def _clip(value: Any, limit: int) -> str:
    text = redact_sensitive_text(str(value or "").strip())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def needs_input_tool(kind: Any, question: Any, options: Any = None, detail: Any = None, *, agent: Any = None) -> str:
    """Record the request on ``agent`` and end its turn. Validation errors go back to the model as tool errors."""
    if agent is None:
        return tool_error("needs_input only works inside a running subagent.")
    if kind not in KINDS:
        return tool_error(f"kind must be one of: {', '.join(KINDS)}.")
    if not isinstance(question, str) or not question.strip():
        return tool_error("question is required: say exactly what you need from the user.")
    if getattr(agent, "_needs_input", None) is not None:
        return json.dumps({"status": "already_recorded", "note": "Your run is ending. Make no further tool calls."})
    choices = [_clip(item, _MAX_OPTION) for item in options[:MAX_OPTIONS]] if isinstance(options, list) else []
    agent._needs_input = {
        "kind": kind, "question": _clip(question, _MAX_QUESTION),
        "options": [item for item in choices if item], "detail": _clip(detail, _MAX_DETAIL),
    }
    agent.interrupt(tool_reason=INTERRUPT_REASON)
    return json.dumps({"status": "recorded", "note": "Your run is ending. Make no further tool calls."})


def attach_needs_input(entry: dict[str, Any], child: Any, result: Mapping[str, Any]) -> None:
    """Add the child's pending request to its result entry. The run ended on its own ``needs_input`` call (the
    turn exit reason says so; a stop that raced it does not count), so it is reported as completed: the interrupt
    that ended the turn is not a failure, and the request is what the dispatcher must act on."""
    pending = getattr(child, "_needs_input", None)
    if not isinstance(pending, dict) or result.get("turn_exit_reason") != EXIT_REASON:
        return
    placeholders = {"", str(result.get("final_response") or "").strip(), "Operation interrupted."}
    work = str(entry.get("summary") or "").strip()
    entry.update(
        status="completed", exit_reason="completed", truncated=False, needs_input=dict(pending),
        summary="" if work in placeholders else work,
    )
    entry["summary"] = entry["summary"] or pending["question"]
    entry.pop("error", None)
    entry.pop("failure_reason", None)


def needs_input_lines(r: Mapping[str, Any]) -> list[str]:
    """The block a completion formatter prints for a result entry that carries ``needs_input``."""
    ask = r["needs_input"]
    lines = [
        f"THIS SUBAGENT STOPPED TO ASK THE USER ({ask.get('kind', 'question')}); the task is NOT done.",
        f"Question: {ask.get('question', '')}",
    ]
    if ask.get("options"):
        lines.append("Options: " + " | ".join(str(option) for option in ask["options"]))
    if ask.get("detail"):
        lines.append(f"Detail: {ask['detail']}")
    lines.append(
        "Relay the question to the user in your own words and wait for their answer; never answer it for them. Then "
        "start a new subagent with the original goal, the work so far (below) and their answer: a finished subagent "
        "cannot be resumed.")
    if ask.get("kind") == "approval":
        lines.append(
            "An approval is not carried over: the new subagent meets the same refusal for the same command. Tell "
            "the user it was blocked and that they can run it themselves or change how approvals work here.")
    return lines


NEEDS_INPUT_SCHEMA = {
    "name": "needs_input",
    "description": (
        "Ask the user something you cannot work out yourself, and end your run. Use it for a decision only they "
        "can make, information only they have, or a command that was refused because nobody is available to "
        "approve it. Call it once; say in the same message what you finished and what is left. Do not call it "
        "for anything you can resolve with your tools."
    ),
    "parameters": {
        "type": "object",
        "required": ["kind", "question"],
        "properties": {
            "kind": {"type": "string", "enum": list(KINDS),
                     "description": "question: a decision or missing fact. approval: a refused command. blocked: "
                                    "something outside your reach stops the work."},
            "question": {"type": "string", "description": "What you need, in one or two sentences."},
            "options": {"type": "array", "maxItems": MAX_OPTIONS, "items": {"type": "string"},
                        "description": "Short answers to pick from, when there are a few."},
            "detail": {"type": "string", "description": "Context the user needs to answer (the command, the error)."},
        },
    },
}


def _registry_handler(args: dict, **_kw: Any) -> Optional[str]:
    return needs_input_tool(args.get("kind"), args.get("question"), args.get("options"), args.get("detail"))


registry.register(
    name="needs_input", toolset="executor", schema=NEEDS_INPUT_SCHEMA,
    # The agent loop runs it inline with the child (agent/inline_tool_executors.py); a bare registry dispatch has
    # no agent to end, so it only reports that.
    handler=_registry_handler, emoji="❓",
)
