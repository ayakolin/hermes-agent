"""``needs_input``: the executor-only tool that hands a question back, and how the dispatcher sees it.

Dispatch goes through the real ``INLINE_TOOL_EXECUTORS`` table, gating through the real toolset tables, and the
completion text through the real ``format_process_notification``.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from agent.inline_tool_executors import INLINE_TOOL_EXECUTORS, InlineToolContext
from toolsets import session_platform_tool_drops
from tools.delegate_tool_toolsets import DELEGATE_BLOCKED_TOOLS
from tools.needs_input_tool import EXIT_REASON, INTERRUPT_REASON, attach_needs_input
from tools.process_registry_notifications import async_delegation_display_text, format_process_notification

CTX = InlineToolContext(effective_task_id="t")


class StubChild:
    """The slice of an agent ``needs_input`` touches: its own record slot and ``interrupt``."""

    def __init__(self):
        self.interrupts: list[dict] = []

    def interrupt(self, message=None, **kwargs):
        self.interrupts.append({"message": message, **kwargs})
        return True


def _call(child, **args):
    return INLINE_TOOL_EXECUTORS["needs_input"](child, args, CTX)


def test_a_call_records_the_request_and_ends_the_childs_turn_with_a_system_reason():
    child = StubChild()

    out = json.loads(_call(child, kind="question", question="Which key?", options=["prod", "staging"], detail="d"))

    assert out["status"] == "recorded"
    assert child._needs_input == {"kind": "question", "question": "Which key?", "options": ["prod", "staging"],
                                  "detail": "d"}
    assert child.interrupts == [{"message": None, "tool_reason": INTERRUPT_REASON}]


def test_bad_calls_are_tool_errors_the_model_can_correct_and_end_nothing():
    child = StubChild()

    assert "kind must be one of" in json.loads(_call(child, kind="chat", question="q"))["error"]
    assert "question is required" in json.loads(_call(child, kind="question", question="  "))["error"]
    assert "question is required" in json.loads(_call(child, kind="blocked"))["error"]
    assert child.interrupts == [] and getattr(child, "_needs_input", None) is None


def test_options_detail_and_question_are_bounded_and_secrets_are_scrubbed():
    child = StubChild()
    secret = "sk-proj-" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4"

    _call(child, kind="blocked", question="q " * 600, options=[f"o{i}" for i in range(9)] + ["x" * 500],
          detail=f"export OPENAI_API_KEY={secret} " + "d" * 4000)

    recorded = child._needs_input
    assert len(recorded["question"]) <= 600 and len(recorded["detail"]) <= 1500
    assert recorded["options"] == ["o0", "o1", "o2", "o3"]
    assert secret not in recorded["detail"]


def test_a_second_call_changes_nothing():
    child = StubChild()
    _call(child, kind="question", question="first")

    again = json.loads(_call(child, kind="approval", question="second"))

    assert again["status"] == "already_recorded" and child._needs_input["question"] == "first"
    assert len(child.interrupts) == 1


def test_a_bare_registry_dispatch_has_no_agent_to_end_and_says_so():
    from tools.registry import registry

    out = json.loads(registry.dispatch("needs_input", {"kind": "question", "question": "q"}))

    assert "only works inside a running subagent" in out["error"]


def test_only_a_subagent_session_ever_carries_the_tool():
    for platform in ("cli", "tui", "telegram", "desktop", None):
        assert "needs_input" in session_platform_tool_drops(platform)
    assert "needs_input" not in session_platform_tool_drops("subagent")

    import model_tools
    assert "needs_input" not in model_tools._select_tool_names(None, None, True)
    assert "needs_input" not in DELEGATE_BLOCKED_TOOLS and "clarify" in DELEGATE_BLOCKED_TOOLS


def _interrupted_result(**extra):
    return {"interrupted": True, "final_response": None, "turn_exit_reason": EXIT_REASON, "api_calls": 1, **extra}


def _entry(summary=None, **extra):
    return {"task_index": 0, "status": "interrupted", "exit_reason": "interrupted", "truncated": False,
            "summary": summary, **extra}


def _asking_child(**request):
    return SimpleNamespace(_needs_input={"kind": "question", "question": "Which key?", "options": [], "detail": "",
                                         **request})


def test_a_run_that_ended_on_needs_input_is_completed_and_carries_the_request():
    entry = _entry(summary="Finished steps 1-2.", error="interrupted note")

    attach_needs_input(entry, _asking_child(), _interrupted_result())

    assert entry["status"] == "completed" and entry["exit_reason"] == "completed" and entry["truncated"] is False
    assert entry["needs_input"]["question"] == "Which key?"
    assert entry["summary"] == "Finished steps 1-2." and "error" not in entry


def test_with_no_work_text_the_question_is_the_summary():
    entry = _entry(summary="Operation interrupted.")

    attach_needs_input(entry, _asking_child(), _interrupted_result(final_response="Operation interrupted."))

    assert entry["status"] == "completed" and entry["summary"] == "Which key?"


@pytest.mark.parametrize("result", [
    _interrupted_result(turn_exit_reason="interrupted_by_user"),  # a stop that raced the call
    {"final_response": "all done", "completed": True, "turn_exit_reason": "text_response(stop)"},
])
def test_a_run_that_ended_any_other_way_is_left_alone(result):
    entry = _entry(summary="partial")

    attach_needs_input(entry, _asking_child(), result)

    assert entry == _entry(summary="partial")


def _event(entry, **extra):
    return {"type": "async_delegation", "delegation_id": "d1", "is_batch": True, "goals": ["deploy the site"],
            "results": [entry], **extra}


def _completed_ask(**request):
    entry = _entry(summary="Built the site; deploy key missing.")
    attach_needs_input(entry, _asking_child(**request), _interrupted_result())
    return entry


def test_the_batch_notification_prints_the_question_and_tells_the_dispatcher_not_to_answer_it():
    text = format_process_notification(_event(_completed_ask(options=["prod", "staging"], detail="needs DEPLOY_KEY")))

    assert "NEEDS INPUT" in text and "--- ? TASK 1/1" in text and "--- ✓ TASK" not in text
    assert "Question: Which key?" in text and "Options: prod | staging" in text and "Detail: needs DEPLOY_KEY" in text
    assert "never answer it for them" in text and "cannot be resumed" in text
    assert "Work so far:\nBuilt the site; deploy key missing." in text


def test_an_approval_request_warns_that_consent_does_not_carry_to_a_new_executor():
    text = format_process_notification(_event(_completed_ask(kind="approval", question="Run rm -rf build?")))

    assert "(approval)" in text and "An approval is not carried over" in text


def test_a_plain_completion_prints_as_before():
    plain = _entry(summary="All done.")
    plain.update(status="completed", exit_reason="completed")

    text = format_process_notification(_event(plain))

    assert "--- ✓ TASK 1/1" in text and "NEEDS INPUT" not in text and "All done." in text


def test_the_single_task_notification_and_the_ui_title_say_it_needs_input_too():
    entry = _completed_ask()
    single = {"type": "async_delegation", "delegation_id": "d2", "goal": "deploy the site", **entry}

    text = format_process_notification(single)

    assert "[NEEDS INPUT]" in text and "Question: Which key?" in text
    assert async_delegation_display_text(_event(entry)).startswith("Subagent Task Needs Input: deploy the site")
