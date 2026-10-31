"""``finalize_turn`` on a dispatch handoff: the hidden boundary row is the answer, not a failure.

A dispatch can land on the last iteration; without ``_turn_completed`` knowing the exit reason, that turn
would read ``completed=False`` (the iteration budget is spent and the reason is not a ``text_response(``).
"""

from agent.turn_dispatch_handoff import DISPATCH_HANDOFF_EXIT
from agent.turn_finalizer import finalize_turn
from tests.agent.test_turn_finalizer_interrupt_alternation import _StubAgent

SAID = "On it. I'm starting that research in the background."


def _handoff_tail():
    return [
        {"role": "user", "content": "research that"},
        {"role": "assistant", "content": SAID,
         "tool_calls": [{"id": "c1", "function": {"name": "delegate_task", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": '{"status": "dispatched"}'},
        {"role": "assistant", "content": SAID, "display_kind": "hidden"},
    ]


def _finalize(agent, messages, reason, final_response=SAID):
    return finalize_turn(
        agent, final_response=final_response, api_call_count=agent.max_iterations, interrupted=False, failed=False,
        messages=messages, conversation_history=None, effective_task_id="task-1", turn_id="turn-1",
        user_message="research that", original_user_message="research that", _should_review_memory=False,
        _turn_exit_reason=reason,
    )


def test_a_handoff_on_the_last_iteration_is_a_completed_turn_with_the_boundary_row_as_its_tail():
    agent, messages = _StubAgent(), _handoff_tail()

    result = _finalize(agent, messages, DISPATCH_HANDOFF_EXIT)

    assert result["completed"] is True and result["failed"] is False
    assert result["turn_exit_reason"] == DISPATCH_HANDOFF_EXIT
    assert result["final_response"] == SAID
    assert len(messages) == 4 and agent.persisted_messages[-1]["display_kind"] == "hidden"


def test_the_same_turn_under_any_other_exit_reason_is_not_completed_at_the_budget():
    result = _finalize(_StubAgent(), _handoff_tail(), "unknown")

    assert result["completed"] is False
