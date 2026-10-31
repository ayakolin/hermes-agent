"""Silence on a completion turn: only a consumer that strips the marker asks the model for one.

A machinery turn may end with a bare marker when the result was stale, already reported, or its task was
cancelled. The marker is dropped by the gateway (``response_filters``) and by the Ink/Desktop final for a
machinery turn; the shared notice formatter feeds surfaces that render whatever they are given (classic
CLI, Bot Mode), so it must stay free of the instruction. A turn a human message reached must never go
silent on a route that expects a reply: the user is waiting for an answer that also answers their text.
"""

from __future__ import annotations

import pytest

from gateway.response_filters import is_intentional_silence_response
from gateway.run import _format_gateway_process_notification, _with_completion_silence_instruction
from tools.process_registry_notifications import format_process_notification


def _completion(**extra):
    event = {"type": "async_delegation", "delegation_id": "deleg-1", "session_key": "route",
             "status": "completed", "summary": "found it", "api_calls": 1, "duration_seconds": 2.0,
             "dispatched_at": 1.0, "completed_at": 3.0}
    event.update(extra)
    return event


def test_the_gateway_notice_asks_for_a_marker_that_it_strips():
    """A model obeying the notice must answer with something the gateway really drops."""
    text = _format_gateway_process_notification(_completion())
    assert is_intentional_silence_response("[SILENT]"), "the marker the notice asks for is recognised"
    instruction = text[len(format_process_notification(_completion())):]
    assert "reply with exactly [SILENT]" in instruction and "\n" in instruction


def test_the_notice_details_without_the_instruction_are_unchanged():
    """The shared formatter's text is the notice: the instruction is an add-on, never a rewrite."""
    gateway_text = _format_gateway_process_notification(_completion())
    assert gateway_text.startswith(format_process_notification(_completion()))


@pytest.mark.parametrize("event", [
    {"type": "heartbeat", "session_id": "proc-1", "seq": 2, "elapsed": 60, "interval": 60, "output": "line"},
    {"type": "watch_match", "session_id": "proc-1", "pattern": "READY", "output": "READY"},
    _completion(task_failure_notice=True, results=[{"task_index": 0, "status": "failed"}]),
])
def test_no_other_notice_kind_asks_for_a_marker(event):
    """A liveness tick is not a result to judge; a failure notice is always reported."""
    text = _format_gateway_process_notification(event)
    assert text is not None
    assert "reply with exactly [SILENT]" not in text
    assert text == _with_completion_silence_instruction(format_process_notification(event), event)


def test_the_shared_formatter_never_asks_for_a_marker():
    """The classic CLI and Bot Mode use this text and never strip: asking would print the marker."""
    for event in (_completion(), _completion(results=[_completion()]), _completion(task_failure_notice=True)):
        assert "[SILENT]" not in format_process_notification(event)


def _runner():
    from gateway.run import GatewayRunner
    return object.__new__(GatewayRunner)


@pytest.mark.parametrize("kind", ["internal_notification", "async_delegation_complete", "process_complete"])
def test_an_intervened_machinery_turn_may_not_go_silent(kind):
    """A completion folded into a running turn also answers the human text that arrived: a bare marker
    would swallow that reply."""
    runner = _runner()
    assert runner._silence_allowed_for_source(None, kind, None, {"user_intervened": True}) is False


@pytest.mark.parametrize("kind", ["internal_notification", "async_delegation_complete", "process_complete"])
@pytest.mark.parametrize("agent_result", [None, {}, {"user_intervened": False}])
def test_an_untouched_completion_turn_keeps_its_silence(kind, agent_result):
    runner = _runner()
    assert runner._silence_allowed_for_source(None, kind, None, agent_result) is True


def test_a_group_chat_that_expects_no_reply_still_allows_silence_after_an_intervention():
    """``reply_expected`` False is the unaddressed-group case: silence was already allowed there."""
    runner = _runner()
    assert runner._silence_allowed_for_source(None, "async_delegation_complete", False,
                                              {"user_intervened": True}) is True


def test_a_human_turn_is_judged_as_before():
    from gateway.config import Platform
    from gateway.session import SessionSource
    runner = _runner()
    source = SessionSource(platform=Platform.DISCORD, chat_id="42", chat_type="dm")
    assert runner._silence_allowed_for_source(source, None, True, {"user_intervened": True}) is False
    assert runner._silence_allowed_for_source(source, None, None, None) is False
