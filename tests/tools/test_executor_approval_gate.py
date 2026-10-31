"""A dispatcher session's executor with nobody to ask gets a definite refusal, not a card nobody will see.

Chat gateways register the approval notifier for the length of a turn, and a dispatcher's turn is short by design.
An executor that hits a dangerous command after it ended used to get ``pending_approval`` (a STOP telling it to wait)
and a ``_pending`` entry nothing reads. Only an executor of a dispatcher session changes: every other delegated child
keeps ``pending_approval`` (pinned in test_execute_code_local_session_approval.py). Real ``tools.approval``.
"""

from __future__ import annotations

import pytest

from agent.delegation_context import delegated_child_context, executor_child_context
from gateway.session_context import clear_session_vars, set_session_vars
from tools import approval as A
from tools import approval_context
from tools.approval_detection import detect_dangerous_command

SESSION_KEY = "executor-gate-session"
COMMAND = "rm -rf /tmp/executor-gate-victim"
SCRIPT = "import os"


@pytest.fixture
def gateway_turn(monkeypatch):
    """A chat-gateway session in manual approval mode, with no notifier registered (the dispatcher's turn ended)."""
    assert detect_dangerous_command(COMMAND)[0], "probe command must trip a per-command guard"
    monkeypatch.setattr(approval_context, "_get_approval_mode", lambda: "manual")
    monkeypatch.setattr(A, "_YOLO_MODE_FROZEN", False)
    monkeypatch.setenv("HERMES_EXEC_ASK", "1")
    for name in ("HERMES_INTERACTIVE", "HERMES_GATEWAY_SESSION", "HERMES_CRON_SESSION"):
        monkeypatch.delenv(name, raising=False)
    tokens = set_session_vars(platform="telegram", session_key=SESSION_KEY, session_id=SESSION_KEY)
    key_token = approval_context.set_current_session_key(SESSION_KEY)
    A.unregister_gateway_notify(SESSION_KEY)
    try:
        yield
    finally:
        approval_context.reset_current_session_key(key_token)
        clear_session_vars(tokens)
        with A._lock:
            A._pending.pop(SESSION_KEY, None)
            A._gateway_queues.pop(SESSION_KEY, None)


def test_an_executor_with_no_approver_is_refused_for_good_and_nothing_is_queued(gateway_turn):
    with delegated_child_context("child"), executor_child_context():
        result = A.check_all_command_guards(COMMAND, "local")

    assert result["approved"] is False
    assert result["outcome"] == "no_approver" and result["user_consent"] is False
    assert result.get("status") != "pending_approval" and "approval_pending" not in result
    assert "needs_input" in result["message"] and "Do NOT retry" in result["message"]
    assert SESSION_KEY not in A._pending


def test_any_other_delegated_child_still_gets_the_pending_approval(gateway_turn):
    with delegated_child_context("child"):
        result = A.check_all_command_guards(COMMAND, "local")

    assert result["status"] == "pending_approval" and result["approval_pending"] is True
    assert A._pending[SESSION_KEY]["command"] == COMMAND


def test_the_whole_script_gate_refuses_an_executor_the_same_way(gateway_turn):
    with delegated_child_context("child"), executor_child_context():
        script = A.check_execute_code_guard(SCRIPT, "local")

    assert script["approved"] is False and script["outcome"] == "no_approver"
    assert SESSION_KEY not in A._pending


def test_an_executor_with_a_live_notifier_is_still_asked_and_the_answer_is_honoured(gateway_turn):
    """The refusal is for the no-approver case only: while a card can reach the user the executor waits for it."""
    cards: list[dict] = []

    def notify(approval_data):
        cards.append(approval_data)
        with A._lock:
            entry = A._gateway_queues[SESSION_KEY][-1]
            entry.result = "once"
            entry.event.set()

    A.register_gateway_notify(SESSION_KEY, notify)
    try:
        with delegated_child_context("child"), executor_child_context():
            result = A.check_all_command_guards(COMMAND, "local")
    finally:
        A.unregister_gateway_notify(SESSION_KEY)

    assert result["approved"] is True and len(cards) == 1
