"""Dispatch-and-end: a dispatcher turn of accepted dispatches ends without a status-line model call.

A real ``AIAgent`` with the ``delegation``, ``todo`` and ``file`` toolsets runs against the loopback
``FakeLLMServer`` and a real ``SessionDB``; only ``delegate_task`` itself is replaced (patched where
``run_agent._dispatch_delegate_task`` imports it). The shared ``real_loop`` fixture of
``test_empty_response_recovery_persistence`` is not reused: it enables only ``file``, so a
``delegate_task`` call there never reaches the tool.
"""

from __future__ import annotations

import contextlib
import itertools
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.turn_dispatch_handoff import DISPATCH_HANDOFF_EXIT
from tests.fakes.fake_llm_provider import FakeLLMServer, Text, ToolCall, write_hermes_home
from tests.fakes.redactor_plugin import COMMENTARY, REWRITE, redactor_home

SAID = "On it. I'm starting that research in the background."
STATUS = "Status: still working."
TODOS = [{"id": "1", "content": "research", "status": "pending"}]
# Keeps the background memory/skill review and the update check from forking extra requests; todo_list
# sits in the request's tools instead of behind tool_search.
_CONFIG = (
    "updates:\n  check: false\n"
    "models_dev:\n  url: http://127.0.0.1:9/api.json\n"
    "skills:\n  creation_nudge_interval: 0\nmemory:\n  nudge_interval: 0\n"
    'tools:\n  tool_search:\n    enabled: "off"\n'
)


def _dispatched(_kw):
    return {"status": "dispatched", "mode": "background", "count": 1, "delegation_id": "d1", "goals": ["g"]}


def _control(status):
    return lambda kw: {"action": kw["action"], "subagent_id": kw["subagent_id"], "status": status, "note": "ok"}


@pytest.fixture
def make_session(monkeypatch):
    """``make(script, ...)`` -> a dispatcher-flagged agent on a loopback provider."""
    from hermes_cli import plugins
    from hermes_state import SessionDB
    from run_agent import AIAgent

    stack = contextlib.ExitStack()

    def make(script, *, extra_config="", dispatcher=True, delegate=_dispatched, redactor=False):
        home = Path(os.environ["HERMES_HOME"])
        srv = stack.enter_context(FakeLLMServer(script))
        if redactor:
            redactor_home(home, srv.base_url, hold=False)
            plugins._reset_plugin_managers_for_tests()
            stack.callback(plugins._reset_plugin_managers_for_tests)
        else:
            write_hermes_home(home, srv.base_url, extra_config=_CONFIG + extra_config)
        db = SessionDB(db_path=home / "state.db")
        stack.callback(db.close)
        calls: list[dict] = []

        def fake_delegate_task(**kw):
            calls.append(kw)
            out = delegate(kw)
            return out if isinstance(out, str) else json.dumps(out)

        monkeypatch.setattr("tools.delegate_tool.delegate_task", fake_delegate_task)
        agent = AIAgent(
            base_url=srv.base_url, api_key="sk-fake-e2e", provider="custom", model="fake-model", quiet_mode=True,
            skip_context_files=True, skip_memory=True, enabled_toolsets=["delegation", "todo", "file"],
            session_db=db, session_id="sess-handoff",
        )
        agent._dispatcher_session = dispatcher
        return SimpleNamespace(agent=agent, srv=srv, db=db, sid="sess-handoff", calls=calls)

    yield make
    stack.close()


def _spawn(text=SAID, **kw):
    return ToolCall("delegate_task", {"goal": "g"}, text=text, **kw)


ENDS = {
    "spawn": (_spawn(), _dispatched),
    "spawn_and_todo": (_spawn(parallel=[("todo_list", {"todos": TODOS})]), _dispatched),
    "steer": (ToolCall("delegate_task", {"action": "steer", "subagent_id": "sa-1", "message": "m"}, text=SAID),
              _control("queued")),
    "stop": (ToolCall("delegate_task", {"action": "stop", "subagent_id": "sa-1"}, text=SAID),
             _control("interrupt_requested")),
}


@pytest.mark.parametrize("case", ENDS)
def test_accepted_dispatches_end_the_turn_without_a_status_call(make_session, case):
    first, delegate = ENDS[case]
    s = make_session([first], delegate=delegate)

    result = s.agent.run_conversation("research that")

    assert len(s.srv.main_requests()) == 1 and result["api_calls"] == 1
    assert result["turn_exit_reason"] == DISPATCH_HANDOFF_EXIT
    assert result["completed"] is True and not result["failed"]
    assert result["final_response"] == SAID
    assert len(s.calls) == 1


def test_the_stored_turn_ends_on_a_hidden_boundary_row_and_replays_without_tool_then_user(make_session):
    s = make_session([_spawn(), Text("You're welcome.")])
    result = s.agent.run_conversation("research that")

    stored = s.db.get_messages_as_conversation(s.sid)
    assert [m["role"] for m in stored] == ["user", "assistant", "tool", "assistant"]
    assert stored[-1]["content"] == SAID and stored[-1]["display_kind"] == "hidden"
    assert result["messages"][-1]["content"] == SAID  # one boundary row, not a second from the tail close

    s.agent.run_conversation("thanks", conversation_history=stored)
    wire = s.srv.main_requests()[-1]["messages"]
    assert [m["role"] for m in wire][-3:] == ["tool", "assistant", "user"]
    assert not any(a["role"] == "tool" and b["role"] == "user" for a, b in itertools.pairwise(wire))


def _dropped_duplicate():
    return _spawn(parallel=[("delegate_task", {"goal": "g"})])


DECLINES = {
    "no_text_before_the_call": (ToolCall("delegate_task", {"goal": "g"}), _dispatched, ""),
    "text_is_a_silence_marker": (_spawn(text="NO_REPLY"), _dispatched, ""),
    "another_tool_in_the_round": (_spawn(parallel=[("read_file", {"path": "missing.txt"})]), _dispatched, ""),
    "ran_synchronously": (_spawn(), lambda kw: {"results": [{"status": "completed"}], "note": "ran inline"}, ""),
    "partly_refused": (_spawn(), lambda kw: {**_dispatched(kw), "refused": [{"error": "no"}]}, ""),
    "inline_results_beside_the_handle": (
        _spawn(), lambda kw: {**_dispatched(kw), "inline_results": [{"status": "completed"}]}, ""),
    "queue_full_error": (_spawn(), lambda kw: {"error": "the dispatch queue is full"}, ""),
    "plugin_appended_a_note": (_spawn(), lambda kw: json.dumps(_dispatched(kw)) + "\n[live note]", ""),
    "list_returns_data": (ToolCall("delegate_task", {"action": "list"}, text=SAID),
                          lambda kw: {"action": "list", "subagents": []}, ""),
    "a_duplicate_call_was_dropped": (_dropped_duplicate(), _dispatched, ""),
    "todo_write_without_a_dispatch": (ToolCall("todo_list", {"todos": TODOS}, text=SAID), _dispatched, ""),
    "switch_off": (_spawn(), _dispatched, "delegation:\n  end_turn_on_dispatch: false\n"),
}


@pytest.mark.parametrize("case", DECLINES)
def test_anything_else_keeps_the_model_call_that_reads_the_outcome(make_session, case):
    first, delegate, extra_config = DECLINES[case]
    s = make_session([first, Text(STATUS)], delegate=delegate, extra_config=extra_config)

    result = s.agent.run_conversation("research that")

    assert len(s.srv.main_requests()) == 2 and result["api_calls"] == 2
    assert result["turn_exit_reason"].startswith("text_response(")
    assert result["final_response"] == STATUS


def test_a_session_that_is_not_a_dispatcher_keeps_the_status_call(make_session):
    s = make_session([_spawn(), Text(STATUS)], dispatcher=False)

    result = s.agent.run_conversation("research that")

    assert result["api_calls"] == 2 and result["final_response"] == STATUS


def test_a_dispatcher_session_does_not_cap_delegate_calls_and_other_sessions_still_do(make_session):
    """The cap silently deleted the calls beyond max_concurrent_children; DISPATCH answers a full queue
    with a tool error instead, so a dispatcher round reaches ``delegate_task`` with every call."""
    cap = "delegation:\n  max_concurrent_children: 2\n"
    many = ToolCall("delegate_task", {"goal": "a"}, text=SAID,
                    parallel=[("delegate_task", {"goal": "b"}), ("delegate_task", {"goal": "c"})])

    dispatcher = make_session([many], extra_config=cap)
    outcome = dispatcher.agent.run_conversation("three jobs")
    assert [c["goal"] for c in dispatcher.calls] == ["a", "b", "c"]
    assert outcome["turn_exit_reason"] == DISPATCH_HANDOFF_EXIT

    plain = make_session([many, Text(STATUS)], extra_config=cap, dispatcher=False)
    plain.agent.run_conversation("three jobs")
    assert [c["goal"] for c in plain.calls] == ["a", "b"]


def _run_shown(s, *, interim, stream):
    """Run the turn with the display callbacks a surface wires; returns (result, interims, streamed text)."""
    interims: list[str] = []
    deltas: list[str] = []
    if interim:
        s.agent.interim_assistant_callback = lambda text, **_: interims.append(text)
    if stream:
        s.agent.stream_delta_callback = lambda text: deltas.append(text) if text else None
    return s.agent.run_conversation("research that"), interims, "".join(deltas)


# (interim callback wired, live stream wired, the user saw the preamble before the turn ended)
SHOWN = {
    "interim_only": (True, False, True),  # messaging / TUI with interim messages on
    "live_stream_only": (False, True, True),  # classic CLI: the panel repaints unless the result says reused
    "both": (True, True, True),
    "neither": (False, False, False),  # the final send is the preamble's only appearance
}


@pytest.mark.parametrize("case", SHOWN)
def test_the_final_is_named_a_reuse_exactly_when_the_user_already_saw_it(make_session, case):
    interim, stream, seen = SHOWN[case]
    s = make_session([_spawn()])

    result, interims, streamed = _run_shown(s, interim=interim, stream=stream)

    assert result["final_response"] == SAID
    assert result["response_previewed"] is seen and result["response_reused"] is seen
    assert (SAID in interims) is interim
    if stream:
        assert streamed.strip() == SAID


def _crash_left(history):
    from gateway.config import Platform
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource

    origin = SessionSource(platform=Platform.SLACK, chat_id="D1", chat_type="dm", user_id="u1")
    return GatewayRunner._crash_left_reply(object(), history, 0, origin)


@pytest.mark.parametrize("seen", [True, False], ids=["user_saw_it", "user_did_not"])
def test_crash_recovery_resends_a_handoff_text_only_when_the_user_never_saw_it(make_session, seen):
    s = make_session([_spawn()])
    result, _, _ = _run_shown(s, interim=seen, stream=False)

    history = s.db.get_messages_as_conversation(s.sid)

    assert history[-1]["display_metadata"] == {"dispatch_handoff": seen}
    assert _crash_left(history) == ("" if seen else SAID)
    # The hidden row is not a row the client can show, so the receipt must not address it.
    from agent.persisted_turn_receipt import persisted_turn_receipt

    receipt = persisted_turn_receipt(
        result["messages"], s.agent._persist_user_message_idx, [], SAID, "complete",
        compression_unchanged=True, prefix_row_matches=lambda before, after: before is after)
    assert receipt is None or "final_assistant_row_id" not in receipt


def test_a_rewritten_final_is_not_a_reuse_and_the_stored_row_holds_the_rewrite(make_session):
    """transform_llm_output reaches T only now (it used to see just the status line): the user saw the raw
    text as an interim, so the rewrite must go out as a new final, and the row is stored rewritten."""
    raw = f"{COMMENTARY} Starting."
    s = make_session([ToolCall("delegate_task", {"goal": "g"}, text=raw)], redactor=True)
    s.agent.interim_assistant_callback = lambda *_a, **_k: None

    result = s.agent.run_conversation("research that")

    assert result["turn_exit_reason"] == DISPATCH_HANDOFF_EXIT
    assert result["response_transformed"] is True and result["response_reused"] is False
    assert "SECRET-777" not in result["final_response"] and "[REDACTED]" in result["final_response"]
    stored = s.db.get_messages_as_conversation(s.sid)[-1]
    assert stored["content"] == result["final_response"] and stored["display_kind"] == "hidden"
