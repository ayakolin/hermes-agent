"""A dispatcher turn that would end on a bare silence marker with nothing shown is nudged once.

The fifth stop gate (``agent.turn_stop_gates``): on a turn that owes the user a reply
(``_ack_obligation``, armed by the surface) and has shown no text, a bare ``NO_REPLY`` is not accepted.
The marker row and a user-role nudge are appended as scaffolding (both flagged ``_ack_nudge_synthetic``),
the model is asked again, and neither row survives into the durable transcript or the returned history,
whether the retry ends in text or in a tool call first. Bounded by ``dispatcher.ack.max_nudges``.
"""

from __future__ import annotations

import copy
import itertools
import json
from unittest.mock import MagicMock

from tests.agent import test_empty_response_recovery_persistence as _recovery
from tests.fakes.fake_llm_provider import FakeLLMServer, Text

real_loop = _recovery.real_loop
_call, _response = _recovery._call, _recovery._response
ANSWER = "On it, checking the logs now."
FLAG = "_ack_nudge_synthetic"


def _run(real_loop, script, text="check the logs", **attrs):
    """Run one turn against scripted model responses; returns (requests sent, result)."""
    agent = real_loop.agent
    agent._ack_obligation = True
    for name, value in attrs.items():
        setattr(agent, name, value)
    requests, pending = [], list(script)

    def create(**kwargs):
        requests.append(copy.deepcopy(kwargs["messages"]))
        return pending.pop(0)

    agent.client = MagicMock()
    agent.client.chat.completions.create.side_effect = create
    return requests, agent.run_conversation(text)


def _stored(real_loop):
    return real_loop.db.get_messages_as_conversation(real_loop.sid)


def _roles(messages):
    return [m["role"] for m in messages if m["role"] != "system"]


def _alternates(messages):
    """No two same-role neighbours, and every tool result follows its assistant call or another result."""
    roles = _roles(messages)
    return all(pair not in {("user", "user"), ("assistant", "assistant")} for pair in itertools.pairwise(roles))


def test_a_silence_marker_with_nothing_shown_is_answered_by_one_more_request(real_loop):
    requests, result = _run(real_loop, [_response("NO_REPLY"), _response(ANSWER)])

    assert result["final_response"] == ANSWER
    assert len(requests) == 2  # exactly one extra request
    stored = _stored(real_loop)
    assert [(m["role"], m["content"]) for m in stored if m["role"] != "system"] == [
        ("user", "check the logs"), ("assistant", ANSWER)]
    assert not any(m.get(FLAG) for m in result["messages"])  # not in the returned history
    assert _roles(result["messages"]) == ["user", "assistant"]
    assert "NO_REPLY" not in json.dumps(stored)


def test_the_nudge_is_bounded_then_the_existing_silence_handling_runs(real_loop):
    requests, result = _run(real_loop, [_response("NO_REPLY"), _response("NO_REPLY")])

    assert len(requests) == 2  # one nudge, then the second marker stands
    assert result["final_response"] == "NO_REPLY"
    assert _alternates(result["messages"]) and _alternates(_stored(real_loop))


def test_max_nudges_zero_turns_the_gate_off(real_loop):
    from agent.dispatcher_session import dispatcher_settings

    requests, result = _run(
        real_loop, [_response("NO_REPLY")], _ack_settings=dispatcher_settings({"dispatcher": {"ack": {"max_nudges": 0}}}))

    assert len(requests) == 1 and result["final_response"] == "NO_REPLY"


def test_a_turn_that_is_not_owed_a_reply_may_stay_silent(real_loop):
    # A completion notification, a cron run or an unaddressed group message never arms the obligation:
    # the bare marker ends the turn with zero extra requests.
    requests, result = _run(real_loop, [_response("NO_REPLY")], _ack_obligation=False)

    assert len(requests) == 1 and result["final_response"] == "NO_REPLY"


def test_a_nudge_answered_by_a_tool_call_then_text_leaves_no_scaffolding_behind(real_loop, tmp_path):
    (tmp_path / "notes.txt").write_text("hello", encoding="utf-8")
    read = _call("read_file", {"path": str(tmp_path / "notes.txt")}, "call_read")
    requests, result = _run(real_loop, [
        _response("NO_REPLY"),
        _response("", finish_reason="tool_calls", tool_calls=[read]),
        _response(ANSWER),
    ])

    assert len(requests) == 3 and result["final_response"] == ANSWER
    # The model saw the marker and the nudge in order, and every request alternated.
    assert [m["role"] for m in requests[2] if m["role"] != "system"] == [
        "user", "assistant", "user", "assistant", "tool"]
    assert all(_alternates(r) for r in requests)
    # Persistence parity: only the genuine rows were written, in order.
    stored = _stored(real_loop)
    assert _roles(stored) == ["user", "assistant", "tool", "assistant"]
    assert stored[-1]["content"] == ANSWER and "NO_REPLY" not in json.dumps(stored)
    # The returned (live) history matches what was persisted, with no flagged rows left buried.
    assert not any(m.get(FLAG) for m in result["messages"])
    assert _roles(result["messages"]) == _roles(stored)


def test_the_nudge_row_is_never_the_real_user_message_compression_anchors_on():
    from agent.conversation_compression import _is_real_user_message

    assert _is_real_user_message({"role": "user", "content": "check the logs"})
    assert not _is_real_user_message({"role": "user", "content": "please reply now", FLAG: True})


def test_a_streamed_marker_is_not_text_the_user_saw():
    # The streaming consumer records the marker's deltas in the accumulator before the final resolves
    # it as silence; counting them as shown text means the nudge never fires where streaming is on.
    with FakeLLMServer([Text("NO_REPLY", chunk_chars=3), Text(ANSWER)]) as server:
        from run_agent import AIAgent

        agent = AIAgent(base_url=server.base_url, api_key="sk-fake-e2e", provider="custom", model="fake-model",
                        quiet_mode=True, skip_context_files=True, skip_memory=True, enabled_toolsets=["todo"])
        shown = []
        agent.stream_delta_callback = shown.append
        agent._ack_obligation = True
        result = agent.run_conversation("check the logs")

        assert len(server.main_requests()) == 2
    assert result["final_response"] == ANSWER
    assert not any(m.get(FLAG) for m in result["messages"])
    assert _alternates(result["messages"])
