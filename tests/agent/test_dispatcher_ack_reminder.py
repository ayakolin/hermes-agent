"""A dispatcher turn that has shown the user nothing is reminded once, on its first tool result.

A dispatcher answers at once and hands long work to executors; a turn that goes straight to tool
calls looks like a dead bot. While the turn owes the user a reply (``_ack_obligation``, armed by the
surface) and no text has reached them, the first tool result committed this turn carries a one-shot
system note, appended to the new row before it is flushed: no byte of the cached prefix moves, and the
note is delivered once per turn however many tool batches follow.
"""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tests.agent.test_plugin_live_notes_delivery import TOOL_OUTPUT, _agent, _response, _run


@pytest.fixture(autouse=True)
def _no_background_titles(monkeypatch):
    monkeypatch.setattr("agent.title_generator.maybe_auto_title", lambda *args, **kwargs: None)


def _calls(*ids):
    # Distinct arguments: identical calls in one batch are de-duplicated before they run.
    return [SimpleNamespace(id=i, type="function",
                            function=SimpleNamespace(name="web_search", arguments=json.dumps({"q": i})))
            for i in ids]


def _tool_response(*ids, content=""):
    message = SimpleNamespace(content=content, tool_calls=_calls(*ids))
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="tool_calls")],
                           model="test/model", usage=None)


def _script(agent, *responses):
    """Answer the model calls in order; return the messages each call was sent."""
    requests, pending = [], list(responses)

    def create(**kwargs):
        requests.append(copy.deepcopy(kwargs["messages"]))
        return pending.pop(0)

    agent.client = MagicMock()
    agent.client.chat.completions.create.side_effect = create
    return requests


def _tool_rows(request):
    return [m for m in request if m.get("role") == "tool"]


def _run_turn(*responses, armed, interim=None):
    agent = _agent()
    agent._ack_obligation = armed
    if interim is not None:
        agent.interim_assistant_callback = interim
    requests = _script(agent, *responses)
    _run(agent, "look it up")
    return requests


def test_the_first_tool_result_of_a_silent_turn_carries_the_reminder_and_no_cached_byte_moves():
    control = _run_turn(_response(tool_call_id="c1"), _response("done"), armed=False)
    armed = _run_turn(_response(tool_call_id="c1"), _response("done"), armed=True)

    plain = _tool_rows(control[1])[0]["content"]
    row = _tool_rows(armed[1])[0]["content"]
    assert row.startswith(plain) and len(row) > len(plain)  # appended after the result, nothing else moved
    assert armed[1][: len(armed[0])] == armed[0]  # the cached prefix of the first request is byte-identical
    assert [m for m in armed[1] if m["role"] != "tool"] == [m for m in control[1] if m["role"] != "tool"]


def test_the_reminder_is_told_once_per_turn_across_batches_and_calls():
    requests = _run_turn(
        _tool_response("c1", "c2"), _tool_response("c3"), _response("done"), armed=True)
    plain = _run_turn(
        _tool_response("c1", "c2"), _tool_response("c3"), _response("done"), armed=False)

    rows = _tool_rows(requests[-1])
    plain_rows = _tool_rows(plain[-1])
    extra = [len(r["content"]) - len(p["content"]) for r, p in zip(rows, plain_rows)]
    assert [bool(n) for n in extra] == [True, False, False]  # first result of the first batch only


def test_text_the_user_already_saw_this_turn_means_no_reminder():
    seen = []
    requests = _run_turn(
        _response("On it, looking that up.", tool_call_id="c1"), _response("done"),
        armed=True, interim=lambda text, already_streamed=False: seen.append(text))
    plain = _run_turn(_response("On it, looking that up.", tool_call_id="c1"), _response("done"), armed=False,
                      interim=lambda text, already_streamed=False: None)

    assert seen == ["On it, looking that up."]
    assert _tool_rows(requests[1])[0]["content"] == _tool_rows(plain[1])[0]["content"]


@pytest.mark.parametrize(("kwargs", "agent_attrs"), [
    ({}, {"_dispatcher_session": False}),  # a classic session
    ({}, {"_dispatcher_session": True, "_delegate_depth": 1}),  # an executor subagent
    ({}, {"_dispatcher_session": True, "platform": "cron"}),  # a scheduled run
    ({}, {"_dispatcher_session": True, "_persist_disabled": True}),  # a persistence-detached fork
    ({"display_kind": "async_delegation_complete"}, {"_dispatcher_session": True}),  # an executor completion
    ({"display_kind": "internal_notification"}, {"_dispatcher_session": True}),
    ({"reply_expected": False}, {"_dispatcher_session": True}),  # a group message not addressed to the bot
    ({"muted": True}, {"_dispatcher_session": True}),  # a heartbeat or muted notification
])
def test_only_a_turn_somebody_waits_on_arms_the_reply_obligation(kwargs, agent_attrs):
    from agent.dispatcher_ack import ack_obligation, arm_ack_obligation

    agent = SimpleNamespace(**{"_delegate_depth": 0, "platform": "telegram", "_persist_disabled": False, **agent_attrs})
    arm_ack_obligation(agent, **kwargs)
    assert ack_obligation(agent) is False


def test_a_human_dispatcher_turn_arms_it_with_the_turns_own_settings():
    from agent.dispatcher_ack import ack_obligation, ack_settings, arm_ack_obligation

    agent = SimpleNamespace(_dispatcher_session=True, _delegate_depth=0, platform="telegram", _persist_disabled=False)
    arm_ack_obligation(agent, config={"dispatcher": {"ack": {"max_nudges": 3}}})
    assert ack_obligation(agent) is True and ack_settings(agent).ack_max_nudges == 3


def test_a_note_other_notices_ride_with_still_reaches_the_tool_row():
    # The reminder shares the queue with plugin go-live notices: both land on the one new row.
    from agent.plugin_live_notes import queue_plugin_notice

    agent = _agent()
    agent._ack_obligation = True
    requests = _script(agent, _response(tool_call_id="c1"), _response("done"))
    _run(agent, "look it up", tool_result=lambda: (
        queue_plugin_notice(agent, "blender", "[System note: plugin blender is live.]"), TOOL_OUTPUT)[-1])

    content = _tool_rows(requests[1])[0]["content"]
    assert "plugin blender is live" in content and content.count("[System note:") == 2
    assert json.dumps(requests[1]).count("plugin blender is live") == 1


class TestEarlyPreambleFlush:
    """``dispatcher.ack.early_interim``: on a messaging platform the text written before the first tool
    call is delivered when that call starts, instead of after the whole tool-call response (arguments
    included) has been generated. The normal emit after persistence neither sends it twice nor skips the
    ``on_interim_message`` plugin hook."""

    def _fire(self, agent, text):
        from agent.dispatcher_ack import flush_preamble

        agent.tool_gen_callback = lambda _name: None
        agent._fire_tool_gen_started("web_search", text)

    def test_the_preamble_is_delivered_when_the_tool_call_starts(self):
        agent = _agent()
        agent._ack_obligation = True
        agent.platform = "telegram"
        shown = []
        agent.interim_assistant_callback = lambda text, already_streamed=False: shown.append((text, already_streamed))

        self._fire(agent, "On it, checking now.")

        assert shown == [("On it, checking now.", False)]
        assert agent._interim_text_was_delivered("On it, checking now.")

    @pytest.mark.parametrize(("platform", "armed", "stored", "config"), [
        ("cli", True, None, None),  # a local viewer rebuilds from the stored rows
        ("desktop", True, None, None),
        ("telegram", False, None, None),  # not a turn that owes a reply
        ("telegram", True, None, {"dispatcher": {"ack": {"early_interim": False}}}),
    ], ids=["local-cli", "desktop", "not-armed", "turned-off"])
    def test_nothing_is_flushed_where_it_does_not_apply(self, platform, armed, stored, config):
        from agent.dispatcher_ack import arm_ack_obligation

        agent = _agent()
        agent.platform = platform
        agent._dispatcher_session = True
        agent._ack_obligation = armed
        if config:
            arm_ack_obligation(agent, config=config)
        agent._ack_obligation = armed
        shown = []
        agent.interim_assistant_callback = lambda text, already_streamed=False: shown.append(text)

        self._fire(agent, "On it, checking now.")

        assert shown == [] and not agent._interim_text_was_delivered("On it, checking now.")

    def test_a_streamed_preamble_or_a_marker_is_not_flushed_again(self):
        agent = _agent()
        agent._ack_obligation = True
        agent.platform = "telegram"
        shown = []
        agent.interim_assistant_callback = lambda text, already_streamed=False: shown.append(text)
        agent._current_streamed_assistant_text = "On it, checking now."

        self._fire(agent, "On it, checking now.")
        agent._current_streamed_assistant_text = ""
        agent._ack_preamble_flushed = False
        self._fire(agent, "NO_REPLY")

        assert shown == []

    def test_the_normal_emit_after_persistence_skips_only_the_second_send(self):
        agent = _agent()
        agent._ack_obligation = True
        agent.platform = "telegram"
        shown, hooks = [], []
        agent.interim_assistant_callback = lambda text, already_streamed=False: shown.append(text)
        self._fire(agent, "On it, checking now.")

        with patch("agent.plugin_stream_hooks.enqueue_plugin_stream_hook",
                   side_effect=lambda event, **fields: hooks.append((event, fields["text"]))):
            agent._emit_interim_assistant_message({"role": "assistant", "content": "On it, checking now."})

        assert shown == ["On it, checking now."]  # once, not twice
        assert hooks == [("on_interim_message", "On it, checking now.")]  # the plugin hook still runs once
