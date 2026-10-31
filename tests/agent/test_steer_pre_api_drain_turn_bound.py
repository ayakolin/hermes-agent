"""The pre-API steer drain delivers into the CURRENT turn only.

``prepare_iteration`` drains a steer accepted during the last request and puts it after the newest tool
result. With no tool row in the current turn yet (a dispatcher turn that has only answered so far), the
newest tool row in the whole history belongs to an OLDER turn: splicing the steer after it lands the
user's words in the middle of finished history, before that turn's closing answer, which moves the cached
prefix and reorders the conversation. The steer must stay buffered instead, for the post-tool drain or the
finalizer's next-turn handoff.
"""

from agent.prompt_builder import STEER_DISPLAY_KIND

_CALL = {"id": "c1", "type": "function", "function": {"name": "t", "arguments": "{}"}}


def _agent(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from hermes_state import SessionDB
    from run_agent import AIAgent

    return AIAgent(session_db=SessionDB(db_path=tmp_path / "proof.db"),
                   model="test-model", provider="openai-compat", api_key="test",
                   base_url="http://127.0.0.1:1/v1", max_iterations=4,
                   quiet_mode=True, skip_context_files=True, skip_memory=True)


def _prepare(agent, messages, user_idx):
    from agent.turn_context import _reset_per_turn_agent_state
    from agent.turn_iteration_prep import prepare_iteration

    _reset_per_turn_agent_state(agent)
    return prepare_iteration(agent, messages=messages, api_call_count=1,
                             user_message=messages[user_idx]["content"], current_turn_user_idx=user_idx)


def _steer_rows(messages):
    return [i for i, m in enumerate(messages) if m.get("display_kind") == STEER_DISPLAY_KIND]


def test_steer_is_not_spliced_after_an_older_turns_tool_row(tmp_path, monkeypatch):
    agent = _agent(tmp_path, monkeypatch)
    try:
        messages = [
            {"role": "user", "content": "old question"},
            {"role": "assistant", "content": "", "tool_calls": [_CALL]},
            {"role": "tool", "tool_call_id": "c1", "content": "old output"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "NEW question"},
        ]
        agent.steer("also check the logs")
        prep = _prepare(agent, messages, user_idx=4)
        assert _steer_rows(prep.messages) == [], "steer was spliced into finished history"
        assert [m["content"] for m in prep.messages][-2:] == ["old answer", "NEW question"]
        assert agent._pending_steer == "also check the logs", "the steer must stay buffered, not be lost"
    finally:
        agent._session_db.close()


def test_steer_still_follows_the_current_turns_tool_row(tmp_path, monkeypatch):
    agent = _agent(tmp_path, monkeypatch)
    try:
        messages = [
            {"role": "user", "content": "old question"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "NEW question"},
            {"role": "assistant", "content": "", "tool_calls": [_CALL]},
            {"role": "tool", "tool_call_id": "c1", "content": "fresh output"},
        ]
        agent.steer("also check the logs")
        prep = _prepare(agent, messages, user_idx=2)
        assert _steer_rows(prep.messages) == [5]
        assert prep.messages[4]["role"] == "tool" and agent._pending_steer is None
    finally:
        agent._session_db.close()
