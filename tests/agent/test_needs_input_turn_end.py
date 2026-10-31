"""A child that calls ``needs_input`` ends its turn there, and its result is ``completed``, not ``failed``.

Real ``AIAgent`` on the real ``file`` + ``executor`` toolsets (platform ``subagent``) and a real ``SessionDB``; only
the provider is scripted. The interrupt the tool raises stops the loop before any further model call, so the
empty-response ladder (which would nudge a model that answered a tool result with nothing) is never entered, and
``_build_result_entry`` then reports the run as the dispatcher must read it.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from hermes_state import SessionDB
from run_agent import AIAgent
from tools.delegate_tool_child_run import _build_result_entry
from tools.needs_input_tool import EXIT_REASON

_DEAD_LOCAL = "http://127.0.0.1:9"
_NO_SCHEMA = SimpleNamespace(schema=None, valid=None, retries=0, errors=None)


def _response(content="", finish_reason="stop", tool_calls=None):
    choice = SimpleNamespace(
        message=SimpleNamespace(content=content, tool_calls=tool_calls), finish_reason=finish_reason, index=0)
    return SimpleNamespace(id="chatcmpl-test", choices=[choice], model="test/model", usage=None)


def _call(name, args, call_id):
    return SimpleNamespace(id=call_id, type="function", function=SimpleNamespace(name=name, arguments=json.dumps(args)))


@pytest.fixture
def executor(tmp_path, monkeypatch):
    for var in ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.setenv(var, _DEAD_LOCAL)
    monkeypatch.setattr("agent.title_generator.maybe_auto_title", lambda *a, **k: None)
    monkeypatch.setattr("agent.title_generator.start_title_upgrade", lambda *a, **k: None)
    monkeypatch.chdir(tmp_path)
    db = SessionDB(db_path=tmp_path / "state.db")
    with patch("agent.process_bootstrap.OpenAI"), patch("agent.model_metadata.fetch_model_metadata", return_value={}):
        agent = AIAgent(
            api_key="test-key", base_url=f"{_DEAD_LOCAL}/v1", model="test/model", quiet_mode=True,
            skip_context_files=True, skip_memory=True, enabled_toolsets=["file", "executor"], platform="subagent",
            side_agent=True, session_db=db, session_id="sess-needs-input",
        )
    agent._cached_system_prompt = "You are an executor."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    assert "needs_input" in agent.valid_tool_names

    def run(script):
        pending, requests = list(script), []
        agent.client = MagicMock()
        agent.client.chat.completions.create.side_effect = lambda **_kw: (requests.append(1), pending.pop(0))[1]
        result = agent.run_conversation("deploy the site")
        return result, len(requests)

    yield SimpleNamespace(agent=agent, run=run)
    db.close()


def _ask(**extra):
    return {"kind": "question", "question": "Which deploy key should I use?", "options": ["prod", "staging"], **extra}


def test_the_turn_ends_on_the_call_with_no_nudge_and_the_batch_siblings_skipped(executor):
    # The follow-ups are empty on purpose: a loop that went on would nudge the model and burn them.
    script = [
        _response("Built the site; the deploy key is missing.", "tool_calls", [
            _call("needs_input", _ask(), "c1"), _call("read_file", {"path": "nothing.txt"}, "c2")]),
        _response(), _response(), _response(),
    ]

    result, provider_calls = executor.run(script)

    assert provider_calls == 1 and result["api_calls"] == 1
    assert result["turn_exit_reason"] == EXIT_REASON and result["interrupted"] is True and result["failed"] is False
    tool_rows = [m["content"] for m in result["messages"] if m.get("role") == "tool"]
    assert json.loads(tool_rows[0])["status"] == "recorded"
    assert "read_file was not started" in tool_rows[1]
    assert not any("nudge" in str(m.get("content")).lower() for m in result["messages"] if m.get("role") == "user")


def test_the_result_entry_is_completed_with_the_childs_last_text_and_the_request(executor):
    script = [_response("Built the site; the deploy key is missing.", "tool_calls", [_call("needs_input", _ask(), "c1")])]

    result, _ = executor.run(script)
    entry = _build_result_entry(executor.agent, result, 0, 1.0, _NO_SCHEMA)

    assert entry["status"] == "completed" and entry["exit_reason"] == "completed" and entry["truncated"] is False
    assert "error" not in entry
    assert entry["summary"] == "Built the site; the deploy key is missing."
    assert entry["needs_input"] == {
        "kind": "question", "question": "Which deploy key should I use?", "options": ["prod", "staging"], "detail": "",
    }


def test_with_no_text_beside_the_call_the_question_is_the_summary_not_a_generic_failure(executor):
    script = [_response("", "tool_calls", [_call("needs_input", _ask(), "c1")]), _response()]

    result, provider_calls = executor.run(script)
    entry = _build_result_entry(executor.agent, result, 0, 1.0, _NO_SCHEMA)

    assert provider_calls == 1
    assert entry["status"] == "completed" and entry["summary"] == "Which deploy key should I use?"
    assert "did not produce a response" not in json.dumps(entry)


def test_a_child_that_never_asks_is_reported_exactly_as_before(executor):
    result, _ = executor.run([_response("All done.")])
    entry = _build_result_entry(executor.agent, result, 0, 1.0, _NO_SCHEMA)

    assert entry["status"] == "completed" and entry["summary"] == "All done." and "needs_input" not in entry
