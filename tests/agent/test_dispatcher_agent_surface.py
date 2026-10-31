"""What a real AIAgent built for a dispatcher session carries (agent/dispatcher_session.py).

The narrow toolset is only half of it: the agent must also take the background ``delegate_task`` variant,
keep that surface through an MCP refresh, and render DISPATCHER_GUIDANCE in its stable prompt tier instead
of the "Finishing the job" block that asks it to do the work itself.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agent.dispatcher_session import DISPATCHER_TOOLSET, agent_is_dispatcher
from agent.prompt_builder import DISPATCHER_GUIDANCE, TASK_COMPLETION_GUIDANCE
from run_agent import AIAgent
from toolsets import resolve_toolset

POSTURE = set(resolve_toolset(DISPATCHER_TOOLSET))
# The tool_search bridge (search / call / describe) is added to any large surface; a dispatcher's own
# whitelist is what the bridge may reach, so it does not count against it.
_BRIDGE = {"tool_search", "tool_call", "tool_describe"}


@pytest.fixture(autouse=True)
def _interactive_session(monkeypatch):
    monkeypatch.delenv("HERMES_SINGLE_QUERY_SESSION", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)


def _agent(*, toolsets, platform="cli", inline=False, **kwargs):
    from tools.delegate_tool_config import gateway_delegation_inline_scope

    with (
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
        gateway_delegation_inline_scope(inline),
    ):
        return AIAgent(
            api_key="test-key-1234567890", base_url="https://openrouter.ai/api/v1", model="test/model",
            quiet_mode=True, skip_context_files=True, skip_memory=True, platform=platform,
            enabled_toolsets=list(toolsets), **kwargs,
        )


def test_the_agent_surface_is_the_posture_whitelist_and_delegates_in_the_background():
    agent = _agent(toolsets=[DISPATCHER_TOOLSET], inline=True)  # an explicit inline config does not apply
    assert agent_is_dispatcher(agent) is True
    assert agent.valid_tool_names - _BRIDGE <= POSTURE
    assert "delegate_task" in agent.valid_tool_names
    from tools.delegate_tool import session_joins_delegation
    assert session_joins_delegation(agent) is False
    assert "BETWEEN your turns" in next(
        t["function"]["description"] for t in agent.tools if t["function"]["name"] == "delegate_task")


def test_a_classic_agent_is_unchanged():
    agent = _agent(toolsets=["hermes-cli"])
    assert agent_is_dispatcher(agent) is False
    assert {"terminal", "patch"} <= agent.valid_tool_names


def test_a_side_agent_never_becomes_the_dispatcher():
    agent = _agent(toolsets=[DISPATCHER_TOOLSET], side_agent=True)
    assert agent_is_dispatcher(agent) is False


def test_extra_toolsets_are_frozen_into_the_surface():
    agent = _agent(toolsets=[DISPATCHER_TOOLSET, "web"])
    assert agent_is_dispatcher(agent) is True
    assert {"web_search", "web_extract"} <= agent.valid_tool_names


def test_a_refresh_never_widens_a_dispatcher(monkeypatch):
    """Every rebuild path (Bot Chat refresh, /reload-mcp, live-session refresh) hands in the CLASSIC
    selection; the one choke point must ignore it for a dispatcher, or the conversation's tools[] flips."""
    from tools.mcp_tool_agent import refresh_agent_mcp_tools

    agent = _agent(toolsets=[DISPATCHER_TOOLSET])
    before = set(agent.valid_tool_names)
    refresh_agent_mcp_tools(agent, enabled_override=["hermes-cli"], quiet_mode=True, content_aware=True)
    assert agent.valid_tool_names == before
    # The operator's disabled_toolsets still applies on a refresh.
    refresh_agent_mcp_tools(agent, disabled_override=["delegation"], quiet_mode=True, content_aware=True)
    assert "delegate_task" not in agent.valid_tool_names


def test_the_stable_tier_carries_the_dispatcher_brief_instead_of_finishing_the_job():
    from agent.system_prompt import build_system_prompt_parts

    agent = _agent(toolsets=[DISPATCHER_TOOLSET])
    stable = build_system_prompt_parts(agent)["stable"]
    assert DISPATCHER_GUIDANCE in stable
    assert TASK_COMPLETION_GUIDANCE not in stable
    # A rebuild (compression, failover) renders the same bytes.
    assert build_system_prompt_parts(agent)["stable"] == stable


def test_a_classic_agent_keeps_finishing_the_job_and_never_sees_the_brief():
    from agent.system_prompt import build_system_prompt_parts

    agent = _agent(toolsets=["hermes-cli"])
    stable = build_system_prompt_parts(agent)["stable"]
    assert TASK_COMPLETION_GUIDANCE in stable
    assert DISPATCHER_GUIDANCE not in stable


def test_the_coding_brief_is_left_out_for_a_dispatcher():
    """It points at patch/write_file, which the dispatcher's surface has no tool for."""
    from agent.system_prompt import _coding_parts

    agent = _agent(toolsets=[DISPATCHER_TOOLSET])
    assert _coding_parts(agent) == ([], [], [])


def test_a_routed_review_fork_is_not_a_dispatcher():
    """A routed fork builds its OWN prompt for a review job; DISPATCHER_GUIDANCE would tell it to delegate
    instead of reviewing."""
    from agent import background_review

    parent = _agent(toolsets=[DISPATCHER_TOOLSET])
    assert agent_is_dispatcher(parent) is True
    fork = MagicMock()
    with (
        patch("run_agent.AIAgent", return_value=fork),
        patch.object(background_review, "_resolve_review_runtime", return_value={"routed": True}),
        patch.object(background_review, "_detach_fork_compression"),
    ):
        review, _runtime, routed = background_review.build_cache_parity_fork(parent, max_iterations=5)
    assert routed is True
    assert review._dispatcher_session is False


def test_a_same_model_review_fork_keeps_the_parents_flag():
    """It replays the parent's cached prompt bytes, so a rebuild of that prompt must agree."""
    from agent import background_review

    parent = _agent(toolsets=[DISPATCHER_TOOLSET])
    fork = MagicMock()
    with (
        patch("run_agent.AIAgent", return_value=fork),
        patch.object(background_review, "_resolve_review_runtime", return_value={}),
        patch.object(background_review, "_detach_fork_compression"),
    ):
        review, _runtime, routed = background_review.build_cache_parity_fork(parent, max_iterations=5)
    assert routed is False
    assert review._dispatcher_session is True
