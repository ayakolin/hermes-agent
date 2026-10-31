"""The executor baseline: what a dispatcher session's delegate_task children may use, and what caps it.

Real imports, real toolset tables and the real tool registry. The baseline is trusted config cut down to the
session ceiling (the operator's own limits), then handed to ``_resolve_child_toolsets``; the tool names are
whatever ``model_tools.get_tool_definitions`` really produces for that selection.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

import model_tools
from agent.executor_baseline import (
    ExecutorBaseline, baseline_child_toolsets, executor_baseline_for, freeze_executor_baseline,
    resolve_executor_baseline,
)
from hermes_cli.config_defaults_delegation import DELEGATION_DEFAULTS
from toolsets import resolve_toolset
from tools.delegate_tool_toolsets import DELEGATE_BLOCKED_TOOLS, _resolve_child_toolsets

DEFAULT_TOOLSETS = tuple(DELEGATION_DEFAULTS["executor"]["toolsets"])
# Everything a full interactive profile allows; the baseline is then limited by its own list only.
FULL_CEILING = ("hermes-cli",)


@pytest.fixture(autouse=True)
def _fresh_warnings(monkeypatch):
    monkeypatch.setattr("agent.executor_baseline._warned", set())


def _dispatcher(**attrs):
    return SimpleNamespace(**{
        "_dispatcher_session": True, "_delegate_depth": 0, "enabled_toolsets": ["todo"], "disabled_toolsets": [],
        "platform": "cli", **attrs,
    })


def _tool_names(parent, baseline):
    enabled, disabled = _resolve_child_toolsets(parent, None, "leaf", executor_baseline=baseline)
    definitions = model_tools.get_tool_definitions(
        enabled_toolsets=enabled, disabled_toolsets=disabled, quiet_mode=True, skip_tool_search_assembly=True)
    return {item["function"]["name"] for item in definitions}


def test_only_valid_names_survive_and_executor_cannot_be_listed(caplog):
    junk = [
        "delegation", "kanban", "clarify", "memory", "cronjob", "hermes-cli", "desktop_ui", "catalog", "project",
        "executor", "no-such-toolset", "terminal",
    ]
    with caplog.at_level(logging.WARNING, logger="agent.executor_baseline"):
        baseline = resolve_executor_baseline({"delegation": {"executor": {"toolsets": junk}}}, FULL_CEILING)

    assert baseline.toolsets == ("terminal",)
    dropped = [record.getMessage() for record in caplog.records]
    assert len(dropped) == len(junk) - 1


def test_a_non_list_falls_back_to_the_default_and_an_empty_list_stays_empty():
    assert resolve_executor_baseline({"delegation": {"executor": {"toolsets": "terminal"}}}, FULL_CEILING).toolsets \
        == DEFAULT_TOOLSETS
    assert resolve_executor_baseline({"delegation": {"executor": {"toolsets": []}}}, FULL_CEILING).toolsets == ()
    # The gateway reads raw YAML with no DEFAULT_CONFIG merge: an absent section still yields the defaults.
    assert resolve_executor_baseline({}, FULL_CEILING).toolsets == DEFAULT_TOOLSETS


def test_baseline_children_get_tools_the_dispatcher_lacks_but_never_the_blocked_ones():
    parent = _dispatcher()
    baseline = resolve_executor_baseline({}, FULL_CEILING)

    names = _tool_names(parent, baseline)

    assert {"terminal", "read_file", "write_file", "needs_input"} <= names
    assert DELEGATE_BLOCKED_TOOLS.isdisjoint(names)
    assert not {name for name in names if name.startswith("kanban_")}
    # Pin the whole surface by what is allowed rather than by a list of what is not.
    allowed = {tool for ts in (*baseline.toolsets, "executor") for tool in resolve_toolset(ts)}
    assert names <= allowed


def test_a_deeper_spawn_depth_does_not_hand_the_executor_delegate_task(monkeypatch):
    monkeypatch.setattr("tools.delegate_tool._get_max_spawn_depth", lambda: 2)
    from tools.delegate_tool import _build_child_agent
    from unittest.mock import MagicMock, patch

    def built_toolsets(parent, **extra):
        parent.model = parent.provider = parent.base_url = parent.api_key = "x"
        with patch("run_agent.AIAgent") as agent_cls, patch("tools.delegate_tool._resolve_child_runtime",
                                                              return_value={"provider": "x", "base_url": "x"}), \
                patch("tools.delegate_tool._resolve_child_credential_pool", return_value=None), \
                patch("tools.delegate_tool._open_child_session_db", return_value=None):
            agent_cls.return_value = MagicMock()
            _build_child_agent(
                task_index=0, goal="g", context=None, toolsets=None, model=None, max_iterations=5, task_count=1,
                parent_agent=parent, **extra)
        return agent_cls.call_args.kwargs["enabled_toolsets"]

    plain = SimpleNamespace(_delegate_depth=0, enabled_toolsets=["hermes-cli"], disabled_toolsets=[], platform="cli")
    # Same depth rules: an ordinary parent's child is an orchestrator here, which is exactly what must not leak.
    assert "delegation" in built_toolsets(plain)
    baseline = resolve_executor_baseline({}, FULL_CEILING)
    assert "delegation" not in built_toolsets(_dispatcher(), executor_baseline=baseline)


def test_the_parents_operator_ban_still_applies_and_the_executor_toolset_survives_the_inherited_list():
    # session_disabled_toolsets gives every non-subagent session ``executor`` in its disabled list; inheriting that
    # verbatim would strip needs_input from the executor.
    parent = _dispatcher(disabled_toolsets=["browser", "executor", "catalog"])
    baseline = resolve_executor_baseline({}, FULL_CEILING)

    _enabled, disabled = baseline_child_toolsets(parent, baseline)
    names = _tool_names(parent, baseline)

    assert "browser" in disabled and "executor" not in disabled and "catalog" not in disabled
    assert "needs_input" in names
    assert "browser_exec" not in names


def test_a_platform_ceiling_cuts_the_baseline_to_what_the_operator_allows():
    config = {"platform_toolsets": {"telegram": ["web"]}}
    parent = _dispatcher()
    parent.platform = "telegram"

    baseline = freeze_executor_baseline(parent, config)

    assert baseline.toolsets == ("web",)
    names = _tool_names(parent, baseline)
    assert {"terminal", "write_file", "patch", "execute_code", "browser_exec"}.isdisjoint(names)
    assert "needs_input" in names


def test_a_hosted_room_policy_is_the_ceiling_while_it_is_bound():
    from gateway.hosted_room_execution_policy import (
        RoomExecutionPolicy, bind_room_execution_policy, execution_policy_mapping, reset_room_execution_policy,
    )

    config = {"platform_toolsets": {"api_server": ["web"]}}
    policy = RoomExecutionPolicy.from_mapping(execution_policy_mapping(target_profile="default", config=config))
    parent = _dispatcher()
    token = bind_room_execution_policy(policy)
    try:
        baseline = freeze_executor_baseline(parent, {})
    finally:
        reset_room_execution_policy(token)

    assert baseline.toolsets == ("web",)


def test_an_explicit_ceiling_wins_over_the_derived_one_and_an_empty_one_grants_nothing():
    parent = _dispatcher()

    assert sorted(freeze_executor_baseline(parent, {}, ceiling=["web", "file"]).toolsets) == ["file", "web"]
    assert freeze_executor_baseline(parent, {}, ceiling=[]).toolsets == ()
    assert baseline_child_toolsets(parent, parent._executor_baseline)[0] == ["executor"]


def test_the_ceiling_is_compared_by_tools_not_by_toolset_name():
    # ``debugging`` is a composite (terminal + process_manage, web, file): no baseline name equals it, yet it holds
    # every tool of terminal, file and web, and none of browser, code_execution or the rest.
    baseline = resolve_executor_baseline({}, ["debugging"])

    assert set(baseline.toolsets) == {"terminal", "file", "web"}


def _mcp_config(**section):
    return {
        "mcp_servers": {"alpha": {"command": "x"}, "beta": {"command": "y"}, "gamma": {"command": "z", "enabled": False}},
        "delegation": {"executor": {"toolsets": ["file"], **section}},
    }


def test_mcp_all_is_the_enabled_servers_none_is_empty_and_a_list_narrows():
    assert resolve_executor_baseline(_mcp_config(), ["file", "alpha", "beta"]).mcp_toolsets == ("alpha", "beta")
    assert resolve_executor_baseline(_mcp_config(mcp="none"), ["file", "alpha", "beta"]).mcp_toolsets == ()
    assert resolve_executor_baseline(_mcp_config(mcp=["beta", "gamma", "nope"]), ["file", "alpha", "beta"]).mcp_toolsets \
        == ("beta",)
    # An unreadable value is refused, never read as "all".
    assert resolve_executor_baseline(_mcp_config(mcp="maybe"), ["file", "alpha", "beta"]).mcp_toolsets == ()


def test_a_per_platform_mcp_allowlist_is_a_ceiling_on_the_executors_servers():
    config = {**_mcp_config(), "platform_toolsets": {"telegram": ["file", "alpha"]}}
    parent = _dispatcher()
    parent.platform = "telegram"

    baseline = freeze_executor_baseline(parent, config)

    assert baseline.mcp_toolsets == ("alpha",)


def test_only_a_top_level_dispatcher_parent_gets_a_baseline_and_it_is_frozen_once():
    assert executor_baseline_for(SimpleNamespace(_delegate_depth=0, enabled_toolsets=["file"])) is None
    assert executor_baseline_for(SimpleNamespace(_dispatcher_session=True, _delegate_depth=1)) is None
    # A truthy look-alike is not the frozen flag.
    assert executor_baseline_for(SimpleNamespace(_dispatcher_session="yes", _delegate_depth=0)) is None

    parent = _dispatcher()
    first = executor_baseline_for(parent)
    assert isinstance(first, ExecutorBaseline) and executor_baseline_for(parent) is first
