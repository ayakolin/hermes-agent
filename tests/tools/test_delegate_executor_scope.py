"""Who gets the executor baseline, and that nothing but a dispatcher session's own delegate_task can widen it.

The baseline relaxes exactly one rule: a delegate_task child of a top-level dispatcher session starts from trusted
config instead of its parent's enabled set. Ordinary sessions, executors themselves, the plugin lifecycle API and
anything the model puts in a task keep today's "a child never exceeds its parent".
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent.delegation_context import executor_child_context, is_delegated_child_context, is_executor_child_context
from agent.executor_baseline import ExecutorBaseline
from agent.subagent_lifecycle import SubagentLaunchRequest, SubagentLifecycleError, SubagentLifecycleService
from tools.delegate_tool import _build_children
from tools.delegate_tool_routes import TaskRoute, child_profile_scope, _clean_route
from agent.delegation_provider import ROUTE_KEYS

CREDS = {"model": "m", "provider": "p", "base_url": "http://x", "api_key": "k", "api_mode": "chat_completions"}


@pytest.fixture(autouse=True)
def _fresh_warnings(monkeypatch):
    monkeypatch.setattr("agent.executor_baseline._warned", set())


def _parent(**attrs):
    return SimpleNamespace(**{
        "_delegate_depth": 0, "enabled_toolsets": ["todo"], "disabled_toolsets": [], "platform": "cli",
        "_active_children": [], **attrs,
    })


def _dispatcher(**attrs):
    return _parent(_dispatcher_session=True, **attrs)


@pytest.fixture
def built(monkeypatch):
    """Every kwargs dict ``_build_children`` hands the child builder; the builder itself is a stub."""
    calls: list[dict] = []

    def fake_build(**kwargs):
        calls.append(kwargs)
        return MagicMock()

    monkeypatch.setattr("tools.delegate_tool._build_child_preserving_parent_tools", fake_build)
    monkeypatch.setattr("tools.delegate_tool._route_overrides", lambda route: {})
    return calls


def _children(parent, tasks=({"goal": "work"},)):
    routes = [TaskRoute(creds=dict(CREDS), routing_cfg={}) for _ in tasks]
    children, error = _build_children(
        list(tasks), [None] * len(tasks), routes, top_role="leaf", max_iterations=5, parent_agent=parent,
        live_deleg_id=None, live_writers=[],
    )
    assert error is None and len(children) == len(tasks)
    return children


def test_an_ordinary_parent_builds_children_exactly_as_before(built):
    _children(_parent(enabled_toolsets=["web"]), tasks=({"goal": "a"}, {"goal": "b"}))

    assert len(built) == 2 and all("executor_baseline" not in kwargs for kwargs in built)
    assert all(kwargs["toolsets"] is None for kwargs in built)


def test_every_child_of_a_dispatcher_parent_carries_the_frozen_baseline_and_leaves_toolsets_to_it(built):
    parent = _dispatcher()

    _children(parent, tasks=({"goal": "a"}, {"goal": "b"}))

    baselines = {id(kwargs["executor_baseline"]) for kwargs in built}
    assert baselines == {id(parent._executor_baseline)} and isinstance(parent._executor_baseline, ExecutorBaseline)
    assert all(kwargs["toolsets"] is None for kwargs in built)


def test_a_task_cannot_carry_its_own_toolsets_baseline_or_command(built):
    parent = _dispatcher()
    smuggled = {"goal": "g", "toolsets": ["terminal"], "executor_baseline": ["terminal"], "acp_command": "sh"}

    _children(parent, tasks=(smuggled,))

    (kwargs,) = built
    assert kwargs["toolsets"] is None and kwargs["executor_baseline"] is parent._executor_baseline
    assert "acp_command" not in kwargs and "override_acp_command" not in kwargs
    # A delegation backend's route cannot reach the tool surface either: toolsets is not a route key.
    assert "toolsets" not in ROUTE_KEYS and "executor_baseline" not in ROUTE_KEYS
    assert _clean_route("p", {"toolsets": ["terminal"], "model": "m"}) == {"model": "m"}


def test_an_executor_cannot_spawn_executors(built):
    parent = _dispatcher(_delegate_depth=1)

    _children(parent)

    assert "executor_baseline" not in built[0]


def test_a_look_alike_attribute_is_never_taken_for_the_baseline(built):
    parent = _dispatcher(_executor_baseline=["terminal", "file"])

    _children(parent)

    (kwargs,) = built
    assert isinstance(kwargs["executor_baseline"], ExecutorBaseline)
    assert kwargs["executor_baseline"] is parent._executor_baseline
    # And on a parent that is not a dispatcher it is simply ignored.
    built.clear()
    _children(_parent(_executor_baseline=["terminal", "file"]))
    assert "executor_baseline" not in built[0]


@pytest.fixture
def lifecycle(monkeypatch):
    parent = SimpleNamespace(
        session_id="parent-1", enabled_toolsets=["file"], _dispatcher_session=True, _delegate_depth=0,
        disabled_toolsets=[], platform="cli",
    )
    seen: list[dict] = []

    def build(**kwargs):
        seen.append(kwargs)
        return SimpleNamespace(
            _subagent_id=f"sa-{len(seen)}", _delegate_role="leaf", _delegate_depth=1, provider="p", model="m",
            interrupt=lambda *a, **k: None,
        )

    monkeypatch.setattr("tools.delegate_tool._build_child_agent", build)
    monkeypatch.setattr("tools.delegate_tool._run_single_child", lambda *a, **k: {"status": "completed"})
    return SubagentLifecycleService(lambda: parent), seen


def test_the_plugin_lifecycle_api_never_gets_the_baseline(lifecycle):
    service, seen = lifecycle

    service.launch(SubagentLaunchRequest(goal="no toolsets named"))

    assert "executor_baseline" not in seen[0] and seen[0]["toolsets"] is None
    with pytest.raises(SubagentLifecycleError, match="broaden parent permissions"):
        service.launch(SubagentLaunchRequest(goal="more", allowed_toolsets=("terminal",)))


def test_the_executor_mark_exists_only_inside_an_executors_own_work():
    executor, ordinary = SimpleNamespace(_executor_child=True), SimpleNamespace(_executor_child=False)
    unmarked = MagicMock()  # a test double: a truthy attribute that is not True is not the mark

    assert not is_executor_child_context()
    with child_profile_scope(executor):
        assert is_executor_child_context()
    with child_profile_scope(ordinary):
        assert not is_executor_child_context()
    with child_profile_scope(unmarked):
        assert not is_executor_child_context()
    assert not is_executor_child_context()
    with executor_child_context():
        assert is_executor_child_context() and not is_delegated_child_context()
