"""``model_tools._last_resolved_tool_names`` is the parent's: a delegated child never overwrites it.

A deferred child builds on a pool thread while the dispatcher's next turn resolves its own tools, so a child that
wrote the global (and relied on a save/restore around its run) could leave the child's narrower list behind.
"""

from __future__ import annotations

import model_tools
from agent.delegation_context import delegated_child_context


def _resolve(toolsets):
    return model_tools.get_tool_definitions(enabled_toolsets=toolsets, quiet_mode=True)


def test_child_resolution_leaves_the_parents_tool_names_alone():
    model_tools._clear_tool_defs_cache()
    parent_tools = _resolve(["file"])
    parent_names = list(model_tools._last_resolved_tool_names)
    assert parent_names and {t["function"]["name"] for t in parent_tools} == set(parent_names)

    with delegated_child_context():
        child_tools = _resolve(["todo"])  # cache miss: computed inside the child context
        assert {t["function"]["name"] for t in child_tools} != set(parent_names)
        _resolve(["todo"])  # cache hit: the other write site
    assert model_tools._last_resolved_tool_names == parent_names


def test_parent_resolution_still_updates_the_tool_names():
    model_tools._clear_tool_defs_cache()
    _resolve(["file"])
    first = list(model_tools._last_resolved_tool_names)
    _resolve(["todo"])
    assert model_tools._last_resolved_tool_names != first
    _resolve(["file"])  # cache hit outside a child context
    assert model_tools._last_resolved_tool_names == first
