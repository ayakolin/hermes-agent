"""Dispatcher mode on the legacy tui_gateway surfaces (tui_gateway/agent_factory.py).

Ink and the Desktop app can reach the same backend: a session built here gets the dispatcher posture when
its conversation is a dispatcher, its read-back RPCs report that surface before the agent exists, and a side
agent (``/bg``) — which does the work the dispatcher hands off — keeps the classic selection.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent.dispatcher_session import DISPATCHER_TOOLSET, SESSION_MARKER_KEY
from tui_gateway import server


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("HERMES_TUI_TOOLSETS", raising=False)


@pytest.fixture
def db(tmp_path):
    from hermes_state import SessionDB

    database = SessionDB(tmp_path / "state.db")
    database.create_session("sid", "tui")
    return database


def test_a_new_tui_conversation_is_a_dispatcher(db):
    selected = server._session_enabled_toolsets("tui", db, "sid")
    assert DISPATCHER_TOOLSET in selected
    assert "project" in selected  # the client surface's own toolset still folds in


@pytest.mark.parametrize("platform", ["cli", "telegram"])
def test_only_the_client_surfaces_are_dispatchers(db, platform):
    assert server._session_enabled_toolsets(platform, db, "sid") == server._load_enabled_toolsets(platform)


def test_a_pinned_selection_keeps_the_classic_toolsets(db, monkeypatch):
    monkeypatch.setenv("HERMES_TUI_TOOLSETS", "web")
    assert server._session_enabled_toolsets("tui", db, "sid") == ["web"]


def test_the_marker_carries_the_decision(db):
    db.patch_session_model_config("sid", {SESSION_MARKER_KEY: {"toolsets": [DISPATCHER_TOOLSET, "web"]}})
    assert DISPATCHER_TOOLSET in server._session_enabled_toolsets("desktop", db, "sid")


def test_a_conversation_that_built_a_prompt_stays_classic(db):
    db.update_system_prompt("sid", "bytes this chat already sends")
    assert server._session_enabled_toolsets("tui", db, "sid") == server._load_enabled_toolsets("tui")


def test_a_background_agent_of_a_dispatcher_keeps_the_terminal(db):
    """``/bg`` runs the prompt a dispatcher would hand to an executor, so it must keep the classic surface
    instead of inheriting the six-tool posture."""
    dispatcher = SimpleNamespace(_dispatcher_session=True, enabled_toolsets=[DISPATCHER_TOOLSET],
                                 enabled_toolsets_frozen=True)
    kwargs = server._background_agent_kwargs(dispatcher, "bg_task")
    assert "terminal" in (kwargs["enabled_toolsets"] or [])


def test_a_background_agent_of_a_classic_session_keeps_its_inherited_toolsets():
    classic = SimpleNamespace(_dispatcher_session=False, enabled_toolsets=["file"])
    assert server._background_agent_kwargs(classic, "bg_task")["enabled_toolsets"] == ["file"]
