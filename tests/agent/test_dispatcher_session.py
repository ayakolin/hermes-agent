"""Dispatcher mode's creation-time decision and session-row marker (agent/dispatcher_session.py).

A dispatcher session answers the user at once and hands long work to background executors, so its tool
surface is the narrow ``dispatcher`` posture. The decision is made once, where the session is created, and
frozen: a tool-surface change mid-conversation would break the prompt cache, and the
``sessions.model_config['dispatcher']`` marker keeps a rebuilt conversation (new process, cache eviction,
compression rotation) on the same surface.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent import dispatcher_session
from agent.dispatcher_session import (
    DISPATCHER_TOOLSET, dispatcher_applies, dispatcher_toolsets, freeze_dispatcher_session, marker_toolsets,
    session_dispatcher_toolsets, session_marker, toolsets_pinned,
)

POSTURE = ["delegate_task", "todo_list", "memory", "cronjob_manage", "read_file", "search_files"]


@pytest.fixture(autouse=True)
def _clear_subagent_and_kanban_markers(monkeypatch):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_SINGLE_QUERY_SESSION", raising=False)


def test_the_posture_toolset_is_the_six_bounded_tools():
    from toolsets import TOOLSETS, resolve_toolset

    assert TOOLSETS[DISPATCHER_TOOLSET]["tools"] == POSTURE
    assert TOOLSETS[DISPATCHER_TOOLSET]["posture"] is True
    assert resolve_toolset(DISPATCHER_TOOLSET) == sorted(POSTURE)


def test_an_interactive_local_session_is_a_dispatcher():
    assert dispatcher_applies({}, surface="cli", push_capable=True) is True


@pytest.mark.parametrize("surface", ["oneshot", "cron", "kanban", "subagent", "acp", "tool", "a2a",
                                     "api_server", "webhook", "msgraph_webhook"])
def test_surfaces_that_cannot_receive_a_later_turn_are_classic(surface):
    assert dispatcher_applies({}, surface=surface, push_capable=True) is False


@pytest.mark.parametrize("config, overrides", [
    ({"dispatcher": {"enabled": False}}, {}),
    ({"agent": {"coding_context": "focus"}}, {}),
    ({"agent": {"coding_context": "on"}}, {}),
    ({"agent": {"disabled_toolsets": ["delegation"]}}, {}),
    ({}, {"push_capable": False}),
    ({}, {"safe_mode": True}),
    ({}, {"explicit_toolsets": True}),
])
def test_every_opt_out_keeps_the_classic_surface(config, overrides):
    kwargs = {"push_capable": True, **overrides}
    assert dispatcher_applies(config, surface="cli", **kwargs) is False


def test_extra_toolsets_are_validated_and_appended():
    config = {"dispatcher": {"extra_toolsets": ["web", "not-a-toolset", "all", DISPATCHER_TOOLSET, "web"]}}
    assert dispatcher_toolsets(config) == (DISPATCHER_TOOLSET, "web")
    assert dispatcher_toolsets({}) == (DISPATCHER_TOOLSET,)


def test_a_saved_platform_selection_pins_the_surface():
    assert toolsets_pinned({"platform_toolsets": {"cli": ["terminal"]}}, "cli") is True
    assert toolsets_pinned({"platform_toolsets": {"cli": []}}, "cli") is True  # an explicit zero-tool chat
    assert toolsets_pinned({"platform_toolsets": {"telegram": ["web"]}}, "cli") is False
    assert toolsets_pinned({}, "cli") is False


def test_the_default_composite_hermes_setup_writes_is_not_a_pin():
    """``hermes setup`` writes ``platform_toolsets.<p>: [hermes-<p>]`` on every install. Counting that as a
    deliberate narrowing made dispatcher mode silently unreachable on the whole machine (found on a live
    deploy: telegram stayed classic with the full tool surface)."""
    assert toolsets_pinned({"platform_toolsets": {"telegram": ["hermes-telegram"]}}, "telegram") is False
    assert toolsets_pinned({"platform_toolsets": {"cli": ["hermes-cli"]}}, "cli") is False
    assert toolsets_pinned({"platform_toolsets": {"discord": ["hermes-discord"]}}, "discord") is False
    # Anything the operator actually changed still pins.
    assert toolsets_pinned({"platform_toolsets": {"telegram": ["hermes-telegram", "web"]}}, "telegram") is True
    assert toolsets_pinned({"platform_toolsets": {"telegram": ["terminal"]}}, "telegram") is True


@pytest.mark.parametrize("agent, expected", [
    (SimpleNamespace(enabled_toolsets=[DISPATCHER_TOOLSET], platform="cli", side_agent=False), True),
    (SimpleNamespace(enabled_toolsets=[DISPATCHER_TOOLSET, "project"], platform="tui", side_agent=False), True),
    (SimpleNamespace(enabled_toolsets=["hermes-cli"], platform="cli", side_agent=False), False),
    # A side agent (/bg, a review fork, a delegated child) is never a dispatcher however it got its toolsets.
    (SimpleNamespace(enabled_toolsets=[DISPATCHER_TOOLSET], platform="cli", side_agent=True), False),
    (SimpleNamespace(enabled_toolsets=[DISPATCHER_TOOLSET], platform="subagent", side_agent=True), False),
])
def test_the_agent_flag_comes_from_what_the_agent_was_built_with(agent, expected):
    assert freeze_dispatcher_session(agent, agent.enabled_toolsets) is expected
    assert agent._dispatcher_session is expected
    # Frozen: a later call with other toolsets (a rebuild) never flips a live conversation.
    assert freeze_dispatcher_session(agent, ["hermes-cli"]) is expected


def test_a_one_shot_session_is_never_a_dispatcher(monkeypatch):
    monkeypatch.setenv("HERMES_SINGLE_QUERY_SESSION", "1")
    agent = SimpleNamespace(enabled_toolsets=[DISPATCHER_TOOLSET], platform="cli", side_agent=False)
    assert freeze_dispatcher_session(agent, agent.enabled_toolsets) is False


def test_only_a_dispatcher_agent_carries_a_marker_and_it_holds_its_toolsets():
    dispatcher = SimpleNamespace(_dispatcher_session=True, enabled_toolsets=[DISPATCHER_TOOLSET, "web", "project"],
                                 platform="desktop")
    assert session_marker(dispatcher) == {"toolsets": sorted([DISPATCHER_TOOLSET, "web"])}
    classic = SimpleNamespace(_dispatcher_session=False, enabled_toolsets=["hermes-cli"], platform="cli")
    assert session_marker(classic) is None
    assert marker_toolsets({"toolsets": ["dispatcher", "web"]}) == ("dispatcher", "web")
    assert marker_toolsets({"toolsets": ["web"]}) is None
    assert marker_toolsets({"toolsets": "dispatcher"}) is None
    assert marker_toolsets(True) is None  # a marker written by an earlier shape names no toolsets


def test_the_marker_is_written_only_for_a_dispatcher_agent(tmp_path):
    from hermes_state import SessionDB

    db = SessionDB(tmp_path / "state.db")
    db.create_session("sid", "telegram")
    dispatcher = SimpleNamespace(_dispatcher_session=True, enabled_toolsets=[DISPATCHER_TOOLSET],
                                 platform="telegram", _session_db=db, session_id="sid")
    dispatcher_session.persist_session_marker(dispatcher)
    assert db.get_session_model_config_value("sid", dispatcher_session.SESSION_MARKER_KEY) == {
        "toolsets": [DISPATCHER_TOOLSET]}
    db.create_session("classic", "telegram")
    classic = SimpleNamespace(_dispatcher_session=False, enabled_toolsets=["hermes-cli"], platform="telegram",
                              _session_db=db, session_id="classic")
    dispatcher_session.persist_session_marker(classic)
    assert db.get_session_model_config_value("classic", dispatcher_session.SESSION_MARKER_KEY) is None


def _db_with(tmp_path, marker=None, **columns):
    from hermes_state import SessionDB

    db = SessionDB(tmp_path / "state.db")
    db.create_session("sid", "telegram", **columns)
    if marker is not None:
        db.patch_session_model_config("sid", {dispatcher_session.SESSION_MARKER_KEY: marker})
    return db


def _decide(calls):
    def decide():
        calls.append(1)
        return (DISPATCHER_TOOLSET, "web")
    return decide


def test_the_marker_decides_the_conversation(tmp_path):
    db = _db_with(tmp_path, marker={"toolsets": [DISPATCHER_TOOLSET, "web"]})
    assert session_dispatcher_toolsets(db, "sid", _decide([])) == (DISPATCHER_TOOLSET, "web")


def test_a_conversation_with_history_and_no_marker_asks_the_live_config(tmp_path):
    """Every chat moves to dispatcher mode, including one that ran before the upgrade.

    History is read-only, so narrowing the main agent's surface changes nothing about stored messages. Only a
    stored marker (``test_a_marker_decides_without_reading_config``) settles the question without the config."""
    db = _db_with(tmp_path, system_prompt="bytes the session already sends")
    assert session_dispatcher_toolsets(db, "sid", _decide(calls := [])) == ("dispatcher", "web")
    assert len(calls) == 1
    pinned = _db_with(tmp_path, system_prompt="bytes the session already sends")
    pinned.update_session_tool_names("sid", {"tools": []})
    assert session_dispatcher_toolsets(pinned, "sid", _decide(calls)) == ("dispatcher", "web")
    assert len(calls) == 2


def test_a_conversation_that_built_nothing_asks_the_live_config_once(tmp_path):
    db = _db_with(tmp_path)  # a row a delivery mirror created, no prompt and no tools[] yet
    calls = []
    assert session_dispatcher_toolsets(db, "sid", _decide(calls)) == (DISPATCHER_TOOLSET, "web")
    assert calls == [1]


def test_an_unreadable_store_answers_classic():
    class _Broken:
        def get_compression_tip(self, session_id):
            raise RuntimeError("state.db is being recovered")

    assert session_dispatcher_toolsets(_Broken(), "sid", _decide([])) is None


def test_a_decision_that_is_classic_returns_none(tmp_path):
    assert session_dispatcher_toolsets(_db_with(tmp_path), "sid", lambda: None) is None
