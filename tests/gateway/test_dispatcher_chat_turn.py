"""The dispatcher decision of a chat-platform turn (gateway/run_turn_dispatcher.py).

A chat has no frozen launch policy, so its conversation's decision lives on the cached agent in this process
and in ``sessions.model_config['dispatcher']`` across a rebuild. ``dispatcher_turn_toolsets`` is asked once
per turn, before ``_resolve_turn_agent`` (the agent-cache signature includes the toolsets), and must answer
the same value every turn of one conversation: a flip would rebuild the agent and break the prompt cache.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent.dispatcher_session import DISPATCHER_TOOLSET, SESSION_MARKER_KEY
from gateway.config import Platform
from gateway.run_turn_dispatcher import dispatcher_turn_toolsets

CLASSIC = ["hermes-telegram", "web"]


def _ctx(session_id="s1", session_key="agent:main:telegram:dm:1", user_config=None):
    return SimpleNamespace(
        source=SimpleNamespace(platform=Platform.TELEGRAM, chat_id="1"), session_id=session_id,
        session_key=session_key, user_config=user_config if user_config is not None else {},
    )


def _runner(db=None, agent=None, cached_sid=None, toolsets_for_source=None):
    cache = {}
    if agent is not None:
        cache["agent:main:telegram:dm:1"] = (agent, "sig", 0, cached_sid)
    adapter = SimpleNamespace(toolsets_for_source=lambda source: toolsets_for_source)
    return SimpleNamespace(
        _agent_cache_lock=__import__("threading").Lock(), _agent_cache=cache, _session_db=db,
        _delivery_adapter_for=lambda source: adapter,
    )


@pytest.fixture
def db(tmp_path):
    from hermes_state import SessionDB

    database = SessionDB(tmp_path / "state.db")
    database.create_session("s1", "telegram")
    return database


def _mark(db, session_id="s1", toolsets=(DISPATCHER_TOOLSET, "web")):
    db.patch_session_model_config(session_id, {SESSION_MARKER_KEY: {"toolsets": list(toolsets)}})


def test_the_first_turn_of_a_new_chat_becomes_a_dispatcher(db):
    runner = _runner(db)
    assert dispatcher_turn_toolsets(runner, _ctx(), "telegram", CLASSIC) == [DISPATCHER_TOOLSET]


def test_a_marker_carries_the_decision_into_later_turns_and_other_processes(db):
    _mark(db)
    assert dispatcher_turn_toolsets(_runner(db), _ctx(), "telegram", CLASSIC) == [DISPATCHER_TOOLSET, "web"]


def test_a_chat_that_ran_before_the_upgrade_moves_to_dispatcher(db):
    """Every chat moves, including one whose prompt was built before dispatcher mode existed.

    A config edit still cannot flip a chat that already settled its answer: that answer is the stored marker,
    written on its first dispatcher turn (see ``test_a_marker_carries_the_decision_into_later_turns``)."""
    db.update_system_prompt("s1", "the prompt this chat already sends")
    enabled = {"dispatcher": {"enabled": True}}
    assert dispatcher_turn_toolsets(_runner(db), _ctx(user_config=enabled), "telegram", CLASSIC) == [DISPATCHER_TOOLSET]
    disabled = {"dispatcher": {"enabled": False}}
    assert dispatcher_turn_toolsets(_runner(db, None), _ctx(session_id="s2", user_config=disabled),
                                    "telegram", CLASSIC) == CLASSIC


def test_a_marker_wins_over_an_edited_config(db):
    _mark(db)
    runner = _runner(db)
    ctx = _ctx(user_config={"dispatcher": {"enabled": False, "extra_toolsets": ["session_search"]}})
    assert dispatcher_turn_toolsets(runner, ctx, "telegram", CLASSIC) == [DISPATCHER_TOOLSET, "web"]


def test_a_cached_agent_decides_without_reading_the_store():
    agent = SimpleNamespace(_dispatcher_session=True, enabled_toolsets=[DISPATCHER_TOOLSET])
    runner = _runner(None, agent=agent, cached_sid="s1")
    assert dispatcher_turn_toolsets(runner, _ctx(), "telegram", CLASSIC) == [DISPATCHER_TOOLSET]


def test_a_cached_agent_of_another_conversation_is_not_read(db):
    """The cache is keyed by the chat and survives a session switch: its flag belongs to another session.
    The conversation's own row (no prompt yet) then decides."""
    agent = SimpleNamespace(_dispatcher_session=False, enabled_toolsets=["hermes-telegram"])
    runner = _runner(db, agent=agent, cached_sid="older-session")
    assert dispatcher_turn_toolsets(runner, _ctx(), "telegram", CLASSIC) == [DISPATCHER_TOOLSET]


def test_a_classic_cached_agent_keeps_the_classic_surface():
    agent = SimpleNamespace(_dispatcher_session=False, enabled_toolsets=["hermes-telegram"])
    runner = _runner(None, agent=agent, cached_sid="s1")
    assert dispatcher_turn_toolsets(runner, _ctx(), "telegram", CLASSIC) == CLASSIC


@pytest.mark.parametrize("adapter_override", [["web"], []])
def test_an_adapters_own_selection_pins_the_surface(db, adapter_override):
    """A webhook's per-route toolsets are the operator's routing decision, not a dispatcher default."""
    ctx = _ctx(user_config={"platform_toolsets": {"webhook": ["web"]}})
    ctx.source = SimpleNamespace(platform=Platform.WEBHOOK, chat_id="hook")
    runner = _runner(db, toolsets_for_source=adapter_override)
    assert dispatcher_turn_toolsets(runner, ctx, "webhook", CLASSIC) == CLASSIC


def test_a_platform_selection_saved_by_hermes_tools_pins_the_surface(db):
    ctx = _ctx(user_config={"platform_toolsets": {"telegram": ["web"]}})
    assert dispatcher_turn_toolsets(_runner(db), ctx, "telegram", CLASSIC) == CLASSIC


def test_an_unreachable_adapter_keeps_the_classic_surface(db):
    runner = _runner(db)
    runner._delivery_adapter_for = lambda source: None
    assert dispatcher_turn_toolsets(runner, _ctx(), "telegram", CLASSIC) == CLASSIC


def test_a_local_session_is_never_decided_here(tmp_path):
    """Local sessions carry their decision in the launch policy; this path must not override it."""
    ctx = _ctx()
    ctx.source = SimpleNamespace(platform=Platform.LOCAL, chat_id="c")
    assert dispatcher_turn_toolsets(_runner(None), ctx, "cli", CLASSIC) == CLASSIC
