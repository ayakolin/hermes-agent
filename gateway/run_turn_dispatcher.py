"""Dispatcher mode for a chat-platform turn (Telegram, Discord, Slack, ...).

A chat session has no frozen launch policy, so its dispatcher decision lives where the conversation does: on
the cached agent while the process holds it, and as ``sessions.model_config['dispatcher']`` for a rebuild
(cache eviction, restart, compression rotation). ``chat_turn_toolsets`` answers both for one turn, ahead of
``_resolve_turn_agent``: the agent-cache signature includes ``enabled_toolsets``, so the selection has to be
the same value on every turn of the conversation.
"""
from __future__ import annotations

import logging

from agent.dispatcher_session import (
    dispatcher_applies, dispatcher_toolsets, session_dispatcher_toolsets, toolsets_pinned,
)
from gateway.config import Platform
from gateway.wake import adapter_supports_push

_NO_AGENT = object()
logger = logging.getLogger(__name__)


def _cached_selection(runner, ctx):
    """What this process's cached agent for the chat says: its toolsets (a dispatcher), None (a classic
    agent) or ``_NO_AGENT``. Only an agent built for THIS session id counts: the cache is keyed by the chat and
    deliberately survives a session switch, so a mismatched entry belongs to another conversation."""
    lock, cache = getattr(runner, "_agent_cache_lock", None), getattr(runner, "_agent_cache", None)
    if not lock or cache is None:
        return _NO_AGENT
    with lock:
        entry = cache.get(ctx.session_key)
    if not entry or len(entry) < 4 or entry[3] != ctx.session_id:
        return _NO_AGENT
    frozen = getattr(entry[0], "_dispatcher_session", None)  # the pending-build sentinel has none
    if not isinstance(frozen, bool):
        return _NO_AGENT
    return tuple(entry[0].enabled_toolsets) if frozen else None


def _toolsets_pinned(adapter, source, config, platform_key) -> bool:
    """Whether the operator fixed this chat's tool selection: a saved ``platform_toolsets.<platform>`` (what
    ``hermes tools`` writes) or an adapter's per-source override. Dispatcher mode does not replace either."""
    if toolsets_pinned(config, platform_key):
        return True
    try:
        return bool(adapter.toolsets_for_source(source))
    except Exception:  # health: allow BLE001 -- a plugin adapter's override must not fail the turn; the toolset resolver treats it as none
        return False


def _decide(runner, ctx, platform_key):
    """The creation-time decision for a conversation that has not built a prompt yet."""
    adapter = runner._delivery_adapter_for(ctx.source)
    if adapter is None:
        return None
    applies = dispatcher_applies(
        ctx.user_config, surface=platform_key, push_capable=adapter_supports_push(adapter),
        explicit_toolsets=_toolsets_pinned(adapter, ctx.source, ctx.user_config, platform_key))
    return dispatcher_toolsets(ctx.user_config) if applies else None


def dispatcher_turn_toolsets(runner, ctx, platform_key, current):
    """*current* (the turn's classic toolsets) or, when the conversation is a dispatcher, its toolset list.

    The conversation's own history decides after its first turn: the cached agent, else the marker on its
    compression tip, else (a conversation that built a prompt without a marker) classic. Only the first turn
    reads the live config, so a later ``dispatcher.enabled`` or ``extra_toolsets`` edit never changes a
    running chat. Local sessions carry their decision in their policy, not here."""
    if ctx.source.platform == Platform.LOCAL:
        return current
    cached = _cached_selection(runner, ctx)
    frozen_reason = "cached agent" if cached is not _NO_AGENT else "session marker or first turn"
    if cached is not _NO_AGENT:
        selected = cached
    else:
        db = runner._session_db
        selected = session_dispatcher_toolsets(
            getattr(db, "_db", db), ctx.session_id, lambda: _decide(runner, ctx, platform_key))
    selected = current if selected is None else list(selected)
    logger.info(
        "dispatcher mode: %s session (source=%s, toolsets=%s, decided by %s)",
        "dispatcher" if selected is not current else "classic", platform_key, list(selected), frozen_reason,
    )
    return selected
