"""Dispatcher mode for a local session (CLI, Ink TUI, Desktop): decided once, when ``build_policy`` freezes it.

The decision rides ``LocalSessionPolicy.dispatcher`` for the life of the session (a tool-surface change mid
conversation would break the prompt cache). Chat-platform sessions and the legacy tui_gateway have no policy
to freeze it in; they use the session-row marker (``gateway/run_turn_dispatcher.py``,
``tui_gateway/agent_factory.py``).
"""
from __future__ import annotations

from agent.dispatcher_session import dispatcher_applies, toolsets_pinned


def local_dispatcher(params, config, source, *, safe_mode):
    """Whether a NEW local session is a dispatcher.

    A ``toolsets`` pick in *params* and a saved ``platform_toolsets.cli`` both pin the selection, and a pinned
    session keeps it. A session a managed worker executes (``gateway.managed_workers``) cannot receive a later
    result as a turn, so it stays classic. ``acp``, ``tool`` and ``oneshot`` are in ``NON_DISPATCHER_SURFACES``:
    an editor, an integration and a finite run have no later turn to deliver to."""
    gateway = config.get('gateway')
    managed = isinstance(gateway, dict) and gateway.get('managed_workers') is True
    pinned = 'toolsets' in params or toolsets_pinned(config, 'cli')
    return dispatcher_applies(config, surface=source, push_capable=not managed, safe_mode=safe_mode,
                              explicit_toolsets=pinned)
