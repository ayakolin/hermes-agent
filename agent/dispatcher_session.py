"""Dispatcher mode: the shared contract every surface and phase reads.

A dispatcher session answers the user at once and hands long work to background executor subagents.
Whether a session is a dispatcher is decided ONCE when it is created and frozen on the agent as
``_dispatcher_session`` (a tool-surface or system-prompt change mid-conversation would break the prompt
cache), so nothing here is a live toggle. Call sites read ``agent_is_dispatcher(agent)``; the creation-time
decision lives in the surface that builds the session (``dispatcher_applies`` and its callers).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Mapping

logger = logging.getLogger(__name__)

DISPATCHER_TOOLSET = "dispatcher"
# ``sessions.model_config`` key of a dispatcher session (absent for a classic one).
SESSION_MARKER_KEY = "dispatcher"

# Sessions that cannot receive a later-result turn, or that are themselves workers.
NON_DISPATCHER_SURFACES = frozenset({
    "oneshot", "cron", "kanban", "subagent", "acp", "tool", "a2a",
    "api_server", "webhook", "msgraph_webhook",
})


@dataclass(frozen=True)
class DispatcherSettings:
    enabled: bool = True
    extra_toolsets: tuple[str, ...] = ()
    ack_reminder: bool = True
    ack_max_nudges: int = 1
    force_interim: bool = True
    early_interim: bool = True
    memory_budget_s: float = 2.0
    pre_llm_call_budget_s: float = 3.0
    skills_recheck_s: float = 30.0
    defer_compaction: bool = True
    defer_ceiling_ratio: float = 0.85
    stream_deltas: bool = True
    busy_policy: str = "dispatch"
    quick_lane: bool = True
    stall_seconds: float = 4.0


def _section(config: Mapping[str, Any] | None, name: str) -> Mapping[str, Any]:
    value = (config or {}).get(name)
    return value if isinstance(value, Mapping) else {}


def _num(value: Any, default: float, *, floor: float = 0.0) -> float:
    try:
        return max(floor, float(value))
    except (TypeError, ValueError):
        return default


def dispatcher_settings(config: Mapping[str, Any] | None) -> DispatcherSettings:
    """Typed view of the ``dispatcher:`` section with the defaults carried in code.

    The gateway reads raw YAML without merging ``DEFAULT_CONFIG``, so the defaults live here too."""
    root = _section(config, "dispatcher")
    ack, lat, busy = (_section(root, k) for k in ("ack", "latency", "busy"))
    defaults = DispatcherSettings()
    extra = root.get("extra_toolsets")
    return DispatcherSettings(
        enabled=bool(root.get("enabled", defaults.enabled)),
        extra_toolsets=tuple(str(t) for t in extra) if isinstance(extra, (list, tuple)) else (),
        ack_reminder=bool(ack.get("reminder", defaults.ack_reminder)),
        ack_max_nudges=int(_num(ack.get("max_nudges"), defaults.ack_max_nudges)),
        force_interim=bool(ack.get("force_interim", defaults.force_interim)),
        early_interim=bool(ack.get("early_interim", defaults.early_interim)),
        memory_budget_s=_num(lat.get("memory_budget_s"), defaults.memory_budget_s),
        pre_llm_call_budget_s=_num(lat.get("pre_llm_call_budget_s"), defaults.pre_llm_call_budget_s),
        skills_recheck_s=_num(lat.get("skills_recheck_s"), defaults.skills_recheck_s),
        defer_compaction=bool(lat.get("defer_compaction", defaults.defer_compaction)),
        defer_ceiling_ratio=_num(lat.get("defer_ceiling_ratio"), defaults.defer_ceiling_ratio),
        stream_deltas=bool(lat.get("stream_deltas", defaults.stream_deltas)),
        busy_policy="inherit" if str(busy.get("policy", "")).strip().lower() == "inherit" else "dispatch",
        quick_lane=bool(busy.get("quick_lane", defaults.quick_lane)),
        stall_seconds=_num(busy.get("stall_seconds"), defaults.stall_seconds),
    )


def agent_is_dispatcher(agent: Any) -> bool:
    """True when ``agent`` was created as a dispatcher session (frozen at creation, never toggled)."""
    return getattr(agent, "_dispatcher_session", False) is True


# ── creation-time decision ────────────────────────────────────────────────────────────────────────────
# Not to be confused with ``agent.delegation_context.is_dispatcher_owned_worker_context``, which is the Kanban
# dispatcher. Here "dispatcher" is the foreground agent of a session that hands long work to executors.

def dispatcher_toolsets(config: Mapping[str, Any] | None) -> tuple[str, ...]:
    """The toolsets a new dispatcher session runs: the posture toolset plus the operator's validated
    ``dispatcher.extra_toolsets``. Called once, when the session is created; the result is frozen in the
    session (policy or row marker) so a later config edit never changes a running conversation's tool array."""
    from hermes_cli.toolset_validation import saved_toolset_resolver
    known = saved_toolset_resolver(dict(config or {}))
    extras: list[str] = []
    for name in dispatcher_settings(config).extra_toolsets:
        if name in {"all", "*", DISPATCHER_TOOLSET} or name in extras:
            continue
        if not known(name):
            logger.warning("dispatcher.extra_toolsets: ignoring unknown toolset %r", name)
            continue
        extras.append(name)
    return (DISPATCHER_TOOLSET, *extras)


def toolsets_pinned(config: Mapping[str, Any] | None, platform: str) -> bool:
    """Whether the operator NARROWED this platform's tool selection, so the surface keeps it and is not a
    dispatcher.

    ``platform_toolsets.<platform>`` alone is not evidence: ``hermes setup`` writes the platform's own default
    composite (``telegram: [hermes-telegram]``) on every install, and a default install must still become a
    dispatcher. Only a list that differs from that default is a deliberate choice. An explicit empty list is
    the operator asking for no tools at all — also deliberate."""
    from hermes_cli.tools_config import (
        _coerce_platform_toolsets_value, _platform_default_toolset, _platform_toolsets_explicitly_saved,
    )
    from toolsets import BUILTIN_TOOLSET_NAMES
    config = dict(config or {})
    if not _platform_toolsets_explicitly_saved(config, platform):
        return False
    saved = _coerce_platform_toolsets_value((config.get("platform_toolsets") or {}).get(platform), platform)
    if not saved:
        return True
    if [str(name) for name in saved] == [_platform_default_toolset(platform)]:
        return False
    # A surface whose saved selection is exactly what the platform would offer anyway is unchanged.
    if len(saved) == 1 and str(saved[0]) in BUILTIN_TOOLSET_NAMES and str(saved[0]) in {
        _platform_default_toolset(platform), f"hermes-{platform}",
    }:
        return False
    return True


def delegation_reachable(disabled: Any) -> bool:
    """Whether ``delegate_task`` survives the operator's ``agent.disabled_toolsets``. Without it a dispatcher
    could never hand work off, so the session stays classic."""
    from model_tools import _select_tool_names
    return "delegate_task" in _select_tool_names([DISPATCHER_TOOLSET], list(disabled or ()) or None, True)


def dispatcher_applies(
    config: Mapping[str, Any] | None, *, surface: str, push_capable: bool, safe_mode: bool = False,
    explicit_toolsets: bool = False, disabled: Any = None,
) -> bool:
    """Whether a NEW session is a dispatcher. Asked once per session, where it is created (local
    ``build_policy``, the first turn of a gateway chat, a legacy ``session.create``); never re-asked.

    ``push_capable``: the session can receive a later result as a new turn. ``explicit_toolsets``: the
    operator or client pinned this session's tool selection, which dispatcher mode does not override."""
    from agent.coding_context import coding_pins_posture
    if not dispatcher_settings(config).enabled or safe_mode or explicit_toolsets or not push_capable:
        return False
    if surface in NON_DISPATCHER_SURFACES or coding_pins_posture(dict(config or {})):
        return False
    if disabled is None:
        from agent.skill_utils import parse_config_string_list
        disabled = parse_config_string_list(((config or {}).get("agent") or {}).get("disabled_toolsets"))
    return delegation_reachable(disabled)


def freeze_dispatcher_session(agent: Any, enabled_toolsets: Any) -> bool:
    """Freeze ``agent._dispatcher_session`` from what the agent was built with, before its first tool build
    (the flag picks the ``delegate_task`` variant and the prompt, both part of the cached prefix).

    A dispatcher is a non-side agent on a surface that can receive a later result and whose toolsets carry
    the posture toolset. Side agents (``/bg``, review forks, delegated children) never are one, however
    they were handed their toolsets. The depth of a delegated child is assigned only after construction, so
    it is not an input: children are side agents."""
    cached = getattr(agent, "_dispatcher_session", None)
    if isinstance(cached, bool):
        return cached
    from agent.delegation_context import owned_kanban_task
    from agent.oneshot_footprint import agent_is_oneshot
    frozen = (
        DISPATCHER_TOOLSET in (enabled_toolsets or ())
        and not getattr(agent, "side_agent", False)
        and str(getattr(agent, "platform", "") or "").lower() not in NON_DISPATCHER_SURFACES
        and not agent_is_oneshot(agent)
        and not owned_kanban_task()
    )
    agent._dispatcher_session = frozen
    return frozen


# ── session-row marker ────────────────────────────────────────────────────────────────────────────────
# Written only for a dispatcher (``sessions.model_config['dispatcher']``). A missing marker on a session that
# already built a prompt is a classic session, so a conversation never flips when the default changes.

def session_marker(agent: Any) -> dict[str, Any] | None:
    """The marker to persist for *agent*, or None for a classic session. It holds the toolsets chosen at
    creation (posture plus operator extras, without the toolsets a client surface folds in), so every
    later rebuild of this conversation selects the same tools."""
    if not agent_is_dispatcher(agent):
        return None
    from toolsets import CLIENT_SURFACE_TOOLSETS, TOOLSET_SESSION_PLATFORMS
    chosen = [name for name in (getattr(agent, "enabled_toolsets", None) or ())
              if name not in CLIENT_SURFACE_TOOLSETS and name not in TOOLSET_SESSION_PLATFORMS]
    return {"toolsets": sorted(set(chosen) | {DISPATCHER_TOOLSET})}


def marker_toolsets(value: Any) -> tuple[str, ...] | None:
    """The toolsets a stored marker names, or None when it is absent or malformed."""
    names = value.get("toolsets") if isinstance(value, Mapping) else None
    if not isinstance(names, list) or DISPATCHER_TOOLSET not in names or not all(isinstance(n, str) for n in names):
        return None
    return tuple(names)


def persist_session_marker(agent: Any) -> None:
    """Patch the marker onto the agent's session row. The row may predate the agent (the gateway creates a
    bare one), in which case the agent's own initial ``model_config`` was ignored by the upsert."""
    marker = session_marker(agent)
    db, session_id = getattr(agent, "_session_db", None), getattr(agent, "session_id", None)
    if marker is None or not db or not session_id:
        return
    try:
        db.patch_session_model_config(session_id, {SESSION_MARKER_KEY: marker})
    except Exception:
        logger.warning("Could not record the dispatcher marker (session=%s)", session_id, exc_info=True)


def session_dispatcher_toolsets(db: Any, session_id: Any, decide: Callable[[], tuple[str, ...] | None]
                                ) -> tuple[str, ...] | None:
    """The dispatcher selection of the conversation *session_id* belongs to, or None for a classic one.

    The marker on the compression tip decides when present: it is written the first time a session is found
    to be a dispatcher, so later turns and other processes read the same answer without consulting live
    config again. A conversation with NO marker asks ``decide()``, whether or not it has history: every chat
    is meant to move to dispatcher mode, and a conversation that ran before the upgrade is exactly what must
    move. Its stored messages are read-only here, so narrowing the main agent's tool surface changes nothing
    about them.

    Only a store that cannot be read answers classic, the behaviour before dispatcher mode existed."""
    if db is None or not session_id:
        return decide()
    from hermes_state_sessions import _parse_model_config
    try:
        row = db.get_session(db.get_compression_tip(session_id) or session_id)
    except Exception:
        logger.warning("Dispatcher marker unreadable (session=%s); using the classic tool surface",
                       session_id, exc_info=True)
        return None
    if row is None:
        return decide()
    stored = marker_toolsets(_parse_model_config(row.get("model_config")).get(SESSION_MARKER_KEY))
    if stored is not None:
        return stored
    return decide()
