"""Dispatcher mode: the shared contract every surface and phase reads.

A dispatcher session answers the user at once and hands long work to background executor subagents.
Whether a session is a dispatcher is decided ONCE when it is created and frozen on the agent as
``_dispatcher_session`` (a tool-surface or system-prompt change mid-conversation would break the prompt
cache), so nothing here is a live toggle. Call sites read ``agent_is_dispatcher(agent)``; the creation-time
decision lives in the surface that builds the session (``dispatcher_applies`` and its callers).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

DISPATCHER_TOOLSET = "dispatcher"

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
