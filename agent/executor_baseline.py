"""Executor baseline: the tool surface of a dispatcher session's ``delegate_task`` children.

A dispatcher session keeps a narrow tool set and hands long work to background executors. Those executors need
real tools, so for them (and only them) ``delegate_task`` replaces "a child never exceeds its parent" with a
baseline read from trusted config (``delegation.executor``). The model never chooses it: ``toolsets`` stays out of
the delegate_task route keys, and ``tools/delegate_tool.py::_build_children`` is the only reader.

Held fixed for an executor, whatever the baseline says: ``DELEGATE_BLOCKED_TOOLS``, kanban, the forced leaf role,
the parent's ``agent.disabled_toolsets`` (an operator ban), and the **session ceiling**. The ceiling is what the
operator allows the session before the dispatcher posture narrowed it (``platform_toolsets.<platform>``, a
hosted-room execution policy, per-platform MCP allowlists). A baseline toolset survives only when every tool of it
sits inside the ceiling, so a telegram profile limited to ``web`` gets web-only executors.

Contract with the session builder: dispatcher narrowing is done with ``enabled_toolsets`` only, never
``disabled_toolsets`` (executors inherit that list), and the pre-narrowing enabled set is the ``ceiling`` handed to
:func:`freeze_executor_baseline` before the dispatcher whitelist is applied. Without one the ceiling is derived from
config for the session's platform.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional

from agent.dispatcher_session import agent_is_dispatcher

logger = logging.getLogger(__name__)

# Added by code, never listed in config: the platform-gated toolset holding the executor-only ``needs_input`` tool.
EXECUTOR_TOOLSET = "executor"

_warned: set[str] = set()


@dataclass(frozen=True)
class ExecutorBaseline:
    """Frozen per dispatcher session: built-in toolsets and MCP server names, already inside the ceiling."""

    toolsets: tuple[str, ...]
    mcp_toolsets: tuple[str, ...] = ()


def _warn_once(message: str, *args: Any) -> None:
    text = message % args if args else message
    if text not in _warned:
        _warned.add(text)
        logger.warning(text)


def _name_problem(name: str) -> Optional[str]:
    """Why ``name`` cannot be in a baseline, or None. Built-in, single-purpose toolsets only: a plugin or MCP name,
    a client-surface or platform-gated toolset, a ``hermes-*``/composite/posture bundle (it would carry blocked
    tools only partly stripped) and anything resolving to a blocked or kanban tool are all refused."""
    from tools.delegate_tool_toolsets import DELEGATE_BLOCKED_TOOLS
    from toolsets import (
        BUILTIN_TOOLSET_NAMES, CLIENT_SURFACE_TOOLSETS, TOOLSET_SESSION_PLATFORMS, TOOLSETS, resolve_toolset,
    )

    if name not in BUILTIN_TOOLSET_NAMES:
        return "not a built-in toolset"
    if name in CLIENT_SURFACE_TOOLSETS or name in TOOLSET_SESSION_PLATFORMS:
        return "a client-surface or platform-gated toolset"
    spec = TOOLSETS[name]
    if name.startswith("hermes-") or spec.get("includes") or spec.get("posture"):
        return "a composite or posture toolset"
    tools = resolve_toolset(name, include_registry=False)
    if not tools:
        return "a toolset whose tools are added at runtime"
    if DELEGATE_BLOCKED_TOOLS.intersection(tools) or any(tool.startswith("kanban_") for tool in tools):
        return "a toolset holding a tool executors never get"
    return None


def _clean_toolsets(raw: Any) -> tuple[str, ...]:
    """The valid, de-duplicated names of ``delegation.executor.toolsets``. A non-list falls back to the default
    list; an empty or fully-invalid list stays empty (fail closed, never silently widened)."""
    from hermes_cli.config_defaults_delegation import DELEGATION_DEFAULTS

    if not isinstance(raw, (list, tuple)):
        _warn_once("delegation.executor.toolsets=%r is not a list; using the default executor toolsets", raw)
        raw = DELEGATION_DEFAULTS["executor"]["toolsets"]
    kept: list[str] = []
    for item in raw:
        name = str(item).strip()
        if name in kept:
            continue
        if (problem := _name_problem(name)) is not None:
            _warn_once("delegation.executor.toolsets: dropping %r (%s)", name, problem)
            continue
        kept.append(name)
    return tuple(kept)


def _clean_mcp(raw: Any, enabled: Iterable[str]) -> tuple[str, ...]:
    """MCP server names for ``delegation.executor.mcp``: ``all`` (every server enabled in the profile), ``none``, or
    a list narrowed to the enabled ones. Anything else is refused as ``none``."""
    enabled = set(enabled)
    if raw is True or (isinstance(raw, str) and raw.strip().lower() == "all"):
        return tuple(sorted(enabled))
    if raw is False or (isinstance(raw, str) and raw.strip().lower() == "none"):
        return ()
    if isinstance(raw, (list, tuple)):
        return tuple(sorted({str(item).strip() for item in raw} & enabled))
    _warn_once('delegation.executor.mcp=%r is not "all", "none" or a list of server names; using "none"', raw)
    return ()


def _ceiling_tools(ceiling: Iterable[str]) -> frozenset[str]:
    from toolsets import resolve_toolset

    return frozenset(tool for name in ceiling for tool in resolve_toolset(name, include_registry=False))


def _inside_ceiling(name: str, ceiling_tools: frozenset[str]) -> bool:
    from toolsets import resolve_toolset

    return set(resolve_toolset(name, include_registry=False)) <= ceiling_tools


def resolve_executor_baseline(config: Mapping[str, Any], ceiling: Iterable[str]) -> ExecutorBaseline:
    """The baseline ``config`` declares, cut down to ``ceiling`` (the session's operator-allowed toolset names).
    Defaults live here, not in DEFAULT_CONFIG: the gateway reads raw YAML without merging them."""
    from hermes_cli.config_defaults_delegation import DELEGATION_DEFAULTS
    from hermes_cli.tools_config import enabled_mcp_server_names

    defaults = DELEGATION_DEFAULTS["executor"]
    delegation = config.get("delegation") if isinstance(config.get("delegation"), Mapping) else {}
    section = delegation.get("executor") if isinstance(delegation.get("executor"), Mapping) else {}
    ceiling = frozenset(ceiling)
    ceiling_tools = _ceiling_tools(ceiling)
    wanted = _clean_toolsets(section.get("toolsets", defaults["toolsets"]))
    toolsets = tuple(name for name in wanted if _inside_ceiling(name, ceiling_tools))
    servers = _clean_mcp(section.get("mcp", defaults["mcp"]), enabled_mcp_server_names(dict(config)))
    mcp_toolsets = tuple(name for name in servers if name in ceiling or f"mcp-{name}" in ceiling)
    if len(toolsets) < len(wanted):
        logger.info("executor baseline narrowed by the session's tool ceiling: dropped %s",
                    ", ".join(name for name in wanted if name not in toolsets))
    return ExecutorBaseline(toolsets=toolsets, mcp_toolsets=mcp_toolsets)


def _config_platform(agent: Any) -> str:
    """The ``platform_toolsets`` key a session's operator toolset list lives under (local surfaces share ``cli``)."""
    platform = str(getattr(agent, "platform", "") or "")
    return "cli" if platform in {"", "cli", "tui", "gui", "desktop", "local"} else platform


def session_ceiling(agent: Any, config: Mapping[str, Any]) -> frozenset[str]:
    """What the operator allows this session: a bound hosted-room policy's toolsets, else the platform's
    configured toolsets (which already carry the per-platform MCP allowlist)."""
    from gateway.hosted_room_execution_policy import current_room_execution_policy
    from hermes_cli.tools_config import _get_platform_tools

    if (policy := current_room_execution_policy()) is not None:
        return frozenset(policy.enabled_toolsets)
    return frozenset(_get_platform_tools(dict(config), _config_platform(agent)))


def freeze_executor_baseline(
    agent: Any, config: Optional[Mapping[str, Any]] = None, *, ceiling: Optional[Iterable[str]] = None,
) -> ExecutorBaseline:
    """Resolve and store the baseline on a dispatcher agent (``agent._executor_baseline``). The session builder
    passes the pre-narrowing enabled set as ``ceiling``; ``config`` and the derived ceiling are read in the
    caller's profile scope."""
    if config is None:
        from hermes_cli.config import load_config_readonly

        config = load_config_readonly()
    baseline = resolve_executor_baseline(config, session_ceiling(agent, config) if ceiling is None else ceiling)
    agent._executor_baseline = baseline
    return baseline


def executor_baseline_for(parent_agent: Any) -> Optional[ExecutorBaseline]:
    """The baseline for ``parent_agent``'s delegate_task children, or None for every other parent (ordinary
    sessions, executors themselves, test doubles): only a top-level dispatcher session relaxes "a child never
    exceeds its parent". Frozen on first use unless the session builder froze it earlier."""
    if not agent_is_dispatcher(parent_agent) or getattr(parent_agent, "_delegate_depth", 0) != 0:
        return None
    frozen = getattr(parent_agent, "_executor_baseline", None)
    return frozen if isinstance(frozen, ExecutorBaseline) else freeze_executor_baseline(parent_agent)


def baseline_child_toolsets(parent_agent: Any, baseline: ExecutorBaseline) -> tuple[list[str], list[str]]:
    """``(enabled_toolsets, disabled_toolsets)`` for an executor, in the shape ``_resolve_child_toolsets`` returns.
    Enabled is the baseline plus ``executor``, through the usual blocked-toolset strip. Disabled is the parent's
    operator list (minus platform-gated names, which the child's own platform recomputes) plus the leaf and kanban
    denies."""
    from toolsets import TOOLSET_SESSION_PLATFORMS
    from tools.delegate_tool_toolsets import _blocked_toolsets_for_role, _strip_blocked_tools

    enabled = _strip_blocked_tools([*baseline.toolsets, *baseline.mcp_toolsets, EXECUTOR_TOOLSET])
    raw = getattr(parent_agent, "disabled_toolsets", None)
    inherited = [str(name) for name in raw] if isinstance(raw, (list, tuple, set)) else []
    inherited = [name for name in inherited if name not in TOOLSET_SESSION_PLATFORMS]
    return enabled, list(dict.fromkeys(inherited + _blocked_toolsets_for_role("leaf") + ["kanban"]))
