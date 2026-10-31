"""Busy-session policy of dispatcher sessions: the startup config snapshot and the per-session predicates.

A dispatcher session never makes a user message wait behind its running turn and never hard-interrupts it
(``agent/busy_input.py`` orders the verbs; ``run_busy_dispatcher_ladder.py`` runs them). Whether a session IS
a dispatcher is decided once at creation and frozen on its agent (``agent_is_dispatcher``); everything here
only reads it. The busy policy and the receipt mode come from config once per routed profile at gateway
startup, never per message and never from ``/busy``, so a live session cannot change them mid-conversation.

``DispatcherBusyMixin`` is inherited by ``GatewayBusySessionMixin``, which keeps ``gateway/run.py`` untouched.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Optional

from agent.dispatcher_session import DispatcherSettings, agent_is_dispatcher, dispatcher_settings

logger = logging.getLogger("gateway.run")  # log-record parity with gateway/run.py

RECEIPT_MODES = ("auto", "reaction", "text", "off")


@dataclass(frozen=True)
class DispatcherBusyCfg:
    settings: DispatcherSettings = field(default_factory=DispatcherSettings)
    receipt: str = "auto"
    receipt_by_platform: tuple[tuple[str, str], ...] = ()

    def receipt_for(self, platform_key: str) -> str:
        return dict(self.receipt_by_platform).get(platform_key, self.receipt)


def _receipt_mode(raw: Any, key: str) -> Optional[str]:
    """One ``dispatcher_receipt`` value; None when unset. YAML reads an unquoted ``off`` as False."""
    if raw is None or raw == "":
        return None
    if raw is False:
        return "off"
    if raw is True:
        return "auto"
    mode = str(raw).strip().lower()
    if mode in RECEIPT_MODES:
        return mode
    logger.warning("%s=%r is not one of %s; using auto", key, raw, "/".join(RECEIPT_MODES))
    return "auto"


def dispatcher_busy_from_config(config: Mapping[str, Any] | None) -> DispatcherBusyCfg:
    """Typed busy and receipt view of one profile's raw gateway YAML (it is not merged with DEFAULT_CONFIG)."""
    root = (config or {}).get("dispatcher")
    busy = root.get("busy") if isinstance(root, Mapping) else None
    policy = busy.get("policy") if isinstance(busy, Mapping) else None
    if policy is not None and str(policy).strip().lower() not in ("", "dispatch", "inherit"):
        logger.warning("dispatcher.busy.policy=%r is not dispatch/inherit; using dispatch", policy)
    display = (config or {}).get("display")
    display = display if isinstance(display, Mapping) else {}
    platforms = display.get("platforms")
    by_platform = []
    for name, overrides in platforms.items() if isinstance(platforms, Mapping) else ():
        if isinstance(overrides, Mapping):
            mode = _receipt_mode(overrides.get("dispatcher_receipt"), f"display.platforms.{name}.dispatcher_receipt")
            if mode is not None:
                by_platform.append((str(name), mode))
    default = _receipt_mode(display.get("dispatcher_receipt"), "display.dispatcher_receipt") or "auto"
    return DispatcherBusyCfg(dispatcher_settings(config), default, tuple(by_platform))


class DispatcherBusyMixin:
    # Class-level defaults: tests build a GatewayRunner with ``__new__`` and never run ``__init__``.
    _dispatcher_busy: DispatcherBusyCfg = DispatcherBusyCfg()
    _dispatcher_busy_by_profile: Mapping[str, DispatcherBusyCfg] = MappingProxyType({})

    def _snapshot_profile_dispatcher_busy(self, profile_name: str, config: dict) -> None:
        """Startup only (``run_adapters`` profile prepare). Deliberately NOT part of
        ``_snapshot_profile_busy_modes``, which ``/busy`` re-runs: that would let a command change the
        policy of sessions already running."""
        self.__dict__.setdefault("_dispatcher_busy_by_profile", {})[profile_name] = dispatcher_busy_from_config(config)

    def _dispatcher_busy_cfg(self, source) -> DispatcherBusyCfg:
        profile = self._busy_profile_name_for_source(source)
        by_profile = self._dispatcher_busy_by_profile
        return by_profile.get(profile, self._dispatcher_busy) if profile else self._dispatcher_busy

    def _dispatcher_session_for(self, session_key: str) -> bool:
        """Whether the chat's session was created as a dispatcher: the frozen flag of its running or cached
        agent, remembered in ``persistent.dispatcher_seen`` for the windows where no agent is visible (a
        turn still building its agent, the gap after /new). A chat that never showed an agent is not one."""
        from gateway.run import _AGENT_PENDING_SENTINEL
        state = self._peek_session_state(session_key)
        if state is None:
            return False
        agent = state.turn.agent
        if agent is None or agent is _AGENT_PENDING_SENTINEL:
            agent = self._cached_agent_for(session_key)
        if agent is not None and agent is not _AGENT_PENDING_SENTINEL:
            state.persistent.dispatcher_seen = agent_is_dispatcher(agent)
        return state.persistent.dispatcher_seen is True

    def _dispatcher_busy_enabled(self, source, session_key: str) -> bool:
        """The ladder, not ``display.busy_input_mode``, decides what a busy message of this session does."""
        return self._dispatcher_busy_cfg(source).settings.busy_policy == "dispatch" and self._dispatcher_session_for(session_key)

    def _dispatcher_text_batch_floor(self, pending) -> bool:
        """An adapter's text-batch hold is skipped for text of a busy dispatcher session."""
        key = self._session_key_for_source(pending.source)
        return self._is_session_running(key) and self._dispatcher_busy_enabled(pending.source, key)

    def _drain_lane_notes(self, session_key: str) -> list[str]:
        """The 'quick reply already sent' notes the next user turn carries (one-shot)."""
        state = self._peek_session_state(session_key)
        if state is None:
            return []
        notes, state.conversation.lane_notes = state.conversation.lane_notes, []
        return list(notes)

    def _cancel_quick_lane(self, session_key: str) -> None:
        from gateway.dispatcher_quick_lane import cancel_quick_lane
        cancel_quick_lane(self, session_key)

    def _dispatcher_stop_funnel(self, session_key: str) -> None:
        """/stop and /new: nothing of this chat's dispatcher busy path outlives them."""
        from gateway.session_receipt import close_route_receipts
        self._cancel_quick_lane(session_key)
        close_route_receipts(self, session_key)
