"""First-token path trims for dispatcher sessions.

A dispatcher session must answer at once, so the synchronous work that runs before the first token is
bounded here. Everything is driven by one frozen ``DispatcherLatency`` that ``agent_init`` attaches as
``agent._dispatcher_latency`` while the agent is built (the same moment the dispatcher flag is frozen, so
a config edit applies to the next agent, not to a cached one). A non-dispatcher session, a delegated
child and every test double have ``None``: each helper then answers "legacy behaviour" and the paths are
unchanged. Nothing here touches the system prompt or the tool schemas.

What is trimmed, and what is deliberately not:

* Compaction (``should_defer_compaction``). Only the summary LLM wait is removed. When a background
  prepared pass (``agent/prepared_compaction.py``) is still writing the summary and the request is under
  the deferral ceiling, the automatic decision sites skip compaction for now. When a prepared summary is
  ready, compaction runs as always and splices it; the lease, pruning, window planning and the rewrite
  stay on the synchronous path (they are not free, and the splice is still a prompt-cache break). With no
  pass at all, or past the ceiling, compaction is the legacy synchronous one. The automatic sites that
  consult it: turn-start preflight, pre-API and post-tool. Exempt on purpose: idle compaction
  (``compression.idle_compact_after_seconds``, opt-in, fires below the trigger so there is nothing to
  wait for), provider-overflow recovery (the request is proven not to fit), manual ``/compress`` and
  engine-driven maintenance. Plugin context engines never get a ``PreparedCompaction``, so deferral is
  a no-op for them. A memory provider that overrides ``on_pre_compress`` supplies summary context, so
  every prepared summary would be discarded: deferral is off for such sessions and no pass is started.
* Recall (``skip_recall_for_turn``). The text of a completion or process notice is machine text; recall
  keyed on it is noise, so both the turn's prefetch and the next turn's queued prefetch are skipped.
  The provider's per-turn tick still runs.
* The bounded memory tick and the ``pre_llm_call`` budget live with the code they bound
  (``MemoryManager.begin_turn_bounded``, ``hermes_cli.plugins_dispatch.hook_budget``) and read their
  numbers from here.

Not trimmed (read and found cheap or off the path): the session lease and the SQLite writes, voice route,
titling, fast mode, and the skills-index comparison (about a millisecond for tens of skills on a warm
index, so a throttle would only trade a measurable cost for stale-index risk).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DispatcherLatency:
    """The ``dispatcher.latency.*`` numbers, frozen when the agent was built."""

    memory_budget_s: float
    pre_llm_call_budget_s: float
    defer_compaction: bool
    defer_ceiling_ratio: float
    stream_deltas: bool


def freeze_dispatcher_latency(agent: Any, config: Optional[Mapping[str, Any]]) -> None:
    """Set ``agent._dispatcher_latency``: the frozen settings for a dispatcher session, else ``None``.

    Call after the dispatcher flag is frozen and the config is loaded, before the memory manager and the
    context engine are built (both read it)."""
    from agent.dispatcher_session import agent_is_dispatcher, dispatcher_settings

    if not agent_is_dispatcher(agent):
        agent._dispatcher_latency = None
        return
    settings = dispatcher_settings(config)
    agent._dispatcher_latency = DispatcherLatency(
        memory_budget_s=settings.memory_budget_s,
        pre_llm_call_budget_s=settings.pre_llm_call_budget_s,
        defer_compaction=settings.defer_compaction,
        defer_ceiling_ratio=min(1.0, settings.defer_ceiling_ratio),
        stream_deltas=settings.stream_deltas,
    )


def dispatcher_latency(agent: Any) -> Optional[DispatcherLatency]:
    """The frozen settings, or ``None`` (not a dispatcher session, or a test double)."""
    value = getattr(agent, "_dispatcher_latency", None)
    return value if isinstance(value, DispatcherLatency) else None


def memory_budget_s(agent: Any) -> float:
    """Seconds the memory tick plus prefetch may take before the first token; 0 means the legacy path."""
    latency = dispatcher_latency(agent)
    return latency.memory_budget_s if latency is not None else 0.0


def pre_llm_call_budget_s(agent: Any) -> float:
    """Total seconds for the ``pre_llm_call`` hook (first-party observer plus plugins); 0 means no budget."""
    latency = dispatcher_latency(agent)
    return latency.pre_llm_call_budget_s if latency is not None else 0.0


def dispatcher_streams_deltas(agent: Any) -> bool:
    """Whether this session streams text deltas on platforms that explicitly opt in to streaming."""
    latency = dispatcher_latency(agent)
    return bool(latency is not None and latency.stream_deltas)


def skip_recall_for_turn(agent: Any) -> bool:
    """True when this turn's text is machine-written (a completion or process notice, a hidden intent) and
    the session is on the bounded memory path. ``agent._turn_display_kind`` is the persisted kind of the
    turn, stamped by ``build_turn_context``."""
    if memory_budget_s(agent) <= 0:
        return False
    from agent.machine_turns import is_machine_text_display_kind

    return is_machine_text_display_kind(getattr(agent, "_turn_display_kind", None))


def memory_supplies_compaction_context(agent: Any) -> bool:
    """True when an external memory provider overrides ``on_pre_compress``: it feeds the summary, so a
    summary prepared in the background would be thrown away."""
    check = getattr(getattr(agent, "_memory_manager", None), "overrides_pre_compress", None)
    return callable(check) and check() is True


def wants_prepared_compaction(agent: Any, prepare_ahead: bool) -> bool:
    """Whether ``agent_init`` should give the compressor a ``PreparedCompaction``: the user's
    ``compression.prepare_ahead``, or a dispatcher session that defers compaction (and has no provider
    that would make every prepared summary useless)."""
    if prepare_ahead:
        return True
    latency = dispatcher_latency(agent)
    return bool(
        latency is not None and latency.defer_compaction and not memory_supplies_compaction_context(agent)
    )


def should_defer_compaction(agent: Any, request_tokens: int) -> bool:
    """True when an automatic compaction decision should wait for the summary a background pass is writing.

    All of: a dispatcher session with ``defer_compaction``; a compressor with prepared-compaction state
    whose pass is running (a ready summary splices at once, no pass means the legacy path); no memory
    provider supplying summary context; and a request below
    ``max(threshold_tokens, defer_ceiling_ratio * context_length)``. A pass that fails or never finishes
    cannot defer past the ceiling, and the synchronous compaction that follows waits for a running pass
    that fits instead of writing the summary twice. Cheap: attribute reads under one short lock, no
    transcript hashing; a prepared summary that no longer matches is found by ``compress()`` itself and
    costs what it cost before."""
    latency = dispatcher_latency(agent)
    if latency is None or not latency.defer_compaction:
        return False
    compressor = getattr(agent, "context_compressor", None)
    prepared = getattr(compressor, "prepared_compaction", None)
    if prepared is None or prepared.state() != "running" or memory_supplies_compaction_context(agent):
        return False
    threshold = int(getattr(compressor, "threshold_tokens", 0) or 0)
    ceiling = max(threshold, int(latency.defer_ceiling_ratio * int(getattr(compressor, "context_length", 0) or 0)))
    if request_tokens >= ceiling:
        return False
    logger.info(
        "Deferring compaction: ~%s tokens is over the %s trigger but under the %s ceiling, and the summary "
        "is being written in the background (session %s)",
        f"{request_tokens:,}", f"{threshold:,}", f"{ceiling:,}", getattr(agent, "session_id", None) or "none",
    )
    return True
