"""The memory part of a dispatcher turn's first-token path: ``on_turn_start`` plus recall in one budget.

``MemoryManager.begin_turn_bounded`` is the entry. The budget is ``dispatcher.latency.memory_budget_s``
and covers the providers' tick and the recall together; the caller stops waiting when it is spent.

* A provider that does not override ``on_turn_start`` (the base class is a no-op, and no in-tree provider
  overrides it) costs nothing: no thread. The bounding therefore protects third-party providers whose
  tick can block; it exists because the legacy tick has no timeout at all.
* A provider that does override it ticks on a context thread (profile and secret scope ride along). At
  most ONE tick per provider is in flight: while the previous turn's tick is still running the new tick is
  skipped (a provider that is wedged or slow for one turn is not handed overlapping calls), and the turn
  also skips recall, since a provider that cannot finish a tick will not finish a query either.
  A tick the caller stopped waiting for finishes on its own; nothing is cancelled.
* Recall runs through ``prefetch_all`` unchanged, including its own ``_PREFETCH_GATE`` call with the real
  provider timeout; only the caller's join is shortened (``deadline``). The gate is not driven with the
  shortened budget, so a slow-but-healthy provider is not marked stuck by rapid messages. A late result is
  dropped: ``build_turn_context`` stamps recall into the user message before the first request, so it
  cannot be appended afterwards, and carrying it to the next turn would answer a different question.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from agent.memory_provider import MemoryProvider, spawn_context_thread

logger = logging.getLogger(__name__)


def overrides_hook(provider: Any, hook: str) -> bool:
    """True when ``provider`` implements ``hook`` itself rather than inheriting ``MemoryProvider``'s default.

    Duck-typed providers (not subclasses) count as overriding when they define it at all; a provider
    without the attribute does not."""
    method = getattr(provider, hook, None)
    if method is None:
        return False
    return getattr(method, "__func__", method) is not getattr(MemoryProvider, hook, None)


def _tick_bounded(manager: Any, turn_number: int, message: str, kwargs: dict[str, Any], deadline: float) -> bool:
    """Tick every overriding provider; True when none is still running at ``deadline``."""
    from agent.memory_manager import _redact_for_provider

    if not manager._owner_allows("on_turn_start"):
        return True  # nothing may reach the providers; prefetch_all makes its own scope decision
    message = _redact_for_provider(message)
    started: list[Any] = []
    for provider in manager._providers:
        if not overrides_hook(provider, "on_turn_start"):
            continue
        previous = manager._tick_threads.get(provider.name)
        if previous is not None and previous.is_alive():
            logger.debug("Memory provider '%s' on_turn_start skipped: the previous tick is still running",
                         provider.name)
            started.append(previous)
            continue
        thread = spawn_context_thread(
            manager._each_provider, name=f"memory-turn-start-{provider.name}",
            args=("on_turn_start failed", lambda p: manager._tick_provider(p, turn_number, message, kwargs)),
            kwargs={"providers": [provider]},
        )
        manager._tick_threads[provider.name] = thread
        thread.start()
        started.append(thread)
    for thread in started:
        thread.join(max(0.0, deadline - time.monotonic()))
    return not any(thread.is_alive() for thread in started)


def begin_turn_bounded(
    manager: Any, turn_number: int, message: str, *, budget_s: float, recall: bool, session_id: str = "",
    **kwargs: Any,
) -> str:
    """Tick the providers and, when ``recall`` and the tick settled in time, return their recall text."""
    deadline = time.monotonic() + max(0.0, budget_s)
    settled = _tick_bounded(manager, turn_number, message, kwargs, deadline)
    if not recall:
        return ""
    if not settled:
        logger.info("Memory recall skipped for this turn: a provider's turn-start did not settle within %.1fs",
                    budget_s)
        return ""
    return manager.prefetch_all(message, session_id=session_id, deadline=deadline)
