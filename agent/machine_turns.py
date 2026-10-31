"""Which persisted user-row kinds mean "the text of this turn was written by Hermes, not by the user".

A completion notice, a process notice, a heartbeat tick or a hidden widget intent is machinery: the agent
still answers it, but semantic recall keyed on that text only injects noise and a memory backend would be
queried for nothing. The kinds are the ones ``gateway/response_filters.py`` names as machinery
(``MACHINERY_DISPLAY_KINDS``) plus ``hidden`` (``gateway/session_display.py``), which is what a completion
whose presentation is suppressed is persisted as. The agent layer reads this module so it never imports
gateway constants; ``tests/agent/test_machine_turns.py`` pins both sets together.

Anything that needs "is this turn machine text" imports :func:`is_machine_text_display_kind`; do not
compare against one literal such as ``"hidden"`` (a real wake turn carries ``internal_notification``,
``process_complete`` or ``async_delegation_complete``).
"""

from __future__ import annotations

from typing import Any

MACHINE_TEXT_DISPLAY_KINDS = frozenset({
    "internal_notification",
    "process_complete",
    "async_delegation_complete",
    "hidden",
})


def is_machine_text_display_kind(display_kind: Any) -> bool:
    """True when ``display_kind`` (the persisted kind of the current user row) marks machine-written text."""
    return display_kind in MACHINE_TEXT_DISPLAY_KINDS
