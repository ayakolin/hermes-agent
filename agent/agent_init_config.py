"""Small readers for the ``config.yaml`` sections ``init_agent`` consumes (kept out of the facade so it
does not grow; see ``agent_init.py``)."""

from __future__ import annotations

from typing import Any


def _cfg_flag(cfg: dict[str, Any], key: str, default: bool) -> bool:
    """Legacy string-set truthiness used by the ``compression`` section."""
    return str(cfg.get(key, default)).lower() in {"true", "1", "yes"}


def _cfg_dict(cfg: dict[str, Any], key: str) -> dict[str, Any]:
    """``cfg[key]`` if it is a mapping, else ``{}`` (malformed sections are ignored)."""
    section = cfg.get(key, {})
    return section if isinstance(section, dict) else {}
