"""Streaming on a dispatcher session (gateway/config.py ``StreamingConfig.enabled_for`` and the stream
consumer setup in gateway/run_turn_runner.py).

A dispatcher session wants the first words on screen while the turn is still running. With
``dispatcher.latency.stream_deltas`` on, an EXPLICIT per-platform ``streaming: true`` streams even though the
global ``streaming.enabled`` switch is off. Everything that was off stays off: a platform that follows the
global, an explicit per-platform ``false``, ``transport: off``, ``stream_deltas: false``, and every
non-dispatcher session (the #53697 contract). The wiring cases run the real ``TurnRunner`` against a real
adapter and read what the agent was given, so a call site that forgot the dispatcher flag fails here.

Not covered, on purpose: the classic CLI. ``display.streaming`` is merged with its default before the CLI
reads it, so an explicit ``false`` cannot be told from unset, and the CLI builds its agent before the
session mode is known; TUI and Desktop stream deltas unconditionally.
"""
from __future__ import annotations

import pytest

from agent.dispatcher_latency import DispatcherLatency
from gateway.config import Platform, StreamingConfig
from tests.gateway.test_run_progress_topics import _run_with_agent


@pytest.mark.parametrize(("enabled", "transport", "override", "dispatcher", "expected"), [
    # the old contract, untouched without the flag
    (False, "auto", True, False, False),
    (False, "auto", None, False, False),
    (True, "auto", None, False, True),
    (True, "auto", False, False, False),
    (True, "off", True, False, False),
    # a dispatcher session: only an explicit per-platform True gets through a global off
    (False, "auto", True, True, True),
    (False, "edit", True, True, True),
    (False, "auto", None, True, False),    # follows the global
    (False, "auto", False, True, False),   # the platform's own opt-out
    (False, "off", True, True, False),     # transport off is final
    # a global on is unchanged by the flag
    (True, "auto", None, True, True),
    (True, "auto", False, True, False),
])
def test_enabled_for_table(enabled, transport, override, dispatcher, expected):
    config = StreamingConfig(enabled=enabled, transport=transport)
    assert config.enabled_for(override, dispatcher_stream=dispatcher) is expected


def test_enabled_for_without_the_keyword_is_the_old_call():
    assert StreamingConfig(enabled=False).enabled_for(True) is False
    assert StreamingConfig(enabled=True).enabled_for(None) is True


class _RecordingAgent:
    """What the gateway handed the agent: whether a delta sink was wired, and what it did with it."""

    _dispatcher_latency = None
    seen: list = []

    def __init__(self, **kwargs):
        self.stream_delta_callback = kwargs.get("stream_delta_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None, **kwargs):
        type(self).seen.append(self.stream_delta_callback is not None)
        if self.stream_delta_callback:
            self.stream_delta_callback("hello there")
        return {"final_response": "hello there", "response_previewed": bool(self.stream_delta_callback),
                "messages": [], "api_calls": 1}


def _agent_cls(latency):
    return type("Agent", (_RecordingAgent,), {"_dispatcher_latency": latency, "seen": []})


def _latency(stream_deltas=True):
    return DispatcherLatency(2.0, 3.0, True, 0.85, stream_deltas)


def _config(*, platform_streaming, transport="auto"):
    return {
        "display": {"tool_progress": "off", "interim_assistant_messages": False,
                    "platforms": {"telegram": {"streaming": platform_streaming}}},
        "streaming": {"enabled": False, "transport": transport, "edit_interval": 0.01, "buffer_threshold": 1},
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(("latency", "platform_streaming", "transport", "streams"), [
    (_latency(), True, "auto", True),             # the new behaviour
    (None, True, "auto", False),                  # a non-dispatcher session: #53697 still holds
    (_latency(stream_deltas=False), True, "auto", False),   # the operator's off switch
    (_latency(), False, "auto", False),           # the platform's own opt-out
    (_latency(), True, "off", False),             # transport off is final
])
async def test_turn_runner_wires_deltas_only_for_a_dispatcher_with_an_explicit_platform_opt_in(
    monkeypatch, tmp_path, latency, platform_streaming, transport, streams,
):
    agent_cls = _agent_cls(latency)
    await _run_with_agent(
        monkeypatch, tmp_path, agent_cls, session_id=f"sess-{streams}-{transport}-{platform_streaming}",
        config_data=_config(platform_streaming=platform_streaming, transport=transport),
    )
    assert agent_cls.seen == [streams]


@pytest.mark.asyncio
async def test_a_dispatcher_platform_that_follows_the_global_does_not_stream(monkeypatch, tmp_path):
    config = _config(platform_streaming=None)
    del config["display"]["platforms"]["telegram"]["streaming"]
    agent_cls = _agent_cls(_latency())
    await _run_with_agent(monkeypatch, tmp_path, agent_cls, session_id="sess-follows-global", config_data=config)
    assert agent_cls.seen == [False]
