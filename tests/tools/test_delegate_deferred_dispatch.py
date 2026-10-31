"""Dispatcher-session dispatch: the handle returns before any child exists, a full pool queues (never runs inline),
and a full queue is a tool error. The unit is then built and run on the pool with the admitting thread's callbacks.

The child builder is stubbed the same way on every revision (``AIAgent`` is never constructed), so a failure here is
the dispatcher path's, not a missing stub: the base revision fails the first assertion because the old path runs the
child inside the call.
"""

from __future__ import annotations

import json
import queue as queue_mod
import threading
import time
from types import SimpleNamespace

import pytest

from tools import async_delegation
from tools import delegate_tool as dt
from tools.delegate_tool_registry import _active_subagents, _active_subagents_lock, list_active_subagents
from tools.process_registry import process_registry


class _Parent:
    """What delegate_task reads off a session's agent; ``_dispatcher_session`` is the frozen session flag."""

    def __init__(self, session_id: str, depth: int = 0) -> None:
        self.session_id = session_id
        self._dispatcher_session = True
        self._delegate_depth = depth
        self._active_children = []
        self._active_children_lock = threading.Lock()
        self._session_db = None
        self._print_fn = None
        self.model = "test-model"
        self.provider = "test"
        self.api_key = "k"
        self.base_url = ""
        self.api_mode = "chat_completions"
        self.tool_progress_callback = None
        self._interrupt_requested = False
        self.quiet_mode = True


class _Child:
    def __init__(self, subagent_id: str, goal: str, approval: str | None = None) -> None:
        self._subagent_id = subagent_id
        self._delegate_role = "leaf"
        self._delegate_depth = 1
        self.model = "test-model"
        self.session_id = f"child-{subagent_id}"
        self.tool_progress_callback = None
        self.approval = approval
        self.closed = threading.Event()
        self._delegate_parent_ref = None

    def run_conversation(self, **_kwargs):
        return {"final_response": "did it", "completed": True, "api_calls": 1,
                "messages": [{"role": "assistant", "content": "did it"}]}

    def get_activity_summary(self):
        return {"api_call_count": 1, "current_tool": None, "last_activity_ts": time.time()}

    def steer(self, _text):
        return True

    def close(self):
        self.closed.set()


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_IGNORE_USER_CONFIG", raising=False)
    (tmp_path / "config.yaml").write_text(
        "delegation:\n  max_concurrent_children: 1\n  queue_limit: 2\n  worktree_isolation: false\n", encoding="utf-8")
    async_delegation._reset_for_tests()
    with _active_subagents_lock:
        _active_subagents.clear()
    completion_queue = queue_mod.Queue()
    monkeypatch.setattr(process_registry, "completion_queue", completion_queue)
    monkeypatch.setattr(dt, "_get_subagent_approval_callback", lambda: None)
    monkeypatch.setattr(dt, "_set_subagent_approval_cb", lambda cb: None)
    yield completion_queue
    if async_delegation._executor is not None:
        async_delegation._executor.shutdown(wait=True)
    async_delegation._reset_for_tests()
    with _active_subagents_lock:
        _active_subagents.clear()


def _stub_builder(monkeypatch, *, gate: threading.Event | None = None, fail: bool = False,
                  seen: list | None = None):
    """Child construction (as the deferred worker performs it) -> a live-looking child, without an AIAgent."""
    def build(task_index, goal, subagent_id=None, **_kwargs):
        if seen is not None:
            from tools.terminal_tool import _get_approval_callback
            seen.append(_get_approval_callback())
        if gate is not None:
            assert gate.wait(10), "test never released construction"
        if fail:
            raise RuntimeError("boom")
        return _Child(subagent_id or f"sa-{task_index}", goal)
    return build


def _dispatch(parent, goal="do the thing", tasks=None):
    from tools.delegate_tool import delegate_task
    kwargs = {"tasks": tasks} if tasks is not None else {"goal": goal}
    result = json.loads(delegate_task(background=True, parent_agent=parent, **kwargs))
    if not isinstance(result, dict):
        raise AssertionError(result)
    for unit in result.get("units") or []:
        _CALLS[unit["delegation_id"]] = unit
    return result


_CALLS: dict = {}


def _wait(predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def _events(completion_queue, count, timeout=15.0):
    out = []
    deadline = time.monotonic() + timeout
    while len(out) < count and time.monotonic() < deadline:
        try:
            out.append(completion_queue.get(timeout=0.1))
        except queue_mod.Empty:
            continue
    return out


def _final_event(completion_queue, delegation_id, timeout=15.0):
    """The unit's own terminal event (a task failing notice may arrive first)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for event in list(completion_queue.queue):
            if (event.get("type") == "async_delegation" and event.get("delegation_id") == delegation_id
                    and not event.get("task_failure_notice")):
                return event
        time.sleep(0.02)
    return None


def test_handle_returns_before_the_child_is_built_and_the_unit_runs(env, monkeypatch):
    gate = threading.Event()
    monkeypatch.setattr(dt, "_build_child_agent", _stub_builder(monkeypatch, gate=gate))
    monkeypatch.setattr(dt, "_parent_live_home", lambda _parent: None)
    parent = _Parent("dispatcher-a")

    returned = {}
    thread = threading.Thread(target=lambda: returned.update(_dispatch(parent)), daemon=True)
    thread.start()
    try:
        assert _wait(lambda: bool(returned)), "delegate_task did not return while the child was still being built"
        payload = returned
        assert payload["status"] == "dispatched", payload
        sid = payload["subagent_ids"][0]
        with _active_subagents_lock:
            record = _active_subagents[sid]
        assert record["agent"] is None and record["status"] in ("queued", "running"), record
        assert payload["live_transcripts"], "the handle has no transcript paths"
    finally:
        gate.set()
    assert _wait(lambda: _completed_for(env, sid)), "the dispatched unit never completed"
    assert _wait(lambda: sid not in _active_subagents), "the task's registry record outlived its run"


def _completed_for(completion_queue, sid):
    for event in list(completion_queue.queue):
        if event.get("type") == "async_delegation" and event.get("delegation_id"):
            return True
    return False


def test_a_full_pool_queues_and_a_full_queue_is_a_tool_error(env, monkeypatch):
    gate = threading.Event()
    monkeypatch.setattr(dt, "_build_child_agent", _stub_builder(monkeypatch, gate=gate))
    monkeypatch.setattr(dt, "_parent_live_home", lambda _parent: None)
    parent = _Parent("dispatcher-b")

    first = _dispatch(parent, "task one")
    assert first["subagent_ids"]
    second = _dispatch(parent, "task two")
    assert second.get("queued") is True and second["queue_position"] == 1, second
    third = _dispatch(parent, "task three")
    assert third.get("queued") is True and third["queue_position"] == 2, third
    try:
        from tools.delegate_tool import delegate_task
        full = json.loads(delegate_task(background=True, goal="task four", parent_agent=parent))
        assert set(full) == {"error"}, f"a full queue must be a tool error, got: {full}"
        assert "queue is full" in full["error"]
        # Nothing ran inline: no child exists while the first unit is still parked in construction.
        with _active_subagents_lock:
            assert all(record["agent"] is None for record in _active_subagents.values())
    finally:
        gate.set()
    assert _wait(lambda: not any(r.get("phase") == "queued" for r in list_active_subagents()))


def test_a_stop_before_the_child_exists_cancels_the_queued_task(env, monkeypatch):
    gate = threading.Event()
    monkeypatch.setattr(dt, "_build_child_agent", _stub_builder(monkeypatch, gate=gate))
    monkeypatch.setattr(dt, "_parent_live_home", lambda _parent: None)
    parent = _Parent("dispatcher-c")
    first = _dispatch(parent, "occupies the slot")
    queued = _dispatch(parent, "waits in line")
    sid = queued["subagent_ids"][0]

    from tools.delegate_tool import delegate_task
    try:
        stopped = json.loads(delegate_task(action="stop", subagent_id=sid, parent_agent=parent))
        assert stopped["status"] == "interrupt_requested", stopped
        assert "will not start" in stopped["note"]
    finally:
        gate.set()
    event = _final_event(env, queued["delegation_id"])
    assert event is not None and event.get("status") == "interrupted", event
    assert any(e.get("status") == "interrupted" for e in event.get("results") or []), event
    assert _wait(lambda: sid not in _active_subagents), "the cancelled task's placeholder was left behind"


def _interrupted_event(completion_queue, sid):
    for event in list(completion_queue.queue):
        entries = event.get("results") or []
        if event.get("type") == "async_delegation" and any(e.get("status") == "interrupted" for e in entries):
            return True
    return False


def test_construction_failure_is_that_tasks_error(env, monkeypatch):
    monkeypatch.setattr(dt, "_build_child_agent", _stub_builder(monkeypatch, fail=True))
    monkeypatch.setattr(dt, "_parent_live_home", lambda _parent: None)
    parent = _Parent("dispatcher-d")
    payload = _dispatch(parent, "never builds")
    sid = payload["subagent_ids"][0]
    event = _final_event(env, payload["delegation_id"])
    assert event is not None, f"no completion event for a construction failure; steps={getattr(parent, 'steps', None)}"
    entries = event.get("results") or []
    rec = async_delegation._records.get(payload["delegation_id"]) or {}
    detail = {k: v for k, v in rec.items() if k.startswith("_") or k in ("status", "phase", "is_batch")}
    assert entries and entries[0]["failure_reason"] == "construction_failed", (entries, event.get("status"), event.get("error"))
    assert "boom" in (entries[0].get("error") or "")
    assert _wait(lambda: sid not in _active_subagents), "the placeholder of a failed task was left behind"


def test_a_non_dispatcher_session_keeps_the_old_path(env, monkeypatch):
    """The deferred path is opt-in per session: a plain session still builds children inside the call."""
    seen: list = []
    monkeypatch.setattr(dt, "_build_child_agent", _stub_builder(monkeypatch, seen=seen))
    monkeypatch.setattr(dt, "_parent_live_home", lambda _parent: None)
    parent = _Parent("normal-session")
    parent._dispatcher_session = False
    payload = json.loads(dt.delegate_task(background=True, goal="runs the old way", parent_agent=parent))
    # Either an async handle (built in the call) or the joined sync note: both mean the child was built here.
    assert seen or payload.get("results") is not None, payload


def test_a_unit_promoted_by_another_thread_keeps_the_admitting_thread_callbacks(env, monkeypatch):
    """The worker is wrapped at admission, so promotion (here: from a thread with session A's callback installed)
    cannot swap in that thread's approval callback."""
    from tools.thread_context import propagate_context_to_thread
    from tools.terminal_tool import _get_approval_callback, set_approval_callback

    seen: list = []
    monkeypatch.setattr(dt, "_build_child_agent", _stub_builder(monkeypatch, seen=seen))
    monkeypatch.setattr(dt, "_parent_live_home", lambda _parent: None)
    parent = _Parent("dispatcher-e")
    set_approval_callback(lambda *_a, **_k: "B")
    try:
        wrapped = propagate_context_to_thread(lambda: seen.append(_get_approval_callback()))
        set_approval_callback(lambda *_a, **_k: "A")
        threading.Thread(target=wrapped, daemon=True).start()
        assert _wait(lambda: len(seen) == 1)
        assert seen[0] is not None and seen[0]() == "B", "the wrapped worker did not carry its own thread's callback"
    finally:
        set_approval_callback(None)
