"""The gateway's idle completion consumer is woken by a producer's put, never by a poll.

These drive the real ``_async_delegation_watcher`` on a real ``ProcessRegistry`` queue. Waits are on events the
code under test (or a stand-in adapter) sets; the only clock involved is the backoff under test.
"""

import asyncio

import pytest

from gateway import run_notifications_wake
from tests.gateway.test_completion_delivery import (
    AdmittingHandler,
    _completion_event,
    _distinct_async_event,
    _persist_pending_completion,
    _runner,
)
from tools.process_registry_awaiting import awaits_notification


@pytest.fixture
def configure(tmp_path, monkeypatch):
    """``configure(window)`` -> the isolated registry, with ``delegation.completion_batch_window_seconds``."""
    import gateway.run as gateway_run
    import tools.process_registry as pr_module
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(pr_module, "CHECKPOINT_PATH", tmp_path / "processes.json")
    registry = pr_module.ProcessRegistry()
    monkeypatch.setattr(pr_module, "process_registry", registry)

    def apply(window):
        (tmp_path / "config.yaml").write_text(f"delegation:\n  completion_batch_window_seconds: {window}\n")
        return registry
    return apply


async def _start_watcher(runner):
    """Run the watcher and return ``(task, parked)`` once it has parked for the first time."""
    parked = asyncio.Event()
    real_park = type(runner)._park_until_completion

    async def park(flag, timeout):
        parked.set()
        return await real_park(flag, timeout)
    runner._park_until_completion = park
    task = asyncio.create_task(runner._async_delegation_watcher())
    await asyncio.wait_for(parked.wait(), 10)
    return task, parked


async def _stop(task):
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_idle_watcher_is_woken_by_a_put_from_another_thread_and_never_sleeps_to_poll(configure, monkeypatch):
    registry = configure(0)
    positive_sleeps = []
    real_sleep = asyncio.sleep

    async def recording_sleep(delay, *args, **kwargs):
        if delay > 0:
            positive_sleeps.append(delay)
        await real_sleep(0)
    monkeypatch.setattr(asyncio, "sleep", recording_sleep)
    delivered = asyncio.Event()
    adapter = type("Adapter", (), {})()
    adapter.handle_message = AdmittingHandler(side_effect=lambda event: delivered.set())
    runner = _runner(adapter)
    event = _distinct_async_event("deleg_woken")
    _persist_pending_completion(event)

    task, _parked = await _start_watcher(runner)
    try:
        await asyncio.to_thread(registry.completion_queue.put, dict(event))
        await asyncio.wait_for(delivered.wait(), 10)
    finally:
        await _stop(task)

    adapter.handle_message.assert_awaited_once()
    assert positive_sleeps == [], "the watcher slept to poll instead of parking on the wake"
    assert not registry.completion_wake._subscribers, "a stopped watcher left its subscription behind"


@pytest.mark.asyncio
async def test_completions_landing_inside_the_window_become_one_turn(configure, monkeypatch):
    registry = configure(0.2)
    first, second = _distinct_async_event("deleg_win_1"), _distinct_async_event("deleg_win_2")
    for event in (first, second):
        _persist_pending_completion(event)
    windows = []
    real_sleep = asyncio.sleep

    async def sleep(delay, *args, **kwargs):
        if delay == 0.2:  # the batching window: the second completion lands while it is open
            windows.append(delay)
            registry.completion_queue.put(dict(second))
        await real_sleep(0)
    monkeypatch.setattr(asyncio, "sleep", sleep)
    delivered = asyncio.Event()
    adapter = type("Adapter", (), {})()
    adapter.handle_message = AdmittingHandler(side_effect=lambda event: delivered.set())
    runner = _runner(adapter)

    task, _parked = await _start_watcher(runner)
    try:
        registry.completion_queue.put(dict(first))
        await asyncio.wait_for(delivered.wait(), 10)
    finally:
        await _stop(task)

    assert windows == [0.2]
    adapter.handle_message.assert_awaited_once()
    text = adapter.handle_message.await_args.args[0].text
    assert "deleg_win_1" in text and "deleg_win_2" in text


@pytest.mark.asyncio
async def test_a_post_turn_drain_requeue_wakes_the_parked_watcher_at_the_retry_time(configure, monkeypatch):
    """The watcher is parked when ANOTHER coroutine (the post-turn drain) has a watch event refused and
    requeues it: nothing is put afterwards, so only the timed wake armed by the requeue can bring it back.
    The park timeout is pinned far away so the fallback timeout cannot be what retries it."""
    from gateway import run_notifications
    registry = configure(0)
    monkeypatch.setattr(run_notifications, "_WATCH_INJECTION_RETRY_BASE_DELAY", 0.05)
    monkeypatch.setattr(run_notifications_wake, "_PARK_FLOOR_S", 30.0)
    calls = []
    retried = asyncio.Event()

    def handler(event):
        calls.append(event)
        if len(calls) == 1:
            raise RuntimeError("transport offline")
        retried.set()
    adapter = type("Adapter", (), {})()
    adapter.handle_message = AdmittingHandler(side_effect=handler)
    runner = _runner(adapter)
    runner._load_background_notifications_mode = lambda: "concise"
    task, _parked = await _start_watcher(runner)
    try:
        event = dict(_completion_event(started_at=1), type="watch_match", pattern="READY", output="READY")
        # Queued without announcing it, as if the watcher's pass had already come and gone.
        import queue
        queue.Queue.put(registry.completion_queue, event)
        await runner._drain_watch_notifications(registry.completion_queue)
        assert len(calls) == 1 and registry.completion_queue.qsize() == 1, "the refused event must be requeued"
        await asyncio.wait_for(retried.wait(), 10)
    finally:
        await _stop(task)

    assert len(calls) == 2
    assert registry.completion_queue.empty()


@pytest.mark.asyncio
async def test_a_process_completion_event_leaves_the_queue_after_one_pass_and_frees_its_owner(configure):
    """A process 'completion' belongs to its per-process watcher: the idle consumer drops it (as it always
    has) instead of requeueing it, which would pin its owner as awaiting a notice for good."""
    registry = configure(0)
    adapter = type("Adapter", (), {})()
    adapter.handle_message = AdmittingHandler()
    runner = _runner(adapter)
    registry.completion_queue.put({"type": "completion", "owner_task_id": "owner-1", "session_id": "proc_x"})
    assert awaits_notification(registry, {"owner-1"})

    await runner._completion_watcher_pass(registry.completion_queue)

    assert registry.completion_queue.empty()
    assert not awaits_notification(registry, {"owner-1"})
    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_respawned_watcher_leaves_exactly_one_subscription(configure):
    registry = configure(0)
    adapter = type("Adapter", (), {})()
    adapter.handle_message = AdmittingHandler()
    runner = _runner(adapter)

    first, _ = await _start_watcher(runner)
    await _stop(first)
    assert not registry.completion_wake._subscribers

    second, _ = await _start_watcher(runner)
    try:
        assert len(registry.completion_wake._subscribers) == 1
    finally:
        await _stop(second)
    assert not registry.completion_wake._subscribers
