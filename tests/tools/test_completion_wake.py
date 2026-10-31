"""A producer's put wakes the completion consumer; a consumer's own retry does not.

The registry's ``completion_wake`` is what an idle consumer parks on. Every producer calls plain
``completion_queue.put``, so the wake has to come from the queue, and a consumer that hands a refused
event back must be able to do so without waking itself (it would spin on its own backoff items).
"""

import asyncio
import threading

import pytest

from tools.process_registry import ProcessRegistry
from tools.process_registry_awaiting import awaits_notification, release, take_ready
from tools.process_registry_wake import requeue


@pytest.fixture
def registry():
    return ProcessRegistry()


async def _subscribed(registry):
    flag = asyncio.Event()
    return flag, registry.completion_wake.subscribe(flag)


@pytest.mark.asyncio
async def test_a_put_from_another_thread_wakes_the_subscriber(registry):
    flag, unsubscribe = await _subscribed(registry)
    try:
        producer = threading.Thread(target=registry.completion_queue.put, args=({"type": "completion"},))
        producer.start()
        await asyncio.wait_for(flag.wait(), 5)
        producer.join(5)
    finally:
        unsubscribe()


@pytest.mark.asyncio
async def test_requeue_does_not_wake_but_a_timed_requeue_does(registry, monkeypatch):
    import tools.process_registry as module
    monkeypatch.setattr(module, "process_registry", registry)
    flag, unsubscribe = await _subscribed(registry)
    try:
        requeue(registry.completion_queue, {"type": "watch_match"})
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert not flag.is_set(), "a consumer's own requeue woke it"
        assert registry.completion_queue.qsize() == 1

        requeue(registry.completion_queue, {"type": "watch_match"}, wake_in=0.05)
        await asyncio.sleep(0)
        assert not flag.is_set(), "a timed wake fired before its time"
        await asyncio.wait_for(flag.wait(), 5)
    finally:
        unsubscribe()


@pytest.mark.asyncio
async def test_a_swapped_in_plain_queue_is_still_reachable_through_the_registry_wake(registry, monkeypatch):
    """A consumer subscribes through the registry, not the queue object, so replacing the queue keeps its
    timed wakes working (the watcher's timeout covers the puts a plain queue never announces)."""
    import queue
    import tools.process_registry as module
    monkeypatch.setattr(module, "process_registry", registry)
    registry.completion_queue = queue.Queue()
    flag, unsubscribe = await _subscribed(registry)
    try:
        requeue(registry.completion_queue, {"type": "watch_match"}, wake_in=0)
        await asyncio.wait_for(flag.wait(), 5)
    finally:
        unsubscribe()


def test_a_subscriber_on_a_closed_loop_is_dropped_and_never_breaks_the_producer(registry):
    loop = asyncio.new_event_loop()
    flag = asyncio.Event()
    holder = {}

    async def subscribe():
        holder["unsubscribe"] = registry.completion_wake.subscribe(flag)
    loop.run_until_complete(subscribe())
    loop.close()

    registry.completion_queue.put({"type": "completion"})  # must not raise into the producer
    assert registry.completion_queue.qsize() == 1
    assert not registry.completion_wake._subscribers
    holder["unsubscribe"]()  # idempotent after the drop


def test_an_event_a_consumer_took_keeps_its_owner_awaiting_until_released(registry):
    """The single-consumer convention is unchanged by the wake: dequeue and in-flight hold are one step."""
    registry.completion_queue.put({"type": "async_delegation", "owner_task_id": "owner-1"})
    assert awaits_notification(registry, {"owner-1"})
    ready = take_ready(registry.completion_queue, 0)
    assert len(ready) == 1 and registry.completion_queue.empty()
    assert awaits_notification(registry, {"owner-1"}), "between dequeue and routing the owner must stay held"
    release(ready)
    assert not awaits_notification(registry, {"owner-1"})


def test_take_ready_with_accept_leaves_the_other_events_queued_in_order(registry):
    kept_a = {"type": "async_delegation", "owner_task_id": "a"}
    kept_b = {"type": "async_delegation", "owner_task_id": "b"}
    registry.completion_queue.put(kept_a)
    registry.completion_queue.put({"type": "watch_match", "owner_task_id": "c"})
    registry.completion_queue.put(kept_b)

    ready = take_ready(registry.completion_queue, 0, accept=lambda evt: evt["type"] != "async_delegation")

    assert [evt["owner_task_id"] for evt in ready] == ["c"]
    assert list(registry.completion_queue.queue) == [kept_a, kept_b]
    assert not awaits_notification(registry, {"x"})
    assert awaits_notification(registry, {"a"}), "a queued event still holds its owner"
    release(ready)
