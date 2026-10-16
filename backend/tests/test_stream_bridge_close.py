from __future__ import annotations

import asyncio

import pytest

from deerflow.runtime import END_SENTINEL, MemoryStreamBridge


@pytest.mark.anyio
async def test_close_wakes_waiting_subscriber_and_rejects_new_work():
    """Closing the bridge must not strand consumers on a cleared stream."""
    bridge = MemoryStreamBridge(queue_maxsize=2, heartbeat_interval=30.0)
    run_id = "run-close-wakes-subscriber"
    stream = bridge._get_or_create_stream(run_id)

    class ObservedCondition(asyncio.Condition):
        def __init__(self):
            super().__init__()
            self.wait_started = asyncio.Event()

        async def wait(self):
            self.wait_started.set()
            return await super().wait()

    condition = ObservedCondition()
    stream.condition = condition
    await bridge.publish(run_id, "before-close", {"n": 1})
    subscriber = bridge.subscribe(run_id, heartbeat_interval=30.0)
    pending = None
    try:
        first = await asyncio.wait_for(anext(subscriber), timeout=1.0)
        assert first.event == "before-close"

        pending = asyncio.create_task(anext(subscriber))
        await asyncio.wait_for(condition.wait_started.wait(), timeout=1.0)
        await bridge.close()

        assert await asyncio.wait_for(pending, timeout=1.0) is END_SENTINEL
        with pytest.raises(StopAsyncIteration):
            await anext(subscriber)
        assert await bridge.stream_exists(run_id) is False
        assert await asyncio.wait_for(anext(bridge.subscribe("after-close")), timeout=1.0) is END_SENTINEL
        with pytest.raises(RuntimeError, match="stream bridge is closed"):
            await bridge.publish("after-close", "event", {})
    finally:
        if pending is not None and not pending.done():
            pending.cancel()
        if pending is not None:
            await asyncio.gather(pending, return_exceptions=True)
        await subscriber.aclose()
