"""Lifecycle regression tests for stream bridge shutdown semantics."""

import pytest

from deerflow.runtime import MemoryStreamBridge, StreamGap
from deerflow.runtime.stream_bridge.redis import RedisStreamBridge


@pytest.mark.anyio
async def test_memory_bridge_close_wakes_readers_and_rejects_new_work():
    bridge = MemoryStreamBridge(queue_maxsize=4)
    run_id = "closed-memory-bridge"
    await bridge.publish(run_id, "metadata", {"run_id": run_id})
    subscriber = bridge.subscribe(run_id, heartbeat_interval=1.0)
    first = await anext(subscriber)

    await bridge.close()
    assert await anext(subscriber) == StreamGap(first.id, None, None)
    with pytest.raises(StopAsyncIteration):
        await anext(subscriber)
    await bridge.close()

    assert await bridge.stream_exists(run_id) is False
    with pytest.raises(RuntimeError, match="closed"):
        await bridge.publish(run_id, "late-event", {})
    with pytest.raises(RuntimeError, match="closed"):
        await bridge.publish_end(run_id)
    with pytest.raises(RuntimeError, match="closed"):
        await anext(bridge.subscribe("new-memory-run"))


@pytest.mark.anyio
async def test_redis_bridge_rejects_new_work_after_close_with_borrowed_client():
    # The borrowed client remains usable after bridge.close(), so the bridge
    # itself must enforce its lifecycle rather than relying on client closure.
    bridge = RedisStreamBridge(redis_url="redis://unused", client=object())
    await bridge.close()
    await bridge.close()

    with pytest.raises(RuntimeError, match="closing"):
        await bridge.publish("closed-redis-bridge", "late-event", {})
    with pytest.raises(RuntimeError, match="closing"):
        await bridge.publish_end("closed-redis-bridge")
    with pytest.raises(RuntimeError, match="closing"):
        await anext(bridge.subscribe("new-redis-run"))
