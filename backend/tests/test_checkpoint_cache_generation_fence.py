"""Read-through checkpoint cache fills must not outlive a thread purge."""

import asyncio
import threading
from typing import Annotated, TypedDict

import pytest
from langgraph.channels import DeltaChannel
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph

from deerflow.runtime.checkpoint_cache.memory import MemoryCheckpointHistoryCache
from deerflow.runtime.checkpointer.cached_saver import CachedHistorySaver


def _fold(state, writes):
    result = list(state)
    for batch in writes:
        result.extend(batch)
    return result


class State(TypedDict):
    items: Annotated[list[str], DeltaChannel(_fold, snapshot_frequency=100)]


def _build_graph(saver):
    builder = StateGraph(State)
    builder.add_node("one", lambda state: {"items": ["saved"]})
    builder.set_entry_point("one")
    builder.set_finish_point("one")
    return builder.compile(checkpointer=saver)


class _BlockingAsyncSaver(InMemorySaver):
    def __init__(self):
        super().__init__()
        self.block_next_get = False
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.captured = None

    async def aget_tuple(self, config):
        result = await super().aget_tuple(config)
        if self.block_next_get:
            self.block_next_get = False
            self.captured = result
            self.started.set()
            await self.release.wait()
        return result


class _BlockingSyncSaver(InMemorySaver):
    def __init__(self):
        super().__init__()
        self.block_next_get = False
        self.started = threading.Event()
        self.release = threading.Event()
        self.captured = None

    def get_tuple(self, config):
        result = super().get_tuple(config)
        if self.block_next_get:
            self.block_next_get = False
            self.captured = result
            self.started.set()
            if not self.release.wait(timeout=5):
                raise TimeoutError("test reader was not released")
        return result


@pytest.mark.asyncio
async def test_async_read_through_fill_started_before_delete_cannot_repopulate():
    inner = _BlockingAsyncSaver()
    cache = MemoryCheckpointHistoryCache()
    saver = CachedHistorySaver(inner, cache, key_prefix="delete-generation-async")
    graph = _build_graph(saver)
    config = {"configurable": {"thread_id": "thread-async"}}
    try:
        await graph.ainvoke({"items": ["input"]}, config)
        inner.block_next_get = True
        read = asyncio.create_task(
            saver.aget_delta_channel_history(config=config, channels=["items"])
        )
        await asyncio.wait_for(inner.started.wait(), timeout=1)
        target = inner.captured
        assert target is not None
        key = saver._key(target, "items")

        await saver.adelete_thread("thread-async")
        inner.release.set()
        await asyncio.wait_for(read, timeout=2)

        assert cache.get_many([key]) == {}
    finally:
        inner.release.set()
        await cache.aclose()


@pytest.mark.asyncio
async def test_sync_read_through_fill_started_before_delete_cannot_repopulate():
    inner = _BlockingSyncSaver()
    cache = MemoryCheckpointHistoryCache()
    saver = CachedHistorySaver(inner, cache, key_prefix="delete-generation-sync")
    graph = _build_graph(saver)
    config = {"configurable": {"thread_id": "thread-sync"}}
    try:
        graph.invoke({"items": ["input"]}, config)
        inner.block_next_get = True
        read = asyncio.create_task(
            asyncio.to_thread(saver.get_delta_channel_history, config=config, channels=["items"])
        )
        assert await asyncio.to_thread(inner.started.wait, 1)
        target = inner.captured
        assert target is not None
        key = saver._key(target, "items")

        saver.delete_thread("thread-sync")
        inner.release.set()
        await asyncio.wait_for(read, timeout=2)

        assert cache.get_many([key]) == {}
    finally:
        inner.release.set()
        await cache.aclose()


class _UnavailableGenerationCache(MemoryCheckpointHistoryCache):
    async def aget_generation(self, key_prefix):
        return None

    async def aget_many(self, keys):
        raise AssertionError("cache reads must be bypassed without a generation")


@pytest.mark.asyncio
async def test_async_reader_bypasses_cache_when_generation_is_unavailable():
    inner = InMemorySaver()
    graph = _build_graph(inner)
    config = {"configurable": {"thread_id": "generation-unavailable"}}
    await graph.ainvoke({"items": ["input"]}, config)

    cache = _UnavailableGenerationCache()
    saver = CachedHistorySaver(inner, cache, key_prefix="generation-unavailable")
    try:
        result = await saver.aget_delta_channel_history(config=config, channels=["items"])
        assert result["items"]["writes"]
        assert cache.stats().entries == 0
    finally:
        await cache.aclose()
