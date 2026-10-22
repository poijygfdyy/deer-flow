"""Regression tests for explicit empty event-type filters across event-store backends."""

import pytest

from deerflow.persistence.engine import close_engine, get_session_factory, init_engine
from deerflow.runtime.events.store.db import DbRunEventStore
from deerflow.runtime.events.store.jsonl import JsonlRunEventStore
from deerflow.runtime.events.store.memory import MemoryRunEventStore


@pytest.mark.anyio
@pytest.mark.parametrize("backend", ["memory", "jsonl", "db"])
async def test_empty_event_types_is_an_empty_filter_across_backends(tmp_path, backend):
    if backend == "db":
        await init_engine(
            "sqlite",
            url=f"sqlite+aiosqlite:///{tmp_path / 'test.db'}",
            sqlite_dir=str(tmp_path),
        )
        event_store = DbRunEventStore(get_session_factory())
    elif backend == "jsonl":
        event_store = JsonlRunEventStore(base_dir=tmp_path / "jsonl")
    else:
        event_store = MemoryRunEventStore()

    try:
        await event_store.put(
            thread_id="t1",
            run_id="r1",
            event_type="llm_end",
            category="trace",
            user_id=None,
        )
        assert len(
            await event_store.list_events(
                "t1", "r1", event_types=None, user_id=None
            )
        ) == 1
        assert await event_store.list_events(
            "t1", "r1", event_types=[], limit=1, user_id=None
        ) == []
    finally:
        if backend == "db":
            await close_engine()
