"""Regression coverage for best-effort Redis checkpoint-cache serialization."""

from typing import Any

import pytest

from deerflow.runtime.checkpoint_cache import redis as redis_cache


class _Pipeline:
    def __init__(self, client: Any) -> None:
        self.client = client
        self.pending: dict[str, bytes] = {}

    def set(self, key: str, value: bytes, ex: int | None = None) -> "_Pipeline":
        self.pending[key] = value
        return self

    async def execute(self) -> None:
        self.client.store.update(self.pending)


class _Redis:
    def __init__(self) -> None:
        self.store: dict[str, bytes] = {}

    def pipeline(self, transaction: bool = False) -> _Pipeline:
        return _Pipeline(self)

    async def aclose(self) -> None:
        pass


class _SelectiveSerde:
    def dumps_typed(self, entry: dict[str, Any]) -> tuple[str, bytes]:
        if entry.get("fail_serialization"):
            raise TypeError("synthetic serializer failure")
        return "json", b"encoded"

    def loads_typed(self, value: tuple[str, bytes]) -> Any:
        raise AssertionError("this regression only exercises writes")


@pytest.mark.anyio
async def test_unserializable_entry_does_not_abort_other_cache_writes(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    client = _Redis()
    monkeypatch.setattr(redis_cache, "_create_client", lambda *args, **kwargs: client)
    cache = redis_cache.RedisCheckpointHistoryCache(
        "redis://unused", serde=_SelectiveSerde(), ttl_seconds=60
    )

    with caplog.at_level("WARNING"):
        await cache.aset_many(
            {
                "bad": {"fail_serialization": True},
                "good": {"writes": []},
            }
        )

    assert set(client.store) == {"good"}
    assert "encode failed" in caplog.text
    assert "TypeError" in caplog.text
