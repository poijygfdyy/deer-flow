"""Optional Redis checkpoint-cache writes must tolerate serializer failures."""

from __future__ import annotations

import pytest

from deerflow.runtime.checkpoint_cache.redis import RedisCheckpointHistoryCache


class _Pipeline:
    def __init__(self, client: "_Client") -> None:
        self.client = client
        self.writes: list[tuple[str, bytes, int | None]] = []

    def set(self, key: str, value: bytes, ex: int | None = None) -> "_Pipeline":
        self.writes.append((key, value, ex))
        return self

    async def execute(self) -> None:
        for key, value, ttl in self.writes:
            self.client.store[key] = value
            self.client.ttls[key] = ttl


class _Client:
    def __init__(self) -> None:
        self.store: dict[str, bytes] = {}
        self.ttls: dict[str, int | None] = {}

    def pipeline(self, transaction: bool = False) -> _Pipeline:
        return _Pipeline(self)


class _SelectiveFailingSerde:
    def dumps_typed(self, entry: dict) -> tuple[str, bytes]:
        if entry["writes"] == ["unserializable"]:
            raise TypeError("test serializer cannot encode this entry")
        return "json", b"ok"


@pytest.mark.anyio
async def test_serialization_failure_skips_bad_entry_but_keeps_other_writes(
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = _Client()
    cache = RedisCheckpointHistoryCache.__new__(RedisCheckpointHistoryCache)
    cache._client = client
    cache._serde = _SelectiveFailingSerde()
    cache._ttl = 60

    with caplog.at_level("WARNING"):
        await cache.aset_many(
            {
                "good-before": {"writes": ["ok"]},
                "bad": {"writes": ["unserializable"]},
                "good-after": {"writes": ["ok"]},
            }
        )

    expected_value = b"json" + bytes([0]) + b"ok"
    assert client.store == {"good-before": expected_value, "good-after": expected_value}
    assert client.ttls == {"good-before": 60, "good-after": 60}
    assert "serialization failed; skipping entry" in caplog.text
    assert "TypeError" in caplog.text
