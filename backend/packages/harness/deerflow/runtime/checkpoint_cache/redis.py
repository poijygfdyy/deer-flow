"""Shared Redis backend for immutable checkpoint-history entries.

Read-through fills capture a namespace epoch before source reads and publish
through an atomic compare-and-write. Thread deletion advances that epoch before
SCANNING/UNLINKing the thread's keys, so a fill that started before deletion
cannot repopulate the cache afterward. The single epoch key is namespace-scoped
rather than thread-scoped, avoiding unbounded per-thread tombstones. A missing
epoch is initialized with a unique token so Redis key eviction cannot reset an
old writer's generation to a reusable value.

The redis import is lazy (module is importable without the optional
``redis`` extra), mirroring runtime/stream_bridge/redis.py.
"""

from __future__ import annotations

import logging
import secrets
from contextlib import suppress
from typing import Any

from deerflow.runtime.checkpoint_cache.base import CheckpointCacheStats, thread_key_stem, valid_history_entry

logger = logging.getLogger(__name__)

REDIS_INSTALL = "redis is required for the redis checkpoint cache backend. Install it with: uv sync --extra redis"

_TAG_SEPARATOR = b"\x00"

_SET_IF_GENERATION_SCRIPT = """
local current = redis.call('GET', KEYS[1])
if not current then current = '' end
if current ~= ARGV[1] then return 0 end
local ttl = tonumber(ARGV[2]) or -1
for i = 3, #ARGV, 2 do
    if ttl > 0 then
        redis.call('SET', ARGV[i], ARGV[i + 1], 'EX', ttl)
    else
        redis.call('SET', ARGV[i], ARGV[i + 1])
    end
end
return 1
"""

_ADVANCE_GENERATION_SCRIPT = "return redis.call('SET', KEYS[1], ARGV[1])"


def _warn(message: str, *args: Any) -> None:
    """Keep optional-cache diagnostics from changing fallback behavior."""
    with suppress(Exception):
        logger.warning(message, *args)


def _create_client(redis_url: str, *, max_connections: int | None) -> Any:
    try:
        import redis.asyncio as redis_async
    except ImportError as exc:
        raise ImportError(REDIS_INSTALL) from exc
    kwargs: dict[str, Any] = {"decode_responses": False}
    if max_connections is not None:
        kwargs["max_connections"] = max_connections
    return redis_async.from_url(redis_url, **kwargs)


def _redis_error() -> type[Exception]:
    """Lazy RedisError import, mirroring the lazy client creation above."""
    try:
        from redis.exceptions import RedisError
    except ImportError as exc:
        raise ImportError(REDIS_INSTALL) from exc
    return RedisError


class RedisCheckpointHistoryCache:
    def __init__(
        self,
        redis_url: str,
        *,
        serde: Any,
        ttl_seconds: int,
        max_connections: int | None = None,
    ) -> None:
        self._client = _create_client(redis_url, max_connections=max_connections)
        self._serde = serde
        # ttl_seconds=0 is an explicit opt-out of expiry (no SETEX) — not the
        # default, and leaked/orphaned keys then rely on redis maxmemory only.
        self._ttl = ttl_seconds if ttl_seconds > 0 else None
        self._hits = 0
        self._misses = 0
        self._write_serialization_failures = 0

    async def aget_many(self, keys: list[str]) -> dict[str, dict[str, Any]]:
        if not keys:
            return {}
        try:
            raws = await self._client.mget(keys)
        except _redis_error() as exc:
            # Performance-only bypass: a redis outage costs hits, never availability.
            _warn("checkpoint history cache mget failed; treating as all-miss: %s", exc)
            self._misses += len(keys)
            return {}
        found: dict[str, dict[str, Any]] = {}
        for key, raw in zip(keys, raws, strict=True):
            if raw is None:
                self._misses += 1
                continue
            try:
                tag, payload = raw.split(_TAG_SEPARATOR, 1)
                entry = self._serde.loads_typed((tag.decode(), payload))
                # Delta materialization unpacks each pending write as a triple.
                # A three-character string also unpacks, silently inventing a
                # channel value, so validate the record rather than just len().
                if not valid_history_entry(entry):
                    raise ValueError("invalid checkpoint history entry or write record")
            except Exception as exc:
                # Cached history is optional. A malformed entry must not prevent
                # rebuilding history from authoritative checkpoints.
                _warn(
                    "checkpoint history cache decode failed; treating as miss (key=%s): %s",
                    key,
                    type(exc).__name__,
                )
                self._misses += 1
                continue
            self._hits += 1
            found[key] = entry
        return found

    @staticmethod
    def _epoch_key(key_prefix: str) -> str:
        # No trailing colon: this metadata key cannot match any thread stem,
        # including a thread literally named "__cache_epoch__".
        return f"{key_prefix}:__cache_epoch__"

    async def aget_generation(self, key_prefix: str) -> str | None:
        epoch_key = self._epoch_key(key_prefix)
        try:
            for _ in range(3):
                raw = await self._client.get(epoch_key)
                if raw is not None:
                    return raw.decode() if isinstance(raw, bytes) else str(raw)
                candidate = secrets.token_hex(16)
                if await self._client.setnx(epoch_key, candidate):
                    return candidate
            return None
        except _redis_error() as exc:
            _warn("checkpoint history cache generation read failed; skipping cache fill: %s", exc)
            return None

    def _encode_entries(self, entries: dict[str, dict[str, Any]]) -> list[tuple[str, bytes]]:
        encoded: list[tuple[str, bytes]] = []
        for key, entry in entries.items():
            if not valid_history_entry(entry):
                _warn("checkpoint history cache invalid entry; skipping (key=%s)", key)
                continue
            try:
                tag, data = self._serde.dumps_typed(entry)
            except Exception as exc:
                self._write_serialization_failures += 1
                # Cache writes are optional; one unserializable entry must not
                # abort authoritative history reads or discard other entries.
                _warn(
                    "checkpoint history cache serialization failed; skipping (key=%s): %s",
                    key,
                    type(exc).__name__,
                )
                continue
            encoded.append((key, tag.encode() + _TAG_SEPARATOR + data))
        return encoded

    async def aset_many(self, entries: dict[str, dict[str, Any]]) -> None:
        if not entries:
            return
        encoded = self._encode_entries(entries)
        if not encoded:
            return
        try:
            pipe = self._client.pipeline(transaction=False)
            for key, value in encoded:
                pipe.set(key, value, ex=self._ttl)
            await pipe.execute()
        except _redis_error() as exc:
            # Direct cache writes remain best-effort; read-through uses the
            # generation-checked method below.
            _warn("checkpoint history cache write failed; skipping: %s", exc)

    async def aset_many_if_generation(
        self,
        key_prefix: str,
        generation: int | str | None,
        entries: dict[str, dict[str, Any]],
    ) -> bool:
        if not entries:
            return True
        if generation is None:
            return False
        encoded = self._encode_entries(entries)
        if not encoded:
            return True
        arguments: list[Any] = [
            str(generation),
            str(self._ttl if self._ttl is not None else -1),
        ]
        for key, value in encoded:
            arguments.extend((key, value))
        try:
            result = await self._client.eval(
                _SET_IF_GENERATION_SCRIPT,
                1,
                self._epoch_key(key_prefix),
                *arguments,
            )
            return bool(result)
        except _redis_error() as exc:
            _warn("checkpoint history cache conditional write failed; skipping: %s", exc)
            return False

    async def adelete_thread(self, key_prefix: str, thread_id: str) -> None:
        """Advance the namespace fence, then SCAN+UNLINK this thread's entries.

        If Redis is unavailable, the source-of-truth delete has already
        happened; residual cache retention follows the configured TTL/eviction
        policy and this method never raises.
        """
        try:
            # A random token avoids generation reuse if Redis evicts the epoch key.
            await self._client.eval(
                _ADVANCE_GENERATION_SCRIPT,
                1,
                self._epoch_key(key_prefix),
                secrets.token_hex(16),
            )
        except _redis_error() as exc:
            _warn("checkpoint history cache thread purge failed; generation fence unavailable: %s", exc)
            return
        stem = thread_key_stem(key_prefix, thread_id)
        # The configured namespace is a literal key prefix, not a Redis glob.
        pattern = "".join("\\" + char if char in "\\*?[]" else char for char in stem) + "*"
        # redis-py encodes SCAN results with the configured connection encoder.
        # A fixed UTF-8 fence silently skips owned keys for other encodings.
        encoded_stem = self._client.connection_pool.get_encoder().encode(stem)
        try:
            cursor = 0
            while True:
                cursor, keys = await self._client.scan(cursor=cursor, match=pattern, count=500)
                keys = [key for key in keys if key.startswith(encoded_stem if isinstance(key, bytes) else stem)]
                if keys:
                    await self._client.unlink(*keys)
                if cursor == 0:
                    break
        except _redis_error() as exc:
            _warn("checkpoint history cache thread purge failed; residual entries expire via TTL: %s", exc)

    def stats(self) -> CheckpointCacheStats:
        return CheckpointCacheStats(
            hits=self._hits,
            misses=self._misses,
            write_serialization_failures=self._write_serialization_failures,
        )

    async def aclose(self) -> None:
        await self._client.aclose()
