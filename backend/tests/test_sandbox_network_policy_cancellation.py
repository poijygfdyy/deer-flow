"""Cancellation ownership for sandbox network-policy mutations."""

from __future__ import annotations

import asyncio
import threading

import pytest
from langgraph.runtime import Runtime

from deerflow.sandbox.middleware import SandboxMiddleware
from deerflow.sandbox.sandbox_provider import SandboxProvider


class _BlockingPolicyProvider(SandboxProvider):
    def __init__(self) -> None:
        self.entered: dict[str, threading.Event] = {
            "consume": threading.Event(),
            "deny": threading.Event(),
            "decide": threading.Event(),
        }
        self.release = threading.Event()
        self.completed: list[str] = []

    def acquire(self, thread_id: str | None = None, *, user_id: str | None = None) -> str:
        del thread_id, user_id
        return "sandbox-1"

    def get(self, sandbox_id: str):
        del sandbox_id
        return None

    def release(self, sandbox_id: str) -> None:
        del sandbox_id

    def _block(self, kind: str, result):
        self.entered[kind].set()
        if not self.release.wait(5):
            raise TimeoutError("test did not release network-policy mutation")
        self.completed.append(kind)
        return result

    def consume_network_policy_events(self, sandbox_id: str) -> list[dict[str, object]]:
        del sandbox_id
        return self._block("consume", [{"request_id": "request-1"}])

    def deny_pending_network_policy_events(self, sandbox_id: str) -> bool:
        del sandbox_id
        return self._block("deny", True)

    def decide_network_policy_request(self, sandbox_id: str, request_id: str, decision: str) -> bool:
        del sandbox_id, request_id, decision
        return self._block("decide", True)


async def _deliver_repeated_cancellation(task: asyncio.Task) -> None:
    task.cancel("first host cancellation")
    await asyncio.sleep(0)
    task.cancel("second host cancellation")
    await asyncio.sleep(0)


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("method_name", "args", "kind"),
    [
        ("consume_network_policy_events_async", ("sandbox-1",), "consume"),
        ("deny_pending_network_policy_events_async", ("sandbox-1",), "deny"),
        (
            "decide_network_policy_request_async",
            ("sandbox-1", "request-1", "deny"),
            "decide",
        ),
    ],
)
async def test_async_network_policy_mutation_drains_before_cancellation(
    method_name: str,
    args: tuple[str, ...],
    kind: str,
) -> None:
    provider = _BlockingPolicyProvider()
    task = asyncio.create_task(getattr(provider, method_name)(*args))

    try:
        assert await asyncio.to_thread(provider.entered[kind].wait, 1)
        await _deliver_repeated_cancellation(task)

        assert not task.done(), "caller cancellation escaped while the policy mutation worker was still running"
        assert provider.completed == []

        provider.release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert provider.completed == [kind]
    finally:
        provider.release.set()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.anyio
async def test_abefore_agent_drains_network_policy_response_before_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    middleware = SandboxMiddleware(
        lazy_init=True,
        owns_agent_skill_projection=False,
    )
    entered = threading.Event()
    release = threading.Event()
    completed = threading.Event()

    def apply_network_policy_response(*_args) -> None:
        entered.set()
        if not release.wait(5):
            raise TimeoutError("test did not release network-policy response")
        completed.set()

    monkeypatch.setattr(
        middleware,
        "_apply_network_policy_response",
        apply_network_policy_response,
    )
    task = asyncio.create_task(
        middleware.abefore_agent(
            {},
            Runtime(context={"thread_id": "thread-1", "user_id": "user-1"}),
        )
    )

    try:
        assert await asyncio.to_thread(entered.wait, 1)
        await _deliver_repeated_cancellation(task)

        assert not task.done(), "caller cancellation escaped while the policy decision worker was still running"
        assert not completed.is_set()

        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert completed.is_set()
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
