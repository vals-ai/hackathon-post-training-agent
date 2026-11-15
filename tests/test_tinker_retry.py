import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

import pytest

import post_training_agent.run as run_module


class StatusError(Exception):
    def __init__(self, status_code: int):
        super().__init__(f"Error code: {status_code}")
        self.status_code = status_code


async def _no_wait(*_: object) -> float:
    return 0


def _run(query: Callable[[int], Awaitable[str]], monkeypatch: pytest.MonkeyPatch) -> int:
    calls = 0
    monkeypatch.setattr(run_module.ExponentialBackoffRetrier, "_calculate_wait_time", _no_wait)

    async def counted() -> str:
        nonlocal calls
        calls += 1
        return await query(calls)

    async def main() -> Any:
        return await run_module.tinker_retry(counted)()

    try:
        asyncio.run(main())
    except Exception:
        pass
    return calls


@pytest.mark.parametrize(("code", "retried"), [(400, False), (422, False), (429, True), (503, True)])
def test_retries_only_transient_status(monkeypatch: pytest.MonkeyPatch, code: int, retried: bool) -> None:
    async def fail_then_ok(calls: int) -> str:
        if calls == 1:
            raise StatusError(code)
        return "ok"

    assert _run(fail_then_ok, monkeypatch) == (2 if retried else 1)


def test_hung_request_times_out_and_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run_module, "TINKER_REQUEST_TIMEOUT", 0.05)

    async def hang_then_ok(calls: int) -> str:
        if calls == 1:
            await asyncio.sleep(60)
        return "ok"

    assert _run(hang_then_ok, monkeypatch) == 2


def test_persistent_hang_gives_up(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run_module, "TINKER_REQUEST_TIMEOUT", 0.05)
    monkeypatch.setattr(run_module, "TINKER_RETRY_MAX_TIME", 0.3)

    async def hang(_calls: int) -> str:
        await asyncio.sleep(60)
        return "ok"

    assert 1 < _run(hang, monkeypatch) < 20
