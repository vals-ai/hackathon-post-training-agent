import argparse
import asyncio
import inspect
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from model_library.agent import Agent

import post_training_agent.run as run_module


def test_locked_model_library_supports_atif_export() -> None:
    assert "atif_export" in inspect.signature(Agent.run).parameters


@pytest.mark.parametrize("error_type", [None, "SampleConversionLimit"])
def test_run_requests_non_scoring_atif_export(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, error_type: str | None
) -> None:
    run_kwargs: dict[str, Any] = {}

    class FakeResult:
        stop_reason = "error" if error_type else "done_tool"
        final_error = SimpleNamespace(type=error_type) if error_type else None

        def model_dump_json(self, indent: int) -> str:
            return "{}"

    class FakeAgent:
        def __init__(self, **kwargs: Any) -> None:
            logs = kwargs["log_dir"]
            assert logs.stat().st_uid == logs.stat().st_gid == 0
            assert logs.stat().st_mode & 0o777 == 0o700

        async def run(self, *args: Any, **kwargs: Any) -> FakeResult:
            run_kwargs.update(kwargs)
            return FakeResult()

    def fake_get_registry_model(model: str) -> SimpleNamespace:
        return SimpleNamespace()

    def fake_submission(problem_path: str) -> SimpleNamespace:
        return SimpleNamespace()

    monkeypatch.setattr(run_module, "get_registry_model", fake_get_registry_model)
    monkeypatch.setattr(run_module, "CppExecutor", lambda: SimpleNamespace())
    monkeypatch.setattr(run_module, "Submission", fake_submission)
    monkeypatch.setattr(run_module, "Agent", FakeAgent)
    monkeypatch.setattr(run_module, "LOGS_DIR", tmp_path / "logs")

    problem_file = tmp_path / "problem.txt"
    problem_file.write_text("Solve this")
    output_file = tmp_path / "result.json"
    solution_file = tmp_path / "solution.cpp"
    solution_file.write_text("saved solution")
    args = argparse.Namespace(
        model="provider/model",
        checkpoint=None,
        max_output_tokens=16384,
        problem_file=str(problem_file),
        problem_dir=str(tmp_path),
        task_id="2024/example",
        output=str(output_file),
    )

    old_umask = os.umask(0o022)
    try:
        asyncio.run(run_module.run(args))
    finally:
        os.umask(old_umask)

    assert run_kwargs["atif_export"] is True
    assert output_file.read_text() == "{}"
    assert solution_file.read_text() == "saved solution"
    assert output_file.stat().st_uid == output_file.stat().st_gid == 0
    assert output_file.stat().st_mode & 0o777 == 0o600
