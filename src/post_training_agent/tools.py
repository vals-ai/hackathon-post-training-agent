import asyncio
import json
import logging
import os
import pwd
import sys
import tempfile
from pathlib import Path
from typing import Any

from model_library.agent import Tool, ToolOutput

from post_training_agent.utils import simple_extract_code

SOLUTION_FILE = "solution.cpp"
SAMPLES_DIR = "samples"
COMPILE_TIMEOUT = 60
RUN_TIMEOUT = 10
OUTPUT_LIMIT = 4000


def _extract(args: dict[str, Any]) -> str:
    text = args.get("cpp_code", "")
    return simple_extract_code(text) or text


def _clip(text: str) -> str:
    return text if len(text) <= OUTPUT_LIMIT else text[:OUTPUT_LIMIT] + f"\n... [{len(text) - OUTPUT_LIMIT} more chars]"


async def _communicate(cmd: list[str], stdin: bytes, timeout: int) -> tuple[int, str, str] | None:
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        str(Path(__file__).with_name("_process.py")),
        str(os.getpid()),
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(stdin), timeout=timeout)
    except TimeoutError:
        return None
    finally:
        try:
            process.terminate()
        except ProcessLookupError:
            pass
        await process.wait()
    return process.returncode or 0, stdout.decode(errors="replace"), stderr.decode(errors="replace")


async def _compile(code: str, workdir: Path) -> str | None:
    source = workdir / "main.cpp"
    source.write_text(code)
    source.chmod(0o644)
    os.chown(workdir, pwd.getpwnam("nobody").pw_uid, -1)
    compiled = await _communicate(
        ["g++", "-std=c++20", "-O2", "-pipe", str(source), "-o", str(workdir / "main")], b"", COMPILE_TIMEOUT
    )
    if compiled is None:
        return f"Compilation timed out after {COMPILE_TIMEOUT} seconds"
    if compiled[0] != 0:
        return f"Compilation failed:\n{_clip(compiled[2])}"
    return None


class CppExecutor(Tool):
    """Compile a C++20 program and run it on the given stdin."""

    name = "cpp_executor"
    description = (
        "Compile a C++20 program (g++ -O2) and run it once with the given stdin. "
        f"Returns stdout, stderr and the exit code. Runs are killed after {RUN_TIMEOUT} seconds. "
        "Use it for experiments, brute-force checkers and generators."
    )
    parameters: dict[str, Any] = {
        "cpp_code": {"type": "string", "description": "The C++20 program, optionally inside a code block."},
        "stdin": {"type": "string", "description": "Text passed to the program on standard input."},
    }
    required = ["cpp_code"]

    async def execute(self, args: dict[str, Any], state: dict[str, Any], logger: logging.Logger) -> ToolOutput:
        code = _extract(args)
        if not code.strip():
            return ToolOutput(output="No C++ code found in the input", error="No C++ code found")
        with tempfile.TemporaryDirectory() as workdir:
            if error := await _compile(code, Path(workdir)):
                return ToolOutput(output=error, error=error)
            ran = await _communicate([str(Path(workdir) / "main")], args.get("stdin", "").encode(), RUN_TIMEOUT)
        if ran is None:
            error = f"Execution timed out after {RUN_TIMEOUT} seconds"
            return ToolOutput(output=error, error=error)
        exit_code, stdout, stderr = ran
        result = {"exit_code": exit_code, "stdout": _clip(stdout), "stderr": _clip(stderr)}
        return ToolOutput(output=json.dumps(result), error="Execution failed" if exit_code else None)


class Submission(Tool):
    """Record a solution and check it against the statement's sample tests."""

    name = "submission"
    description = (
        "Submit a complete C++20 solution. It is compiled, run on the sample tests from the statement, and "
        "saved as your final answer; your last submission that compiles is the one graded on the hidden tests. "
        "Only sample results are reported, never hidden test results."
    )
    parameters: dict[str, Any] = {
        "cpp_code": {"type": "string", "description": "The full C++20 solution, optionally inside a code block."},
    }
    required = ["cpp_code"]

    def __init__(self, problem_path: str, max_submissions: int = 50):
        super().__init__()
        self.workspace = Path(problem_path)
        self.max_submissions = max_submissions
        self.submission_count = 0

    def _samples(self) -> list[tuple[str, Path, Path]]:
        return [
            (given.stem, given, given.with_suffix(".out"))
            for given in sorted((self.workspace / SAMPLES_DIR).glob("*.in"))
            if given.with_suffix(".out").is_file()
        ]

    async def execute(self, args: dict[str, Any], state: dict[str, Any], logger: logging.Logger) -> ToolOutput:
        if self.submission_count >= self.max_submissions:
            error = f"Submission limit of {self.max_submissions} reached; your last accepted submission stands."
            return ToolOutput(output=error, error=error, done=True)
        code = _extract(args)
        if not code.strip():
            return ToolOutput(output="No C++ code found in the input", error="No C++ code found")
        self.submission_count += 1
        remaining = self.max_submissions - self.submission_count
        with tempfile.TemporaryDirectory() as workdir:
            if error := await _compile(code, Path(workdir)):
                message = f"{error}\nNot saved; your previous submission stands. Submissions remaining: {remaining}"
                return ToolOutput(output=message, error="Compilation failed", done=remaining == 0)
            (self.workspace / SOLUTION_FILE).write_text(code)
            state["has_submissions"] = True
            samples: dict[str, str] = {}
            for name, given, wanted in self._samples():
                ran = await _communicate([str(Path(workdir) / "main")], given.read_bytes(), RUN_TIMEOUT)
                if ran is None:
                    samples[name] = "time limit exceeded"
                elif ran[0] != 0:
                    samples[name] = f"runtime error (exit code {ran[0]}): {_clip(ran[2])}"
                elif ran[1].split() != wanted.read_text().split():
                    samples[name] = f"wrong answer, got:\n{_clip(ran[1])}"
                else:
                    samples[name] = "passed"
        state["samples_passed"] = sum(verdict == "passed" for verdict in samples.values())
        result = {"saved": True, "samples": samples, "submissions_remaining": remaining}
        logger.info(f"Submission {self.submission_count}: {samples}")
        return ToolOutput(output=json.dumps(result), done=remaining == 0)
