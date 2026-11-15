import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

import post_training_agent.tools as tools_module
from post_training_agent.tools import CppExecutor, Submission

LOGGER = logging.getLogger("test")
ECHO_SUM = '#include <cstdio>\nint main(){long long a,b;scanf("%lld %lld",&a,&b);printf("%lld\\n",a+b);}\n'


@pytest.fixture
def tool_tmp_path(tmp_path: Path) -> Path:
    tmp_path.parent.parent.chmod(0o755)
    tmp_path.parent.chmod(0o755)
    tmp_path.chmod(0o777)
    return tmp_path


def workspace(tmp_path: Path) -> Path:
    (tmp_path / "samples").mkdir()
    (tmp_path / "samples/sum.in").write_text("2 3\n")
    (tmp_path / "samples/sum.out").write_text("5\n")
    return tmp_path


def test_executor_runs_with_stdin():
    output = asyncio.run(CppExecutor().execute({"cpp_code": ECHO_SUM, "stdin": "40 2"}, {}, LOGGER))
    assert json.loads(output.output)["stdout"] == "42\n"


def test_executor_has_no_api_key_or_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TINKER_API_KEY", "test-secret")
    code = '''#include <arpa/inet.h>
#include <cstdlib>
#include <cstdio>
#include <sys/socket.h>
int main(){
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(443);
    inet_pton(AF_INET, "1.1.1.1", &address.sin_addr);
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if(fd < 0) return 1;
    printf("%d %d\\n", getenv("TINKER_API_KEY") == nullptr,
           connect(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == -1);
}
'''
    output = asyncio.run(CppExecutor().execute({"cpp_code": code}, {}, LOGGER))
    result = json.loads(output.output)
    assert result["exit_code"] == 0
    assert result["stdout"].split() == ["1", "1"]


def test_executor_reports_compile_error_and_timeout():
    broken = asyncio.run(CppExecutor().execute({"cpp_code": "int main("}, {}, LOGGER))
    assert "error" in broken.output.lower()
    spin = asyncio.run(CppExecutor().execute({"cpp_code": "int main(){for(;;);}"}, {}, LOGGER))
    assert "time" in spin.output.lower()


def test_submission_saves_and_runs_only_samples(tmp_path: Path):
    root = workspace(tmp_path)
    state: dict[str, object] = {}
    tool = Submission(problem_path=str(root), max_submissions=2)
    saved = json.loads((asyncio.run(tool.execute({"cpp_code": f"```\n{ECHO_SUM}```"}, state, LOGGER))).output)
    assert saved["samples"] == {"sum": "passed"} and (root / "solution.cpp").read_text().strip() == ECHO_SUM.strip()
    rejected = asyncio.run(tool.execute({"cpp_code": "int main("}, state, LOGGER))
    assert rejected.error == "Compilation failed" and (root / "solution.cpp").read_text().strip() == ECHO_SUM.strip()
    assert state == {"has_submissions": True, "samples_passed": 1} and rejected.done


def test_executor_timeout_kills_descendants(monkeypatch: pytest.MonkeyPatch, tool_tmp_path: Path) -> None:
    ready = tool_tmp_path / "ready"
    survived = tool_tmp_path / "survived"
    code = f'''#include <unistd.h>
#include <fstream>
int main(){{
    if(fork()==0){{
        std::ofstream(R"({ready})") << "ready";
        sleep(2);
        std::ofstream(R"({survived})") << "survived";
        _exit(0);
    }}
    sleep(10);
}}
'''
    monkeypatch.setattr(tools_module, "RUN_TIMEOUT", 1)

    async def attempt() -> None:
        result = await CppExecutor().execute({"cpp_code": code}, {}, LOGGER)
        assert result.error == "Execution timed out after 1 seconds"
        assert ready.exists()
        await asyncio.sleep(2.2)
        assert not survived.exists()

    asyncio.run(attempt())


@pytest.mark.parametrize("stop", ["SIGTERM", "SIGKILL_GROUP", "exit"])
def test_agent_exit_kills_live_tools_and_descendants(tool_tmp_path: Path, stop: str) -> None:
    ready = tool_tmp_path / "pids.json"
    finish = tool_tmp_path / "finish"
    program = (
        "import json,os,time; from pathlib import Path; child=os.fork(); "
        f"Path({str(ready)!r}).write_text(json.dumps([os.getppid(),os.getpid(),child])) if child else None; "
        "time.sleep(60)"
    )
    agent = (
        "import asyncio,os,sys; from pathlib import Path; from post_training_agent.tools import _communicate\n"
        "async def main():\n"
        f" task=asyncio.create_task(_communicate(['/usr/bin/python3','-c',{program!r}],b'',60))\n"
        f" while not Path({str(finish)!r}).exists(): await asyncio.sleep(0.02)\n"
        " os._exit(0)\n"
        "asyncio.run(main())\n"
    )
    process = subprocess.Popen([sys.executable, "-c", agent], start_new_session=True)
    pids: list[int] = []

    def ps() -> list[str]:
        return subprocess.run(
            ["ps", "-o", "pid=,ppid=,pgid=,stat=", "-p", ",".join(map(str, pids))],
            capture_output=True, text=True, check=False,
        ).stdout.splitlines()

    try:
        deadline = time.monotonic() + 10
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        pids = json.loads(ready.read_text())
        before = ps()
        assert len(before) == 3 and all(not line.split()[-1].startswith("Z") for line in before)
        if stop == "exit":
            finish.touch()
        elif stop == "SIGKILL_GROUP":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.terminate()
        process.wait(timeout=5)
        deadline = time.monotonic() + 5
        after = ps()
        while any(not line.split()[-1].startswith("Z") for line in after) and time.monotonic() < deadline:
            time.sleep(0.02)
            after = ps()
        print(f"PROCESS_CLEANUP {stop} before={before} after={after}")
        assert all(line.split()[-1].startswith("Z") for line in after)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        if len(pids) == 3:
            try:
                os.killpg(pids[1], signal.SIGKILL)
            except ProcessLookupError:
                pass
