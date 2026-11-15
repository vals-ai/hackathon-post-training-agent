import asyncio
import logging
import os
import pwd
import subprocess
from pathlib import Path

import pytest

from post_training_agent import tools
from post_training_agent._permissions import secure_directory


def test_usage_and_trajectory_are_root_owned_and_protected(tmp_path: Path) -> None:
    tmp_path.parent.parent.chmod(0o755)
    tmp_path.parent.chmod(0o755)
    tmp_path.chmod(0o755)
    logs = tmp_path / "logs"
    workspace = tmp_path / "workspace"
    nobody = pwd.getpwnam("nobody")
    paths = [logs / "vals_format/turns.jsonl", logs / "result.json", logs / "post_training_agent/atif/trajectory.json"]
    for path in [*paths, workspace / "solution.cpp"]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("original")
        os.chown(path, nobody.pw_uid, nobody.pw_gid)
        path.chmod(0o666)
    for directory in [logs, *[p for p in logs.rglob("*") if p.is_dir()], workspace]:
        os.chown(directory, nobody.pw_uid, nobody.pw_gid)
        directory.chmod(0o777)

    secure_directory(logs, private=True)
    secure_directory(workspace, private=False)
    old_umask = os.umask(0o077)
    try:
        new_dir = logs / "post_training/new-turn"
        new_dir.mkdir(parents=True)
        (new_dir / "result.json").write_text("new")
    finally:
        os.umask(old_umask)

    for path in [logs, *logs.rglob("*"), workspace, workspace / "solution.cpp"]:
        assert path.stat().st_uid == path.stat().st_gid == 0
        assert path.stat().st_mode & 0o022 == 0

    operations = [
        "p.write_text('forged')",
        "p.rename(p.with_name('replaced'))",
        "p.unlink()",
        "p.chmod(0o777)",
        "p.parent.joinpath('injected').write_text('forged')",
    ]
    for path in [*paths, new_dir / "result.json", workspace / "solution.cpp"]:
        for operation in operations:
            result = subprocess.run(
                ["/usr/bin/python3", "-c", f"from pathlib import Path; p=Path({str(path)!r}); {operation}"],
                user="nobody", group="nogroup", extra_groups=[], capture_output=True, text=True,
            )
            assert result.returncode != 0, (path, operation)
            assert "PermissionError" in result.stderr
        assert path.read_text() in {"original", "new"}
    result = subprocess.run(
        ["/usr/bin/python3", "-c", f"from pathlib import Path; Path({str(logs)!r}).rename({str(tmp_path / 'moved')!r})"],
        user="nobody", group="nogroup", extra_groups=[], capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "PermissionError" in result.stderr


def test_output_directory_rejects_symlinks(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.write_text("unchanged")
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "result.json").symlink_to(target)
    with pytest.raises(ValueError, match="symlinks"):
        secure_directory(logs, private=True)
    assert target.read_text() == "unchanged"


def test_compiler_source_is_written_before_directory_changes_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    released = False

    def chown(path: Path, uid: int, gid: int) -> None:
        nonlocal released
        assert uid == pwd.getpwnam("nobody").pw_uid
        assert (path / "main.cpp").read_text() == "source"
        assert (path / "main.cpp").stat().st_mode & 0o777 == 0o644
        released = True

    async def communicate(cmd: list[str], stdin: bytes, timeout: int) -> tuple[int, str, str]:
        assert released
        return 0, "", ""

    monkeypatch.setattr(tools.os, "chown", chown)
    monkeypatch.setattr(tools, "_communicate", communicate)
    result = asyncio.run(tools.CppExecutor().execute({"cpp_code": "source"}, {}, logging.getLogger("test")))
    assert result.error is None
