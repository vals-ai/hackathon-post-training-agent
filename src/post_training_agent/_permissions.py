import os
from pathlib import Path

LOGS_DIR = Path("/logs")
WORKSPACE_DIR = Path("/workspace")


def secure_directory(path: Path, *, private: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for entry in [path, *path.rglob("*")]:
        if entry.is_symlink():
            raise ValueError(f"Output paths cannot be symlinks: {entry}")
        os.chown(entry, 0, 0)
        entry.chmod((0o700 if private else 0o755) if entry.is_dir() else (0o600 if private else 0o644))


if __name__ == "__main__":
    secure_directory(LOGS_DIR, private=True)
    secure_directory(WORKSPACE_DIR, private=False)
