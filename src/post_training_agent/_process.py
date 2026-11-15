import ctypes
import os
import signal
import subprocess
import sys
from contextlib import suppress
from types import FrameType
from typing import NoReturn


def _stop(_signal: int, _frame: FrameType | None) -> NoReturn:
    raise SystemExit(143)


def _unblock() -> None:
    signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGTERM})


def main() -> int:
    signal.signal(signal.SIGTERM, _stop)
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    if os.getppid() != int(sys.argv[1]):
        return 143
    os.unshare(os.CLONE_NEWNET)
    process: subprocess.Popen[bytes] | None = None
    try:
        signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM})
        process = subprocess.Popen(
            sys.argv[2:],
            process_group=0,
            preexec_fn=_unblock,
            user="nobody",
            group="nogroup",
            extra_groups=[],
            env={"PATH": "/usr/local/bin:/usr/bin:/bin"},
        )
        _unblock()
        status = process.wait()
    finally:
        if process is not None:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
    if status < 0:
        signal.signal(-status, signal.SIG_DFL)
        os.kill(os.getpid(), -status)
    return status


if __name__ == "__main__":
    sys.exit(main())
