"""Process helpers for workers: start a job process in its own group, kill the whole
tree on cancel (COLMAP and friends are grandchildren), and check whether a pid lives."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from typing import Any


def group_kwargs() -> dict[str, Any]:
    """Popen kwargs that put the child in a new process group/session."""
    if sys.platform == "win32":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def kill_tree(proc: subprocess.Popen, timeout: float = 10.0) -> None:
    """Terminate `proc` and every process it started; escalate to a hard kill."""
    if proc.poll() is not None:
        return
    if sys.platform == "win32":
        # taskkill /T walks the tree; TerminateProcess alone would orphan COLMAP.
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    else:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        if sys.platform != "win32":
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        proc.kill()
        proc.wait(timeout=timeout)


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        k32 = ctypes.windll.kernel32
        handle = k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        try:
            code = wintypes.DWORD()
            if not k32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            return code.value == STILL_ACTIVE
        finally:
            k32.CloseHandle(handle)
    try:
        os.kill(pid, 0)   # signal 0: existence check only (POSIX)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
