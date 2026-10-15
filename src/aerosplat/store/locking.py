"""Cross-process exclusive file lock (no third-party dependency).

Used to serialise catalog migrations: `aerosplat up` starts the server and its
workers at the same moment, and each of them opens the catalog.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def file_lock(path: Path, *, timeout: float = 120.0, poll: float = 0.05) -> Iterator[None]:
    """Hold an exclusive lock on `path` (created if missing) for the `with` block."""
    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout
    with open(path, "a+b") as f:
        if sys.platform == "win32":
            import msvcrt

            while True:
                try:
                    f.seek(0)
                    msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    if time.monotonic() > deadline:
                        raise TimeoutError(f"could not lock {path} within {timeout:.0f}s") from None
                    time.sleep(poll)
            try:
                yield
            finally:
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            while True:
                try:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() > deadline:
                        raise TimeoutError(f"could not lock {path} within {timeout:.0f}s") from None
                    time.sleep(poll)
            try:
                yield
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
