"""Shared logging / console helpers (replaces 8 copies of _fix_console_encoding)."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def fix_console_encoding() -> None:
    """Avoid UnicodeEncodeError on Windows consoles with legacy codepages.

    Pipeline scripts log Cyrillic and emoji from the first line, and a crash in
    a long stage (training) loses the run, so this runs before any output.
    """
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name)
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except Exception:
                pass


class _StdoutHandler(logging.StreamHandler):
    """Always writes to the *current* sys.stdout (survives reconfigure / capture)."""

    def __init__(self) -> None:
        super().__init__(sys.stdout)

    @property
    def stream(self):
        return sys.stdout

    @stream.setter
    def stream(self, value) -> None:
        pass


def get_logger(name: str = "aerosplat", log_file: Path | None = None,
               level: int = logging.INFO) -> logging.Logger:
    """Console logger (plain message format) with an optional run-directory file."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        console = _StdoutHandler()
        console.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(console)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        if not any(isinstance(h, logging.FileHandler)
                   and Path(h.baseFilename) == log_file.resolve()
                   for h in logger.handlers):
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(logging.Formatter(
                "%(asctime)s %(levelname)s %(message)s"))
            logger.addHandler(fh)
    return logger
