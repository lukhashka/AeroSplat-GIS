"""`manifest.json`: what ran, with which config, how long, and what it produced."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

SCHEMA = 1


def fingerprint(values: dict[str, Any]) -> str:
    blob = json.dumps(values, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _git_commit(cwd: Path) -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=cwd,
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def describe_path(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    if path.is_dir():
        return {"path": str(path), "exists": True, "kind": "dir"}
    st = path.stat()
    return {"path": str(path), "exists": True, "kind": "file", "bytes": st.st_size}


class Manifest:
    def __init__(self, path: Path, data: dict[str, Any]):
        self.path = path
        self.data = data

    @classmethod
    def load(cls, path: Path) -> "Manifest":
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if data.get("schema") == SCHEMA:
                    return cls(path, data)
            except (OSError, ValueError):
                pass
        return cls(path, {"schema": SCHEMA, "stages": {}})

    def start_run(self, dataset: str, cfg: dict[str, Any], project_dir: Path) -> None:
        from aerosplat import __version__
        self.data.update({
            "dataset": dataset,
            "aerosplat_version": __version__,
            "git_commit": _git_commit(project_dir),
            "python": sys.version.split()[0],
            "config": cfg,
            "updated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        })
        self.save()

    def record(self, name: str, *, status: str, fingerprint: str, seconds: float = 0.0,
               inputs: Sequence[Path] = (), outputs: Sequence[Path] = (),
               error: str | None = None) -> None:
        self.data["stages"][name] = {
            "status": status,
            "fingerprint": fingerprint,
            "seconds": round(seconds, 2),
            "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "inputs": [describe_path(p) for p in inputs],
            "outputs": [describe_path(p) for p in outputs],
            "error": error,
        }
        self.data["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.save()

    def is_fresh(self, name: str, fp: str, outputs: list[Path]) -> bool:
        """Stage finished OK earlier with the same config and its outputs still exist."""
        rec = self.data["stages"].get(name)
        return bool(rec and rec["status"] == "ok" and rec["fingerprint"] == fp
                    and all(p.exists() for p in outputs))

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.path)
