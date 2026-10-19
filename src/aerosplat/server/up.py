"""`aerosplat up`: the whole platform on one workstation — API + web UI and workers.

    aerosplat up                      # http://127.0.0.1:8000 , GPU + CPU workers, render service :8001
    aerosplat up --no-cpu-worker --no-render
    aerosplat up --projects-root D:/aerosplat/projects

Workers and the render service run as separate processes (a crashed or OOM-killed job never touches the
API). A worker that exits is restarted with a growing delay, so the UI never
silently loses its GPU slot; all workers stop together with the server.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

from aerosplat.jobs.proctree import group_kwargs, kill_tree
from aerosplat.logging_utils import fix_console_encoding
from aerosplat.paths import PROJECT_DIR

RESTART_DELAYS_S = (2.0, 5.0, 15.0, 30.0, 60.0)
HEALTHY_AFTER_S = 120.0     # a worker that lived this long resets its restart backoff
POLL_S = 1.0


RENDER = "render"           # the render service is supervised like a worker


def with_gsplat_env(cmd: list[str]) -> list[str]:
    """`cmd` run inside tools/gsplat_env.bat when gsplat would JIT-build its CUDA extension.

    On Windows that build only works with VS 2022 and the CUDA toolkit torch was built for,
    which a plain shell does not have. A prebuilt gsplat (`gsplat.csrc`) needs no toolchain,
    and without gsplat there is nothing to build. Whatever the wrapper achieves, the pipeline
    checks gsplat again before the first stage (runner.check_rasterizer)."""
    env_bat = PROJECT_DIR / "tools" / "gsplat_env.bat"
    if os.name != "nt" or not env_bat.is_file():
        return cmd
    try:
        import gsplat  # noqa: F401
    except ImportError:
        return cmd
    try:
        from gsplat import csrc  # noqa: F401  (prebuilt: no toolchain needed)
    except ImportError:
        return ["cmd.exe", "/c", str(env_bat), *cmd]
    return cmd


def render_service_command(port: int) -> list[str]:
    """Command line of the render service (renders with gsplat)."""
    return with_gsplat_env([sys.executable, "-m", "aerosplat.render.service", "--port", str(port)])


def worker_command(resource: str) -> list[str]:
    """Command line of a worker. The GPU worker trains (`rasterizer: gsplat` included), so it
    gets the gsplat build environment like the render service; the CPU worker never needs it."""
    cmd = [sys.executable, "-m", "aerosplat.jobs.worker", "--resource", resource]
    return with_gsplat_env(cmd) if resource == "gpu" else cmd


class WorkerSupervisor(threading.Thread):
    """Keep one process per resource class (and the render service) alive until stop()."""

    def __init__(self, resources: list[str], commands: Optional[dict[str, list[str]]] = None):
        super().__init__(name="worker-supervisor", daemon=True)
        self.resources = resources
        self.commands = commands or {}
        self.procs: dict[str, subprocess.Popen] = {}
        self.started: dict[str, float] = {}
        self.failures: dict[str, int] = {r: 0 for r in resources}
        self.next_start: dict[str, float] = {}
        self._stopping = threading.Event()
        self._lock = threading.Lock()

    def _spawn(self, resource: str) -> None:
        cmd = self.commands.get(resource) or worker_command(resource)
        self.procs[resource] = subprocess.Popen(
            cmd, env={**os.environ, "PYTHONIOENCODING": "utf-8"}, **group_kwargs())
        self.started[resource] = time.monotonic()

    def start_all(self) -> None:
        with self._lock:
            for r in self.resources:
                self._spawn(r)
        self.start()

    def run(self) -> None:
        while not self._stopping.wait(POLL_S):
            with self._lock:
                if self._stopping.is_set():
                    return
                now = time.monotonic()
                for r in self.resources:
                    proc = self.procs.get(r)
                    if proc is not None and proc.poll() is None:
                        continue
                    if proc is not None:              # just died: schedule a restart
                        lived = now - self.started[r]
                        self.failures[r] = 0 if lived > HEALTHY_AFTER_S else self.failures[r] + 1
                        delay = RESTART_DELAYS_S[min(self.failures[r], len(RESTART_DELAYS_S) - 1)]
                        print(f"⚠️  {r.upper()} завершився (код {proc.returncode}) — "
                              f"перезапуск через {delay:.0f} с", flush=True)
                        self.procs.pop(r)
                        self.next_start[r] = now + delay
                    elif now >= self.next_start.get(r, 0.0):
                        self._spawn(r)
                        print(f"🔄 {r.upper()} перезапущено", flush=True)

    def stop(self) -> None:
        with self._lock:
            self._stopping.set()
            for p in self.procs.values():
                kill_tree(p)


def main(argv: Optional[list[str]] = None) -> int:
    fix_console_encoding()
    ap = argparse.ArgumentParser(prog="aerosplat up", description=__doc__.split("\n")[0])
    ap.add_argument("--host", default="127.0.0.1",
                    help="адреса сервера; API без автентифікації — лише localhost (ADR-0010)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--projects-root", type=Path, default=None)
    ap.add_argument("--no-gpu-worker", action="store_true")
    ap.add_argument("--no-cpu-worker", action="store_true")
    ap.add_argument("--no-render", action="store_true", help="do not start the render service")
    ap.add_argument("--render-port", type=int, default=8001)
    args = ap.parse_args(argv)

    if args.projects_root is not None:
        # Before importing the server: it resolves the catalog root at import time.
        os.environ["AEROSPLAT_PROJECTS"] = str(args.projects_root.resolve())

    # Migrate once, here, before anything else opens the catalog: a schema error
    # then stops `up` with one clear message instead of killing a worker.
    from aerosplat.store.catalog import Catalog
    from aerosplat.store.paths import default_projects_root

    Catalog.open(default_projects_root()).close()

    resources = [r for r, off in (("gpu", args.no_gpu_worker), ("cpu", args.no_cpu_worker))
                 if not off]
    commands: dict[str, list[str]] = {r: worker_command(r) for r in resources}
    if "gpu" in commands and commands["gpu"][0] == "cmd.exe":
        print("ℹ️  GPU-воркер запускається через tools\\gsplat_env.bat (у gsplat немає зібраного "
              "csrc, rasterizer=gsplat збирає його при першому виклику)", flush=True)
    if not args.no_render:
        resources.append(RENDER)
        commands[RENDER] = render_service_command(args.render_port)
        os.environ.setdefault("AEROSPLAT_RENDER_URL", f"http://127.0.0.1:{args.render_port}")
    supervisor = WorkerSupervisor(resources, commands)
    supervisor.start_all()
    try:
        import uvicorn

        from aerosplat.server.main import app, describe_scenes

        describe_scenes()
        print(f"\n🚀 AeroSplat-GIS: http://{args.host}:{args.port}/   "
              f"(процеси: {', '.join(resources) or '—'})\n", flush=True)
        uvicorn.run(app, host=args.host, port=args.port)
    finally:
        supervisor.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
