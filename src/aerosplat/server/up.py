"""`aerosplat up`: the whole platform on one workstation — API + web UI and workers.

    aerosplat up                      # http://127.0.0.1:8000 , GPU worker + CPU worker
    aerosplat up --no-cpu-worker
    aerosplat up --projects-root D:/aerosplat/projects

Workers run as separate processes (a crashed or OOM-killed job never touches the
API) and are stopped together with the server.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

from aerosplat.jobs.proctree import group_kwargs, kill_tree
from aerosplat.logging_utils import fix_console_encoding


def main(argv: Optional[list[str]] = None) -> int:
    fix_console_encoding()
    ap = argparse.ArgumentParser(prog="aerosplat up", description=__doc__.split("\n")[0])
    ap.add_argument("--host", default="127.0.0.1",
                    help="адреса сервера; API без автентифікації — лише localhost (ADR-0010)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--projects-root", type=Path, default=None)
    ap.add_argument("--no-gpu-worker", action="store_true")
    ap.add_argument("--no-cpu-worker", action="store_true")
    args = ap.parse_args(argv)

    if args.projects_root is not None:
        # Before importing the server: it resolves the catalog root at import time.
        os.environ["AEROSPLAT_PROJECTS"] = str(args.projects_root.resolve())

    workers: list[subprocess.Popen] = []
    for resource, off in (("gpu", args.no_gpu_worker), ("cpu", args.no_cpu_worker)):
        if not off:
            workers.append(subprocess.Popen(
                [sys.executable, "-m", "aerosplat.jobs.worker", "--resource", resource],
                env={**os.environ, "PYTHONIOENCODING": "utf-8"}, **group_kwargs()))
    try:
        import uvicorn

        from aerosplat.server.main import app, describe_scenes

        describe_scenes()
        print(f"\n🚀 AeroSplat-GIS: http://{args.host}:{args.port}/   "
              f"(воркери: {', '.join(r for r, off in (('gpu', args.no_gpu_worker), ('cpu', args.no_cpu_worker)) if not off) or '—'})\n")
        uvicorn.run(app, host=args.host, port=args.port)
    finally:
        for p in workers:
            kill_tree(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
