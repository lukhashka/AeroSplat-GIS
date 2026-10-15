"""`aerosplat render-service`: the long-lived render process.

    aerosplat render-service                       # http://127.0.0.1:8001
    aerosplat render-service --idle-seconds 120    # give the GPU back sooner

On this Windows workstation gsplat is JIT-built, so start it through
`tools\\gsplat_env.bat` (`aerosplat up` does that itself).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

from aerosplat.logging_utils import fix_console_encoding

DEFAULT_PORT = 8001


def main(argv: Optional[list[str]] = None) -> int:
    fix_console_encoding()
    ap = argparse.ArgumentParser(prog="aerosplat render-service", description=__doc__.split("\n")[0])
    ap.add_argument("--host", default="127.0.0.1", help="localhost only: there is no authentication")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--projects-root", type=Path, default=None)
    ap.add_argument("--data-dir", type=Path, default=None, help="legacy data/ with *_splat.pt")
    ap.add_argument("--idle-seconds", type=float, default=600.0,
                    help="unload a scene from the GPU after this long without requests")
    ap.add_argument("--max-scenes", type=int, default=1)
    args = ap.parse_args(argv)

    if args.projects_root is not None:
        os.environ["AEROSPLAT_PROJECTS"] = str(args.projects_root.resolve())

    import uvicorn

    from aerosplat.paths import DATA_DIR
    from aerosplat.render.app import create_app
    from aerosplat.render.engine import RenderEngine
    from aerosplat.store.paths import default_projects_root

    root = default_projects_root()
    data_dir = args.data_dir or Path(os.environ.get("AEROSPLAT_DATA", str(DATA_DIR)))
    engine = RenderEngine(root, data_dir, idle_s=args.idle_seconds, max_scenes=args.max_scenes)
    print(f"🎞️  [Render] http://{args.host}:{args.port}/  каталог: {root}; "
          f"вивантаження після {args.idle_seconds:.0f} с простою", flush=True)
    uvicorn.run(create_app(engine), host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
