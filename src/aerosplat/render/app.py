"""HTTP face of the render engine (localhost only, like the rest of the platform)."""

from __future__ import annotations

import io
import json
import threading
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from PIL import Image

from aerosplat import __version__
from aerosplat.render.engine import GpuOutOfMemory, RenderEngine, RenderError, SceneNotFound
from aerosplat.render.models import RenderRequest

META_HEADER = "X-Render-Meta"
SWEEP_S = 15.0


def encode_png(image) -> bytes:
    """uint8 (H,W,3) / (H,W) → RGB / L PNG; uint16 (H,W) → 16-bit grayscale PNG."""
    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format="PNG", compress_level=1)
    return buf.getvalue()


def create_app(engine: RenderEngine) -> FastAPI:
    stop = threading.Event()

    def sweeper() -> None:
        while not stop.wait(SWEEP_S):
            engine.unload_idle()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        t = threading.Thread(target=sweeper, name="render-idle-sweeper", daemon=True)
        t.start()
        yield
        stop.set()

    app = FastAPI(title="AeroSplat render service", version=__version__, lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "version": __version__, **engine.status()}

    @app.get("/scenes")
    def scenes() -> list[str]:
        return sorted(k for k, s in engine.scenes(fresh=True).items() if s.splat)

    @app.post("/unload")
    def unload() -> dict[str, str]:
        engine.unload_all()
        return {"status": "unloaded"}

    # `def`, not `async def`: rendering blocks on the GPU, FastAPI runs it in a worker thread
    # and the engine lock serialises the actual GPU work.
    @app.post("/scenes/{scene:path}/render")
    def render(scene: str, req: RenderRequest) -> Response:
        try:
            out = engine.render(scene, req)
        except SceneNotFound as e:
            raise HTTPException(404, str(e)) from None
        except RenderError as e:
            raise HTTPException(422, str(e)) from None
        except GpuOutOfMemory as e:
            raise HTTPException(503, str(e)) from None
        return Response(content=encode_png(out.image), media_type="image/png",
                        headers={META_HEADER: json.dumps(out.meta, separators=(",", ":"))})

    return app
