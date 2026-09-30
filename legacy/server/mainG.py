import io
import os
import time
import torch
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="AeroSplat-3DGS Stream Engine", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
splat_path = os.path.join(project_dir, "data", "zaporizhzhia_splat.pt")

# --- Глобальне завантаження НАВЧЕНИХ Гаусіанів у VRAM ---
if os.path.exists(splat_path):
    print(f"🔮 [Server] Завантаження НАВЧЕНОЇ 3DGS-сцени з {splat_path} у VRAM...")
    splat_data = torch.load(splat_path, map_location=DEVICE)
    
    # Витягуємо всі 5 нейромережевих параметрів
    xyz = splat_data['xyz'].to(torch.float32)
    scale = splat_data['scale'].to(torch.float32)
    rotation = splat_data['rotation'].to(torch.float32)
    opacity = splat_data['opacity'].to(torch.float32)
    rgb = splat_data['rgb'].to(torch.float32)
    
    NUM_GAUSSIANS = xyz.shape[0]
    
    # Склеюємо все в один гігантський матричний тензор продуктивності [N, 14]
    # Порядок: XYZ (3) + Scale (3) + Rotation (4) + Opacity (1) + RGB (3) = 14 флотів
    GLOBAL_SPLAT_DATA = torch.cat([xyz, scale, rotation, opacity, rgb], dim=1).cpu()
    print(f"✅ [Server] 3DGS Модель готова до стрімінгу. Активно: {NUM_GAUSSIANS:,} Гаусіанів!")
else:
    print("❌ Помилка: Файл zaporizhzhia_splat.pt не знайдено! Запусти спочатку gs_trainer.py")
    exit(1)


@app.get("/api/v1/spatial-chunk")
async def stream_splat_chunk(lat: float = Query(...), lon: float = Query(...), radius: float = Query(...)):
    start_time = time.time()
    
    # Вивантажуємо всю упаковану матрицю Гаусіанів у сирі байти
    binary_data = GLOBAL_SPLAT_DATA.numpy().tobytes()
    buffer = io.BytesIO(binary_data)
    
    process_time = (time.time() - start_time) * 1000
    print(f"🛸 Стрімінг 3DGS: відправлено {NUM_GAUSSIANS:,} об'ємних Гаусіанів | {process_time:.2f} ms")
    
    return StreamingResponse(
        buffer,
        media_type="application/octet-stream",
        headers={
            "X-Processing-Time-Ms": f"{process_time:.2f}",
            "X-Gaussians-Count": str(NUM_GAUSSIANS),
            "Access-Control-Expose-Headers": "X-Processing-Time-Ms, X-Gaussians-Count"
        }
    )

app.mount("/web", StaticFiles(directory=os.path.join(project_dir, "web"), html=True), name="web")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)