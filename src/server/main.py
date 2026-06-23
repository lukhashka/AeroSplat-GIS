import io
import os
import time
import torch
from fastapi import FastAPI, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="AeroSplat-GIS Dual Engine", version="2.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

cloud_path = os.path.join(project_dir, "data", "zaporizhzhia_cloud.pt")
splat_path = os.path.join(project_dir, "data", "zaporizhzhia_splat.pt")

GLOBAL_DENSE_BYTES = None
NUM_DENSE_POINTS = 0

GLOBAL_SPLAT_BYTES = None
NUM_SPLAT_POINTS = 0

# 📊 1. СИНХРОНІЗАЦІЯ ВАРІАНТА А (Dense Cloud)
if os.path.exists(cloud_path):
    print(f"📦 [Server] Завантаження щільної хмари (Варіант А)...")
    cloud_data = torch.load(cloud_path, map_location=DEVICE)
    
    # Семплюємо кожну 7-му точку для оптимізації FPS
    raw_xyz = cloud_data['xyz'][::7]
    dense_rgb = cloud_data['rgb'][::7].to(torch.float32)
    
    # 🎨 Нормалізація кольорів під стандарт WebGL (0.0 - 1.0)
    if dense_rgb.max() > 1.0:
        dense_rgb = dense_rgb / 255.0
        
    NUM_DENSE_POINTS = raw_xyz.shape[0]
    
    # 📐 Залізобетонна матриця орієнтації: інвертуємо Z через мінус, щоб підняти будинки догори
    dx = (raw_xyz[:, 0] - raw_xyz[:, 0].mean()) * 20.0
    dy = -(raw_xyz[:, 2] - raw_xyz[:, 2].mean()) * 20.0  # 🔥 ОСЬ ВІН, РЯТІВНИЙ МІНУС!
    dz = (raw_xyz[:, 1] - raw_xyz[:, 1].mean()) * 20.0
    dense_xyz = torch.stack([dx, dy, dz], dim=1).to(torch.float32)
    
    GLOBAL_DENSE_BYTES = torch.cat([dense_xyz, dense_rgb], dim=1).cpu().numpy().tobytes()
    print(f"✅ [Server] Варіант А синхронізовано: {NUM_DENSE_POINTS:,} точок.")

# 🔮 2. СИНХРОНІЗАЦІЯ ВАРІАНТА Б (3D Gaussian Splatting)
if os.path.exists(splat_path):
    print(f"🔮 [Server] Завантаження Гаусіанів (Варіант Б)...")
    splat_data = torch.load(splat_path, map_location=DEVICE)
    
    s_xyz = splat_data['xyz'][::7].to(torch.float32)
    s_scale = splat_data['scale'][::7].to(torch.float32)
    s_rot = splat_data['rotation'][::7].to(torch.float32)
    s_opac = splat_data['opacity'][::7].to(torch.float32)
    s_rgb = splat_data['rgb'][::7].to(torch.float32)
    
    # 🎨 Нормалізація кольорів Гаусіанів
    if s_rgb.max() > 1.0:
        s_rgb = s_rgb / 255.0
        
    NUM_SPLAT_POINTS = s_xyz.shape[0]
    
    # 📐 ПРИМУСОВО застосовуємо ТОЧНО ТАКУ САМУ трансформацію, щоб моделі збігалися до міліметра!
    sx = (s_xyz[:, 0] - s_xyz[:, 0].mean()) * 20.0
    sy = -(s_xyz[:, 2] - s_xyz[:, 2].mean()) * 20.0  # 🔥 ТОЙ САМИЙ МІНУС ДЛЯ ОРБІТИ
    sz = (s_xyz[:, 1] - s_xyz[:, 1].mean()) * 20.0
    splat_xyz = torch.stack([sx, sy, sz], dim=1)
    
    GLOBAL_SPLAT_BYTES = torch.cat([splat_xyz, s_scale, s_rot, s_opac, s_rgb], dim=1).cpu().numpy().tobytes()
    print(f"✅ [Server] Варіант Б синхронізовано: {NUM_SPLAT_POINTS:,} Гаусіанів.")


@app.get("/api/v1/spatial-chunk")
async def stream_spatial_chunk(mode: str = Query("splat")):
    start_time = time.time()
    
    if mode == "dense":
        if GLOBAL_DENSE_BYTES is None:
            return Response(status_code=404, content="Dense data not loaded")
        data_to_send = GLOBAL_DENSE_BYTES
        points_count = NUM_DENSE_POINTS
    else:
        if GLOBAL_SPLAT_BYTES is None:
            return Response(status_code=404, content="Splat data not loaded")
        data_to_send = GLOBAL_SPLAT_BYTES
        points_count = NUM_SPLAT_POINTS
        
    process_time = (time.time() - start_time) * 1000
    
    return Response(
        content=data_to_send,
        media_type="application/octet-stream",
        headers={
            "X-Processing-Time-Ms": f"{process_time:.2f}",
            "X-Gaussians-Count": str(points_count),
            "Access-Control-Expose-Headers": "X-Processing-Time-Ms, X-Gaussians-Count"
        }
    )

app.mount("/web", StaticFiles(directory=os.path.join(project_dir, "web"), html=True), name="web")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)