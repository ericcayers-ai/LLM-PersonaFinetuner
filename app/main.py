from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api import datasets, inference, personas, system, training
from app.config import ensure_dirs, settings

ensure_dirs()

app = FastAPI(title=settings.app_name, version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(system.router)
app.include_router(datasets.router)
app.include_router(personas.router)
app.include_router(training.router)
app.include_router(inference.router)

frontend_path = settings.frontend_dir
if frontend_path.exists():
    app.mount("/css", StaticFiles(directory=str(frontend_path / "css")), name="css")
    app.mount("/js", StaticFiles(directory=str(frontend_path / "js")), name="js")


@app.get("/")
async def serve_index():
    index = frontend_path / "index.html"
    if index.exists():
        return FileResponse(index)
    return {"message": "PersonaFinetuner API is running. Frontend not found."}


@app.get("/api/health")
async def api_health():
    return {"status": "ok", "app": settings.app_name}
