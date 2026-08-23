from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api import datasets, inference, personas, system, training, voice
from app.config import ensure_dirs, settings
from app.errors import http_exception_handler, validation_exception_handler

ensure_dirs()

app = FastAPI(title=settings.app_name, version=settings.app_version)

app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)

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
app.include_router(voice.router)

frontend_path = settings.frontend_dir
if frontend_path.exists():
    app.mount("/css", StaticFiles(directory=str(frontend_path / "css")), name="css")
    app.mount("/js", StaticFiles(directory=str(frontend_path / "js")), name="js")


@app.get("/")
async def serve_index():
    index = frontend_path / "index.html"
    if index.exists():
        return FileResponse(index)
    return {
        "error": "frontend_missing",
        "message": "PersonaFinetuner API is running but the frontend was not found.",
        "hint": "Ensure the frontend/ directory exists next to app/.",
    }


@app.get("/api/health")
async def api_health():
    from app.api.system import health

    return health()
