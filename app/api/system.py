from fastapi import APIRouter

from app.core.gpu import get_gpu_info
from app.models.schemas import GPUInfo, HealthResponse
from app.config import settings

router = APIRouter(prefix="/api/system", tags=["system"])


@router.get("/gpu", response_model=GPUInfo)
def gpu_status() -> GPUInfo:
    return get_gpu_info()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        app_name=settings.app_name,
        available_models=settings.available_models,
    )
