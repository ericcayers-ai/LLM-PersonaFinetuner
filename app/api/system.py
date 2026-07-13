from fastapi import APIRouter

from app.core.gpu import get_gpu_info
from app.core.parsers import SUPPORTED_EXTENSIONS
from app.models.schemas import GPUInfo, HealthResponse
from app.config import settings

router = APIRouter(prefix="/api/system", tags=["system"])


def _data_dirs_ok() -> bool:
    for path in (
        settings.data_dir,
        settings.uploads_dir,
        settings.outputs_dir,
    ):
        try:
            path.mkdir(parents=True, exist_ok=True)
            test = path / ".write_test"
            test.write_text("ok", encoding="utf-8")
            test.unlink()
        except OSError:
            return False
    return True


@router.get("/gpu", response_model=GPUInfo)
def gpu_status() -> GPUInfo:
    return get_gpu_info()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    gpu = get_gpu_info()
    return HealthResponse(
        status="ok",
        app_name=settings.app_name,
        version=settings.app_version,
        available_models=settings.available_models,
        cuda_available=gpu.cuda_available,
        gpu_name=gpu.device_name,
        data_dirs_ok=_data_dirs_ok(),
        supported_formats=sorted(SUPPORTED_EXTENSIONS),
    )
