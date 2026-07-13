"""CUDA / GPU availability checks."""

from typing import Optional

from app.models.schemas import GPUInfo

# VRAM recommendations by model size class
VRAM_REQUIREMENTS = {
    "3b": 6.0,
    "7b": 10.0,
    "default": 8.0,
}


def _model_vram_requirement(model_name: str) -> float:
    lower = model_name.lower()
    if "3b" in lower or "1b" in lower or "1.5b" in lower:
        return VRAM_REQUIREMENTS["3b"]
    if "7b" in lower or "8b" in lower:
        return VRAM_REQUIREMENTS["7b"]
    return VRAM_REQUIREMENTS["default"]


def get_gpu_info(model_name: Optional[str] = None) -> GPUInfo:
    try:
        import torch
    except ImportError:
        return GPUInfo(
            cuda_available=False,
            warning="PyTorch is not installed. Install torch with CUDA support to train locally.",
        )

    if not torch.cuda.is_available():
        return GPUInfo(
            cuda_available=False,
            warning="No CUDA device found. This app trains locally on an NVIDIA GPU.",
        )

    try:
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        total_gb = props.total_memory / (1024 ** 3)
        free_bytes, _ = torch.cuda.mem_get_info(device)
        free_gb = free_bytes / (1024 ** 3)
        name = props.name

        warning = None
        if model_name:
            required = _model_vram_requirement(model_name)
            if total_gb < required:
                warning = (
                    f"GPU has {total_gb:.1f} GB VRAM; "
                    f"{required:.0f} GB recommended for {model_name}. "
                    "Training may run out of memory — try a smaller model or lower batch size."
                )

        return GPUInfo(
            cuda_available=True,
            device_name=name,
            vram_total_gb=round(total_gb, 2),
            vram_free_gb=round(free_gb, 2),
            warning=warning,
        )
    except Exception as e:
        return GPUInfo(
            cuda_available=False,
            warning=f"Could not query GPU: {e}",
        )


def require_cuda() -> None:
    info = get_gpu_info()
    if not info.cuda_available:
        raise RuntimeError(
            info.warning or "No CUDA device found. This app trains locally on an NVIDIA GPU."
        )
