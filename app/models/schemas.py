from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class PersonaStatus(str, Enum):
    DRAFT = "draft"
    TRAINING = "training"
    TRAINED = "trained"
    FAILED = "failed"


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class GPUInfo(BaseModel):
    cuda_available: bool
    device_name: Optional[str] = None
    vram_total_gb: Optional[float] = None
    vram_free_gb: Optional[float] = None
    warning: Optional[str] = None


class UploadResponse(BaseModel):
    upload_id: str
    filename: str
    file_type: str
    size_bytes: int
    detected_format: str
    format_label: str
    requires_target_name: bool = False
    sample_hint: str = ""


class DatasetBuildRequest(BaseModel):
    upload_ids: list[str]
    persona_name: str
    system_prompt: str
    chunk_min_tokens: int = 512
    chunk_max_tokens: int = 1024
    target_name: Optional[str] = None
    generate_synthetic_qa: bool = False
    synthetic_base_model: Optional[str] = None


class DatasetPreviewExample(BaseModel):
    messages: list[dict[str, str]]


class DatasetPreviewResponse(BaseModel):
    dataset_id: str
    persona_id: str
    total_examples: int
    train_examples: int
    val_examples: int
    warnings: list[str] = Field(default_factory=list)
    examples: list[DatasetPreviewExample]


class PersonaCreate(BaseModel):
    name: str
    system_prompt: str
    dataset_id: Optional[str] = None
    base_model: Optional[str] = None


class PersonaResponse(BaseModel):
    id: str
    name: str
    system_prompt: str
    dataset_id: Optional[str] = None
    base_model: str
    status: PersonaStatus
    adapter_path: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class TrainingConfig(BaseModel):
    persona_id: str
    base_model: Optional[str] = None
    epochs: int = 3
    learning_rate: float = 2e-4
    batch_size: int = 2
    max_seq_length: int = 2048
    lora_r: int = 16
    lora_alpha: int = 32
    gradient_accumulation_steps: int = 4


class TrainingStartResponse(BaseModel):
    job_id: str
    persona_id: str
    status: JobStatus


class TrainingProgress(BaseModel):
    job_id: str
    persona_id: str
    status: JobStatus
    epoch: int = 0
    total_epochs: int = 0
    step: int = 0
    total_steps: int = 0
    loss: Optional[float] = None
    eta_seconds: Optional[float] = None
    message: str = ""
    error: Optional[str] = None


class LossPoint(BaseModel):
    step: int
    epoch: int
    loss: float


class LossHistoryResponse(BaseModel):
    job_id: str
    points: list[LossPoint]


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    persona_id: str
    messages: list[ChatMessage]
    max_tokens: int = 512
    temperature: float = 0.7
    compare_base: bool = False


class ChatResponse(BaseModel):
    persona_id: str
    response: str
    base_response: Optional[str] = None


class ExportResponse(BaseModel):
    persona_id: str
    status: str
    message: str
    export_dir: str
    gguf_path: Optional[str] = None
    limitations: list[str] = Field(default_factory=list)
    manual_steps: list[str] = Field(default_factory=list)
    ollama_command: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    app_name: str
    available_models: list[str]
