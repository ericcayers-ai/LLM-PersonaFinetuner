from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PF_", env_file=".env", extra="ignore")

    app_name: str = "PersonaFinetuner"
    allowed_origins: list[str] = [
        "http://127.0.0.1:8000",
        "http://localhost:8000",
    ]
    data_dir: Path = Path("data")
    uploads_dir: Path = Path("data/uploads")
    personas_dir: Path = Path("data/personas")
    datasets_dir: Path = Path("data/datasets")
    outputs_dir: Path = Path("outputs")
    adapters_dir: Path = Path("outputs/adapters")
    voices_dir: Path = Path("data/voices")

    default_base_model: str = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
    available_models: list[str] = [
        "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
        "unsloth/Mistral-7B-Instruct-v0.3-bnb-4bit",
        "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    ]

    default_lora_r: int = 16
    default_lora_alpha: int = 32
    default_epochs: int = 3
    default_learning_rate: float = 2e-4
    default_batch_size: int = 2
    default_max_seq_length: int = 2048
    default_gradient_accumulation_steps: int = 4

    chunk_min_tokens: int = 512
    chunk_max_tokens: int = 1024
    chunk_overlap_tokens: int = 64
    train_val_split: float = 0.9
    min_samples_warning: int = 20

    frontend_dir: Path = Path("frontend")
    app_version: str = "1.2.0"
    max_upload_bytes: int = 50 * 1024 * 1024  # 50 MB
    max_voice_upload_bytes: int = 25 * 1024 * 1024  # 25 MB
    max_call_utterance_bytes: int = 12 * 1024 * 1024  # 12 MB
    max_tts_characters: int = 1200
    stt_model: str = "turbo"
    stt_device: str = "auto"
    stt_compute_type: str = "auto"
    tts_model: str = "turbo"
    tts_device: str = "auto"


settings = Settings()


def ensure_dirs() -> None:
    for path in (
        settings.data_dir,
        settings.uploads_dir,
        settings.personas_dir,
        settings.datasets_dir,
        settings.outputs_dir,
        settings.adapters_dir,
        settings.voices_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)
