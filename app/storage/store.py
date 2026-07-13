"""JSON-backed metadata and file storage."""

import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from app.config import settings
from app.models.schemas import PersonaStatus


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def _meta_path(kind: str, item_id: str) -> Path:
    return settings.data_dir / kind / f"{item_id}.json"


# --- Uploads ---


def save_upload(filename: str, content: bytes) -> dict:
    upload_id = str(uuid.uuid4())
    ext = Path(filename).suffix.lower()
    dest = settings.uploads_dir / f"{upload_id}{ext}"
    dest.write_bytes(content)
    record = {
        "id": upload_id,
        "filename": filename,
        "path": str(dest),
        "file_type": ext.lstrip("."),
        "size_bytes": len(content),
        "uploaded_at": _now().isoformat(),
    }
    _write_json(_upload_meta_path(upload_id), record)
    return record


def _upload_meta_path(upload_id: str) -> Path:
    return settings.uploads_dir / "_meta" / f"{upload_id}.json"


def get_upload(upload_id: str) -> Optional[dict]:
    return _read_json(_upload_meta_path(upload_id))


def update_upload_meta(upload_id: str, **fields: Any) -> Optional[dict]:
    record = get_upload(upload_id)
    if not record:
        return None
    record.update(fields)
    _write_json(_upload_meta_path(upload_id), record)
    return record


# --- Datasets ---


def save_dataset(
    persona_id: str,
    train_path: Path,
    val_path: Path,
    meta: dict,
) -> dict:
    dataset_id = str(uuid.uuid4())
    record = {
        "id": dataset_id,
        "persona_id": persona_id,
        "train_path": str(train_path),
        "val_path": str(val_path),
        "created_at": _now().isoformat(),
        **meta,
    }
    _write_json(_meta_path("datasets", dataset_id), record)
    return record


def get_dataset(dataset_id: str) -> Optional[dict]:
    return _read_json(_meta_path("datasets", dataset_id))


def read_jsonl_preview(path: str, limit: int = 10) -> list[dict]:
    examples: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))
            if len(examples) >= limit:
                break
    return examples


# --- Personas ---


def create_persona(
    name: str,
    system_prompt: str,
    dataset_id: Optional[str] = None,
    base_model: Optional[str] = None,
) -> dict:
    persona_id = str(uuid.uuid4())
    now = _now().isoformat()
    record = {
        "id": persona_id,
        "name": name,
        "system_prompt": system_prompt,
        "dataset_id": dataset_id,
        "base_model": base_model or settings.default_base_model,
        "status": PersonaStatus.DRAFT.value,
        "adapter_path": None,
        "created_at": now,
        "updated_at": now,
    }
    _write_json(_meta_path("personas", persona_id), record)
    return record


def list_personas() -> list[dict]:
    personas_dir = settings.data_dir / "personas"
    if not personas_dir.exists():
        return []
    records = []
    for path in sorted(personas_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        data = _read_json(path)
        if data:
            records.append(data)
    return records


def get_persona(persona_id: str) -> Optional[dict]:
    return _read_json(_meta_path("personas", persona_id))


def update_persona(persona_id: str, **fields: Any) -> Optional[dict]:
    record = get_persona(persona_id)
    if not record:
        return None
    record.update(fields)
    record["updated_at"] = _now().isoformat()
    _write_json(_meta_path("personas", persona_id), record)
    return record


def adapter_dir_for(persona_id: str) -> Path:
    return settings.adapters_dir / persona_id


def delete_persona(persona_id: str) -> bool:
    path = _meta_path("personas", persona_id)
    if not path.exists():
        return False
    path.unlink()
    adapter_dir = adapter_dir_for(persona_id)
    if adapter_dir.exists():
        shutil.rmtree(adapter_dir)
    return True
