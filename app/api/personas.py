from fastapi import APIRouter, HTTPException

from app.core.gguf_export import export_for_ollama
from app.models.schemas import ExportResponse, PersonaCreate, PersonaResponse, PersonaStatus
from app.storage import store

router = APIRouter(prefix="/api/personas", tags=["personas"])


def _to_response(record: dict) -> PersonaResponse:
    return PersonaResponse(
        id=record["id"],
        name=record["name"],
        system_prompt=record["system_prompt"],
        dataset_id=record.get("dataset_id"),
        base_model=record.get("base_model", ""),
        status=PersonaStatus(record.get("status", "draft")),
        adapter_path=record.get("adapter_path"),
        created_at=record["created_at"],
        updated_at=record["updated_at"],
    )


@router.post("", response_model=PersonaResponse)
def create_persona(req: PersonaCreate) -> PersonaResponse:
    if not req.name.strip():
        raise HTTPException(status_code=400, detail="Persona name is required")
    if not req.system_prompt.strip():
        raise HTTPException(status_code=400, detail="System prompt is required")

    record = store.create_persona(
        name=req.name.strip(),
        system_prompt=req.system_prompt.strip(),
        dataset_id=req.dataset_id,
        base_model=req.base_model,
    )
    return _to_response(record)


@router.get("", response_model=list[PersonaResponse])
def list_personas() -> list[PersonaResponse]:
    return [_to_response(r) for r in store.list_personas()]


@router.get("/{persona_id}", response_model=PersonaResponse)
def get_persona(persona_id: str) -> PersonaResponse:
    record = store.get_persona(persona_id)
    if not record:
        raise HTTPException(status_code=404, detail="Persona not found")
    return _to_response(record)


@router.post("/{persona_id}/export-gguf", response_model=ExportResponse)
def export_persona_gguf(persona_id: str) -> ExportResponse:
    try:
        result = export_for_ollama(persona_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {e}") from e

    return ExportResponse(
        persona_id=result["persona_id"],
        status=result["status"],
        message=result["message"],
        export_dir=result["export_dir"],
        gguf_path=result.get("gguf_path"),
        limitations=result.get("limitations", []),
        manual_steps=result.get("manual_steps", []),
        ollama_command=result.get("ollama_command"),
    )
