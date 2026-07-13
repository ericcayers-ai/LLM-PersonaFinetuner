from fastapi import APIRouter

from app.core.inference import inference_engine
from app.errors import api_error
from app.models.schemas import ChatRequest, ChatResponse
from app.storage import store

router = APIRouter(prefix="/api/inference", tags=["inference"])


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    persona = store.get_persona(req.persona_id)
    if not persona:
        raise api_error(404, "persona_not_found", "Persona not found.")

    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    if not messages:
        raise api_error(400, "no_messages", "At least one message is required.")

    try:
        result = inference_engine.chat(
            persona_id=req.persona_id,
            messages=messages,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            compare_base=req.compare_base,
        )
    except RuntimeError as e:
        raise api_error(503, "inference_unavailable", str(e), hint="Train a persona first or check GPU availability.") from e
    except ValueError as e:
        raise api_error(400, "inference_error", str(e)) from e
    except Exception as e:
        raise api_error(500, "inference_failed", f"Inference failed: {e}") from e

    return ChatResponse(
        persona_id=req.persona_id,
        response=result["response"],
        base_response=result.get("base_response"),
    )
