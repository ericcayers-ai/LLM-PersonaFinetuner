from fastapi import APIRouter, HTTPException

from app.core.inference import inference_engine
from app.models.schemas import ChatRequest, ChatResponse
from app.storage import store

router = APIRouter(prefix="/api/inference", tags=["inference"])


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    persona = store.get_persona(req.persona_id)
    if not persona:
        raise HTTPException(status_code=404, detail="Persona not found")

    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    if not messages:
        raise HTTPException(status_code=400, detail="At least one message is required")

    try:
        result = inference_engine.chat(
            persona_id=req.persona_id,
            messages=messages,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            compare_base=req.compare_base,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}") from e

    return ChatResponse(
        persona_id=req.persona_id,
        response=result["response"],
        base_response=result.get("base_response"),
    )
