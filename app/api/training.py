import asyncio
import json
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from app.core.trainer import job_manager
from app.models.schemas import (
    JobStatus,
    LossHistoryResponse,
    LossPoint,
    TrainingConfig,
    TrainingProgress,
    TrainingStartResponse,
)
from app.storage import store

router = APIRouter(prefix="/api/training", tags=["training"])


@router.post("/start", response_model=TrainingStartResponse)
def start_training(config: TrainingConfig) -> TrainingStartResponse:
    persona = store.get_persona(config.persona_id)
    if not persona:
        raise HTTPException(status_code=404, detail="Persona not found")
    if not persona.get("dataset_id"):
        raise HTTPException(status_code=400, detail="Persona has no dataset. Complete the Source step first.")

    job = job_manager.create(
        persona_id=config.persona_id,
        config=config.model_dump(),
    )
    job.status = JobStatus.RUNNING
    job.progress.status = JobStatus.PENDING
    job_manager.start(job)

    return TrainingStartResponse(
        job_id=job.job_id,
        persona_id=config.persona_id,
        status=JobStatus.PENDING,
    )


@router.get("/{job_id}/stream")
async def stream_training(job_id: str) -> EventSourceResponse:
    job = job_manager.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")

    queue: asyncio.Queue[TrainingProgress | None] = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def on_progress(progress: TrainingProgress) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, progress)

    job_manager.subscribe(job_id, on_progress)

    async def event_generator() -> AsyncGenerator[dict, None]:
        try:
            while True:
                progress = await queue.get()
                if progress is None:
                    break
                payload = progress.model_dump(mode="json")
                yield {"event": "progress", "data": json.dumps(payload)}
                if progress.status in (
                    JobStatus.COMPLETED,
                    JobStatus.FAILED,
                    JobStatus.CANCELLED,
                ):
                    break
        finally:
            job_manager.unsubscribe(job_id, on_progress)

    return EventSourceResponse(event_generator())


@router.post("/{job_id}/cancel")
def cancel_training(job_id: str) -> dict:
    job = job_manager.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    if not job_manager.cancel(job_id):
        raise HTTPException(status_code=400, detail="Job cannot be cancelled")
    return {"job_id": job_id, "status": "cancelled"}


@router.get("/{job_id}/status", response_model=TrainingProgress)
def training_status(job_id: str) -> TrainingProgress:
    job = job_manager.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    return job.progress


@router.get("/{job_id}/loss", response_model=LossHistoryResponse)
def training_loss_history(job_id: str) -> LossHistoryResponse:
    job = job_manager.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    points = [
        LossPoint(step=p.step, epoch=p.epoch, loss=p.loss)
        for p in job.loss_history
    ]
    return LossHistoryResponse(job_id=job_id, points=points)
