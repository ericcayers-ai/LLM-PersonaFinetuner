"""Unsloth LoRA trainer with background jobs and SSE progress."""

from __future__ import annotations

import threading
import time
import traceback
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from app.config import settings
from app.core.gpu import require_cuda
from app.models.schemas import JobStatus, TrainingProgress
from app.storage import store


@dataclass
class LossPoint:
    step: int
    epoch: int
    loss: float


@dataclass
class TrainingJob:
    job_id: str
    persona_id: str
    config: dict[str, Any]
    status: JobStatus = JobStatus.PENDING
    progress: TrainingProgress = field(default_factory=lambda: TrainingProgress(
        job_id="", persona_id="", status=JobStatus.PENDING
    ))
    cancel_requested: bool = False
    thread: Optional[threading.Thread] = None
    listeners: list[Callable[[TrainingProgress], None]] = field(default_factory=list)
    loss_history: list[LossPoint] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        self.progress.job_id = self.job_id
        self.progress.persona_id = self.persona_id


class JobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, TrainingJob] = {}
        self._lock = threading.Lock()

    def get(self, job_id: str) -> Optional[TrainingJob]:
        with self._lock:
            return self._jobs.get(job_id)

    def create(self, persona_id: str, config: dict[str, Any]) -> TrainingJob:
        job_id = str(uuid.uuid4())
        job = TrainingJob(job_id=job_id, persona_id=persona_id, config=config)
        with self._lock:
            self._jobs[job_id] = job
        return job

    def subscribe(self, job_id: str, callback: Callable[[TrainingProgress], None]) -> bool:
        job = self.get(job_id)
        if not job:
            return False
        with job._lock:
            job.listeners.append(callback)
            callback(deepcopy(job.progress))
        return True

    def unsubscribe(self, job_id: str, callback: Callable[[TrainingProgress], None]) -> None:
        job = self.get(job_id)
        if job:
            with job._lock:
                if callback in job.listeners:
                    job.listeners.remove(callback)

    def _notify(self, job: TrainingJob) -> None:
        snapshot = deepcopy(job.progress)
        with job._lock:
            listeners = list(job.listeners)
        for cb in listeners:
            try:
                cb(snapshot)
            except Exception:
                pass

    def _update(self, job: TrainingJob, **kwargs: Any) -> None:
        with job._lock:
            for k, v in kwargs.items():
                setattr(job.progress, k, v)
        self._notify(job)

    def start(self, job: TrainingJob) -> None:
        thread = threading.Thread(target=self._run_job, args=(job,), daemon=True)
        job.thread = thread
        thread.start()

    def cancel(self, job_id: str) -> bool:
        job = self.get(job_id)
        if not job:
            return False
        if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
            return False
        job.cancel_requested = True
        self._update(job, status=JobStatus.CANCELLED, message="Cancellation requested…")
        return True

    def _run_job(self, job: TrainingJob) -> None:
        try:
            require_cuda()
            self._update(job, status=JobStatus.RUNNING, message="Loading model…")
            store.update_persona(job.persona_id, status="training")
            self._train(job)
            if job.cancel_requested:
                self._update(job, status=JobStatus.CANCELLED, message="Training cancelled.")
                store.update_persona(job.persona_id, status="draft")
                return
            adapter_path = str(store.adapter_dir_for(job.persona_id))
            store.update_persona(
                job.persona_id,
                status="trained",
                adapter_path=adapter_path,
            )
            self._update(
                job,
                status=JobStatus.COMPLETED,
                message="Training complete. Adapter saved.",
            )
        except Exception as e:
            tb = traceback.format_exc()
            store.update_persona(job.persona_id, status="failed")
            self._update(
                job,
                status=JobStatus.FAILED,
                error=str(e),
                message=f"Training failed: {e}",
            )
            print(f"Training job {job.job_id} failed:\n{tb}")

    def _train(self, job: TrainingJob) -> None:
        # Lazy imports — app starts without GPU/torch
        from datasets import load_dataset
        from trl import SFTTrainer
        from trl.trainer.sft_config import SFTConfig

        from unsloth import FastLanguageModel

        cfg = job.config
        persona = store.get_persona(job.persona_id)
        if not persona:
            raise ValueError(f"Persona {job.persona_id} not found")

        dataset_id = persona.get("dataset_id")
        if not dataset_id:
            raise ValueError("Persona has no dataset. Build a dataset first.")

        dataset_meta = store.get_dataset(dataset_id)
        if not dataset_meta:
            raise ValueError(f"Dataset {dataset_id} not found")

        train_path = dataset_meta["train_path"]
        base_model = cfg.get("base_model") or persona.get("base_model") or settings.default_base_model
        max_seq_length = cfg.get("max_seq_length", settings.default_max_seq_length)
        epochs = cfg.get("epochs", settings.default_epochs)
        learning_rate = cfg.get("learning_rate", settings.default_learning_rate)
        batch_size = cfg.get("batch_size", settings.default_batch_size)
        lora_r = cfg.get("lora_r", settings.default_lora_r)
        lora_alpha = cfg.get("lora_alpha", settings.default_lora_alpha)
        grad_accum = cfg.get("gradient_accumulation_steps", settings.default_gradient_accumulation_steps)

        adapter_dir = store.adapter_dir_for(job.persona_id)
        adapter_dir.mkdir(parents=True, exist_ok=True)

        self._update(job, message=f"Loading {base_model}…", total_epochs=epochs)

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=True,
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_r,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_alpha=lora_alpha,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )

        ds = load_dataset("json", data_files={"train": train_path}, split="train")

        def formatting_func(examples: dict) -> list[str]:
            texts = []
            for messages in examples["messages"]:
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                texts.append(text)
            return texts

        total_steps = max(1, (len(ds) // max(batch_size, 1)) * epochs)
        self._update(job, total_steps=total_steps, step=0, epoch=0)

        start_time = time.time()
        step_counter = {"n": 0}

        from transformers import TrainerCallback

        class ProgressCallback(TrainerCallback):
            def on_log(self, args: Any, state: Any, control: Any, logs: Optional[dict] = None, **kwargs: Any) -> None:
                if job.cancel_requested:
                    control.should_training_stop = True
                    return
                if not logs:
                    return
                step_counter["n"] += 1
                loss = logs.get("loss")
                epoch = int(state.epoch) if state.epoch else 0
                elapsed = time.time() - start_time
                steps_done = step_counter["n"]
                eta = None
                if steps_done > 0 and total_steps > steps_done:
                    eta = (elapsed / steps_done) * (total_steps - steps_done)
                rounded_loss = round(loss, 4) if loss is not None else None
                if rounded_loss is not None:
                    with job._lock:
                        job.loss_history.append(
                            LossPoint(step=steps_done, epoch=epoch, loss=rounded_loss)
                        )
                job_manager._update(
                    job,
                    step=steps_done,
                    epoch=epoch,
                    loss=rounded_loss,
                    eta_seconds=round(eta, 1) if eta else None,
                    message=f"Epoch {epoch}/{epochs} — step {steps_done}/{total_steps}",
                )

        progress_cb = ProgressCallback()

        sft_config = SFTConfig(
            output_dir=str(adapter_dir),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            learning_rate=learning_rate,
            logging_steps=1,
            save_strategy="epoch",
            fp16=False,
            bf16=True,
            optim="adamw_8bit",
            warmup_ratio=0.05,
            max_seq_length=max_seq_length,
            dataset_text_field=None,
            report_to="none",
        )

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=ds,
            formatting_func=formatting_func,
            args=sft_config,
            callbacks=[progress_cb],
        )

        if job.cancel_requested:
            return

        trainer.train()
        model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))


job_manager = JobManager()
