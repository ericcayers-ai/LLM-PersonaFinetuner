"""Optional synthetic Q&A generation from writing samples using a base model."""

from __future__ import annotations

import random
from typing import Any

from app.config import settings
from app.core.gpu import get_gpu_info

FALLBACK_QUESTIONS = [
    "What is your perspective on this topic?",
    "How would you explain this to a friend?",
    "What stands out to you about this?",
    "Can you share your thoughts in your own words?",
    "What would you say about this situation?",
]


def _fallback_qa_from_text(text: str, count: int = 3) -> list[dict[str, Any]]:
    """Template-based Q&A when GPU/model generation is unavailable."""
    sentences = [s.strip() for s in text.replace("\n", " ").split(".") if len(s.strip()) > 40]
    if not sentences:
        sentences = [text[:500]]
    rng = random.Random(42)
    examples = []
    for i in range(min(count, len(sentences))):
        answer = sentences[i % len(sentences)].strip()
        if not answer.endswith("."):
            answer += "."
        question = FALLBACK_QUESTIONS[i % len(FALLBACK_QUESTIONS)]
        examples.append({
            "type": "chat",
            "messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ],
        })
    return examples


def generate_synthetic_qa(
    text_samples: list[str],
    base_model: str | None = None,
    max_pairs: int = 10,
) -> tuple[list[dict[str, Any]], list[str]]:
    """
    Generate Q&A chat pairs from writing samples.
    Uses the base model when CUDA is available; otherwise falls back to templates.
    """
    warnings: list[str] = []
    if not text_samples:
        return [], ["No text samples available for synthetic Q&A generation."]

    combined = "\n\n".join(t.strip() for t in text_samples if t.strip())[:4000]
    gpu = get_gpu_info(base_model)

    if not gpu.cuda_available:
        warnings.append(
            "No CUDA GPU — using template-based synthetic Q&A instead of model generation."
        )
        pairs = _fallback_qa_from_text(combined, count=min(max_pairs, 5))
        return pairs, warnings

    model_name = base_model or settings.default_base_model
    try:
        pairs = _generate_with_model(combined, model_name, max_pairs)
        warnings.append(f"Generated {len(pairs)} synthetic Q&A pair(s) with {model_name}.")
        return pairs, warnings
    except Exception as e:
        warnings.append(f"Model Q&A generation failed ({e}); using template fallback.")
        return _fallback_qa_from_text(combined, count=min(max_pairs, 5)), warnings


def _generate_with_model(text: str, model_name: str, max_pairs: int) -> list[dict[str, Any]]:
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    prompt = (
        "Read the following writing sample. Generate "
        f"{min(max_pairs, 5)} diverse question-and-answer pairs where the answers "
        "sound like the same author. Format each pair as:\n"
        "Q: <question>\nA: <answer>\n\n"
        f"Writing sample:\n{text[:3000]}\n\nPairs:"
    )

    import torch

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.8,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    generated = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return _parse_qa_pairs(generated)


def _parse_qa_pairs(text: str) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    blocks = text.split("\nQ:")
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        if block.startswith("Q:"):
            block = block[2:].strip()
        if "\nA:" not in block and "A:" not in block:
            continue
        if "\nA:" in block:
            q, a = block.split("\nA:", 1)
        else:
            q, a = block.split("A:", 1)
        q, a = q.strip(), a.strip()
        if q and a:
            pairs.append({
                "type": "chat",
                "messages": [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": a},
                ],
            })
    return pairs
