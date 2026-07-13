"""Build SFT chat JSONL from raw text and chat examples."""

import json
import random
import re
from pathlib import Path
from typing import Any

from app.config import settings

VARIED_USER_PROMPTS = [
    "Continue writing in this person's voice and style.",
    "Reply as this person would.",
    "Write a short message in this voice.",
    "Express the following idea the way this person typically would.",
    "Draft a paragraph that sounds like this author.",
    "How would this person phrase their thoughts on this topic?",
    "Capture this person's tone in a few sentences.",
    "Write naturally, as this person would speak.",
]


def _estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars per token for English)."""
    return max(1, len(text) // 4)


def _split_text_into_chunks(
    text: str,
    min_tokens: int = 512,
    max_tokens: int = 1024,
    overlap_tokens: int = 64,
) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        paragraphs = [text.strip()]

    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        candidate = f"{current}\n\n{para}".strip() if current else para
        if _estimate_tokens(candidate) <= max_tokens:
            current = candidate
            continue

        if current and _estimate_tokens(current) >= min_tokens:
            chunks.append(current)
            overlap = _tail_tokens(current, overlap_tokens)
            current = f"{overlap}\n\n{para}".strip() if overlap else para
        else:
            if _estimate_tokens(para) > max_tokens:
                sub_chunks = _split_long_paragraph(para, max_tokens, overlap_tokens)
                if current:
                    chunks.append(current)
                    current = ""
                chunks.extend(sub_chunks[:-1])
                current = sub_chunks[-1] if sub_chunks else ""
            else:
                current = candidate

    if current.strip():
        chunks.append(current.strip())

    return [c for c in chunks if c.strip()]


def _tail_tokens(text: str, token_count: int) -> str:
    words = text.split()
    approx_words = token_count * 4 // 5  # rough
    return " ".join(words[-approx_words:]) if approx_words < len(words) else text


def _split_long_paragraph(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current = ""
    for sent in sentences:
        candidate = f"{current} {sent}".strip() if current else sent
        if _estimate_tokens(candidate) <= max_tokens:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = sent
    if current:
        chunks.append(current)
    return chunks


def _to_sft_example(system_prompt: str, user_prompt: str, assistant_content: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_content},
        ]
    }


def build_sft_examples(
    parsed_items: list[dict[str, Any]],
    system_prompt: str,
    chunk_min_tokens: int | None = None,
    chunk_max_tokens: int | None = None,
) -> list[dict]:
    min_t = chunk_min_tokens or settings.chunk_min_tokens
    max_t = chunk_max_tokens or settings.chunk_max_tokens
    overlap = settings.chunk_overlap_tokens

    examples: list[dict] = []
    rng = random.Random(42)

    for item in parsed_items:
        if item.get("type") == "chat":
            messages = item["messages"]
            assistant_parts = [m["content"] for m in messages if m.get("role") == "assistant"]
            user_parts = [m["content"] for m in messages if m.get("role") == "user"]
            assistant_text = "\n".join(assistant_parts).strip()
            user_text = "\n".join(user_parts).strip() or rng.choice(VARIED_USER_PROMPTS)
            if assistant_text:
                examples.append(_to_sft_example(system_prompt, user_text, assistant_text))
        elif item.get("type") == "text":
            chunks = _split_text_into_chunks(
                item["content"], min_tokens=min_t, max_tokens=max_t, overlap_tokens=overlap
            )
            for chunk in chunks:
                user_prompt = rng.choice(VARIED_USER_PROMPTS)
                examples.append(_to_sft_example(system_prompt, user_prompt, chunk))

    rng.shuffle(examples)
    return examples


def split_train_val(examples: list[dict], train_ratio: float | None = None) -> tuple[list[dict], list[dict]]:
    ratio = train_ratio if train_ratio is not None else settings.train_val_split
    if len(examples) <= 1:
        return examples, []
    split_idx = max(1, int(len(examples) * ratio))
    train = examples[:split_idx]
    val = examples[split_idx:]
    return train, val


def write_jsonl(examples: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def build_dataset_files(
    parsed_items: list[dict[str, Any]],
    system_prompt: str,
    output_dir: Path,
    chunk_min_tokens: int = 512,
    chunk_max_tokens: int = 1024,
) -> tuple[Path, Path, dict]:
    examples = build_sft_examples(
        parsed_items,
        system_prompt,
        chunk_min_tokens=chunk_min_tokens,
        chunk_max_tokens=chunk_max_tokens,
    )
    train, val = split_train_val(examples)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"
    write_jsonl(train, train_path)
    write_jsonl(val, val_path)

    warnings: list[str] = []
    if len(examples) < settings.min_samples_warning:
        warnings.append(
            f"Only {len(examples)} training examples. "
            f"50+ chunks are recommended for stronger personality capture."
        )

    meta = {
        "total_examples": len(examples),
        "train_examples": len(train),
        "val_examples": len(val),
        "warnings": warnings,
    }
    return train_path, val_path, meta
