"""File parsers for uploads — plain text, JSONL, and chat exports."""

from __future__ import annotations

import json
import re
from enum import Enum
from pathlib import Path
from typing import Any


USER_ASSISTANT_PATTERN = re.compile(
    r"^(?:User|Human)\s*:\s*(.+?)(?:\n(?:Assistant|AI)\s*:\s*(.+?))(?=\n(?:User|Human)\s*:|\Z)",
    re.DOTALL | re.MULTILINE | re.IGNORECASE,
)

WHATSAPP_PATTERNS = [
    # [DD/MM/YYYY, HH:MM:SS] Name: message
    re.compile(
        r"^\[(\d{1,2}/\d{1,2}/\d{2,4},\s*\d{1,2}:\d{2}(?::\d{2})?)\]\s*([^:]+?):\s*(.*)$"
    ),
    # [M/D/YY, H:MM AM/PM] Name: message
    re.compile(
        r"^\[(\d{1,2}/\d{1,2}/\d{2,4},\s*\d{1,2}:\d{2}\s*(?:AM|PM))\]\s*([^:]+?):\s*(.*)$",
        re.IGNORECASE,
    ),
]

SUPPORTED_EXTENSIONS = {".txt", ".jsonl", ".json"}


class DetectedFormat(str, Enum):
    PLAIN_TEXT = "plain_text"
    USER_ASSISTANT = "user_assistant"
    JSONL = "jsonl"
    DISCORD = "discord"
    WHATSAPP = "whatsapp"
    INSTAGRAM = "instagram"
    UNKNOWN = "unknown"


FORMAT_META: dict[DetectedFormat, dict[str, Any]] = {
    DetectedFormat.PLAIN_TEXT: {
        "label": "Plain text",
        "requires_target_name": False,
        "hint": "Upload essays, transcripts, or writing samples as .txt",
    },
    DetectedFormat.USER_ASSISTANT: {
        "label": "User/Assistant blocks",
        "requires_target_name": False,
        "hint": "Lines like User: … / Assistant: … in a .txt file",
    },
    DetectedFormat.JSONL: {
        "label": "JSONL chat dataset",
        "requires_target_name": False,
        "hint": 'One JSON object per line with a "messages" array',
    },
    DetectedFormat.DISCORD: {
        "label": "Discord chat export",
        "requires_target_name": True,
        "hint": "Discord JSON export — set the target username to extract as the persona",
    },
    DetectedFormat.WHATSAPP: {
        "label": "WhatsApp chat export",
        "requires_target_name": True,
        "hint": "[DD/MM/YYYY, HH:MM:SS] Name: message — set the contact name for the persona",
    },
    DetectedFormat.INSTAGRAM: {
        "label": "Instagram DM export",
        "requires_target_name": True,
        "hint": "Instagram message_*.json — set the sender username for the persona",
    },
    DetectedFormat.UNKNOWN: {
        "label": "Unknown format",
        "requires_target_name": False,
        "hint": "Could not detect format — try .txt, .jsonl, or a supported chat export",
    },
}


def get_format_meta(fmt: DetectedFormat) -> dict[str, Any]:
    return FORMAT_META.get(fmt, FORMAT_META[DetectedFormat.UNKNOWN])


def _normalize_name(name: str) -> str:
    return name.strip().lower()


def _names_match(a: str, b: str) -> bool:
    return _normalize_name(a) == _normalize_name(b)


def _chat_item(user: str, assistant: str) -> dict[str, Any]:
    return {
        "type": "chat",
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
    }


def detect_format(path: Path, content: str | None = None) -> DetectedFormat:
    ext = path.suffix.lower()
    text = content if content is not None else path.read_text(encoding="utf-8", errors="replace")
    stripped = text.strip()

    if ext == ".jsonl":
        return DetectedFormat.JSONL

    if ext == ".json":
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError:
            return DetectedFormat.UNKNOWN
        if _looks_like_discord(data):
            return DetectedFormat.DISCORD
        if _looks_like_instagram(data):
            return DetectedFormat.INSTAGRAM
        return DetectedFormat.UNKNOWN

    if ext == ".txt":
        if _looks_like_whatsapp(stripped):
            return DetectedFormat.WHATSAPP
        if _parse_user_assistant_blocks(stripped) or _parse_line_blocks(stripped):
            return DetectedFormat.USER_ASSISTANT
        return DetectedFormat.PLAIN_TEXT

    return DetectedFormat.UNKNOWN


def _looks_like_whatsapp(text: str) -> bool:
    lines = [ln for ln in text.splitlines() if ln.strip()][:20]
    if not lines:
        return False
    matches = sum(1 for ln in lines if any(p.match(ln.strip()) for p in WHATSAPP_PATTERNS))
    return matches >= max(2, len(lines) // 2)


def _looks_like_discord(data: Any) -> bool:
    messages = data if isinstance(data, list) else data.get("messages") if isinstance(data, dict) else None
    if not isinstance(messages, list) or not messages:
        return False
    sample = messages[0]
    return isinstance(sample, dict) and "author" in sample and "content" in sample


def _looks_like_instagram(data: Any) -> bool:
    if not isinstance(data, dict):
        return False
    return "messages" in data and isinstance(data["messages"], list) and (
        "participants" in data or any(
            isinstance(m, dict) and "sender_name" in m for m in data["messages"][:5]
        )
    )


def parse_txt(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return []

    blocks = _parse_user_assistant_blocks(text)
    if blocks:
        return blocks

    line_blocks = _parse_line_blocks(text)
    if line_blocks:
        return line_blocks

    return [{"type": "text", "content": text}]


def _parse_user_assistant_blocks(text: str) -> list[dict[str, Any]]:
    matches = list(USER_ASSISTANT_PATTERN.finditer(text))
    if not matches:
        return []

    examples = []
    for m in matches:
        user_msg = m.group(1).strip()
        assistant_msg = m.group(2).strip() if m.group(2) else ""
        if user_msg and assistant_msg:
            examples.append(_chat_item(user_msg, assistant_msg))
    return examples


def _parse_line_blocks(text: str) -> list[dict[str, Any]]:
    lines = text.splitlines()
    examples: list[dict[str, Any]] = []
    current_user: str | None = None
    current_assistant: list[str] = []

    def flush() -> None:
        nonlocal current_user, current_assistant
        if current_user and current_assistant:
            examples.append(_chat_item(current_user, "\n".join(current_assistant).strip()))
        current_user = None
        current_assistant = []

    for line in lines:
        lower = line.strip().lower()
        if lower.startswith("user:") or lower.startswith("human:"):
            flush()
            current_user = line.split(":", 1)[1].strip()
        elif lower.startswith("assistant:") or lower.startswith("ai:"):
            current_assistant.append(line.split(":", 1)[1].strip())
        elif current_assistant:
            current_assistant.append(line)
        elif current_user:
            current_user += "\n" + line

    flush()
    return examples


def parse_jsonl(path: Path) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {i}: {e}") from e

            if "messages" in row:
                examples.append({"type": "chat", "messages": row["messages"]})
            elif "text" in row:
                examples.append({"type": "text", "content": row["text"]})
            elif "instruction" in row and "output" in row:
                examples.append(_chat_item(row["instruction"], row["output"]))
            else:
                raise ValueError(f"Unsupported JSONL format on line {i}")
    return examples


def _extract_discord_messages(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("messages"), list):
        return data["messages"]
    raise ValueError("Discord export must be a message array or object with a messages field")


def parse_discord(path: Path, target_author: str | None = None) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    raw_messages = _extract_discord_messages(data)

    parsed: list[tuple[str, str, str]] = []
    skipped_notes = 0

    for msg in raw_messages:
        if not isinstance(msg, dict):
            continue
        author_obj = msg.get("author") or {}
        author = author_obj.get("name") or author_obj.get("username") or "unknown"
        content = (msg.get("content") or "").strip()

        attachments = msg.get("attachments") or []
        embeds = msg.get("embeds") or []
        if not content:
            if attachments or embeds:
                skipped_notes += 1
                parts = []
                if attachments:
                    parts.append(f"{len(attachments)} attachment(s)")
                if embeds:
                    parts.append(f"{len(embeds)} embed(s)")
                content = f"[{', '.join(parts)} omitted]"
            else:
                continue

        if target_author and not _names_match(author, target_author):
            continue

        parsed.append((author, content, msg.get("timestamp", "")))

    if target_author and not parsed:
        raise ValueError(f"No messages found for Discord author '{target_author}'")

    if not target_author:
        # Return as plain text corpus from all authors
        all_text = "\n".join(content for _, content, _ in parsed)
        if not all_text.strip():
            raise ValueError("No text content found in Discord export")
        return [{"type": "text", "content": all_text}]

    return _pair_chat_messages(parsed, target_author, platform="Discord", skipped_notes=skipped_notes)


def parse_whatsapp(path: Path, target_contact: str | None = None) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    parsed: list[tuple[str, str, str]] = []

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        match = None
        for pattern in WHATSAPP_PATTERNS:
            match = pattern.match(line)
            if match:
                break
        if not match:
            continue

        sender = match.group(2).strip()
        message = match.group(3).strip()
        if not message or message in ("<Media omitted>", "image omitted", "video omitted"):
            continue
        if target_contact and not _names_match(sender, target_contact):
            continue
        parsed.append((sender, message, match.group(1)))

    if target_contact and not parsed:
        raise ValueError(f"No messages found for WhatsApp contact '{target_contact}'")

    if not target_contact:
        all_text = "\n".join(msg for _, msg, _ in parsed)
        if not all_text.strip():
            raise ValueError("No messages found in WhatsApp export")
        return [{"type": "text", "content": all_text}]

    return _pair_chat_messages(parsed, target_contact, platform="WhatsApp")


def parse_instagram(path: Path, target_sender: str | None = None) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    if not isinstance(data, dict):
        raise ValueError("Instagram export must be a JSON object")

    raw_messages = data.get("messages") or []
    parsed: list[tuple[str, str, str]] = []
    skipped_notes = 0

    for msg in raw_messages:
        if not isinstance(msg, dict):
            continue
        sender = (msg.get("sender_name") or "unknown").strip()
        content = (msg.get("content") or "").strip()

        if not content:
            if msg.get("photos") or msg.get("videos") or msg.get("share") or msg.get("reactions"):
                skipped_notes += 1
                content = "[media/share omitted]"
            else:
                continue

        if target_sender and not _names_match(sender, target_sender):
            continue

        ts = str(msg.get("timestamp_ms", ""))
        parsed.append((sender, content, ts))

    # Instagram exports are often newest-first — reverse for chronological pairing
    parsed.reverse()

    if target_sender and not parsed:
        raise ValueError(f"No messages found for Instagram sender '{target_sender}'")

    if not target_sender:
        all_text = "\n".join(content for _, content, _ in parsed)
        if not all_text.strip():
            raise ValueError("No messages found in Instagram export")
        return [{"type": "text", "content": all_text}]

    return _pair_chat_messages(parsed, target_sender, platform="Instagram", skipped_notes=skipped_notes)


def _pair_chat_messages(
    messages: list[tuple[str, str, str]],
    target_name: str,
    platform: str = "",
    skipped_notes: int = 0,
) -> list[dict[str, Any]]:
    """Build user/assistant pairs where the target persona is always the assistant."""
    examples: list[dict[str, Any]] = []
    pending_user: list[str] = []

    def flush_user_to_assistant(assistant_text: str) -> None:
        nonlocal pending_user
        if not pending_user or not assistant_text.strip():
            return
        user_text = "\n".join(pending_user).strip()
        if user_text:
            examples.append(_chat_item(user_text, assistant_text.strip()))
        pending_user = []

    for author, content, _ts in messages:
        if _names_match(author, target_name):
            if pending_user:
                flush_user_to_assistant(content)
            else:
                # Target spoke without prior user context — use a generic prompt
                examples.append(_chat_item("Say something in your usual style.", content))
        else:
            pending_user.append(content)

    warnings: list[str] = []
    if skipped_notes:
        warnings.append(f"{skipped_notes} message(s) with attachments/media noted as omitted")
    if platform:
        warnings.append(f"Parsed {len(examples)} conversation pair(s) from {platform} export")

    if not examples:
        raise ValueError(
            f"No conversation pairs could be built for '{target_name}'. "
            "Ensure the target name matches the export and they have replies in the chat."
        )

    result: list[dict[str, Any]] = examples
    if warnings:
        result.append({"type": "meta", "warnings": warnings})
    return result


def parse_upload(
    path: Path,
    fmt: DetectedFormat | None = None,
    target_name: str | None = None,
) -> list[dict[str, Any]]:
    detected = fmt or detect_format(path)
    meta = get_format_meta(detected)

    if meta["requires_target_name"] and not target_name:
        raise ValueError(
            f"{meta['label']} requires a target username/contact name. "
            "Provide target_name when building the dataset."
        )

    if detected == DetectedFormat.JSONL:
        return parse_jsonl(path)
    if detected == DetectedFormat.DISCORD:
        return parse_discord(path, target_author=target_name)
    if detected == DetectedFormat.WHATSAPP:
        return parse_whatsapp(path, target_contact=target_name)
    if detected == DetectedFormat.INSTAGRAM:
        return parse_instagram(path, target_sender=target_name)
    if detected in (DetectedFormat.PLAIN_TEXT, DetectedFormat.USER_ASSISTANT):
        return parse_txt(path)

    ext = path.suffix.lower()
    if ext == ".txt":
        return parse_txt(path)
    if ext == ".jsonl":
        return parse_jsonl(path)
    if ext == ".json":
        # Last-chance auto detect
        text = path.read_text(encoding="utf-8", errors="replace")
        redetected = detect_format(path, text)
        return parse_upload(path, fmt=redetected, target_name=target_name)

    raise ValueError(
        f"Unsupported file type: {ext}. Use .txt, .jsonl, or .json chat exports."
    )


def extract_meta_warnings(items: list[dict[str, Any]]) -> list[str]:
    warnings: list[str] = []
    for item in items:
        if item.get("type") == "meta" and item.get("warnings"):
            warnings.extend(item["warnings"])
    return warnings


def strip_meta_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item for item in items if item.get("type") != "meta"]
