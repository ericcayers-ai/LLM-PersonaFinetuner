"""Tests for chat export parsers."""

from pathlib import Path

import pytest

from app.core import parsers
from app.core.dataset_builder import build_sft_examples

FIXTURES = Path(__file__).parent / "fixtures"


class TestFormatDetection:
    def test_detect_discord(self):
        path = FIXTURES / "discord_export.json"
        assert parsers.detect_format(path) == parsers.DetectedFormat.DISCORD

    def test_detect_whatsapp(self):
        path = FIXTURES / "whatsapp_export.txt"
        assert parsers.detect_format(path) == parsers.DetectedFormat.WHATSAPP

    def test_detect_instagram(self):
        path = FIXTURES / "instagram_export.json"
        assert parsers.detect_format(path) == parsers.DetectedFormat.INSTAGRAM


class TestDiscordParser:
    def test_parse_with_target_author(self):
        items = parsers.parse_discord(FIXTURES / "discord_export.json", target_author="target_user")
        chat_items = parsers.strip_meta_items(items)
        assert len(chat_items) >= 2
        for item in chat_items:
            assert item["type"] == "chat"
            roles = [m["role"] for m in item["messages"]]
            assert "user" in roles and "assistant" in roles
            assistant = next(m["content"] for m in item["messages"] if m["role"] == "assistant")
            assert "writing" in assistant.lower() or "ideas" in assistant.lower()

    def test_requires_target_raises(self):
        with pytest.raises(ValueError, match="target username"):
            parsers.parse_upload(FIXTURES / "discord_export.json", fmt=parsers.DetectedFormat.DISCORD)


class TestWhatsAppParser:
    def test_parse_24h_format(self):
        items = parsers.parse_whatsapp(FIXTURES / "whatsapp_export.txt", target_contact="Target Person")
        chat_items = parsers.strip_meta_items(items)
        assert len(chat_items) >= 2
        assistant_msgs = [
            m["content"]
            for item in chat_items
            for m in item["messages"]
            if m["role"] == "assistant"
        ]
        assert any("porch" in m.lower() or "stillness" in m.lower() for m in assistant_msgs)

    def test_parse_12h_format(self):
        items = parsers.parse_whatsapp(FIXTURES / "whatsapp_export.txt", target_contact="Target Person")
        chat_items = parsers.strip_meta_items(items)
        assert any("stillness" in m["content"].lower() for item in chat_items for m in item["messages"])


class TestInstagramParser:
    def test_parse_with_sender_filter(self):
        items = parsers.parse_instagram(FIXTURES / "instagram_export.json", target_sender="Target Sender")
        chat_items = parsers.strip_meta_items(items)
        assert len(chat_items) >= 2
        for item in chat_items:
            assistant = next(m["content"] for m in item["messages"] if m["role"] == "assistant")
            assert len(assistant) > 0


class TestDatasetPipeline:
    def test_chat_exports_to_sft(self):
        items = parsers.parse_discord(FIXTURES / "discord_export.json", target_author="target_user")
        items = parsers.strip_meta_items(items)
        examples = build_sft_examples(items, "You are target_user.")
        assert len(examples) >= 2
        for ex in examples:
            assert "messages" in ex
            assert ex["messages"][0]["role"] == "system"
            assert ex["messages"][2]["role"] == "assistant"


class TestSyntheticQA:
    def test_fallback_qa_without_gpu(self, monkeypatch):
        from app.core.synthetic_qa import generate_synthetic_qa
        from app.models.schemas import GPUInfo

        monkeypatch.setattr(
            "app.core.synthetic_qa.get_gpu_info",
            lambda *_: GPUInfo(cuda_available=False),
        )
        pairs, warnings = generate_synthetic_qa(["This is a sample writing about creativity and flow."])
        assert len(pairs) > 0
        assert any("template" in w.lower() or "cuda" in w.lower() for w in warnings)
