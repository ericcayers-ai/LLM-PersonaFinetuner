"""API smoke tests using FastAPI TestClient."""

import io
import json
import wave

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)
FIXTURES = __import__("pathlib").Path(__file__).parent / "fixtures"


class TestHealth:
    def test_root_serves_html_or_json(self):
        res = client.get("/")
        assert res.status_code == 200

    def test_health_endpoint(self):
        res = client.get("/api/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "ok"
        assert data["app_name"] == "PersonaFinetuner"
        assert data["version"] == "1.2.0"
        assert "supported_formats" in data
        assert ".txt" in data["supported_formats"]

    def test_system_health_alias(self):
        res = client.get("/api/system/health")
        assert res.status_code == 200
        assert res.json()["status"] == "ok"

    def test_gpu_endpoint(self):
        res = client.get("/api/system/gpu")
        assert res.status_code == 200
        assert "cuda_available" in res.json()


class TestDatasetsAPI:
    def test_upload_empty_file_rejected(self):
        res = client.post(
            "/api/datasets/upload",
            files={"file": ("empty.txt", b"", "text/plain")},
        )
        assert res.status_code == 400
        body = res.json()
        assert body["error"] == "empty_file"

    def test_upload_unsupported_extension(self):
        res = client.post(
            "/api/datasets/upload",
            files={"file": ("data.csv", b"a,b,c", "text/csv")},
        )
        assert res.status_code == 400
        assert res.json()["error"] == "unsupported_format"

    def test_upload_discord_fixture(self):
        content = (FIXTURES / "discord_export.json").read_bytes()
        res = client.post(
            "/api/datasets/upload",
            files={"file": ("discord_export.json", content, "application/json")},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["detected_format"] == "discord"
        assert data["requires_target_name"] is True

    def test_build_requires_uploads(self):
        res = client.post(
            "/api/datasets/build",
            json={
                "upload_ids": [],
                "persona_name": "Test",
                "system_prompt": "You are Test.",
            },
        )
        assert res.status_code == 400
        assert res.json()["error"] == "no_uploads"

    def test_build_pipeline(self):
        content = (FIXTURES / "discord_export.json").read_bytes()
        upload = client.post(
            "/api/datasets/upload",
            files={"file": ("discord_export.json", content, "application/json")},
        ).json()

        res = client.post(
            "/api/datasets/build",
            json={
                "upload_ids": [upload["upload_id"]],
                "persona_name": "API Test Persona",
                "system_prompt": "You are target_user.",
                "target_name": "target_user",
            },
        )
        assert res.status_code == 200
        data = res.json()
        assert data["total_examples"] >= 2
        assert len(data["examples"]) > 0


class TestPersonasAPI:
    def test_list_personas(self):
        res = client.get("/api/personas")
        assert res.status_code == 200
        assert isinstance(res.json(), list)

    def test_create_persona_validation(self):
        res = client.post("/api/personas", json={"name": "", "system_prompt": ""})
        assert res.status_code == 400


class TestTrainingAPI:
    def test_start_without_persona(self):
        res = client.post(
            "/api/training/start",
            json={"persona_id": "00000000-0000-0000-0000-000000000000"},
        )
        assert res.status_code == 404

    def test_job_not_found(self):
        res = client.get("/api/training/bad-id/status")
        assert res.status_code == 404
        assert res.json()["error"] == "job_not_found"


class TestInferenceAPI:
    def test_chat_missing_persona(self):
        res = client.post(
            "/api/inference/chat",
            json={
                "persona_id": "00000000-0000-0000-0000-000000000000",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )
        assert res.status_code == 404

    def test_chat_empty_messages(self):
        res = client.post(
            "/api/inference/chat",
            json={"persona_id": "x", "messages": []},
        )
        assert res.status_code in (400, 404, 422)


def _wav_bytes(duration_seconds: int = 4, sample_rate: int = 16000) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x00\x00" * sample_rate * duration_seconds)
    return output.getvalue()


def _voice_persona(*, trained: bool = False) -> dict:
    from app.storage import store

    persona = store.create_persona("Voice Test", "You are a concise voice assistant.")
    if trained:
        persona = store.update_persona(
            persona["id"],
            status="trained",
            adapter_path=f"outputs/adapters/{persona['id']}",
        )
    return persona


class TestVoiceAPI:
    def test_capabilities_describe_single_integrated_stack(self):
        res = client.get("/api/voice/capabilities")
        assert res.status_code == 200
        data = res.json()
        assert data["tts_engine"] == "Chatterbox"
        assert data["stt_engine"] == "faster-whisper + Silero VAD"
        assert data["transport"] == "websocket"
        assert ".wav" in data["supported_voice_formats"]

    def test_voice_profile_requires_consent(self):
        persona = _voice_persona()
        res = client.post(
            f"/api/voice/{persona['id']}/profile",
            files={"file": ("reference.wav", _wav_bytes(), "audio/wav")},
            data={"consent_confirmed": "false", "language": "en"},
        )
        assert res.status_code == 400
        assert res.json()["error"] == "consent_required"

    def test_voice_profile_rejects_short_reference(self):
        persona = _voice_persona()
        res = client.post(
            f"/api/voice/{persona['id']}/profile",
            files={"file": ("reference.wav", _wav_bytes(1), "audio/wav")},
            data={"consent_confirmed": "true", "language": "en"},
        )
        assert res.status_code == 400
        assert res.json()["error"] == "invalid_reference_duration"

    def test_upload_and_get_voice_profile(self):
        persona = _voice_persona()
        upload = client.post(
            f"/api/voice/{persona['id']}/profile",
            files={"file": ("reference.wav", _wav_bytes(), "audio/wav")},
            data={"consent_confirmed": "true", "language": "en"},
        )
        assert upload.status_code == 200
        assert upload.json()["consent_confirmed"] is True
        assert "sample_path" not in upload.json()

        get_profile = client.get(f"/api/voice/{persona['id']}/profile")
        assert get_profile.status_code == 200
        assert get_profile.json()["sample_filename"] == "reference.wav"

        get_persona = client.get(f"/api/personas/{persona['id']}")
        assert get_persona.json()["voice_profile"]["language"] == "en"

    def test_transcribe_uses_shared_stt_engine(self, monkeypatch):
        from app.api import voice

        monkeypatch.setattr(
            voice.stt_engine,
            "transcribe",
            lambda _content, **_kwargs: {
                "text": "hello from speech",
                "language": "en",
                "language_probability": 0.99,
                "duration_seconds": 4.0,
                "segments": [{"start": 0, "end": 1, "text": "hello from speech"}],
            },
        )
        res = client.post(
            "/api/voice/transcribe",
            files={"file": ("speech.wav", _wav_bytes(), "audio/wav")},
        )
        assert res.status_code == 200
        assert res.json()["text"] == "hello from speech"

    def test_synthesize_uses_persona_voice(self, monkeypatch):
        from app.api import voice

        persona = _voice_persona()
        client.post(
            f"/api/voice/{persona['id']}/profile",
            files={"file": ("reference.wav", _wav_bytes(), "audio/wav")},
            data={"consent_confirmed": "true", "language": "en"},
        )
        monkeypatch.setattr(
            voice.voice_engine,
            "synthesize",
            lambda text, reference_path, **kwargs: b"RIFF-test-audio",
        )
        res = client.post(
            f"/api/voice/{persona['id']}/synthesize",
            json={"text": "A voice preview", "language": "en"},
        )
        assert res.status_code == 200
        assert res.headers["content-type"] == "audio/wav"
        assert res.content == b"RIFF-test-audio"

    def test_live_call_runs_stt_llm_tts_pipeline(self, monkeypatch):
        from app.api import voice
        from app.storage import store

        persona = _voice_persona(trained=True)
        store.save_voice_profile(
            persona["id"],
            "reference.wav",
            _wav_bytes(),
            language="en",
            consent_confirmed=True,
        )
        monkeypatch.setattr(
            voice.stt_engine,
            "transcribe",
            lambda _content: {
                "text": "How are you?",
                "language": "en",
                "language_probability": 1.0,
                "duration_seconds": 1.0,
                "segments": [],
            },
        )
        monkeypatch.setattr(
            voice.inference_engine,
            "chat",
            lambda *_args, **_kwargs: {"response": "Doing great."},
        )
        monkeypatch.setattr(
            voice.voice_engine,
            "synthesize",
            lambda *_args, **_kwargs: b"RIFF-call-audio",
        )

        with client.websocket_connect(f"/api/voice/call/{persona['id']}") as websocket:
            assert websocket.receive_json()["type"] == "ready"
            websocket.send_bytes(_wav_bytes())
            assert websocket.receive_json() == {"type": "state", "state": "transcribing"}
            assert websocket.receive_json()["text"] == "How are you?"
            assert websocket.receive_json() == {"type": "state", "state": "thinking"}
            assert websocket.receive_json()["text"] == "Doing great."
            assert websocket.receive_json() == {"type": "state", "state": "speaking"}
            assert websocket.receive_json()["type"] == "audio"
            assert websocket.receive_bytes() == b"RIFF-call-audio"
            assert websocket.receive_json()["type"] == "ready"
