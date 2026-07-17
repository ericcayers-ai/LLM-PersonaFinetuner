"""API smoke tests using FastAPI TestClient."""

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

    def test_start_persists_base_model(self, monkeypatch):
        content = (FIXTURES / "discord_export.json").read_bytes()
        upload = client.post(
            "/api/datasets/upload",
            files={"file": ("discord_export.json", content, "application/json")},
        ).json()
        built = client.post(
            "/api/datasets/build",
            json={
                "upload_ids": [upload["upload_id"]],
                "persona_name": "Base Model Persist",
                "system_prompt": "You are target_user.",
                "target_name": "target_user",
            },
        ).json()

        monkeypatch.setattr("app.api.training.job_manager.start", lambda _job: None)

        chosen = "unsloth/Mistral-7B-Instruct-v0.3-bnb-4bit"
        res = client.post(
            "/api/training/start",
            json={"persona_id": built["persona_id"], "base_model": chosen},
        )
        assert res.status_code == 200

        persona = client.get(f"/api/personas/{built['persona_id']}").json()
        assert persona["base_model"] == chosen


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
