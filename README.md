# PersonaFinetuner

A local-first web app for fine-tuning small open-weight LLMs with LoRA/QLoRA to capture a person's writing voice. Upload samples, train on your NVIDIA GPU, and chat with the result.

## Features

- **Source → Train → Test** wizard with a voice-transfer UI
- Upload plain text, JSONL, or chat exports (Discord, WhatsApp, Instagram)
- Auto-detect export format with target username/contact filtering
- Optional synthetic Q&A generation before training
- LoRA/QLoRA training via Unsloth with SSE progress and loss chart
- Side-by-side base vs fine-tuned comparison in Test step
- Export adapter for Ollama (with GGUF conversion when tooling is available)

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA (6+ GB VRAM for 3B models, 10+ GB for 7B)
- Windows or Linux

## Install

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

If Unsloth fails on Windows, see [Unsloth docs](https://github.com/unslothai/unsloth) for CUDA-specific install notes.

## Run

```bash
uvicorn app.main:app --reload
```

Open http://localhost:8000

## Supported upload formats

| Format | Extension | Notes |
|--------|-----------|-------|
| Plain text | `.txt` | Essays, transcripts, writing samples |
| User/Assistant | `.txt` | `User: …` / `Assistant: …` blocks |
| JSONL | `.jsonl` | `{"messages": [...]}` per line |
| Discord | `.json` | Array of messages with `author`, `content`, `timestamp` |
| WhatsApp | `.txt` | `[DD/MM/YYYY, HH:MM:SS] Name: message` or `[M/D/YY, H:MM AM/PM]` |
| Instagram | `.json` | `message_*.json` from data download |

For chat exports, set the **target username/contact name** so that person's messages become the assistant in training pairs.

### Format examples

**Discord** (`discord_export.json`):
```json
[{"author": {"name": "alice"}, "content": "Hello!", "timestamp": "..."}]
```

**WhatsApp**:
```
[15/01/2024, 10:00:00] Bob: Hey there
[15/01/2024, 10:01:00] Alice: Hi! How are you?
```

**Instagram** (`message_1.json`):
```json
{"participants": [{"name": "alice"}], "messages": [{"sender_name": "alice", "content": "Hi"}]}
```

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/system/gpu` | GPU info |
| POST | `/api/datasets/upload` | Upload file (returns detected format) |
| POST | `/api/datasets/build` | Build SFT dataset |
| POST | `/api/training/start` | Start training |
| GET | `/api/training/{id}/stream` | SSE progress |
| GET | `/api/training/{id}/loss` | Loss history for chart |
| POST | `/api/inference/chat` | Chat (optional `compare_base`) |
| POST | `/api/personas/{id}/export-gguf` | Export for Ollama |

## Ollama export limitations

- Ollama requires GGUF; LoRA adapters must be merged with the base model first
- 4-bit quantized bases may need full-precision merge before conversion
- Auto GGUF export works when Unsloth GGUF tooling is available; otherwise manual steps are provided in the export manifest

## Testing

```bash
pip install pytest httpx
pytest tests/ -v
```

## Project structure

```
app/           FastAPI backend
frontend/      Vanilla JS UI
tests/         Parser unit tests
data/          Runtime uploads (gitignored)
outputs/       Trained adapters (gitignored)
```
