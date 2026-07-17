# PersonaFinetuner

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A local-first web app for fine-tuning small open-weight LLMs with LoRA/QLoRA to capture a person's writing voice. Upload samples, train on your NVIDIA GPU, and chat with the result.

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to set up a dev environment and open a PR. Licensed under the [MIT License](LICENSE).

## Quick start (Windows)

1. **Setup** (first time only): double-click `setup.bat` or run `.\setup.ps1`
2. **Run**: double-click `run.bat` or run `.\run.ps1`
3. Your browser opens to **http://localhost:8000**

Linux/macOS: `make setup && make run`

## Features (v1.2.0)

- **Source → Train → Test** wizard with clearer step navigation and soft gating
- Progressive disclosure for advanced training settings; quieter product chrome
- Actionable empty states, continue CTAs, and collapsible voice preview
- Upload plain text, JSONL, or chat exports (Discord, WhatsApp, Instagram)
- Auto-detect export format with target username/contact filtering
- Optional synthetic Q&A generation before training
- LoRA/QLoRA training via Unsloth with SSE progress and loss chart
- Side-by-side base vs fine-tuned comparison in Test step
- Export adapter for Ollama (with GGUF conversion when tooling is available)
- One-click Windows launchers (`run.bat`, `setup.bat`)

## Requirements

| Requirement | Details |
|-------------|---------|
| Python | 3.10+ |
| GPU | NVIDIA with CUDA (6+ GB VRAM for 3B, 10+ GB for 7B) |
| OS | Windows (primary), Linux |

## Manual install

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Supported upload formats

| Format | Extension | Notes |
|--------|-----------|-------|
| Plain text | `.txt` | Essays, transcripts, writing samples |
| User/Assistant | `.txt` | `User: …` / `Assistant: …` blocks |
| JSONL | `.jsonl` | `{"messages": [...]}` per line |
| Discord | `.json` | Array of messages with `author`, `content`, `timestamp` |
| WhatsApp | `.txt` | `[DD/MM/YYYY, HH:MM:SS] Name: message` |
| Instagram | `.json` | `message_*.json` from data download |

For chat exports, set the **target username/contact name** so that person's messages become the assistant in training pairs.

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | App health, version, GPU summary |
| GET | `/api/system/gpu` | GPU info |
| POST | `/api/datasets/upload` | Upload file (returns detected format) |
| POST | `/api/datasets/build` | Build SFT dataset |
| POST | `/api/training/start` | Start training |
| GET | `/api/training/{id}/stream` | SSE progress |
| GET | `/api/training/{id}/loss` | Loss history for chart |
| POST | `/api/inference/chat` | Chat (optional `compare_base`) |
| POST | `/api/personas/{id}/export-gguf` | Export for Ollama |

API errors return a consistent JSON shape: `{ error, message, detail, hint }`.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| No GPU detected | Install NVIDIA drivers + CUDA PyTorch (`setup.bat` handles this) |
| Unsloth fails on Windows | See [Unsloth docs](https://github.com/unslothai/unsloth) |
| Training job lost after restart | Jobs are in-memory; restart training after server reboot |
| Upload rejected | Max 50 MB per file; check format and non-empty content |
| Chat export empty | Verify target username matches export exactly |

## Screenshots

<!-- Add screenshots of Source, Train, and Test steps here -->

## Ollama export limitations

- Ollama requires GGUF; LoRA adapters must be merged with the base model first
- 4-bit quantized bases may need full-precision merge before conversion
- Auto GGUF export works when Unsloth GGUF tooling is available

## Testing

```bash
pip install pytest httpx
pytest tests/ -v
```

## Project structure

```
app/           FastAPI backend
frontend/      Vanilla JS UI
tests/         Parser + API tests
run.bat        One-click Windows launcher
setup.bat      First-time setup
data/          Runtime uploads (gitignored)
outputs/       Trained adapters (gitignored)
```
