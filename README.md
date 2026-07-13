# PersonaFinetuner

A local-first web app for fine-tuning small open-weight LLMs with LoRA/QLoRA to capture a person's writing style, cloning a consented speaking voice, and holding live browser voice calls with the result.

## Quick start (Windows)

1. **Setup** (first time only): double-click `setup.bat` or run `.\setup.ps1`
2. **Run**: double-click `run.bat` or run `.\run.ps1`
3. Your browser opens to **http://localhost:8000**

Linux/macOS: `make setup && make run`

## Features (v1.2)

- **Source → Train → Test** wizard with voice-transfer UI
- Upload plain text, JSONL, or chat exports (Discord, WhatsApp, Instagram)
- Auto-detect export format with target username/contact filtering
- Optional synthetic Q&A generation before training
- LoRA/QLoRA training via Unsloth with SSE progress and loss chart
- Side-by-side base vs fine-tuned comparison in Test step
- Zero-shot voice cloning with **Chatterbox Turbo 0.1.7** (MIT, PerTh-watermarked output)
- Local speech-to-text with **faster-whisper 1.2.1** and integrated Silero VAD
- Live, hands-free browser calls over one bidirectional WebSocket pipeline
- Voice reference recording, TTS preview, and standalone dictation in the Voice step
- Export adapter for Ollama (with GGUF conversion when tooling is available)
- One-click Windows launchers (`run.bat`, `setup.bat`)

## Requirements

| Requirement | Details |
|-------------|---------|
| Python | 3.10+ |
| GPU | NVIDIA with CUDA (8+ GB recommended for a 3B persona plus speech models; 12+ GB for smoother calls) |
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

The first transcription and synthesis request downloads the configured open weights from Hugging Face. All inference and call audio then remain local.

## Voice architecture

The app uses one shared pipeline for previews, dictation, and calls:

`browser microphone → adaptive silence detection → faster-whisper + Silero VAD → fine-tuned persona → Chatterbox → browser audio`

- **TTS:** Chatterbox Turbo is the default because it is the low-latency voice-agent model. Set `PF_TTS_MODEL=multilingual-v3` for Chatterbox Multilingual V3 and its 23 supported languages.
- **STT:** faster-whisper's `turbo` checkpoint defaults to CUDA `int8_float16`, with an automatic CPU `int8` fallback.
- **Transport:** a single FastAPI WebSocket carries turns, transcripts, state, and WAV responses. There is no duplicate chat service or cloud speech provider.
- **Turn detection:** microphone audio is captured as mono PCM WAV; an adaptive local noise floor detects speech and sends a turn after 700 ms of silence.

This is a local browser-to-app call path, not PSTN/SIP telephony. It is designed for one user talking to a local persona without requiring a separate media server.

### Voice configuration

| Environment variable | Default | Purpose |
|----------------------|---------|---------|
| `PF_TTS_MODEL` | `turbo` | `turbo` or `multilingual-v3` |
| `PF_TTS_DEVICE` | `auto` | `auto`, `cuda`, or `cpu` |
| `PF_STT_MODEL` | `turbo` | Any faster-whisper model name or local model path |
| `PF_STT_DEVICE` | `auto` | `auto`, `cuda`, or `cpu` |
| `PF_STT_COMPUTE_TYPE` | `auto` | CTranslate2 compute type |
| `PF_MAX_VOICE_UPLOAD_BYTES` | `26214400` | Voice-reference upload limit |
| `PF_ALLOWED_ORIGINS` | local app URLs | JSON list of trusted cross-origin API clients |

For best cloning, record one consenting speaker for 8–15 seconds in a quiet room without music, reverb, or other voices. References must be uncompressed PCM WAV and are checked to be longer than 5 seconds and no more than 30 seconds.

### Responsible voice cloning

The API and UI require explicit confirmation that the voice owner consented. Chatterbox adds its built-in imperceptible PerTh watermark to generated audio. Only clone voices you own or have clear permission to use.

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
| GET | `/api/voice/capabilities` | Installed speech engines and active models |
| POST | `/api/voice/{persona_id}/profile` | Save a consented voice reference |
| POST | `/api/voice/transcribe` | Transcribe local audio |
| POST | `/api/voice/{persona_id}/synthesize` | Generate watermarked cloned speech |
| WS | `/api/voice/call/{persona_id}` | Live STT → persona → TTS call |
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
| Speech engine missing | Re-run `pip install -r requirements.txt` in the app environment |
| First voice request is slow | The speech model is downloading and warming up; later requests reuse it |
| Call runs out of VRAM | Use the 3B persona, set `PF_STT_DEVICE=cpu`, or use a GPU with more VRAM |
| Multilingual TTS rejected | Set `PF_TTS_MODEL=multilingual-v3` before starting the server |

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
