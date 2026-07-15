# Contributing to PersonaFinetuner

Thanks for helping improve PersonaFinetuner. This guide covers how to propose changes, open issues, and keep the local voice-first workflow healthy.

## Before you start

1. Read the [Code of Conduct](CODE_OF_CONDUCT.md).
2. Search existing [issues](https://github.com/ericcayers-ai/LLM-PersonaFinetuner/issues) and [pull requests](https://github.com/ericcayers-ai/LLM-PersonaFinetuner/pulls) to avoid duplicates.
3. For substantial features or architecture changes, open an issue first so we can align on scope.

## Development setup

```bash
# Windows
.\setup.ps1
.\run.ps1

# Linux / macOS
make setup
make run
```

The app serves at `http://localhost:8000`.

Run tests before opening a pull request:

```bash
pip install pytest httpx
pytest tests/ -v
python -m compileall app tests
```

## Project layout

| Path | Purpose |
|------|---------|
| `app/` | FastAPI backend, training, inference, voice, and storage |
| `frontend/` | Vanilla JS UI (Source → Train → Test → Voice → Call) |
| `tests/` | Parser and API/WebSocket smoke tests |
| `data/` and `outputs/` | Local runtime artifacts (gitignored) |

## Contribution guidelines

### Code

- Prefer extending existing persona, inference, STT, and TTS paths. Avoid parallel speech stacks or duplicate chat endpoints.
- Match nearby style: typed FastAPI routes, consistent `{ error, message, detail, hint }` API errors, and small vanilla ES modules on the frontend.
- Keep voice cloning responsible: preserve consent checks, PCM WAV reference validation, and watermarked Chatterbox output.
- Do not commit secrets, model weights, personal voice samples, chat exports, or `.env` files.

### Docs and templates

- Update `README.md` when behavior, APIs, environment variables, or setup steps change.
- Keep examples reproducible on a local CUDA machine when possible.

### Commits and branches

- Use focused branches and clear commit messages.
- One logical change per pull request when practical.

## Pull request process

1. Fork or create a feature branch from `main`.
2. Make your change with tests for new API behavior when possible.
3. Fill out the pull request template completely.
4. Link related issues with `Fixes #123` / `Closes #123`.
5. Expect review feedback around correctness, docs, and local-first safety.

## Reporting issues

Use the issue templates:

- **Bug report** for broken behavior
- **Feature request** for new capabilities

Include OS, GPU, Python version, and reproduction steps. Scrub private chat and voice content from screenshots or logs.

## Responsible voice AI

Only contribute samples, fixtures, or demos for voices you own or have explicit permission to use. Do not upload third-party speech for cloning without consent.

## Questions

Open a GitHub issue for project questions. Report Code of Conduct concerns to eric.c.ayers@gmail.com.
