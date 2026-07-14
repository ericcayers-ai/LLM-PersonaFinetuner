# Contributing to PersonaFinetuner

Thanks for your interest in contributing. This guide covers setup, workflow, and what kinds of changes are welcome.

Please follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## Development setup

**Windows (recommended):**

1. Double-click `setup.bat` or run `.\setup.ps1` (first time only)
2. Double-click `run.bat` or run `.\run.ps1`
3. Open http://localhost:8000

**Linux/macOS:**

```bash
make setup && make run
```

**Manual install** details are in the [README](README.md). You need Python 3.10+ and, for training, an NVIDIA GPU with CUDA.

## Branch and pull request workflow

1. Open an issue first for larger changes (use the [bug](.github/ISSUE_TEMPLATE/bug_report.yml) or [feature](.github/ISSUE_TEMPLATE/feature_request.yml) templates when applicable).
2. Fork the repo (or create a branch if you have write access).
3. Create a focused branch from `main`, e.g. `fix/parser-whatsapp` or `docs/contributing`.
4. Make your changes with clear commits.
5. Open a pull request against `main` and fill out the [PR template](.github/PULL_REQUEST_TEMPLATE.md).
6. Keep PRs small and reviewable when possible.

## Coding conventions

- **Backend:** Python FastAPI under `app/`. Prefer clear modules, typed schemas in `app/models/`, and consistent API error shapes (`{ error, message, detail, hint }`).
- **Frontend:** Vanilla JS under `frontend/` (no framework). Keep step modules in `frontend/js/steps/` and shared API helpers in `frontend/js/api.js`.
- **Parsers:** Chat/text parsers live in `app/core/parsers.py`. Match existing patterns for format detection and target-username filtering.
- Prefer readable names and small functions over clever abstractions. Match the style of nearby code.

## Testing

Install test deps and run the suite:

```bash
pip install pytest httpx
pytest tests/ -v
```

- Add or update tests under `tests/` for parser changes and API behavior.
- Include briefly how you verified UI changes (e.g. Source → Train → Test wizard).

## What kinds of PRs are welcome

- New or improved **upload/chat parsers** (Discord, WhatsApp, Instagram, etc.)
- **Bug fixes** and error-handling improvements
- **QOL** for the local wizard UI and one-click Windows launchers
- **Docs** (README, troubleshooting, contribution docs)
- Tests that catch format or API regressions

Please do **not** commit secrets, personal chat dumps, trained adapters, or anything under `data/` / `outputs/` (those paths are gitignored for a reason).

## Questions

Use GitHub Issues for bugs and feature ideas. For conduct concerns, see [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
