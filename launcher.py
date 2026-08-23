"""Standalone launcher, packaged into PersonaFinetuner.exe via PyInstaller.

Mirrors run.ps1/run.bat: it does NOT bundle torch/unsloth/transformers into
the exe (those are multi-GB CUDA-linked packages that PyInstaller cannot
package reliably). Instead the exe launches the project's existing .venv
(built by setup.bat/setup.ps1) and opens the browser, exactly like the
current launch flow -- just double-clickable instead of a script.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import webbrowser
from pathlib import Path


def app_root() -> Path:
    if getattr(sys, "frozen", False):
        # PyInstaller onefile: the exe lives next to the project directory
        # the user launches it from (they copy/keep it in the repo root).
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def main() -> int:
    root = app_root()
    venv_python = root / ".venv" / "Scripts" / "python.exe"

    if not venv_python.exists():
        print("=" * 60)
        print("PersonaFinetuner: virtual environment not found.")
        print(f"Expected: {venv_python}")
        print("Run setup.bat (or setup.ps1) once before launching the exe.")
        print("=" * 60)
        input("Press Enter to exit...")
        return 1

    port = os.environ.get("PF_PORT", "8000")
    url = f"http://127.0.0.1:{port}"

    print(f"PersonaFinetuner starting at {url} (close this window to stop)")

    proc = subprocess.Popen(
        [str(venv_python), "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", port],
        cwd=str(root),
    )

    time.sleep(1.5)
    webbrowser.open(url)

    try:
        return proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
