"""Voice cloning pipeline: SANITIZE -> TTS BACKEND -> UPSCALE -> DENOISE.

Ported from voice_clone_mini.py and voice_clone_10.py. Every heavy TTS/
upscaling engine runs in its own isolated `uv run` subprocess so conflicting
numpy/torch/python-version requirements never touch each other or this app's
own environment.
"""
