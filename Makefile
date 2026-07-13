.PHONY: setup run test lint clean

PYTHON := .venv/Scripts/python
PIP := .venv/Scripts/pip
UVICORN := .venv/Scripts/uvicorn
PORT ?= 8000

setup:
	python -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install torch --index-url https://download.pytorch.org/whl/cu124
	$(PIP) install -r requirements.txt

run:
	$(UVICORN) app.main:app --host 127.0.0.1 --port $(PORT) --reload

test:
	$(PYTHON) -m pytest tests/ -v

lint:
	$(PYTHON) -m compileall app tests

clean:
	rm -rf .pytest_cache __pycache__ app/__pycache__ tests/__pycache__
