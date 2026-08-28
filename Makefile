.PHONY: install install-mvp check test lint doctor

install:
	python -m pip install -e ".[dev]"

install-mvp:
	python -m pip install -e ".[dev,mvp]"

check: lint test

lint:
	ruff check .

test:
	pytest

doctor:
	python scripts/doctor.py
