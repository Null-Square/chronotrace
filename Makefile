.PHONY: install install-mvp check test lint doctor audit assets assets-check reviewer reviewer-full paper

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

audit:
	python scripts/audit_release.py

assets:
	python scripts/generate_release_assets.py --write

assets-check:
	python scripts/generate_release_assets.py --check

reviewer: audit assets-check check

paper:
	$(MAKE) -C paper pdf

reviewer-full: reviewer paper
