.PHONY: install lint format typecheck test

install:
	uv venv --python 3.12
	uv sync

lint:
	uv run ruff check .

format:
	uv run ruff check --fix .

typecheck:
	uv run basedpyright

test:
	sudo $(CURDIR)/.venv/bin/python -m pytest
