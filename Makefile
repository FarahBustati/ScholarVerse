.PHONY: setup test lint format demo serve

setup:
	uv sync

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .
	uv run ruff check --fix .

demo:
	SCHOLARPATH_FAKE_LLM=1 uv run python -m scholarpath demo

serve:
	uv run uvicorn scholarpath.api:app --reload
