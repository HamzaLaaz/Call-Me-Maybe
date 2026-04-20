install:
	uv sync

run:
	uv run python -m src

debug:
	uv run python -m pdb -m src

setup:
	mkdir -p ~/goinfre/uv
	mkdir -p ~/goinfre/.venv
	mkdir -p ~/goinfre/huggingface

	mkdir -p ~/.cache

	rm -rf ~/.cache/uv
	ln -sfn ~/goinfre/uv ~/.cache/uv

	rm -rf ~/.cache/huggingface
	ln -sfn ~/goinfre/huggingface ~/.cache/huggingface

	rm -rf ~/.cache/.venv
	ln -sfn ~/goinfre/.venv ~/.cache/.venv

lint:
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores \
	    --ignore-missing-imports --disallow-untyped-defs \
	    --check-untyped-defs

lint-strict:
	uv run flake8 .
	uv run mypy . --strict

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf .mypy_cache

.PHONY: install run debug lint clean
