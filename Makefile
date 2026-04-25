install:
	uv sync

run:
	uv run python -m src

debug:
	uv run python -m pdb -m src

setup:
	rm -rf .venv
	rm -rf ~/.cache/uv
	rm -rf ~/.cache/huggingface

	mkdir -p ~/goinfre/uv
	mkdir -p ~/goinfre/huggingface
	mkdir -p ~/goinfre/venvs

	mkdir -p ~/.cache

	ln -sfn ~/goinfre/uv ~/.cache/uv
	ln -sfn ~/goinfre/huggingface ~/.cache/huggingface

	mkdir -p ~/goinfre/venvs/call_me_maybe
	ln -s ~/goinfre/venvs/call_me_maybe .venv

	UV_PROJECT_ENVIRONMENT=~/goinfre/venvs/call_me_maybe uv sync

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


# for run the moulinette:
# 	uv run python3 -m src --functions_definition moulinette/data/input/functions_definition.json --input moulinette/data/input/function_calling_tests.json
# 	 uv run python -m moulinette grade_student_answers ~/github_project/Call-Me-Maybe/data/output/function_calls.json --set private
