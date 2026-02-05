.PHONY: setup test docs lint clean

setup:
	uv sync --extra dev --extra ml

test:
	uv run pytest tests

lint:
	uv run ruff tasa_churn tests

docs:
	sphinx-apidoc -o docs/source/ tasa_churn
	make html -C docs

clean:
	rm -rf models/*
