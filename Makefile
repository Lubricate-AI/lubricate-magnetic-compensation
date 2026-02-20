VENV_PATH ?= .venv
SOURCE_PATH ?= lmc
TEST_PATH ?= tests

.PHONY: help install lint lint-python lint-spellcheck lint-yaml lint-security type-checking format test coverage docs-serve docs-build

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Create uv environment and install dependencies
	uv sync --dev

lint: lint-python lint-spellcheck lint-yaml lint-security type-checking ## Run all linting checks

lint-python: ## Run ruff linting and formatting checks (passive)
	uv run ruff check $(SOURCE_PATH) $(TEST_PATH)
	uv run ruff format --check $(SOURCE_PATH) $(TEST_PATH)

lint-spellcheck: ## Run typos spellchecker
	uv run typos .

lint-yaml: ## Run YAML linting
	uv run yamllint -d "{extends: relaxed, rules: {line-length: {max: 120}}}" .github

lint-security: ## Run bandit security linting
	uv run bandit -c pyproject.toml -r $(SOURCE_PATH)

type-checking: ## Run pyright type checking
	uv run pyright $(SOURCE_PATH) $(TEST_PATH)

format: ## Run ruff linting and formatting (active fixes)
	uv run ruff check --fix $(SOURCE_PATH) $(TEST_PATH)
	uv run ruff check --select I --fix $(SOURCE_PATH) $(TEST_PATH)
	uv run ruff format $(SOURCE_PATH) $(TEST_PATH)

test: ## Run all unit tests
	uv run pytest $(TEST_PATH)

coverage: ## Run tests with coverage report
	uv run pytest --cov=$(SOURCE_PATH) --cov-report=term-missing $(TEST_PATH)

docs-serve: ## Serve documentation locally with live reload
	uv run mkdocs serve

docs-build: ## Build documentation to site/ directory
	uv run mkdocs build
