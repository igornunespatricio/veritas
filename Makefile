.PHONY: test test-coverage test-unit test-integration lint lint-fix format check help

# Run all tests with coverage
test:
	PYTHONPATH=. pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing

# Run all tests with coverage (alias)
test-coverage:
	PYTHONPATH=. pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing

# Run only unit tests
test-unit:
	PYTHONPATH=. pytest tests/unit/ -v --cov=src --cov-report=html --cov-report=term-missing

# Run only integration tests
test-integration:
	PYTHONPATH=. pytest tests/integration/ -v --cov=src --cov-report=html --cov-report=term-missing

# Run ruff linter
lint:
	ruff check src/ tests/

# Run ruff with auto-fix
lint-fix:
	ruff check --fix src/ tests/

# Run ruff formatter
format:
	ruff format src/ tests/

# Run all code quality checks
check: lint format

# Show help
help:
	@echo "Available targets:"
	@echo "  test           - Run all tests with coverage"
	@echo "  test-coverage  - Run all tests with coverage (same as test)"
	@echo "  test-unit      - Run unit tests only"
	@echo "  test-integration - Run integration tests only"
	@echo "  lint           - Run ruff linter"
	@echo "  lint-fix       - Run ruff linter with auto-fix"
	@echo "  format         - Run ruff formatter"
	@echo "  check          - Run lint and format checks"
	@echo "  help           - Show this help message"
