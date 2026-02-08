.PHONY: test test-coverage test-unit test-integration help

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

# Show help
help:
	@echo "Available targets:"
	@echo "  test           - Run all tests with coverage"
	@echo "  test-coverage  - Run all tests with coverage (same as test)"
	@echo "  test-unit      - Run unit tests only"
	@echo "  test-integration - Run integration tests only"
	@echo "  help           - Show this help message"
