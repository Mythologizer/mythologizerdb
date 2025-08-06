# Mythologizer PostgreSQL Makefile
# Provides convenient commands for development and database management

# Configuration
ENV_FILE ?= .env
COMPOSE_FILE ?= docker-compose.yaml
DIM ?= 384
PYTHON ?= uv run python
CLI ?= uv run mythologizer_postgres

# Colors for output
GREEN = \033[0;32m
RED = \033[0;31m
YELLOW = \033[1;33m
BLUE = \033[0;34m
NC = \033[0m # No Color

.PHONY: help install build dev-install clean test lint format check

# Default target
help: ## Show this help message
	@echo "$(BLUE)Mythologizer PostgreSQL - Available Commands:$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-15s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(YELLOW)Environment Variables:$(NC)"
	@echo "  ENV_FILE      Database environment file (default: .env)"
	@echo "  COMPOSE_FILE  Docker compose file (default: docker-compose.yaml)"
	@echo "  DIM           Embedding dimensionality (default: 384)"
	@echo "  PYTHON        Python command (default: uv run python)"
	@echo "  CLI           CLI command (default: uv run mythologizer_postgres)"

# Development Setup
install: ## Install package in development mode
	@echo "$(BLUE)Installing package in development mode...$(NC)"
	uv pip install -e .

dev-install: install ## Install package and dependencies for development
	@echo "$(BLUE)Installing development dependencies...$(NC)"
	uv pip install -e ".[dev]"

build: ## Build the package distribution
	@echo "$(BLUE)Building package distribution...$(NC)"
	uv build

clean: ## Clean build artifacts
	@echo "$(BLUE)Cleaning build artifacts...$(NC)"
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -name __pycache__ -type d -not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -type f -not -path "./.venv/*" -delete 2>/dev/null || true

# Database Management
db-start: ## Start the database using docker compose
	@echo "$(BLUE)Starting database...$(NC)"
	$(CLI) start --file $(COMPOSE_FILE) --env-file $(ENV_FILE)

db-stop: ## Stop the database containers
	@echo "$(BLUE)Stopping database...$(NC)"
	$(CLI) stop --file $(COMPOSE_FILE)

db-restart: db-stop db-start ## Restart the database

db-destroy: ## Stop containers and remove all volumes (deletes persistent data)
	@echo "$(RED)WARNING: This will delete all persistent data!$(NC)"
	$(CLI) destroy --file $(COMPOSE_FILE) --env-file $(ENV_FILE) --yes

db-status: ## Check database status and table information
	@echo "$(BLUE)Checking database status...$(NC)"
	$(CLI) status --dim $(DIM)

db-ping: ## Test database connectivity
	@echo "$(BLUE)Testing database connectivity...$(NC)"
	$(CLI) ping

db-url: ## Show database connection URL
	@echo "$(BLUE)Database connection URL:$(NC)"
	$(CLI) show-url

db-url-reveal: ## Show database connection URL with password
	@echo "$(BLUE)Database connection URL (with password):$(NC)"
	$(CLI) show-url --reveal-password

# Database Schema and Data
db-setup: ## Apply database schema for the given dimensionality
	@echo "$(BLUE)Setting up database schema with dimension $(DIM)...$(NC)"
	$(CLI) setup --dim $(DIM)

db-clear: ## Delete all rows from all tables (with confirmation)
	@echo "$(RED)WARNING: This will remove all data from all tables!$(NC)"
	$(CLI) clear

db-clear-force: ## Delete all rows from all tables (no confirmation)
	@echo "$(RED)Clearing all data from database...$(NC)"
	$(CLI) clear --yes

# Development Workflows
dev-setup: dev-install db-start ## Complete development setup
	@echo "$(BLUE)Waiting for database to be ready...$(NC)"
	@until $(CLI) ping >/dev/null 2>&1; do sleep 2; done
	@echo "$(GREEN)Database is ready!$(NC)"
	@echo "$(BLUE)Setting up database schema...$(NC)"
	$(CLI) setup --dim $(DIM)
	@echo "$(GREEN)Development environment is ready!$(NC)"

dev-reset: db-clear-force db-setup ## Reset database (clear data and reapply schema)
	@echo "$(GREEN)Database reset complete!$(NC)"

dev-fresh: db-destroy db-start ## Fresh start (destroy and recreate everything)
	@echo "$(BLUE)Waiting for database to be ready...$(NC)"
	@until $(CLI) ping >/dev/null 2>&1; do sleep 2; done
	@echo "$(GREEN)Database is ready!$(NC)"
	@echo "$(BLUE)Setting up database schema...$(NC)"
	$(CLI) setup --dim $(DIM)
	@echo "$(GREEN)Fresh development environment is ready!$(NC)"

# Testing and Quality
test: ## Run tests
	@echo "$(BLUE)Running tests...$(NC)"
	$(PYTHON) -m pytest tests/ -v

test-cov: ## Run tests with coverage
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	$(PYTHON) -m pytest tests/ -v --cov=mythologizer_postgres --cov-report=html --cov-report=term

lint: ## Run linting
	@echo "$(BLUE)Running linting...$(NC)"
	$(PYTHON) -m flake8 mythologizer_postgres/ tests/
	$(PYTHON) -m mypy mythologizer_postgres/

format: ## Format code
	@echo "$(BLUE)Formatting code...$(NC)"
	$(PYTHON) -m black mythologizer_postgres/ tests/
	$(PYTHON) -m isort mythologizer_postgres/ tests/

check: format lint test ## Run all quality checks

# Package Management
publish: build ## Build and publish package (placeholder)
	@echo "$(YELLOW)Package publishing not configured. Run 'make build' to create distribution files.$(NC)"

# Utility Commands
shell: ## Start Python shell with package imported
	@echo "$(BLUE)Starting Python shell...$(NC)"
	$(PYTHON) -c "import mythologizer_postgres; print('Package imported successfully!')"
	$(PYTHON)

logs: ## Show database logs
	@echo "$(BLUE)Showing database logs...$(NC)"
	docker compose -f $(COMPOSE_FILE) logs -f

# Quick Commands
up: db-start ## Alias for db-start
down: db-stop ## Alias for db-stop
status: db-status ## Alias for db-status
setup: db-setup ## Alias for db-setup
reset: dev-reset ## Alias for dev-reset
fresh: dev-fresh ## Alias for dev-fresh
