# Makefile Usage Guide

This Makefile provides convenient commands for managing the Mythologizer PostgreSQL project. It integrates with the CLI commands and provides comprehensive development workflows.

## Quick Start

```bash
# Show all available commands
make help

# Complete development setup
make dev-setup

# Start database
make up

# Check database status
make status

# Stop database
make down
```

## Environment Variables

You can customize the behavior by setting these environment variables:

```bash
# Use a different environment file
ENV_FILE=.env.prod make db-start

# Use a different docker compose file
COMPOSE_FILE=docker-compose.prod.yaml make db-start

# Set embedding dimensionality
DIM=768 make db-setup

# Use different Python/CLI commands
PYTHON=python3 make test
CLI=mythologizer_postgres make db-status
```

## Database Management

### Basic Operations
```bash
make db-start      # Start database
make db-stop       # Stop database
make db-restart    # Restart database
make db-status     # Check database status
make db-ping       # Test connectivity
```

### Schema and Data
```bash
make db-setup      # Apply database schema
make db-clear      # Clear all data (with confirmation)
make db-clear-force # Clear all data (no confirmation)
make db-destroy    # Destroy everything (volumes, containers)
```

### Connection Information
```bash
make db-url        # Show connection URL (password hidden)
make db-url-reveal # Show connection URL (with password)
```

## Development Workflows

### Setup
```bash
make install       # Install package in development mode
make dev-install   # Install with development dependencies
make dev-setup     # Complete setup (install + start DB + schema)
```

### Reset Options
```bash
make dev-reset     # Clear data and reapply schema
make dev-fresh     # Destroy everything and start fresh
```

### Quality Assurance
```bash
make format        # Format code (black + isort)
make lint          # Run linting (flake8 + mypy)
make test          # Run tests
make test-cov      # Run tests with coverage
make check         # Run all quality checks
```

## Package Management

```bash
make build         # Build distribution packages
make clean         # Clean build artifacts
make publish       # Build and publish (placeholder)
```

## Utility Commands

```bash
make shell         # Start Python shell with package imported
make logs          # Show database logs
```

## Common Workflows

### New Developer Setup
```bash
# 1. Clone the repository
git clone <repository-url>
cd mythologizerDB

# 2. Complete setup
make dev-setup

# 3. Verify everything works
make status
```

### Daily Development
```bash
# Start working
make up
make status

# Make changes and test
make test

# Reset if needed
make reset

# Stop when done
make down
```

### Code Quality
```bash
# Before committing
make check

# Or run individually
make format
make lint
make test
```

### Troubleshooting
```bash
# If database is in a bad state
make fresh

# If you need to see what's happening
make logs

# If you need to check connectivity
make db-ping
```

## Command Aliases

For convenience, these shorter commands are available:

```bash
make up      # Alias for db-start
make down    # Alias for db-stop
make status  # Alias for db-status
make setup   # Alias for db-setup
make reset   # Alias for dev-reset
make fresh   # Alias for dev-fresh
```

## Integration with CLI

The Makefile uses your CLI commands under the hood:

- `make db-start` → `uv run mythologizer_postgres start`
- `make db-status` → `uv run mythologizer_postgres status`
- `make db-setup` → `uv run mythologizer_postgres setup --dim 384`

This ensures consistency between direct CLI usage and Makefile commands. 