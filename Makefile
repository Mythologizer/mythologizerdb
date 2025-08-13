ENV_FILE ?= .env.test
COMPOSE   = docker compose --env-file $(ENV_FILE) -f docker-compose.test.yaml
UV        = uv run --env-file $(ENV_FILE)

.PHONY: build fresh resetdb apply_schemas up down wait test_setup test

# 1) force a clean image build
build:
	$(COMPOSE) build --pull --no-cache

# 2) bring everything up fresh (new containers, new anon volumes)
fresh: down build
	$(COMPOSE) up -d --force-recreate --renew-anon-volumes
	$(MAKE) wait apply_schemas

# 3) in case you want to keep the container but wipe schema
resetdb:
	$(COMPOSE) exec -T postgres psql -U $$POSTGRES_USER -d $$POSTGRES_DB \
		-c 'DROP SCHEMA public CASCADE; CREATE SCHEMA public;'
	$(MAKE) apply_schemas

# 3b) do migrations strictly from base to head (no relying on drop)
apply_schemas:
	$(UV) mythologizer_postgres setup --dim $(shell grep EMBEDDING_DIM $(ENV_FILE) | cut -d'=' -f2)

up:
	$(COMPOSE) up -d

down:
	$(COMPOSE) down -v --remove-orphans

wait:
	@echo "Waiting for test DB to be ready..."
	@until $(COMPOSE) exec -T postgres pg_isready -U $$POSTGRES_USER -d $$POSTGRES_DB >/dev/null 2>&1; do \
		sleep 1; \
	done
	@echo "DB is ready."

test_setup:
	$(UV) pytest tests

.PHONY: benchmark
benchmark: fresh
	$(UV) python mythologizer_postgres/benchmark.py

.PHONY: benchmark-quick
benchmark-quick:
	$(UV) python mythologizer_postgres/benchmark.py

.PHONY: test
test: fresh test_setup
	$(COMPOSE) down -v --remove-orphans
