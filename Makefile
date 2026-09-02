# =============================================================================
# Automatos AI — local edition
# =============================================================================
# Every target that builds also cleans up after itself: a rebuild leaves the
# previous image untagged and grows the build cache, which is what turns a
# 3.7 GB stack into 12 GB of disk. `make up` never leaves that behind.
#
#   make up      build (if needed), start, tidy up
#   make dev     the same with hot reload (source-mounted dev images)
#   make down    stop the stack, keep your data
#   make clean   reclaim dangling images + unused build cache (never your data)
#   make reset   DESTRUCTIVE — stop and delete all data volumes
#   make status  what is running, and what it costs on disk
#
# Nothing here is required: plain `docker compose up` still works.
# =============================================================================

COMPOSE      ?= docker compose
DEV_COMPOSE  ?= docker compose -f docker-compose.yml -f docker-compose.dev.yml

.PHONY: up dev down clean reset status logs cli-host

up:
	$(COMPOSE) up -d --build --remove-orphans
	@$(MAKE) --no-print-directory clean
	@$(MAKE) --no-print-directory status

dev:
	$(DEV_COMPOSE) up -d --build --remove-orphans
	@$(MAKE) --no-print-directory clean

down:
	$(COMPOSE) down --remove-orphans

# Safe at any time, stack up or down:
#   - dangling images only. NEVER `-a`, which deletes images your other
#     projects still need.
#   - unused build cache. Cached layers still referenced by a live build stay.
#   - NO volume pruning. A stopped stack's named volumes look "unused" to
#     Docker, so `docker volume prune` here would delete your database. Data is
#     only ever removed by `make reset`, which asks first.
clean:
	@before=$$(docker system df --format '{{.Type}} {{.Size}}' 2>/dev/null | tr '\n' ' '); \
	docker image prune -f >/dev/null; \
	docker builder prune -f >/dev/null; \
	echo "→ cleaned: dangling images and unused build cache reclaimed."

reset:
	@echo "This deletes the database, object storage and all local data. Ctrl-C to abort."
	@read -r -p "Type 'reset' to confirm: " ans; [ "$$ans" = "reset" ] || { echo "aborted"; exit 1; }
	$(COMPOSE) down -v --remove-orphans
	@$(MAKE) --no-print-directory clean

status:
	@$(COMPOSE) ps --format '{{.Service}}\t{{.Status}}'
	@echo ""
	@docker system df

# PRD-234 Session mode — run tickets as YOUR OWN Claude Code sessions on this
# machine (local edition only; CLI_RUNTIME_ENABLED=true in .env).
#   make cli-host PAIR=XXXX-XXXX   first time — the code comes from Settings → Session mode
#   make cli-host                  afterwards
# Registers ./workspaces (the compose default) as a working directory. Add real
# repositories with CLI_HOST_ARGS="--allow /path/to/repo". Standard library
# Python 3.9+; nothing to install. Stop with Ctrl-C.
cli-host:
	@mkdir -p "$(CURDIR)/workspaces"
	@cd services/cli-host && python3 -m automatos_cli_host --allow "$(CURDIR)/workspaces" $(if $(PAIR),--pair $(PAIR),) $(CLI_HOST_ARGS)
