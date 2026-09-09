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

.PHONY: up dev down clean reset status logs cli-host cli-host-install cli-host-uninstall cli-host-status cli-host-restart cli-host-nudge

# PRD-234 S2: LOCAL_PROJECTS_DIR (root .env) is the owner's projects folder; the
# default bind source ./workspaces/projects must exist before compose mounts it.
LOCAL_PROJECTS_DIR ?= $(shell sed -n 's/^LOCAL_PROJECTS_DIR=//p' .env 2>/dev/null | tail -1 | tr -d '"')
# The deliverables root (root .env AUTOMATOS_WORKSPACE_DIR, default ./workspaces):
# compose mounts it as the local workspace's root, so your machine sees
# artifacts/, reports/, sessions/… directly. Exported ABSOLUTE so the backend can
# map session file paths and Settings → Session mode can show it.
AUTOMATOS_WORKSPACE_DIR ?= $(shell sed -n 's/^AUTOMATOS_WORKSPACE_DIR=//p' .env 2>/dev/null | tail -1 | tr -d '"')
ifeq ($(strip $(AUTOMATOS_WORKSPACE_DIR)),)
AUTOMATOS_WORKSPACE_DIR := $(CURDIR)/workspaces
endif
AUTOMATOS_WORKSPACE_DIR := $(patsubst ~%,$(HOME)%,$(AUTOMATOS_WORKSPACE_DIR))
export AUTOMATOS_WORKSPACE_DIR
DEFAULT_WORKSPACE_ID ?= $(shell sed -n 's/^DEFAULT_WORKSPACE_ID=//p' .env 2>/dev/null | tail -1 | tr -d '"')
ifeq ($(strip $(DEFAULT_WORKSPACE_ID)),)
DEFAULT_WORKSPACE_ID := 00000000-0000-0000-0000-0000000000c1
endif

# One-time layout migration: before 2026-09-09 the host folder held
# <workspace_id>/artifacts/…; it now IS the workspace root. Move the old
# subtree up one level, once, and only when the root has no layout of its own.
define migrate_workspace_layout
@if [ -d "$(AUTOMATOS_WORKSPACE_DIR)/$(DEFAULT_WORKSPACE_ID)" ] && [ ! -d "$(AUTOMATOS_WORKSPACE_DIR)/artifacts" ]; then \
  echo "deliverables: moving $(AUTOMATOS_WORKSPACE_DIR)/$(DEFAULT_WORKSPACE_ID)/* up to $(AUTOMATOS_WORKSPACE_DIR)/ (the workspace root is the folder itself now)"; \
  for entry in "$(AUTOMATOS_WORKSPACE_DIR)/$(DEFAULT_WORKSPACE_ID)"/* "$(AUTOMATOS_WORKSPACE_DIR)/$(DEFAULT_WORKSPACE_ID)"/.[!.]*; do \
    [ -e "$$entry" ] || continue; \
    case "$$(basename "$$entry")" in projects) rmdir "$$entry" 2>/dev/null || true; continue;; esac; \
    mv -n "$$entry" "$(AUTOMATOS_WORKSPACE_DIR)/"; \
  done; \
  rmdir "$(AUTOMATOS_WORKSPACE_DIR)/$(DEFAULT_WORKSPACE_ID)" 2>/dev/null || echo "deliverables: $(AUTOMATOS_WORKSPACE_DIR)/$(DEFAULT_WORKSPACE_ID) kept (not empty) — check it and remove it by hand"; \
fi
endef

up:
	@mkdir -p "$(AUTOMATOS_WORKSPACE_DIR)/projects"
	$(migrate_workspace_layout)
	$(COMPOSE) up -d --build --remove-orphans
	@$(MAKE) --no-print-directory clean
	@$(MAKE) --no-print-directory status
	@$(MAKE) --no-print-directory cli-host-nudge

dev:
	@mkdir -p "$(AUTOMATOS_WORKSPACE_DIR)/projects"
	$(migrate_workspace_layout)
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
# Registers the deliverables root (AUTOMATOS_WORKSPACE_DIR, default ./workspaces)
# as a working directory; a ticket without one runs in <that root>/sessions/<ticket>,
# which Deliverables → Explorer shows live. Your own repositories: LOCAL_PROJECTS_DIR
# in .env (browsable under projects/) or CLI_HOST_ARGS="--allow /path/to/repo". Standard library
# Python 3.9+; nothing to install. Stop with Ctrl-C.
cli-host:
	@mkdir -p "$(AUTOMATOS_WORKSPACE_DIR)"
	@cd services/cli-host && python3 -m automatos_cli_host --allow "$(AUTOMATOS_WORKSPACE_DIR)" $(if $(LOCAL_PROJECTS_DIR),--allow "$(LOCAL_PROJECTS_DIR)",) $(if $(PAIR),--pair $(PAIR),) $(CLI_HOST_ARGS)

# PRD-235 W3: the host as a login service — starts at login, restarts on exit,
# restarts itself when its code or the backend's contract changed. Pair once
# with `make cli-host PAIR=<code>` (Ctrl-C after "paired"), then install.
cli-host-install:
	@mkdir -p "$(AUTOMATOS_WORKSPACE_DIR)"
	@cd services/cli-host && python3 -m automatos_cli_host --install --allow "$(AUTOMATOS_WORKSPACE_DIR)" $(if $(LOCAL_PROJECTS_DIR),--allow "$(LOCAL_PROJECTS_DIR)",) $(CLI_HOST_ARGS)

cli-host-uninstall:
	@cd services/cli-host && python3 -m automatos_cli_host --uninstall

cli-host-status:
	@cd services/cli-host && python3 -m automatos_cli_host --service-status

cli-host-restart:
	@cd services/cli-host && python3 -m automatos_cli_host --restart-service

# After the app rebuilt: a running host drains and comes back on the new code.
# Silent when no host is running.
cli-host-nudge:
	-@cd services/cli-host && python3 -m automatos_cli_host --nudge >/dev/null 2>&1 || true
