# =============================================================================
# Automatos AI — local edition
# =============================================================================
# Every target that builds also cleans up after itself: a rebuild leaves the
# previous image untagged and grows the build cache, which is what turns a
# 3.7 GB stack into 12 GB of disk. `make up` never leaves that behind. The
# build cache of the last week stays, so the next source build after a git pull
# reuses its dependency installs (PRD-262 US-039).
#
#   make env     check Docker first: Engine 25 or later and Compose 2.24.4 or
#                later, or stop and say so (PRD-262 US-038). Then create .env
#                from .env.example and fill POSTGRES_PASSWORD, REDIS_PASSWORD and
#                API_KEY with random values while they are empty or CHANGE_ME,
#                before the first start only (PRD-262 US-037).
#                make up, make build and make dev run it first.
#   make up      the published images for this checkout, pulled from GHCR, then
#                start and tidy up (PRD-262 US-036). It picks one tag for the
#                three app images, says which and why, builds only what CI has
#                not published, and writes the tag to .env (AUTOMATOS_IMAGE_TAG)
#                so plain docker compose runs the same images. Pin a tag with
#                AUTOMATOS_IMAGE_TAG=sha-xxxxxxx make up (edge follows edge).
#   make build   build the three app images from your checkout instead, as the
#                production stages CI publishes, under a local-<sha7> tag. postgres,
#                redis and minio start first, so they come up while the images build
#                (PRD-262 US-039)
#   make dev     for editing the code: the development stages with your source
#                mounted, uvicorn --reload and the Next.js dev server
#                (docker-compose.dev.yml)
#   make down    stop the stack, keep your data
#   make clean   reclaim dangling images + build cache unused for a week (never your data)
#   make reset   DESTRUCTIVE — stop and delete all data volumes
#   make status  what is running, and what it costs on disk
#   make logs    follow the logs (make logs s=backend for one service)
#
# Nothing here is required: plain `docker compose up` still works (the tag in
# .env, or :edge), and self-hosting.md §1b gives the commands behind each target.
# =============================================================================

COMPOSE      ?= docker compose
DEV_COMPOSE  ?= docker compose -f docker-compose.yml -f docker-compose.dev.yml
# Picks the app images' tag for make up and make build (PRD-262 US-036).
STACK_IMAGES ?= python3 scripts/stack_images.py
# Writes .env's three secrets for make env (PRD-262 US-037).
STACK_ENV    ?= sh scripts/stack_env.sh
# Stops make env, and so make up, build and dev, on a Docker older than the floor:
# Engine 25 and Compose 2.24.4, which the compose healthchecks need (PRD-262 US-038).
STACK_FLOOR  ?= sh scripts/stack_docker_floor.sh

.PHONY: env up build dev down clean reset status logs cli-host cli-host-install cli-host-uninstall cli-host-status cli-host-restart cli-host-nudge

# The commit a local build is made from: docker-compose.yml stamps it on each app
# image as org.opencontainers.image.revision (PRD-262 US-036). HEAD, with -dirty
# when orchestrator, frontend or services/workspace-worker has uncommitted
# changes; empty outside a git checkout, where compose stamps `unknown`. Set
# once, when make starts, and passed to every recipe.
APP_CONTEXTS := orchestrator frontend services/workspace-worker
ifeq ($(origin AUTOMATOS_REVISION),undefined)
AUTOMATOS_REVISION := $(shell git rev-parse --verify -q HEAD 2>/dev/null)$(if $(shell git status --porcelain -- $(APP_CONTEXTS) 2>/dev/null),-dirty,)
endif
export AUTOMATOS_REVISION

# An empty AUTOMATOS_IMAGE_TAG in the shell would make compose run :edge instead
# of the tag make up wrote to .env, so an empty one is not passed on.
ifeq ($(strip $(AUTOMATOS_IMAGE_TAG)),)
unexport AUTOMATOS_IMAGE_TAG
endif

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

# First the Docker floor (scripts/stack_docker_floor.sh): docker-compose.yml's
# healthchecks probe every second while a service starts (start_interval), which
# Compose refuses on an engine older than 25, so an older Docker stops here with the
# floor named, before anything else runs. Then .env with the three secrets
# (scripts/stack_env.sh): generated before the first start, never changed after it.
# An install that started on a CHANGE_ME placeholder stops here with how to rotate it.
# up, build and dev run it as their first recipe line, not as a prerequisite, so the
# #818 guide test still parses `up:`.
env:
	@$(STACK_FLOOR)
	@$(STACK_ENV)

up:
	@$(MAKE) --no-print-directory env
	@mkdir -p "$(AUTOMATOS_WORKSPACE_DIR)/projects"
	$(migrate_workspace_layout)
	@$(STACK_IMAGES) up
	$(COMPOSE) up -d --no-build --remove-orphans
	@$(MAKE) --no-print-directory clean
	@$(MAKE) --no-print-directory status
	@$(MAKE) --no-print-directory cli-host-nudge

# Builds the three app images from this checkout (the production stages CI
# publishes) under local-<sha7>, written to .env first so the build and every
# later compose command use it. Compose never builds under an official tag.
# postgres, redis and minio start before the build (PRD-262 US-039), so their
# first-run initdb and health gate pass while the images build, not after them;
# the backend still waits for all three to be healthy (depends_on). A failed
# build leaves those three running: make down stops them.
build:
	@$(MAKE) --no-print-directory env
	@mkdir -p "$(AUTOMATOS_WORKSPACE_DIR)/projects"
	$(migrate_workspace_layout)
	@$(STACK_IMAGES) build
	$(COMPOSE) up -d postgres redis minio
	$(COMPOSE) build backend frontend workspace-worker
	$(COMPOSE) up -d --no-build --remove-orphans
	@$(MAKE) --no-print-directory clean
	@$(MAKE) --no-print-directory status
	@$(MAKE) --no-print-directory cli-host-nudge

dev:
	@$(MAKE) --no-print-directory env
	@mkdir -p "$(AUTOMATOS_WORKSPACE_DIR)/projects"
	$(migrate_workspace_layout)
	$(DEV_COMPOSE) up -d --build --remove-orphans
	@$(MAKE) --no-print-directory clean

down:
	$(COMPOSE) down --remove-orphans

# Safe at any time, stack up or down:
#   - dangling images only. NEVER `-a`, which deletes images your other
#     projects still need.
#   - build cache nothing has used for a week (until=168h). A plain
#     `docker builder prune -f` deletes every cache record no image holds, and
#     the dependency installs are in no image: the frontend's npm install and
#     the backend's pip install run in build stages whose output the production
#     images only copy. The next make build or make dev after a git pull then
#     installed every dependency again. The week keeps them; CI's
#     source-build-cache job rebuilds after this target and needs both CACHED
#     (PRD-262 US-039). `docker builder prune -f` alone still reclaims it all.
#   - NO volume pruning. A stopped stack's named volumes look "unused" to
#     Docker, so `docker volume prune` here would delete your database. Data is
#     only ever removed by `make reset`, which asks first.
clean:
	@before=$$(docker system df --format '{{.Type}} {{.Size}}' 2>/dev/null | tr '\n' ' '); \
	docker image prune -f >/dev/null; \
	docker builder prune -f --filter until=168h >/dev/null; \
	echo "→ cleaned: dangling images and build cache unused for a week reclaimed."

reset:
	@echo "This deletes the database, object storage and all local data. Ctrl-C to abort."
	@read -r -p "Type 'reset' to confirm: " ans; [ "$$ans" = "reset" ] || { echo "aborted"; exit 1; }
	$(COMPOSE) down -v --remove-orphans
	@$(MAKE) --no-print-directory clean

status:
	@$(COMPOSE) ps --format '{{.Service}}\t{{.Status}}'
	@echo ""
	@docker system df

# Follow the stack's logs. `make logs s=backend` limits it to one service.
logs:
	@$(COMPOSE) logs -f --tail=200 $(s)

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
	@cd services/cli-host && python3 -m automatos_cli_host --allow "$(AUTOMATOS_WORKSPACE_DIR)" --default-root "$(AUTOMATOS_WORKSPACE_DIR)" $(if $(LOCAL_PROJECTS_DIR),--allow "$(LOCAL_PROJECTS_DIR)",) $(if $(PAIR),--pair $(PAIR),) $(CLI_HOST_ARGS)

# PRD-235 W3: the host as a login service — starts at login, restarts on exit,
# restarts itself when its code or the backend's contract changed. Pair once
# with `make cli-host PAIR=<code>` (Ctrl-C after "paired"), then install.
cli-host-install:
	@mkdir -p "$(AUTOMATOS_WORKSPACE_DIR)"
	@cd services/cli-host && python3 -m automatos_cli_host --install --allow "$(AUTOMATOS_WORKSPACE_DIR)" --default-root "$(AUTOMATOS_WORKSPACE_DIR)" $(if $(LOCAL_PROJECTS_DIR),--allow "$(LOCAL_PROJECTS_DIR)",) $(CLI_HOST_ARGS)

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
