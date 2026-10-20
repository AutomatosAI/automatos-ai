#!/bin/sh
# =============================================================================
# The Docker floor: Docker Engine 25 and Docker Compose 2.24.4 (PRD-262 US-038)
# =============================================================================
# docker-compose.yml's healthchecks probe each service every second while it starts
# (healthcheck start_interval), so each service starts within about a second of the
# one it waits for being ready. Compose refuses that key on an engine older than 25,
# whose API is older than 1.44 ("can't set healthcheck.start_interval as feature
# require Docker Engine v25 or later"). Compose 2.24.4 is the oldest Compose the
# stack is documented on (optional env_file entries need 2.24).
#
# `make env` runs this first, and `make up`, `make build` and `make dev` run
# `make env` first, so it is the first thing each of them checks.
#
#   sh scripts/stack_docker_floor.sh
#
# POSIX sh and the docker CLI. Exit 0, silently: both are at or above the floor.
# Exit 1: Docker does not answer, Compose v2 is missing, or either is older than the
# floor; the message names the floor and what this machine runs. A version this
# cannot read (a development build) passes with a warning, because Compose itself
# still refuses an engine that is too old.
# =============================================================================
set -eu

ENGINE_FLOOR=25
# The engine API version Docker Engine 25 speaks, which is what Compose checks.
ENGINE_API_FLOOR=1.44
COMPOSE_FLOOR=2.24.4
FLOOR="Docker Engine ${ENGINE_FLOOR} or later and Docker Compose ${COMPOSE_FLOOR} or later"
INSTALL_HINT="Update Docker Desktop, or on Linux install Docker Engine and its compose plugin from Docker's own repository (https://docs.docker.com/engine/install/): a distribution's docker.io package can be older."

# Prints each argument as a line on stderr and exits 1.
stop() {
    for line in "$@"; do echo "$line"; done >&2
    exit 1
}

warn() { echo "Docker floor: $*" >&2; }

# The leading dotted number of a version: v2.24.6-desktop.1 -> 2.24.6, 24.0.7+dfsg1 -> 24.0.7.
numeric() {
    printf '%s\n' "$1" | sed -e 's/^v//' -e 's/[^0-9.].*$//' -e 's/\.*$//'
}

# Whether the dotted version $1 is at least $2, field by field (a missing field is 0).
at_least() {
    have=$1
    need=$2
    while [ -n "$need" ]; do
        h=${have%%.*}
        n=${need%%.*}
        if [ "${h:-0}" -gt "$n" ]; then return 0; fi
        if [ "${h:-0}" -lt "$n" ]; then return 1; fi
        case $have in *.*) have=${have#*.} ;; *) have='' ;; esac
        case $need in *.*) need=${need#*.} ;; *) need='' ;; esac
    done
    return 0
}

# The engine: its API version decides, as it does for Compose; the release names it.
check_engine() {
    answer=$(docker version --format '{{.Server.Version}} {{.Server.APIVersion}}' 2>&1) \
        || stop "Docker does not answer (docker version failed). Start Docker Desktop, or the docker service on Linux (sudo systemctl start docker), then run this again." \
                "docker said: ${answer}"
    release=$(numeric "${answer%% *}")
    api=$(numeric "${answer##* }")
    if [ -n "$api" ]; then
        at_least "$api" "$ENGINE_API_FLOOR" && return 0
    elif [ -n "$release" ]; then
        at_least "$release" "$ENGINE_FLOOR" && return 0
    else
        warn "could not read the Docker Engine version from '${answer}'; carrying on"
        return 0
    fi
    stop "Automatos needs ${FLOOR}: this machine runs Docker Engine ${answer%% *} (API ${answer##* })." \
         "docker-compose.yml probes each service every second while it starts (healthcheck start_interval), and Compose refuses that on an engine older than ${ENGINE_FLOOR}." \
         "$INSTALL_HINT"
}

check_compose() {
    answer=$(docker compose version --short 2>&1) \
        || stop "Automatos needs ${FLOOR}: docker compose version failed, so the Compose v2 plugin is missing (the old docker-compose v1 binary is not supported)." \
                "$INSTALL_HINT"
    version=$(numeric "$answer")
    if [ -z "$version" ]; then
        warn "could not read the Docker Compose version from '${answer}'; carrying on"
        return 0
    fi
    at_least "$version" "$COMPOSE_FLOOR" && return 0
    stop "Automatos needs ${FLOOR}: this machine runs Docker Compose ${answer}." "$INSTALL_HINT"
}

check_engine
check_compose
