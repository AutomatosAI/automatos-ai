#!/bin/sh
# =============================================================================
# make env: .env with the three secrets (PRD-262 US-037)
# =============================================================================
# `make up`, `make build` and `make dev` run this first, so `git clone && make up`
# is the whole install, and no install runs on the passwords .env.example ships.
#
#   1. No .env: it is created from .env.example, readable by you alone (mode 600).
#   2. POSTGRES_PASSWORD, REDIS_PASSWORD and API_KEY: each one whose .env value is
#      empty or starts with CHANGE_ME (SETUP.md's `^KEY=(CHANGE_ME.*)?$` rule) gets
#      `openssl rand -hex 24`, but only while the automatos_postgres_data volume
#      does not exist yet, because Postgres keeps the password it was created with.
#      A key the shell or the make command line sets is left alone: compose runs
#      that value. No other line of .env is touched.
#   3. Once the volume exists, nothing is filled. An install that started on a
#      placeholder stops here: orchestrator/docker-secrets-guard.sh, which the
#      backend's entrypoint runs too, prints how to rotate it, and this exits 1
#      with .env unchanged.
#
#   sh scripts/stack_env.sh [directory]   (default: this repository's root)
#
# POSIX sh, openssl and the docker CLI. Exit 0: compose will run three secrets that
# are not placeholders. Exit 1: it would not, and the message says why.
# =============================================================================
set -eu
umask 077

SCRIPT_DIR=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
ROOT=${1:-$(dirname -- "$SCRIPT_DIR")}
ENV_FILE="$ROOT/.env"
EXAMPLE="$ROOT/.env.example"
GUARD="$SCRIPT_DIR/../orchestrator/docker-secrets-guard.sh"
SECRET_KEYS="POSTGRES_PASSWORD REDIS_PASSWORD API_KEY"
DATABASE_VOLUME="automatos_postgres_data"
SECRET_BYTES=24

say() { echo "make env: $*"; }

stop() {
    echo "make env: $*" >&2
    exit 1
}

# The value .env gives KEY, as compose reads it: the last KEY= line, quotes removed.
file_value() {
    sed -n "s/^$1=//p" "$ENV_FILE" | tail -n 1 | sed -e 's/^"\(.*\)"$/\1/' -e "s/^'\(.*\)'\$/\1/"
}

# Whether the shell (or the make command line) sets KEY: compose runs that value then.
in_shell() { printenv "$1" > /dev/null 2>&1; }

# The value compose runs for KEY: the shell's when it sets one, else .env's.
effective() {
    if in_shell "$1"; then printenv "$1"; else file_value "$1"; fi
}

# The secrets make env may fill: not set in the shell, empty or a placeholder in .env.
keys_to_fill() {
    for key in $SECRET_KEYS; do
        if in_shell "$key"; then
            continue
        fi
        case "$(file_value "$key")" in
            "" | CHANGE_ME*) printf '%s ' "$key" ;;
        esac
    done
    return 0
}

# yes, no, or unknown when Docker does not answer: does this machine have the
# stack's database volume, from this checkout or from another one?
database_volume() {
    volumes=$(docker volume ls -q 2> /dev/null) || {
        echo unknown
        return 0
    }
    if printf '%s\n' "$volumes" | grep -qx "$DATABASE_VOLUME"; then echo yes; else echo no; fi
}

new_secret() {
    secret=$(openssl rand -hex "$SECRET_BYTES") || return 1
    printf '%s\n' "$secret" | grep -qxE "[0-9a-f]{$((SECRET_BYTES * 2))}" || return 1
    printf '%s' "$secret"
}

# KEY=VALUE over every KEY= line of .env, or at its end when it has none. The file is
# rewritten in place, so it keeps its mode, and a symlinked .env stays a symlink. The
# working copy is private (umask 077) and named as .gitignore's `.env.*.local`.
set_value() {
    tmp="$ENV_FILE.make-env.$$.local"
    if grep -q "^$1=" "$ENV_FILE"; then
        sed "s|^$1=.*|$1=$2|" "$ENV_FILE" > "$tmp"
    else
        cp "$ENV_FILE" "$tmp"
        [ -z "$(tail -c 1 "$ENV_FILE")" ] || echo >> "$tmp"
        echo "$1=$2" >> "$tmp"
    fi
    cat "$tmp" > "$ENV_FILE"
    rm -f "$tmp"
}

# Every value is generated before .env is written, so a failure leaves it as it was.
fill_secrets() {
    command -v openssl > /dev/null 2>&1 \
        || stop "openssl is not installed, so make env cannot generate $*: install it, or set them in .env by hand (self-hosting.md §1b). .env is unchanged."
    pairs=""
    for key in "$@"; do
        value=$(new_secret) || stop "openssl rand -hex $SECRET_BYTES failed while generating $key. .env is unchanged."
        pairs="$pairs $key=$value"
    done
    for pair in $pairs; do
        set_value "${pair%%=*}" "${pair#*=}"
    done
    say "generated $* in .env (openssl rand -hex $SECRET_BYTES). Keep .env private, and keep it with your data: the database only opens with this POSTGRES_PASSWORD."
}

# The guard, on the values compose would run.
run_guard() {
    env POSTGRES_PASSWORD="$(effective POSTGRES_PASSWORD)" REDIS_PASSWORD="$(effective REDIS_PASSWORD)" \
        API_KEY="$(effective API_KEY)" POSTGRES_USER="$(effective POSTGRES_USER)" \
        POSTGRES_DB="$(effective POSTGRES_DB)" sh "$GUARD"
}

create_env_file() {
    [ -f "$EXAMPLE" ] || stop "there is no .env, and no .env.example to create it from, in $ROOT"
    cp "$EXAMPLE" "$ENV_FILE"
    chmod 600 "$ENV_FILE"
    say "created .env from .env.example (readable by you alone)"
}

main() {
    [ -f "$GUARD" ] || stop "$GUARD is missing: make env runs it to refuse placeholder secrets"
    [ -f "$ENV_FILE" ] || create_env_file
    fill=$(keys_to_fill)
    fill=${fill% }
    started=no
    if [ -n "$fill" ]; then
        case "$(database_volume)" in
            unknown) stop "Docker did not answer \`docker volume ls\`, so make env cannot tell whether this install has started; .env is unchanged. Start Docker, then run make up again." ;;
            yes) started=yes ;;
            # $fill is a list of key names, split on purpose.
            # shellcheck disable=SC2086
            no) fill_secrets $fill ;;
        esac
    fi
    if ! run_guard; then
        [ "$started" = no ] || stop "the $DATABASE_VOLUME volume exists, so this install has started (from this checkout or another one on this machine), and Postgres keeps the password it was created with. .env is unchanged: rotate as above, or copy in the .env of the checkout that created it."
        stop "the shell or the make command line sets a CHANGE_ME value (above): unset it, or set a real one."
    fi
    [ "$started" = no ] || stop "empty in .env: $fill. The $DATABASE_VOLUME volume exists, so this install has started, and make env generates secrets only before the first start: set the value this install was created with (from the .env that started it). .env is unchanged."
}

main
