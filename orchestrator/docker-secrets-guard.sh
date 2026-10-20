#!/bin/sh
# =============================================================================
# Automatos AI - the placeholder-secret guard (PRD-262 US-037)
# =============================================================================
# .env.example ships POSTGRES_PASSWORD, REDIS_PASSWORD and API_KEY as CHANGE_ME_…
# placeholders, and anyone can read them. This exits 1 when any of the three starts
# with CHANGE_ME, printing which ones and how to rotate them, and exits 0 otherwise.
# An unset or empty value passes: Helm and Railway set DATABASE_URL and REDIS_URL
# instead of the two passwords, and a one-off `docker run --rm <image> python -c …`
# runs with no env at all. Compose itself stops on an empty one (`${VAR:?…}`).
#
# Two callers, one message:
#   - the backend's entrypoint (docker-entrypoint.sh, next to this script in the
#     image), before it runs any command but `migrate`, the migration Job's;
#   - `make env` (scripts/stack_env.sh) on the host, with the values compose would
#     use, once the install has started and its .env still holds a placeholder.
# POSIX sh, so it runs in the image and on macOS, Linux and WSL2 alike.
# =============================================================================
set -eu

SECRET_KEYS="POSTGRES_PASSWORD REDIS_PASSWORD API_KEY"
PLACEHOLDER_PREFIX="CHANGE_ME"

placeholders=""
for key in $SECRET_KEYS; do
    value=$(printenv "$key" || true)
    case "$value" in
        "$PLACEHOLDER_PREFIX"*) placeholders="${placeholders:+$placeholders }$key" ;;
    esac
done

if [ -z "$placeholders" ]; then
    exit 0
fi

db_user="${POSTGRES_USER:-postgres}"
db_name="${POSTGRES_DB:-orchestrator_db}"
{
    echo "❌ Placeholder secrets: $placeholders. Each still starts with CHANGE_ME, the"
    echo "   value .env.example ships with, which anyone can read. Automatos does not run on it."
    echo ""
    echo "   A new install with no data to keep: \`docker compose down -v\` deletes its"
    echo "   volumes, then \`make up\` writes three random secrets to .env (make env) and"
    echo "   starts again."
    echo ""
    echo "   An install whose data you keep: rotate each one, then \`make up\`"
    echo "   (or \`docker compose up -d\`), which restarts what changed."
    echo "   - POSTGRES_PASSWORD: Postgres keeps the password it was created with, so"
    echo "     change it there first, then put the same value in .env:"
    echo "       new=\$(openssl rand -hex 24)"
    echo "       docker compose exec postgres psql -U $db_user -d $db_name -c \"ALTER USER $db_user PASSWORD '\$new'\""
    echo "       sed -i.bak \"s|^POSTGRES_PASSWORD=.*|POSTGRES_PASSWORD=\$new|\" .env && rm -f .env.bak"
    echo "   - REDIS_PASSWORD and API_KEY: a new value in .env alone (openssl rand -hex 24);"
    echo "     both are read at every start."
} >&2
exit 1
