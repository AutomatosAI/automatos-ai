#!/bin/bash
set -e

# The local edition mounts your deliverables folder (AUTOMATOS_WORKSPACE_DIR) inside the
# volume as the local workspace's root, LOCAL_WORKSPACE_ROOT (docker-compose.yml). The folder
# is yours, so nothing here re-owns it (PRD-262 US-040, US-045). On a Linux host whose user
# is not uid 1000 the worker could not write to it, and its first file listing failed with
# Permission denied. When uid 1000 cannot write the folder, the worker takes the folder
# owner's uid and gid instead, and writes your files as you, as Docker Desktop's file
# sharing already does on macOS and Windows. Never root's: when root owns the folder the
# worker stays uid 1000, and the log says what to do.
if [ -n "${LOCAL_WORKSPACE_ROOT:-}" ] && [ -d "$LOCAL_WORKSPACE_ROOT" ] \
    && ! setpriv --reuid=worker --regid=worker --init-groups test -w "$LOCAL_WORKSPACE_ROOT"; then
    folder_uid="$(stat -c '%u' "$LOCAL_WORKSPACE_ROOT")"
    folder_gid="$(stat -c '%g' "$LOCAL_WORKSPACE_ROOT")"
    if [ "$folder_uid" = "0" ]; then
        echo "[entrypoint] root owns your deliverables folder (AUTOMATOS_WORKSPACE_DIR), so the worker cannot write to it: give the folder to a user other than root (sudo chown -R <user>: <folder>) and restart the worker" >&2
    elif [ "$folder_uid" != "$(id -u worker)" ]; then
        echo "[entrypoint] uid $(id -u worker) cannot write $LOCAL_WORKSPACE_ROOT: the worker runs as its owner, $folder_uid:$folder_gid"
        groupmod -o -g "$folder_gid" worker && usermod -o -u "$folder_uid" worker \
            || echo "[entrypoint] could not run the worker as $folder_uid:$folder_gid" >&2
    fi
fi

# Railway mounts persistent volumes as root. Fix ownership at runtime
# so the non-root worker process can write to the workspace directory.
if [ -d "$WORKSPACE_VOLUME_PATH" ] && [ "$(stat -c '%u' "$WORKSPACE_VOLUME_PATH" 2>/dev/null)" != "$(id -u worker)" ]; then
    echo "[entrypoint] Fixing ownership of $WORKSPACE_VOLUME_PATH (skipping mounted projects folders) ..."
    # PRD-239: <workspace>/projects is the OWNER'S folder bind-mounted in (LOCAL_PROJECTS_DIR)
    # — never re-own it: on a Linux host its files would change owner, and the crawl over a
    # 40 GB development folder alone kept the worker unhealthy for minutes.
    # PRD-262 US-045: chown -h re-owns a symlink itself, never the file it points to (an
    # agent or a cloned repo can link to /etc/passwd); ! -user worker skips what the worker
    # already owns. A chown that fails is logged, and the worker still starts.
    # PRD-262 US-040: your deliverables folder, LOCAL_WORKSPACE_ROOT, is skipped the same way
    # (unset, as on Railway, the empty pattern matches nothing).
    find "$WORKSPACE_VOLUME_PATH" -mindepth 1 \( -path '*/projects' -o -path "${LOCAL_WORKSPACE_ROOT:-}" \) -prune -o ! -user worker -exec chown -h worker:worker {} + || echo "[entrypoint] some files could not be re-owned" >&2
    chown worker:worker "$WORKSPACE_VOLUME_PATH" 2>/dev/null || true
fi

# Drop to worker user and exec the main process
exec gosu worker python -m main "$@"
