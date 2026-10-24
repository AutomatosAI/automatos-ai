#!/bin/bash
set -e

# Railway mounts persistent volumes as root. Fix ownership at runtime
# so the non-root worker process can write to the workspace directory.
if [ -d "$WORKSPACE_VOLUME_PATH" ] && [ "$(stat -c '%u' "$WORKSPACE_VOLUME_PATH" 2>/dev/null)" != "1000" ]; then
    echo "[entrypoint] Fixing ownership of $WORKSPACE_VOLUME_PATH (skipping mounted projects folders) ..."
    # PRD-239: <workspace>/projects is the OWNER'S folder bind-mounted in (LOCAL_PROJECTS_DIR)
    # — never re-own it: on a Linux host its files would change owner, and the crawl over a
    # 40 GB development folder alone kept the worker unhealthy for minutes.
    # PRD-262 US-045: chown -h re-owns a symlink itself, never the file it points to (an
    # agent or a cloned repo can link to /etc/passwd); ! -user worker skips what the worker
    # already owns. A chown that fails is logged, and the worker still starts.
    find "$WORKSPACE_VOLUME_PATH" -mindepth 1 -path '*/projects' -prune -o ! -user worker -exec chown -h worker:worker {} + || echo "[entrypoint] some files could not be re-owned" >&2
    chown worker:worker "$WORKSPACE_VOLUME_PATH" 2>/dev/null || true
fi

# Drop to the worker user and exec the main process. setpriv, from util-linux in the
# base image, replaced gosu (PRD-262, row 47 of the size breakdown: 2.9 MB). It sets
# all three uids and gids and the worker's groups, but unlike gosu it leaves HOME as
# root's, so HOME is set here: the canvas's Claude CLI and ~/.cache read it.
# --no-new-privs (F452, #1309): the worker runs agent code, and the image carries
# setuid-root binaries (su, mount, passwd, newgrp…). With no_new_privs set, exec never
# grants a privilege again, so none of them can take the worker back to root. It holds
# on Railway and in compose, where the chart's securityContext doesn't apply.
exec setpriv --reuid=worker --regid=worker --init-groups --no-new-privs env HOME=/home/worker python -m main "$@"
