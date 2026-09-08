#!/bin/bash
set -e

# Railway mounts persistent volumes as root. Fix ownership at runtime
# so the non-root worker process can write to the workspace directory.
if [ -d "$WORKSPACE_VOLUME_PATH" ] && [ "$(stat -c '%u' "$WORKSPACE_VOLUME_PATH" 2>/dev/null)" != "1000" ]; then
    echo "[entrypoint] Fixing ownership of $WORKSPACE_VOLUME_PATH (skipping mounted projects folders) ..."
    # PRD-239: <workspace>/projects is the OWNER'S folder bind-mounted in (LOCAL_PROJECTS_DIR)
    # — never re-own it: on a Linux host its files would change owner, and the crawl over a
    # 40 GB development folder alone kept the worker unhealthy for minutes.
    find "$WORKSPACE_VOLUME_PATH" -mindepth 1 -path '*/projects' -prune -o -exec chown worker:worker {} + 2>/dev/null || true
    chown worker:worker "$WORKSPACE_VOLUME_PATH" 2>/dev/null || true
fi

# Drop to worker user and exec the main process
exec gosu worker python -m main "$@"
