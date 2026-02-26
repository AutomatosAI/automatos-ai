#!/bin/bash
set -e

# Railway mounts persistent volumes as root. Fix ownership at runtime
# so the non-root worker process can write to the workspace directory.
if [ -d "$WORKSPACE_VOLUME_PATH" ] && [ "$(stat -c '%u' "$WORKSPACE_VOLUME_PATH" 2>/dev/null)" != "1000" ]; then
    echo "[entrypoint] Fixing ownership of $WORKSPACE_VOLUME_PATH ..."
    chown -R worker:worker "$WORKSPACE_VOLUME_PATH" 2>/dev/null || true
fi

# Drop to worker user and exec the main process
exec gosu worker python -m main "$@"
