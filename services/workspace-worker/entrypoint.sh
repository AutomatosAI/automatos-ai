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
#
# Nor any other id the image already gives an account (PRD-262 US-040's security review):
# a folder owned by www-data (33) or nobody (65534) would make the worker that account,
# with what it owns, so the worker stays uid 1000 and the log says why. A gid that is
# root's group (0) or an account's (shadow, 42; users, 100) is not taken either: the
# worker takes the owner's uid alone, which is what writing the folder needs. A free id
# needs no `-o` (a duplicate id), so usermod and groupmod run without it, and refuse a
# taken id themselves.

# Status 0, and why on stdout, when the worker cannot take this uid (passwd) or gid
# (group): the image already gives it to an account other than the worker. getent exits
# 2 for an id no account has (status 1); any other failure counts as taken, so no id is
# taken unchecked.
id_taken_by() {  # passwd|group ID
    local entry status=0
    entry="$(getent "$1" "$2")" || status=$?
    case "$status" in
        0) [ "${entry%%:*}" != worker ] && echo "${entry%%:*}'s in the worker's image" ;;
        2) return 1 ;;
        *) echo "getent $1 exited $status" ;;
    esac
}

# The worker takes the uid of the folder's owner, and its gid when that is free too.
run_as_folder_owner() {  # UID GID
    local uid="$1" gid="$2" taken
    if taken="$(id_taken_by passwd "$uid")"; then
        echo "[entrypoint] uid $uid owns your deliverables folder (AUTOMATOS_WORKSPACE_DIR), but the worker cannot take that uid ($taken), so it stays uid $(id -u worker) and cannot write to the folder: give the folder to your own user (sudo chown -R <user>: <folder>) and restart the worker" >&2
        return 0
    fi
    taken="$(id_taken_by group "$gid")" || taken=""
    [ "$gid" != "0" ] || taken="root's group"
    if [ -n "$taken" ]; then
        echo "[entrypoint] uid $(id -u worker) cannot write $LOCAL_WORKSPACE_ROOT: the worker runs as its owner, uid $uid, and keeps its own group, not gid $gid ($taken)"
        usermod -u "$uid" worker || echo "[entrypoint] could not run the worker as uid $uid" >&2
    else
        echo "[entrypoint] uid $(id -u worker) cannot write $LOCAL_WORKSPACE_ROOT: the worker runs as its owner, $uid:$gid"
        groupmod -g "$gid" worker && usermod -u "$uid" worker \
            || echo "[entrypoint] could not run the worker as $uid:$gid" >&2
    fi
}

if [ -n "${LOCAL_WORKSPACE_ROOT:-}" ] && [ -d "$LOCAL_WORKSPACE_ROOT" ] \
    && ! setpriv --reuid=worker --regid=worker --init-groups test -w "$LOCAL_WORKSPACE_ROOT"; then
    # Owner and group from one stat, so both are the same moment's.
    folder_owner="$(stat -c '%u %g' "$LOCAL_WORKSPACE_ROOT")"
    read -r folder_uid folder_gid <<< "$folder_owner"
    if [ "$folder_uid" = "0" ]; then
        echo "[entrypoint] root owns your deliverables folder (AUTOMATOS_WORKSPACE_DIR), so the worker cannot write to it: give the folder to a user other than root (sudo chown -R <user>: <folder>) and restart the worker" >&2
    elif [ "$folder_uid" != "$(id -u worker)" ]; then
        run_as_folder_owner "$folder_uid" "$folder_gid"
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

# Drop to the worker user and exec the main process. setpriv, from util-linux in the
# base image, replaced gosu (PRD-262, row 47 of the size breakdown: 2.9 MB). It sets
# all three uids and gids and the worker's groups, but unlike gosu it leaves HOME as
# root's, so HOME is set here: the canvas's Claude CLI and ~/.cache read it.
# --no-new-privs (F452, #1309): the worker runs agent code, and the image carries
# setuid-root binaries (su, mount, passwd, newgrp…). With no_new_privs set, exec never
# grants a privilege again, so none of them can take the worker back to root. It holds
# on Railway and in compose, where the chart's securityContext doesn't apply. The image
# also carries no setuid or setgid file any more (the Dockerfile's last install step
# strips them), and nothing after the drop needs one: no sudo or su, and git over ssh
# authenticates with the workspace's key, which needs no ssh-keysign.
exec setpriv --reuid=worker --regid=worker --init-groups --no-new-privs env HOME=/home/worker python -m main "$@"
