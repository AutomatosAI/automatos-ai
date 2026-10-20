"""PRD-262 US-045: the worker boots safely and reports healthy quickly.

- entrypoint.sh re-owns the workspace volume with ``chown -h`` (a symlink itself,
  never the file it points to: an agent or a cloned repo can link to /etc/passwd) and
  only what the worker does not own yet. A chown that fails stays non-fatal under
  ``set -e``, as ``2>/dev/null || true`` made it, but is now logged;
- the Dockerfile copies the entrypoint with ``--chmod=755`` (no chmod RUN and its
  empty layer), probes every 2 s while the worker starts (``--start-interval``), and
  carries ``# syntax=docker/dockerfile:1`` on line 1, without which a build on Engine
  24's built-in frontend rejects ``--start-interval``;
- test.yml's prod-image-parity runs the real block as root in throwaway containers:
  a symlink to a root-owned file, a read-only volume, and a deliverables folder uid 1000
  cannot write.

setpriv replaced gosu (PRD-262, row 47 of the 19 Oct size breakdown): it is in the base
image's util-linux, and the exec sets HOME to the worker's, which setpriv, unlike gosu,
leaves as root's.

US-040: compose names your deliverables folder, LOCAL_WORKSPACE_ROOT. The walk never
enters it, and when uid 1000 cannot write it (a Linux host whose user is uid 1001, say)
the worker takes the folder owner's uid and gid instead of re-owning anything; a folder
root owns keeps uid 1000 and the log says why. The security review's three fixes: a gid
that is root's group (0) is never taken, nor any uid or gid the image already gives an
account (www-data's 33, nobody's 65534, the shadow group's 42), and the folder's owner
and group come from one ``stat -c '%u %g'``.

The entrypoint itself runs here under bash with stand-in stat, find, chown, setpriv, id,
getent, groupmod and usermod, so its control flow is proved in the required suite. The smoke's
20 s bar and /metrics 200 are in test_prd262_smoke_worker_health.py.
"""
from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKER = ROOT / "services" / "workspace-worker"
ENTRYPOINT = WORKER / "entrypoint.sh"
DOCKERFILE = (WORKER / "Dockerfile").read_text()
SYNTAX_LINE = "# syntax=docker/dockerfile:1"
LOG_LINE = "[entrypoint] some files could not be re-owned"
DROP = "setpriv --reuid=worker --regid=worker --init-groups --no-new-privs env HOME=/home/worker python -m main"
OWNERSHIP_FIND = (
    "find \"$WORKSPACE_VOLUME_PATH\" -mindepth 1 \\( -path '*/projects' -o -path \"${LOCAL_WORKSPACE_ROOT:-}\" \\) "
    "-prune -o ! -user worker -exec chown -h worker:worker {} + || echo \"" + LOG_LINE + "\" >&2"
)
# find's arguments as the stand-in logs them; the pattern is empty when compose names no folder.
WALK = ("find {volume} -mindepth 1 ( -path */projects -o -path {local} ) -prune -o ! -user worker "
        "-exec chown -h worker:worker {{}} +")
# The local workspace, whose root compose mounts from your deliverables folder.
DEFAULT_ID = "00000000-0000-0000-0000-0000000000c1"
# Whether uid 1000 can write your deliverables folder, asked as the worker.
PROBE = "setpriv --reuid=worker --regid=worker --init-groups test -w {local}"
START = "gosu worker python -m main --flag"
ROOT_OWNS = "[entrypoint] root owns your deliverables folder (AUTOMATOS_WORKSPACE_DIR)"
# What getent finds in the worker's image (python:3.12-slim's base-passwd, and the worker):
# "database:id:name" each. Any other id is free.
IMAGE_ACCOUNTS = (
    "passwd:0:root group:0:root passwd:33:www-data group:33:www-data group:42:shadow "
    "group:100:users passwd:65534:nobody group:65534:nogroup passwd:1000:worker group:1000:worker"
)
TAKEN_UID = ("[entrypoint] uid {uid} owns your deliverables folder (AUTOMATOS_WORKSPACE_DIR), "
             "but the worker cannot take that uid ({why})")
OWNER_ONLY = ("[entrypoint] uid 1000 cannot write {local}: the worker runs as its owner, uid {uid}, "
              "and keeps its own group, not gid {gid} ({why})")
IN_THE_IMAGE = "{}'s in the worker's image"


def _instructions(text: str) -> list[str]:
    """Instructions with continuation lines joined; comment lines dropped."""
    code = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    return [line.strip() for line in re.sub(r"\\[ \t]*\n", " ", code).splitlines() if line.strip()]


INSTRUCTIONS = _instructions(DOCKERFILE)


def _only(predicate) -> str:
    found = [line for line in INSTRUCTIONS if predicate(line)]
    assert len(found) == 1, f"expected one matching instruction, found {found}"
    return found[0]


# --------------------------------------------------------------------------- entrypoint.sh


def test_the_ownership_walk_skips_owned_files_never_follows_a_link_and_logs_a_failure():
    lines = [line.strip() for line in ENTRYPOINT.read_text().splitlines()]
    assert OWNERSHIP_FIND in lines
    walks = [line for line in lines if line.startswith("find ")]
    assert walks == [OWNERSHIP_FIND], "one walk, and it no longer hides chown's errors"
    assert lines[1] == "set -e"
    assert lines[-1] == f'exec {DROP} "$@"'
    assert not [line for line in lines if "gosu" in line and not line.startswith("#")]


_LOGGED = '#!/bin/sh\necho "$(basename "$0") $*" >> "$CALL_LOG"\n'
_STANDINS = {
    # stat -c FORMAT PATH: the volume's owner, or the deliverables folder's owner and group,
    # which it answers only to the one stat that reads both (any other format fails the boot).
    "stat": '#!/bin/sh\nif [ "$3" != "$LOCAL_WORKSPACE_ROOT" ]; then echo "$FAKE_UID"\n'
            'elif [ "$2" = "%u %g" ]; then echo "$FOLDER_UID $FOLDER_GID"\n'
            'else echo "stat -c $2 of the folder: read its owner and group in one stat" >&2; exit 1; fi\n',
    "find": '#!/bin/sh\necho "find $*" >> "$CALL_LOG"\nexit "$FIND_EXIT"\n',
    "chown": _LOGGED,
    # setpriv is the drop for both: the folder-writable probe (``test -w``, logged, exits
    # on FOLDER_UNWRITABLE) and the final exec (echoed, as it was before the probe existed).
    "setpriv": '#!/bin/sh\ncase "$*" in *" test -w "*) echo "setpriv $*" >> "$CALL_LOG"\n'
               '  exit "${FOLDER_UNWRITABLE:-0}";; esac\necho "setpriv $*"\n',
    "groupmod": _LOGGED,
    # usermod -u UID worker: id -u worker answers UID from then on, 1000 before.
    "usermod": _LOGGED + 'echo "$2" > "$WORKER_UID_FILE"\n',
    "id": '#!/bin/sh\ncat "$WORKER_UID_FILE" 2>/dev/null || echo 1000\n',
    # getent passwd|group ID: "name:x:ID:" for an account in IMAGE_ACCOUNTS, else exit 2, as
    # getent does for an id no account has; GETENT_EXIT, when set, is a getent that fails.
    "getent": '#!/bin/sh\n[ -z "$GETENT_EXIT" ] || exit "$GETENT_EXIT"\nfor account in $IMAGE_ACCOUNTS; do\n'
              '  case "$account" in "$1:$2:"*) echo "${account#"$1:$2:"}:x:$2:"; exit 0;; esac\ndone\nexit 2\n',
}


def _boot(tmp_path: Path, uid: str, find_exit: int, folder: tuple[str, str, bool] | None = None,
          getent_exit: str = "") -> tuple[subprocess.CompletedProcess, list[str], Path]:
    """Run entrypoint.sh with the stand-ins first on PATH; (result, stand-in calls, volume).

    ``folder`` is the deliverables folder compose names, as (uid, gid, writable by uid 1000);
    without it LOCAL_WORKSPACE_ROOT is unset, as on Railway. ``getent_exit`` makes every
    getent fail with that status.
    """
    bin_dir, volume, log = tmp_path / "bin", tmp_path / "volume", tmp_path / "calls.log"
    bin_dir.mkdir()
    volume.mkdir()
    log.touch()
    for name, body in _STANDINS.items():
        (bin_dir / name).write_text(body, encoding="utf-8")
        (bin_dir / name).chmod((bin_dir / name).stat().st_mode | stat.S_IXUSR)
    env = {"PATH": f"{bin_dir}{os.pathsep}/usr/bin{os.pathsep}/bin", "CALL_LOG": str(log),
           "FAKE_UID": uid, "FIND_EXIT": str(find_exit), "WORKSPACE_VOLUME_PATH": str(volume),
           "WORKER_UID_FILE": str(tmp_path / "worker-uid"), "IMAGE_ACCOUNTS": IMAGE_ACCOUNTS,
           "GETENT_EXIT": getent_exit}
    if folder:
        local = volume / DEFAULT_ID
        local.mkdir()
        env.update(LOCAL_WORKSPACE_ROOT=str(local), FOLDER_UID=folder[0], FOLDER_GID=folder[1],
                   FOLDER_UNWRITABLE="0" if folder[2] else "1")
    proc = subprocess.run(["bash", str(ENTRYPOINT), "--flag"], env=env, stdin=subprocess.DEVNULL,
                          capture_output=True, text=True, timeout=30)
    return proc, log.read_text(encoding="utf-8").splitlines(), volume


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_a_failed_chown_is_logged_and_the_worker_still_starts(tmp_path):
    proc, calls, volume = _boot(tmp_path, uid="0", find_exit=1)
    assert proc.returncode == 0, proc.stderr
    assert LOG_LINE in proc.stderr.splitlines()
    assert proc.stdout.splitlines()[-1] == f"{DROP} --flag"
    assert calls[0] == WALK.format(volume=volume, local="")


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_a_clean_walk_logs_nothing(tmp_path):
    proc, calls, volume = _boot(tmp_path, uid="0", find_exit=0)
    assert proc.returncode == 0, proc.stderr
    assert LOG_LINE not in proc.stderr
    assert calls[-1] == f"chown worker:worker {volume}"
    assert proc.stdout.splitlines()[-1] == f"{DROP} --flag"


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_a_volume_the_worker_already_owns_is_not_walked(tmp_path):
    proc, calls, _ = _boot(tmp_path, uid="1000", find_exit=1)
    assert proc.returncode == 0, proc.stderr
    assert calls == []
    assert proc.stdout.splitlines() == [f"{DROP} --flag"]


# US-040: your deliverables folder.


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_a_deliverables_folder_uid_1000_can_write_changes_nothing(tmp_path):
    """A Linux user who is uid 1000, or Docker Desktop's file sharing (it shows root, lets anyone write)."""
    proc, calls, volume = _boot(tmp_path, uid="1000", find_exit=0, folder=("0", "0", True))
    assert proc.returncode == 0, proc.stderr
    assert calls == [PROBE.format(local=volume / DEFAULT_ID)]
    assert proc.stdout.splitlines() == [f"{DROP} --flag"]
    assert ROOT_OWNS not in proc.stderr


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_a_folder_another_user_owns_makes_the_worker_that_user_and_is_never_walked(tmp_path):
    proc, calls, volume = _boot(tmp_path, uid="1000", find_exit=0, folder=("1001", "1002", False))
    local = volume / DEFAULT_ID
    assert proc.returncode == 0, proc.stderr
    assert calls == [
        PROBE.format(local=local),
        "groupmod -g 1002 worker",
        "usermod -u 1001 worker",
        # The volume, still uid 1000's, is walked for the worker's new uid, without your folder.
        WALK.format(volume=volume, local=local),
        f"chown worker:worker {volume}",
    ]
    out = proc.stdout.splitlines()
    assert out[0] == f"[entrypoint] uid 1000 cannot write {local}: the worker runs as its owner, 1001:1002"
    assert out[-1] == f"{DROP} --flag"


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_a_folder_root_owns_keeps_the_worker_at_1000_and_says_why(tmp_path):
    proc, calls, volume = _boot(tmp_path, uid="1000", find_exit=0, folder=("0", "0", False))
    assert proc.returncode == 0, proc.stderr
    assert calls == [PROBE.format(local=volume / DEFAULT_ID)]
    assert [line for line in proc.stderr.splitlines() if line.startswith(ROOT_OWNS)]
    assert proc.stdout.splitlines() == [f"{DROP} --flag"]


# US-040's security review: no gid 0, no id an image account has, one stat.


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_a_folder_in_root_s_group_gives_the_worker_the_owner_s_uid_and_keeps_its_gid(tmp_path):
    """1001:0, as a folder made under a root-group umask is: the worker's gid stays 1000."""
    proc, calls, volume = _boot(tmp_path, uid="1000", find_exit=0, folder=("1001", "0", False))
    local = volume / DEFAULT_ID
    assert proc.returncode == 0, proc.stderr
    assert calls == [
        PROBE.format(local=local),
        "usermod -u 1001 worker",
        WALK.format(volume=volume, local=local),
        f"chown worker:worker {volume}",
    ]
    assert not [call for call in calls if call.startswith("groupmod")], "the worker's gid is unchanged"
    out = proc.stdout.splitlines()
    assert out[0] == OWNER_ONLY.format(local=local, uid="1001", gid="0", why="root's group")
    assert out[-1] == START


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
@pytest.mark.parametrize(("owner", "name"), [("33", "www-data"), ("65534", "nobody")])
def test_a_folder_an_image_account_owns_keeps_the_worker_at_1000_and_says_why(tmp_path, owner, name):
    proc, calls, volume = _boot(tmp_path, uid="1000", find_exit=0, folder=(owner, owner, False))
    assert proc.returncode == 0, proc.stderr
    assert calls == [PROBE.format(local=volume / DEFAULT_ID)], "no usermod, no groupmod"
    taken = TAKEN_UID.format(uid=owner, why=IN_THE_IMAGE.format(name))
    assert [line for line in proc.stderr.splitlines() if line.startswith(taken)]
    assert proc.stdout.splitlines() == [START]


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
@pytest.mark.parametrize(("gid", "group"), [("42", "shadow"), ("100", "users"), ("65534", "nogroup")])
def test_an_image_group_s_gid_is_never_taken(tmp_path, gid, group):
    proc, calls, volume = _boot(tmp_path, uid="1000", find_exit=0, folder=("1001", gid, False))
    local = volume / DEFAULT_ID
    assert proc.returncode == 0, proc.stderr
    assert calls[:2] == [PROBE.format(local=local), "usermod -u 1001 worker"]
    assert not [call for call in calls if call.startswith("groupmod")]
    assert proc.stdout.splitlines()[0] == OWNER_ONLY.format(local=local, uid="1001", gid=gid,
                                                            why=IN_THE_IMAGE.format(group))


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_an_id_getent_cannot_check_is_never_taken(tmp_path):
    proc, calls, volume = _boot(tmp_path, uid="1000", find_exit=0, folder=("1001", "1002", False), getent_exit="1")
    assert proc.returncode == 0, proc.stderr
    assert calls == [PROBE.format(local=volume / DEFAULT_ID)]
    assert TAKEN_UID.format(uid="1001", why="getent passwd exited 1") in proc.stderr


def test_the_folder_s_owner_and_group_come_from_one_stat_and_no_id_is_duplicated():
    """One ``stat -c '%u %g'`` (the stand-in answers no other format for the folder), and no ``-o``."""
    text = ENTRYPOINT.read_text()
    reads = [line.strip() for line in text.splitlines() if "stat -c" in line and "LOCAL_WORKSPACE_ROOT" in line]
    assert reads == ['folder_owner="$(stat -c \'%u %g\' "$LOCAL_WORKSPACE_ROOT")"']
    commands = [line for line in text.splitlines() if re.search(r"\b(usermod|groupmod)\b", line)
                and not line.lstrip().startswith("#")]
    assert commands and not [line for line in commands if re.search(r"\s-o\b", line)]


def test_compose_names_the_folder_it_mounts_and_the_hosted_edition_names_none():
    """The worker's LOCAL_WORKSPACE_ROOT is the target of its deliverables bind mount."""
    worker = yaml.safe_load((ROOT / "docker-compose.yml").read_text())["services"]["workspace-worker"]
    local_root = f"/workspaces/${{DEFAULT_WORKSPACE_ID:-{DEFAULT_ID}}}"
    assert worker["environment"]["LOCAL_WORKSPACE_ROOT"] == local_root
    assert f"${{AUTOMATOS_WORKSPACE_DIR:-./workspaces}}:{local_root}" in worker["volumes"]
    chart = "\n".join(path.read_text() for path in (ROOT / "charts").rglob("*.yaml"))
    assert "LOCAL_WORKSPACE_ROOT" not in chart, "no bind mount in the hosted edition: the volume walk as before"


# --------------------------------------------------------------------------- the Dockerfile


def test_the_entrypoint_is_copied_executable_with_no_chmod_layer():
    copy = _only(lambda line: line.startswith("COPY") and "entrypoint.sh" in line)
    assert copy == "COPY --chmod=755 entrypoint.sh /usr/local/bin/entrypoint.sh"
    assert not [line for line in INSTRUCTIONS if line.startswith("RUN") and "chmod +x" in line]
    assert INSTRUCTIONS.index(copy) < INSTRUCTIONS.index("COPY --chown=1000:1000 . .")
    assert INSTRUCTIONS[-2:] == ['ENTRYPOINT ["entrypoint.sh"]', "CMD []"]


def test_the_healthcheck_probes_every_2_s_while_the_worker_starts():
    healthcheck = _only(lambda line: line.startswith("HEALTHCHECK"))
    flags = dict(re.findall(r"--([a-z-]+)=(\S+)", healthcheck.split(" CMD ", 1)[0]))
    assert flags["start-interval"] == "2s"
    # Docker probes at the start interval only inside the start period.
    assert re.fullmatch(r"[1-9]\d*s", flags["start-period"]), flags
    assert "http://localhost:8081/health" in healthcheck.split(" CMD ", 1)[1]


def test_the_syntax_line_comes_first():
    """Engine 24's built-in frontend (BuildKit 0.11) rejects --start-interval; 0.12 parses it."""
    assert DOCKERFILE.splitlines()[0] == SYNTAX_LINE


# --------------------------------------------------------------------------- the parity job


def test_the_parity_job_runs_the_ownership_block_in_throwaway_containers():
    steps = yaml.safe_load((ROOT / ".github" / "workflows" / "test.yml").read_text())["jobs"]["prod-image-parity"]["steps"]
    step = next(step for step in steps if "entrypoint re-owns the volume" in step.get("name", ""))
    script = step["run"]
    for needle in (
        "docker volume create parity-wk-readonly",
        "-v parity-wk-readonly:/srv/readonly:ro --entrypoint bash parity-wk -s",
        'ln -s /tmp/outside "$vol/ws/link"',
        'check "the root-owned file the symlink points to" "$(owner /tmp/outside)" root',
        'check "the symlink" "$(owner "$vol/ws/link")" worker',
        'check "projects/" "$(owner "$vol/ws/projects")" root',
        'check "the ctime of a file the worker already owned"',
        "out=$(boot /srv/readonly); code=$?",
        "some files could not be re-owned$",
        '"started as uid 1000"',
        'check "the worker\'s HOME" "$(grep -o \'HOME=[^ ]*\' <<< "$out")" "HOME=/home/worker"',
        'out=$(LOCAL_WORKSPACE_ROOT="$mine" boot /tmp/volume2); code=$?',
        'check "the worker with that folder" "$(grep -o \'started as uid [0-9]*\' <<< "$out")" "started as uid 1001"',
        'check "a uid 1000 file in that folder" "$(stat -c %u "$mine/older")" 1000',
        # The security review's fixes, in the real image: no image account's uid, no root group.
        'chown 33:33 "$theirs"',
        'check "the worker with a folder www-data owns" "$(grep -o \'started as uid [0-9]*\' <<< "$out")" "started as uid 1000"',
        '"$(grep -c "cannot take that uid (www-data\'s in the worker\'s image)" <<< "$out")" 1',
        'chown 1002:0 "$rootgroup"',
        '"$(grep -o \'started as uid [0-9]* gid [0-9]*\' <<< "$out")" "started as uid 1002 gid 1001"',
        # F452, in the Railway build: no setuid or setgid file, and the drop sets no_new_privs.
        'setid=$(find / -xdev -perm /6000 -type f)',
        'check "files with a setuid or setgid bit" "${setid:-none}" none',
        'no_new_privs=$(sed -n "s/^NoNewPrivs:[[:space:]]*//p" /proc/$$/status)',
        'check "the worker\'s no_new_privs" "$(grep -o \'no_new_privs=[0-9]*\' <<< "$out")" "no_new_privs=1"',
    ):
        assert needle in script, needle
    assert script.rstrip().endswith('exit "$bad"\nSH')
    # It runs after the step that builds the image it needs.
    builds = "docker build -t parity-wk services/workspace-worker/"
    built_at = next(i for i, other in enumerate(steps) if builds in other.get("run", ""))
    assert built_at < steps.index(step)
