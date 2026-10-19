"""PRD-262 US-037: `make up` writes the three secrets and refuses placeholders.

``make env`` (``scripts/stack_env.sh``), which ``make up``, ``make build`` and ``make dev``
run as their first recipe line:

* creates ``.env`` from ``.env.example`` when it is missing, readable by its owner alone;
* fills POSTGRES_PASSWORD, REDIS_PASSWORD and API_KEY with ``openssl rand -hex 24`` when
  a value is empty or starts with CHANGE_ME, only while the ``automatos_postgres_data``
  volume does not exist (Postgres keeps its first password), and never touches any other
  value, nor a key the shell sets (compose runs the shell's value);
* on an install that has started with a placeholder, changes nothing and stops with the
  backend entrypoint's own message (orchestrator/docker-secrets-guard.sh): how to rotate.

The script runs for real under sh against a copy of ``.env.example`` in a temporary
directory, with ``docker`` and ``openssl`` replaced by stubs that record their calls:
``FAKE_VOLUMES`` is what ``docker volume ls -q`` lists, ``FAKE_DOCKER_RC`` makes Docker
not answer, and ``FAKE_OPENSSL_RC`` makes openssl fail. The CI side (make up with no
secrets exported finds three 48-hex values; make env on a started install leaves .env as
it was) is in test_prd262_smoke_images.py.
"""
from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO / "scripts" / "stack_env.sh"
_GUARD = _REPO / "orchestrator" / "docker-secrets-guard.sh"
_EXAMPLE = _REPO / ".env.example"
_MAKEFILE = _REPO / "Makefile"
_KEYS = ("POSTGRES_PASSWORD", "REDIS_PASSWORD", "API_KEY")
_VOLUME = "automatos_postgres_data"
_SECRET = re.compile(r"[0-9a-f]{48}")
_REAL = "ab" * 24
_FIRST_LINE = "@$(MAKE) --no-print-directory env"

_STUBS = {
    "docker": (
        'echo "docker $*" >> "$CALL_LOG"\n'
        '[ "${FAKE_DOCKER_RC:-0}" = 0 ] || exit "$FAKE_DOCKER_RC"\n'
        'if [ "$*" = "volume ls -q" ]; then for volume in ${FAKE_VOLUMES:-}; do echo "$volume"; done; fi'
    ),
    # A different 48-hex value per call, as openssl rand -hex 24 gives.
    "openssl": (
        'calls=$(grep -c "^openssl" "$CALL_LOG" || true)\n'
        'echo "openssl $*" >> "$CALL_LOG"\n'
        '[ -z "${FAKE_OPENSSL_RC:-}" ] || exit "$FAKE_OPENSSL_RC"\n'
        "printf '%048x\\n' \"$((calls + 1))\""
    ),
}

pytestmark = pytest.mark.skipif(shutil.which("sh") is None or shutil.which("bash") is None,
                                reason="needs sh and bash")


@pytest.fixture()
def checkout(tmp_path: Path) -> Path:
    """A directory holding the repository's .env.example, as a fresh clone has it."""
    root = tmp_path / "checkout"
    root.mkdir()
    shutil.copyfile(_EXAMPLE, root / ".env.example")
    return root


def _make_env(root: Path, **env: str) -> tuple[int, list[str], str]:
    """``sh scripts/stack_env.sh <root>`` with the stubs: (exit code, calls, output)."""
    bin_dir = root.parent / "bin"
    if not bin_dir.exists():
        bin_dir.mkdir()
        for name, body in _STUBS.items():
            path = bin_dir / name
            path.write_text(f"#!/bin/bash\n{body}\n", encoding="utf-8")
            path.chmod(path.stat().st_mode | stat.S_IXUSR)
    log = root.parent / "calls.log"
    log.write_text("", encoding="utf-8")
    proc = subprocess.run(["sh", str(_SCRIPT), str(root)],
                          env={"PATH": f"{bin_dir}{os.pathsep}/usr/bin{os.pathsep}/bin", "CALL_LOG": str(log), **env},
                          stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=60)
    return proc.returncode, log.read_text(encoding="utf-8").splitlines(), proc.stdout + proc.stderr


def _values(text: str) -> dict:
    """The last value each key has in a .env text."""
    values = {}
    for line in text.splitlines():
        key, sep, value = line.partition("=")
        if sep and not key.startswith("#"):
            values[key] = value
    return values


def _without_secrets(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.partition("=")[0] not in _KEYS]


def _placeholder_env(root: Path) -> bytes:
    shutil.copyfile(_EXAMPLE, root / ".env")
    return (root / ".env").read_bytes()


# --------------------------------------------------------------------------- a fresh checkout


def test_a_fresh_checkout_gets_a_private_env_with_three_generated_secrets(checkout):
    code, calls, out = _make_env(checkout)
    assert code == 0, out
    env = checkout / ".env"
    assert stat.S_IMODE(env.stat().st_mode) == 0o600
    values = _values(env.read_text(encoding="utf-8"))
    secrets = [values[key] for key in _KEYS]
    assert all(_SECRET.fullmatch(secret) for secret in secrets), secrets
    assert len(set(secrets)) == 3, "each secret is generated on its own"
    assert calls == ["docker volume ls -q"] + ["openssl rand -hex 24"] * 3
    assert "created .env from .env.example" in out
    assert "generated POSTGRES_PASSWORD REDIS_PASSWORD API_KEY in .env" in out


def test_every_other_line_of_env_example_is_kept_as_it_is(checkout):
    code, _, out = _make_env(checkout)
    assert code == 0, out
    written = (checkout / ".env").read_text(encoding="utf-8")
    assert _without_secrets(written) == _without_secrets(_EXAMPLE.read_text(encoding="utf-8"))


def test_a_second_run_changes_nothing_and_asks_neither_docker_nor_openssl(checkout):
    assert _make_env(checkout)[0] == 0
    before = (checkout / ".env").read_bytes()
    code, calls, out = _make_env(checkout)
    assert code == 0, out
    assert (checkout / ".env").read_bytes() == before
    assert calls == [] and out == ""


def test_real_secrets_and_every_other_value_are_never_touched(checkout):
    env = checkout / ".env"
    env.write_text(f"POSTGRES_PASSWORD={_REAL}\nREDIS_PASSWORD=mine\nAPI_KEY={_REAL}\n"
                   "MINIO_ROOT_PASSWORD=CHANGE_ME_too\nDEFAULT_WORKSPACE_ID=\n", encoding="utf-8")
    env.chmod(0o640)
    before = env.read_bytes()
    code, calls, out = _make_env(checkout)
    assert code == 0, out
    assert env.read_bytes() == before and stat.S_IMODE(env.stat().st_mode) == 0o640
    assert calls == []


def test_empty_quoted_and_missing_secrets_are_filled_and_nothing_else(checkout):
    env = checkout / ".env"
    env.write_text('POSTGRES_PASSWORD=\nREDIS_PASSWORD="CHANGE_ME_quoted"\nOTHER=CHANGE_ME_kept', encoding="utf-8")
    env.chmod(0o640)
    code, _, out = _make_env(checkout)
    assert code == 0, out
    text = env.read_text(encoding="utf-8")
    values = _values(text)
    assert all(_SECRET.fullmatch(values[key]) for key in _KEYS), values
    assert values["OTHER"] == "CHANGE_ME_kept"
    assert text.splitlines()[2] == "OTHER=CHANGE_ME_kept" and text.splitlines()[3].startswith("API_KEY=")
    assert stat.S_IMODE(env.stat().st_mode) == 0o640, "an existing .env keeps its mode"
    assert not list(checkout.glob(".env.make-env.*")), "no working copy is left behind"


def test_a_key_the_shell_sets_is_left_to_the_shell(checkout):
    _placeholder_env(checkout)
    code, calls, out = _make_env(checkout, API_KEY=_REAL)
    assert code == 0, out
    values = _values((checkout / ".env").read_text(encoding="utf-8"))
    assert values["API_KEY"].startswith("CHANGE_ME"), "compose runs the shell's value"
    assert _SECRET.fullmatch(values["POSTGRES_PASSWORD"]) and _SECRET.fullmatch(values["REDIS_PASSWORD"])
    assert calls.count("openssl rand -hex 24") == 2


def test_a_placeholder_the_shell_sets_is_refused(checkout):
    _placeholder_env(checkout)
    code, _, out = _make_env(checkout, API_KEY="CHANGE_ME_from_the_shell")
    assert code != 0
    assert "Placeholder secrets: API_KEY. Each still starts with CHANGE_ME" in out
    assert "the shell or the make command line sets a CHANGE_ME value" in out


# --------------------------------------------------------------------------- an install that has started


def test_a_started_install_on_placeholders_is_left_unchanged_with_how_to_rotate(checkout):
    before = _placeholder_env(checkout)
    code, calls, out = _make_env(checkout, FAKE_VOLUMES=f"other_volume {_VOLUME}")
    assert code != 0
    assert (checkout / ".env").read_bytes() == before
    assert calls == ["docker volume ls -q"], "nothing is generated"
    assert "Placeholder secrets: POSTGRES_PASSWORD REDIS_PASSWORD API_KEY. Each still starts with CHANGE_ME" in out
    assert "ALTER USER postgres PASSWORD" in out and "REDIS_PASSWORD and API_KEY: a new value in .env alone" in out
    assert f"the {_VOLUME} volume exists, so this install has started" in out and ".env is unchanged" in out


def test_the_refusal_is_the_entrypoint_guards_own_message(checkout):
    _placeholder_env(checkout)
    _, _, out = _make_env(checkout, FAKE_VOLUMES=_VOLUME)
    example = _values(_EXAMPLE.read_text(encoding="utf-8"))
    guard = subprocess.run(["sh", str(_GUARD)], env={"PATH": "/usr/bin:/bin", **{key: example[key] for key in _KEYS}},
                           capture_output=True, text=True, timeout=30)
    assert guard.returncode != 0 and guard.stderr.strip()
    assert guard.stderr in out


def test_a_started_install_with_an_empty_secret_stops_without_generating_one(checkout):
    env = checkout / ".env"
    env.write_text(f"POSTGRES_PASSWORD=\nREDIS_PASSWORD={_REAL}\nAPI_KEY={_REAL}\n", encoding="utf-8")
    before = env.read_bytes()
    code, calls, out = _make_env(checkout, FAKE_VOLUMES=_VOLUME)
    assert code != 0
    assert env.read_bytes() == before and "openssl rand -hex 24" not in calls
    assert f"POSTGRES_PASSWORD is empty in .env, and the {_VOLUME} volume exists" in out


def test_real_secrets_on_a_started_install_pass_without_asking_docker(checkout):
    (checkout / ".env").write_text(f"POSTGRES_PASSWORD={_REAL}\nREDIS_PASSWORD={_REAL}\nAPI_KEY={_REAL}\n",
                                   encoding="utf-8")
    code, calls, out = _make_env(checkout, FAKE_VOLUMES=_VOLUME)
    assert code == 0, out
    assert calls == []


# --------------------------------------------------------------------------- failures change nothing


def test_docker_not_answering_changes_nothing(checkout):
    before = _placeholder_env(checkout)
    code, calls, out = _make_env(checkout, FAKE_DOCKER_RC="1")
    assert code != 0
    assert (checkout / ".env").read_bytes() == before
    assert "Docker did not answer `docker volume ls`" in out and "openssl rand -hex 24" not in calls


def test_openssl_failing_changes_nothing(checkout):
    before = _placeholder_env(checkout)
    code, _, out = _make_env(checkout, FAKE_OPENSSL_RC="1")
    assert code != 0
    assert (checkout / ".env").read_bytes() == before
    assert "openssl rand -hex 24 failed while generating POSTGRES_PASSWORD. .env is unchanged." in out


# --------------------------------------------------------------------------- the Makefile


def _recipe(target: str) -> list[str]:
    makefile = _MAKEFILE.read_text(encoding="utf-8")
    body = makefile.split(f"\n{target}:\n", 1)[1]
    return [line.strip() for line in body.split("\n\n", 1)[0].splitlines() if line.startswith("\t")]


def test_make_env_runs_the_script():
    makefile = _MAKEFILE.read_text(encoding="utf-8")
    assert re.search(r"^STACK_ENV\s*\?=\s*sh scripts/stack_env\.sh$", makefile, re.M)
    assert _recipe("env") == ["@$(STACK_ENV)"]
    phony = next(line for line in makefile.splitlines() if line.startswith(".PHONY:"))
    assert "env" in phony.split()


@pytest.mark.parametrize("target", ["up", "build", "dev"])
def test_up_build_and_dev_run_make_env_as_their_first_recipe_line(target):
    """A recipe line, not a prerequisite: the #818 guide test's target regex skips `up: env`."""
    makefile = _MAKEFILE.read_text(encoding="utf-8")
    assert re.search(rf"^{target}:\s*$", makefile, re.M), f"{target} must keep a bare `{target}:` header"
    assert _recipe(target)[0] == _FIRST_LINE
