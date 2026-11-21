"""F452 (P256-FIX-SEC-3): nothing the workspace worker runs can regain privileges.

/workspaces/{id}/exec and /git run commands a caller supplies, as the worker (uid 1000).
Before this fix the drop to uid 1000 left no_new_privs off, the image shipped nine
setuid-root binaries (su, mount, umount, passwd, chsh, chfn, newgrp, gpasswd,
ssh-keysign), and the Helm chart's worker container kept the runtime's default
capabilities with privilege escalation allowed. Pinned here as file reads:

- entrypoint.sh drops to the worker with ``setpriv ... --no-new-privs``;
- the Dockerfile's production stage strips every setuid and setgid bit after its last
  install, and no RUN follows that could bring one back;
- the chart's worker container sets ``allowPrivilegeEscalation: false``, drops every
  capability and adds none, and its pod runs as uid 1000 (PRD-280 US-201;
  charts/automatos/tests/worker_frontend_test.yaml renders it). Compose, which starts the
  worker as root on purpose, keeps the four the entrypoint's root part uses, its comment
  naming the command that needs each;
- scripts/ci/image_worker_toolchain.py, run in the image images.yml builds on both arches,
  finds no setuid or setgid file and reads NoNewPrivs 1 in the dropped process; test.yml's
  prod-image-parity checks both on the Railway build (test_prd262_worker_boot.py).

The kernel's part, setuid bits ignored under no_new_privs, shows only in a container: CI.
"""
from __future__ import annotations

import importlib.util
import re
import shlex
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKER = ROOT / "services" / "workspace-worker"
ENTRYPOINT = (WORKER / "entrypoint.sh").read_text()
DOCKERFILE = (WORKER / "Dockerfile").read_text()
CHART_WORKER = (ROOT / "charts" / "automatos" / "templates" / "worker.yaml").read_text()
COMPOSE = (ROOT / "docker-compose.yml").read_text()
TOOLCHAIN_SCRIPT = "scripts/ci/image_worker_toolchain.py"
NO_NEW_PRIVS = "--no-new-privs"
STRIP = "RUN find / -xdev -perm /6000 -type f -exec chmod a-s {} +"
SETID_BITS = 0o6000
# Each capability compose keeps for the root start, in the order its `cap_add` names them:
# (a phrase from compose's comment above the list, the entrypoint code that needs it while
# it runs as root). DAC_OVERRIDE, not DAC_READ_SEARCH: #1311 measured the built image and
# found the volume walk needs DAC_OVERRIDE on a mode-700 folder.
CAPABILITIES = {
    "CHOWN": ("re-own the workspace volume (CHOWN", "-exec chown -h worker:worker {} +"),
    "DAC_OVERRIDE": ("DAC_OVERRIDE to walk folders it doesn't own", 'find "$WORKSPACE_VOLUME_PATH" -mindepth 1'),
    "SETUID": ("uid 1000 (SETUID, SETGID)", "setpriv --reuid=worker"),
    "SETGID": ("uid 1000 (SETUID, SETGID)", "--regid=worker --init-groups"),
}


def _load(relative: str):
    """A CI script, loaded by path as the other image tests load theirs."""
    name = Path(relative).stem
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


toolchain = _load(TOOLCHAIN_SCRIPT)


def _code_lines(text: str) -> list[str]:
    """Lines with continuations joined; comment lines dropped."""
    code = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    return [line.strip() for line in re.sub(r"\\[ \t]*\n", " ", code).splitlines() if line.strip()]


def _production_stage() -> list[str]:
    instructions = _code_lines(DOCKERFILE)
    start = next(i for i, line in enumerate(instructions) if line.startswith("FROM ") and line.endswith(" AS production"))
    return instructions[start:]


# --------------------------------------------------------------------------- entrypoint.sh


def test_the_drop_to_the_worker_sets_no_new_privs():
    words = shlex.split(_code_lines(ENTRYPOINT)[-1])
    assert words[:2] == ["exec", "setpriv"]
    options = words[2:words.index("env")]
    assert NO_NEW_PRIVS in options, f"the drop runs {options}"
    assert {"--reuid=worker", "--regid=worker", "--init-groups"} <= set(options)
    assert words[words.index("env"):] == ["env", "HOME=/home/worker", "python", "-m", "main", "$@"]


def test_every_setpriv_that_runs_more_than_the_write_probe_sets_no_new_privs():
    calls = [line for line in _code_lines(ENTRYPOINT) if "setpriv " in line]
    others = [line for line in calls if " test -w " not in line]
    assert others and all(NO_NEW_PRIVS in line for line in others), others


# --------------------------------------------------------------------------- the Dockerfile


def test_the_production_stage_strips_setuid_and_setgid_after_its_last_install():
    stage = _production_stage()
    assert stage.count(STRIP) == 1 and _code_lines(DOCKERFILE).count(STRIP) == 1
    at = stage.index(STRIP)
    runs = [line for line in stage if line.startswith("RUN ")]
    assert runs[-1] == STRIP, "no RUN after the strip: an install there could add a setuid file back"
    installs = [i for i, line in enumerate(stage) if line.startswith("RUN ") and ("install" in line or "useradd" in line)]
    assert installs and max(installs) < at
    assert not [line for line in stage[at + 1:] if line.startswith("ADD ")]
    modes = [int(mode, 8) for line in stage[at + 1:] for mode in re.findall(r"--chmod=([0-7]{3,4})\b", line)]
    assert not [mode for mode in modes if mode & SETID_BITS], "a COPY after the strip sets a setuid or setgid bit"


def test_the_image_runs_as_uid_1000_by_default():
    """PRD-280 US-201: the production stage's one USER line is uid and gid 1000, after its last
    RUN, so the built image starts as the worker unless it is told to start as root."""
    stage = _production_stage()
    users = [line for line in stage if line.startswith("USER ")]
    assert users == ["USER 1000:1000"], users
    runs = [i for i, line in enumerate(stage) if line.startswith("RUN ")]
    assert max(runs) < stage.index("USER 1000:1000")


def test_the_hosted_worker_is_started_as_root_on_its_root_owned_volume():
    """Railway mounts a volume owned by root and starts a non-root image as its USER, so the
    hosted worker, now uid 1000 by default, must be told to start as root (RAILWAY_RUN_UID=0)
    and re-owns its volume as before (docs.railway.com/volumes#permissions). The manifest
    records the value the service requires, beside the key."""
    manifest = yaml.safe_load((ROOT / "infrastructure" / "railway-manifest.json").read_text())
    worker = manifest["services"]["agent-workspace-worker"]
    assert [volume["mount"] for volume in worker["volumes"]] == ["/workspaces"]
    assert worker["required_env_values"] == {"RAILWAY_RUN_UID": "0"}
    assert set(worker["required_env_values"]) <= set(worker["env_keys"])
    assert worker["env_keys"] == sorted(worker["env_keys"])


# --------------------------------------------------------------------------- the chart


def _worker_security_contexts() -> list[dict]:
    """The worker's two securityContext blocks, the pod's then the container's, parsed."""
    blocks = [match.group(0) for match in re.finditer(r"^( +)securityContext:\n(?:\1 .*\n)+", CHART_WORKER, re.M)]
    assert len(blocks) == 2, f"the worker sets {len(blocks)} securityContext blocks, not the pod's and the container's"
    # A Helm control line (`{{- if … }}`, `{{- end }}`) is not YAML: each block is parsed
    # as it renders with the chart's defaults, where every such condition holds.
    as_rendered = (
        "\n".join(line for line in block.splitlines() if not line.strip().startswith("{{")) for block in blocks)
    return [yaml.safe_load(text)["securityContext"] for text in as_rendered]


def test_the_chart_s_worker_starts_as_uid_1000_and_gains_no_privilege_or_capability():
    pod, container = _worker_security_contexts()
    assert pod == {
        "runAsNonRoot": True, "runAsUser": 1000, "runAsGroup": 1000,
        "fsGroup": 1000, "fsGroupChangePolicy": "OnRootMismatch",
    }
    assert container == {
        "allowPrivilegeEscalation": False,
        "readOnlyRootFilesystem": True,
        "capabilities": {"drop": ["ALL"]},
        "seccompProfile": {"type": "RuntimeDefault"},
    }
    assert "privileged" not in CHART_WORKER


def test_each_capability_compose_keeps_names_the_entrypoint_command_that_needs_it():
    """Compose documents the four capabilities of the root start with one comment above its
    `cap_drop` and `cap_add` lists."""
    worker = COMPOSE[COMPOSE.index("\n  workspace-worker:\n"):]
    above = worker[:worker.index("    cap_drop:")].splitlines()
    comment = " ".join(line.strip().lstrip("#").strip() for line in above if line.strip().startswith("#"))
    entrypoint = "\n".join(_code_lines(ENTRYPOINT))
    assert yaml.safe_load(COMPOSE)["services"]["workspace-worker"]["cap_add"] == list(CAPABILITIES)
    for capability, (named, code) in CAPABILITIES.items():
        assert named in comment, (capability, comment)
        assert code in entrypoint, f"{capability}: the entrypoint no longer runs {code!r}"


# --------------------------------------------------------------------------- the in-image checks


NONE = "0000000000000000"


def test_the_in_image_drop_sets_no_new_privs_and_reads_it_back(monkeypatch):
    drop = list(toolchain.SETPRIV_DROP)
    assert NO_NEW_PRIVS in drop[:drop.index("env")]
    assert toolchain.NO_NEW_PRIVS == "1" and "NoNewPrivs:" in toolchain.DROPPED_PROBE
    monkeypatch.setattr(toolchain.shutil, "which", lambda name: None)
    monkeypatch.setattr(toolchain, "must", lambda argv, cwd: f"1000 1000 /home/worker 0 {NONE} {NONE}")
    assert toolchain.check_setpriv(Path("/")) == [
        f"setpriv ran the command as '1000 1000 /home/worker 0 {NONE} {NONE}' (uid gid HOME NoNewPrivs CapPrm "
        f"CapEff), not '1000 1000 /home/worker 1 {NONE} {NONE}'"]
    monkeypatch.setattr(toolchain, "must", lambda argv, cwd: f"1000 1000 /home/worker 1 {NONE} {NONE}")
    assert toolchain.check_setpriv(Path("/")) == []


def test_the_in_image_drop_leaves_the_worker_no_capability(monkeypatch):
    """Security review: compose adds CHOWN, DAC_OVERRIDE, SETGID and SETUID back for the root part;
    past the drop the worker's permitted and effective sets must be empty."""
    assert "CapPrm:" in toolchain.DROPPED_PROBE and "CapEff:" in toolchain.DROPPED_PROBE
    monkeypatch.setattr(toolchain.shutil, "which", lambda name: None)
    kept = "00000000000000c0"                                   # CAP_SETGID | CAP_SETUID still held
    monkeypatch.setattr(toolchain, "must", lambda argv, cwd: f"1000 1000 /home/worker 1 {kept} {kept}")
    assert toolchain.check_setpriv(Path("/")) != []


def test_the_in_image_check_lists_every_setuid_or_setgid_file(monkeypatch):
    assert toolchain.SETID_FILES == ("find", "/", "-xdev", "-perm", "/6000", "-type", "f")
    assert toolchain.CHECKS[-1][1] is toolchain.check_no_setid_files
    asked = []
    monkeypatch.setattr(toolchain, "must", lambda argv, cwd: asked.append(tuple(argv)) or "/usr/bin/su\n/usr/bin/passwd")
    assert toolchain.check_no_setid_files(Path("/")) == [
        "/usr/bin/su keeps its setuid or setgid bit", "/usr/bin/passwd keeps its setuid or setgid bit"]
    assert asked == [toolchain.SETID_FILES]
    monkeypatch.setattr(toolchain, "must", lambda argv, cwd: "")
    assert toolchain.check_no_setid_files(Path("/")) == []
