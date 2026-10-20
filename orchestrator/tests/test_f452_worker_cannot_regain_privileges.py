"""F452 (P256-FIX-SEC-3): nothing the workspace worker runs can regain privileges.

/workspaces/{id}/exec and /git run commands a caller supplies, as the worker (uid 1000).
Before this fix the drop to uid 1000 left no_new_privs off, the image shipped nine
setuid-root binaries (su, mount, umount, passwd, chsh, chfn, newgrp, gpasswd,
ssh-keysign), and the Helm chart's worker container kept the runtime's default
capabilities with privilege escalation allowed. Pinned here as file reads:

- entrypoint.sh drops to the worker with ``setpriv ... --no-new-privs``;
- the Dockerfile's production stage strips every setuid and setgid bit after its last
  install, and no RUN follows that could bring one back;
- the chart's worker container sets ``allowPrivilegeEscalation: false`` and drops every
  capability except the four the entrypoint's root part uses, each commented with the
  command that needs it (charts/automatos/tests/worker_frontend_test.yaml renders it);
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
TOOLCHAIN_SCRIPT = "scripts/ci/image_worker_toolchain.py"
NO_NEW_PRIVS = "--no-new-privs"
STRIP = "RUN find / -xdev -perm /6000 -type f -exec chmod a-s {} +"
SETID_BITS = 0o6000
# Each capability the chart keeps, in the order its `add` list names them: (a phrase from
# the chart's comment above the list, the entrypoint code that needs it while it runs as
# root). DAC_OVERRIDE, not DAC_READ_SEARCH: #1311 measured the built image and found the
# volume walk needs DAC_OVERRIDE on a mode-700 folder (see worker.yaml's reconciliation
# note); an earlier draft of this fix used DAC_READ_SEARCH, untested against the image.
CAPABILITIES = {
    "CHOWN": ("re-owns the volume", "-exec chown -h worker:worker {} +"),
    "DAC_OVERRIDE": ("walk folders it doesn't", 'find "$WORKSPACE_VOLUME_PATH" -mindepth 1'),
    "SETUID": ("setpriv drops to uid 1000", "setpriv --reuid=worker"),
    "SETGID": ("setpriv drops to uid 1000", "--regid=worker --init-groups"),
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


# --------------------------------------------------------------------------- the chart


def _worker_security_context() -> tuple[dict, list[str]]:
    """The worker container's securityContext block, parsed, and its lines as written."""
    block = re.search(r"^( +)securityContext:\n(?:\1 .*\n)+", CHART_WORKER, re.M)
    assert block, "the worker container sets no securityContext"
    lines = block.group(0).splitlines()
    return yaml.safe_load("\n".join(lines)), lines


def test_the_chart_s_worker_gains_no_privileges_and_keeps_only_what_the_entrypoint_needs():
    parsed, _ = _worker_security_context()
    assert parsed == {"securityContext": {
        "allowPrivilegeEscalation": False,
        "capabilities": {"drop": ["ALL"], "add": list(CAPABILITIES)},
        "seccompProfile": {"type": "RuntimeDefault"},
    }}
    assert "privileged" not in CHART_WORKER


def test_each_capability_the_chart_keeps_names_the_entrypoint_command_that_needs_it():
    """The chart documents the four capabilities with one comment above the flat `add`
    list (not one comment per item, as an earlier draft of this change had it)."""
    _, lines = _worker_security_context()
    entrypoint = "\n".join(_code_lines(ENTRYPOINT))
    comment = " ".join(line.strip() for line in lines if line.strip().startswith("#"))
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
    """Security review: the chart adds CHOWN, DAC_READ_SEARCH, SETGID and SETUID back for the root part;
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
