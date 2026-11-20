"""F452 (#1309), PRD-280 US-201: the capabilities each way of starting the worker gives it.

The worker runs agent code. Compose starts the image as root on purpose (its ``user``), and
the entrypoint's root part re-owns the workspace volume (CHOWN, and DAC_OVERRIDE to walk
folders it doesn't own) and drops to uid 1000 (SETUID, SETGID) with ``setpriv
--no-new-privs``. Measured on the built image: with those four the volume is re-owned and
the drop works; without DAC_OVERRIDE the walk fails on a mode-700 folder. Compose keeps
exactly these.

The chart never starts the worker as root (US-201): its pod runs as uid 1000, and its
container adds no capability back (charts/automatos/tests/worker_frontend_test.yaml renders it).
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
NEEDED = ["CHOWN", "DAC_OVERRIDE", "SETUID", "SETGID"]


def _compose_worker() -> dict:
    return yaml.safe_load((ROOT / "docker-compose.yml").read_text())["services"]["workspace-worker"]


def test_compose_drops_every_capability_but_the_entrypoints_four():
    worker = _compose_worker()
    assert worker["cap_drop"] == ["ALL"]
    assert worker["cap_add"] == NEEDED


def test_compose_states_that_it_starts_the_worker_as_root():
    """The image starts as uid 1000 (its Dockerfile's USER); compose's root start is written down."""
    assert _compose_worker()["user"] == "0:0"


def test_the_chart_adds_no_capability_to_the_worker():
    template = (ROOT / "charts" / "automatos" / "templates" / "worker.yaml").read_text()
    assert not re.findall(r"^\s+add:", template, re.MULTILINE)
    assert 'drop: ["ALL"]' in template
    for capability in NEEDED:
        assert capability not in template, capability
