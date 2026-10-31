"""F452 (#1309): compose gives the worker the same four capabilities as the chart.

The worker runs agent code. Its entrypoint starts as root only to re-own the workspace
volume (CHOWN, and DAC_OVERRIDE to walk folders it doesn't own) and to drop to uid 1000
(SETUID, SETGID) with ``setpriv --no-new-privs``. Measured on the built image: with
those four the volume is re-owned and the drop works; without DAC_OVERRIDE the walk
fails on a mode-700 folder. The chart's worker securityContext keeps exactly these (its
unit test pins it); compose keeps the same, so the two editions' runtimes can't drift.
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
NEEDED = ["CHOWN", "DAC_OVERRIDE", "SETUID", "SETGID"]


def test_compose_drops_every_capability_but_the_entrypoints_four():
    worker = yaml.safe_load((ROOT / "docker-compose.yml").read_text())["services"]["workspace-worker"]
    assert worker["cap_drop"] == ["ALL"]
    assert worker["cap_add"] == NEEDED


def test_the_chart_keeps_the_same_four():
    template = (ROOT / "charts" / "automatos" / "templates" / "worker.yaml").read_text()
    [added] = re.findall(r"^\s+add: \[([^\]]*)\]", template, re.MULTILINE)
    assert [name.strip().strip('"') for name in added.split(",")] == NEEDED
