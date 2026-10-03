"""F228 (2 Oct, TESTER's build 2): a backend boot with migrations and seeds took
147 s. The compose health budget was 40 s + 3 x 30 s, so the backend was marked
unhealthy and the frontend, which waits for a healthy backend, never started.

The budget now covers a slow boot. A probe that passes inside start_period marks
the backend healthy at once, so a fast boot waits for nothing more.
"""
from __future__ import annotations

import pathlib
import re

import yaml

_REPO = pathlib.Path(__file__).resolve().parents[2]
SLOWEST_BOOT_SEEN_S = 147          # build 2, 2 Oct
HEADROOM = 2


def _seconds(value) -> int:
    match = re.fullmatch(r"(\d+)s", str(value).strip())
    assert match, f"healthcheck durations are written in seconds: {value!r}"
    return int(match.group(1))


def test_the_backends_health_budget_covers_a_slow_boot():
    compose = yaml.safe_load((_REPO / "docker-compose.yml").read_text())
    check = compose["services"]["backend"]["healthcheck"]
    budget = _seconds(check["start_period"]) + int(check["retries"]) * _seconds(check["interval"])
    assert budget >= SLOWEST_BOOT_SEEN_S * HEADROOM, budget


def test_the_frontend_still_waits_for_a_healthy_backend():
    compose = yaml.safe_load((_REPO / "docker-compose.yml").read_text())
    assert compose["services"]["frontend"]["depends_on"]["backend"]["condition"] == "service_healthy"
