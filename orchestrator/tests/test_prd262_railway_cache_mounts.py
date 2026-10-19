"""PRD-262 US-044: no BuildKit cache mount that Railway would refuse.

Railway builds five Dockerfiles from this repository with its own BuildKit. Its
builder fails a `RUN --mount=type=cache` that has no `id`, or whose id lacks the
hard-coded `s/<service-id>-` prefix, and the id cannot come from a variable.
None of the five service ids is known yet (PRD-262 §10), so none of these files
has a cache mount today. When the operator supplies an id, a mount takes the form
`--mount=type=cache,id=s/<service-id>-/root/.cache/uv,target=/root/.cache/uv`,
and this check keeps every mount in that form.

The Dockerfile list is the one test_prd262_production_last_stage.py checks
against infrastructure/railway-manifest.json, so a new Railway service joins both.

Pure file reads: no Docker, no network.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests.test_prd262_production_last_stage import RAILWAY_DOCKERFILES

ROOT = Path(__file__).resolve().parents[2]
_CONTINUATION = re.compile(r"\\[ \t]*\n")
_MOUNT = re.compile(r"--mount=(\S+)")


def _instructions(text: str) -> list[str]:
    """Instructions with continuation lines joined; comment lines dropped."""
    code = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    return [line.strip() for line in _CONTINUATION.sub(" ", code).splitlines() if line.strip()]


def railway_cache_mount_problems(text: str) -> list[str]:
    """Each cache mount whose id does not start with `s/`, as Railway requires."""
    problems = []
    for line in _instructions(text):
        if not line.upper().startswith("RUN "):
            continue
        for spec in _MOUNT.findall(line):
            options = dict(part.split("=", 1) if "=" in part else (part, "") for part in spec.split(","))
            if options.get("type") != "cache":
                continue
            if not options.get("id", "").startswith("s/"):
                problems.append(f"--mount={spec}: Railway needs id=s/<service-id>-<path> (PRD-262 §10)")
    return problems


@pytest.mark.parametrize("path", RAILWAY_DOCKERFILES)
def test_every_cache_mount_in_a_railway_dockerfile_carries_a_railway_id(path):
    assert railway_cache_mount_problems((ROOT / path).read_text()) == []


@pytest.mark.parametrize(("dockerfile", "problems"), [
    ("FROM python:3.12-slim\nRUN pip install -r requirements.txt\n", 0),
    ("FROM python:3.12-slim\nRUN --mount=type=cache,target=/root/.cache/uv uv pip install x\n", 1),
    ("FROM python:3.12-slim\nRUN --mount=target=/root/.cache/uv,type=cache uv pip install x\n", 1),
    ("FROM python:3.12-slim\nRUN --mount=type=cache,id=uv,target=/root/.cache/uv uv pip install x\n", 1),
    ("FROM python:3.12-slim\nRUN --mount=type=cache,id=${SERVICE}-uv,target=/root/.cache/uv true\n", 1),
    ("FROM python:3.12-slim\nRUN --mount=type=cache,id=s/0b1c-/root/.cache/uv,target=/root/.cache/uv true\n", 0),
    ("FROM python:3.12-slim\nRUN --mount=type=bind,from=locks,source=/a,target=/b true\n", 0),
    ("FROM node:24\nRUN --mount=type=cache,id=s/0b1c-/root/.npm,target=/root/.npm \\\n"
     "    --mount=type=cache,target=/app/.next/cache npm ci\n", 1),
    ("FROM python:3.12-slim\n# RUN --mount=type=cache,target=/root/.cache/uv uv pip install x\n", 0),
])
def test_the_rule_on_fixtures(dockerfile, problems):
    assert len(railway_cache_mount_problems(dockerfile)) == problems
