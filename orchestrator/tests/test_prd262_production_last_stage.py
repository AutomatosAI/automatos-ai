"""PRD-262 US-010: production is the last stage of every Dockerfile Railway builds.

Railway's config file has no target key. Its build keys are builder,
dockerfilePath, buildCommand, watchPatterns and the Nixpacks/Railpack ones, so
Railway builds the LAST stage of each Dockerfile. railway.json used to set
`dockerfileTarget: production`, which Railway never read, and a test asserted
that value. Production shipped only because it happened to be the last stage. A
stage added after it (a debug or test stage) would have shipped to production
with every check green.

This suite checks the order itself:

- the Dockerfile of every Railway service built from this repo ends with the
  stage named `production`: the API, the frontend and media-render, which the
  PRD names, plus the workspace worker and agent-opt-worker;
- that list matches the in-repo services in infrastructure/railway-manifest.json,
  so a new Railway service has to join it;
- any other tracked Dockerfile that has a `production` stage keeps it last (FR-8);
- railway.json no longer claims a target;
- the check itself fails on fixtures with a stage after production.

Pure file reads: no Docker, no network.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
RAILWAY_JSON = ROOT / "railway.json"
MANIFEST = ROOT / "infrastructure" / "railway-manifest.json"
THIS_REPO = "AutomatosAI/automatos-ai"
PRODUCTION = "production"

# The Dockerfiles Railway builds from this repo. The first three are the ones
# PRD-262 US-010 names; the two workers became multi-stage in US-006 and US-007.
RAILWAY_DOCKERFILES = (
    "orchestrator/Dockerfile",
    "frontend/Dockerfile",
    "services/media-render/Dockerfile",
    "services/workspace-worker/Dockerfile",
    "services/agent-opt-worker/Dockerfile",
    "services/voice-tts/Dockerfile",
)

# Dockerfile, Dockerfile.<variant> and <name>.Dockerfile; not a Dockerfile's own
# .dockerignore, nor a document named after one.
_DOCKERFILE_NAME = re.compile(
    r"(?:^|/)(?:Dockerfile(?:\.(?!md$|txt$|dockerignore$)[\w-]+)?|[\w.-]+\.[Dd]ockerfile)$"
)
_CONTINUATION = re.compile(r"\\[ \t]*\n")
_FROM_WORD = re.compile(r"^FROM\s", re.I)
# FROM [--platform=…] <image> [AS <name>]
_FROM = re.compile(r"^FROM\s+(?:--\S+\s+)*\S+(?:\s+AS\s+(\S+))?\s*$", re.I)


def _instructions(text: str) -> list[str]:
    """The Dockerfile's instructions, with continuation lines joined and comments dropped."""
    code = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    return [line.strip() for line in _CONTINUATION.sub(" ", code).splitlines() if line.strip()]


def _shown(names: list[str]) -> list[str]:
    return [name or "<unnamed>" for name in names]


def stage_names(text: str) -> list[str]:
    """Each stage's name in build order, in lower case as BuildKit compares them.

    An unnamed stage is ''. A FROM line this parser cannot read raises ValueError,
    so a check never passes by skipping a stage.
    """
    froms = [line for line in _instructions(text) if _FROM_WORD.match(line)]
    matches = [_FROM.match(line) for line in froms]
    unreadable = [line for line, match in zip(froms, matches) if not match]
    if unreadable:
        raise ValueError(f"FROM lines this check cannot read: {unreadable}")
    return [(match.group(1) or "").lower() for match in matches]


def production_last_problem(names: list[str]) -> str | None:
    """Why these stages do not end with `production`, or None when they do."""
    if names and names[-1] == PRODUCTION:
        return None
    if PRODUCTION not in names:
        return f"no stage is named {PRODUCTION}; the stages are {_shown(names)}"
    after = names[len(names) - names[::-1].index(PRODUCTION):]
    return (
        f"stage(s) {_shown(after)} come after {PRODUCTION}. Railway builds the last stage "
        f"(its config has no target key), so the last one would ship to production"
    )


def _manifest_dockerfiles() -> set[str]:
    """The Dockerfile of each in-repo service in the Railway manifest, relative to the repo."""
    services = json.loads(MANIFEST.read_text())["services"].values()
    ours = [service for service in services if service.get("source", {}).get("repo") == THIS_REPO]
    default_name = json.loads(RAILWAY_JSON.read_text())["build"]["dockerfilePath"]
    return {
        (service.get("dockerfile") or f"{service['source']['root_dir']}/{default_name}").strip("/")
        for service in ours
    }


def _tracked_dockerfiles() -> list[str]:
    proc = subprocess.run(
        ["git", "ls-files"], cwd=str(ROOT), capture_output=True, text=True, timeout=30
    )
    assert proc.returncode == 0, f"git ls-files failed: {proc.stderr}"
    return [path for path in proc.stdout.splitlines() if _DOCKERFILE_NAME.search(path)]


# ── the real Dockerfiles ────────────────────────────────────────────────────


@pytest.mark.parametrize("path", RAILWAY_DOCKERFILES)
def test_production_is_the_last_stage_of_every_dockerfile_railway_builds(path):
    problem = production_last_problem(stage_names((ROOT / path).read_text()))
    assert problem is None, f"{path}: {problem}"


def test_the_list_matches_the_railway_services_built_from_this_repo():
    assert _manifest_dockerfiles() == set(RAILWAY_DOCKERFILES), (
        "infrastructure/railway-manifest.json and RAILWAY_DOCKERFILES disagree: "
        "every Dockerfile Railway builds must end with the production stage"
    )


def test_every_tracked_dockerfile_with_a_production_stage_keeps_it_last():
    tracked = _tracked_dockerfiles()
    assert set(RAILWAY_DOCKERFILES) <= set(tracked), "the Dockerfile search missed a Railway file"
    for path in tracked:
        names = stage_names((ROOT / path).read_text())
        if PRODUCTION in names:
            assert production_last_problem(names) is None, f"{path}: {production_last_problem(names)}"


def test_railway_config_claims_only_what_railway_reads():
    build = json.loads(RAILWAY_JSON.read_text())["build"]
    assert "dockerfileTarget" not in build, (
        "Railway's config has no target key and builds the last stage; this file's "
        "tests check the stage order instead"
    )
    # It was never a Railway file and nothing read it. Railway's watch paths are build.watchPatterns.
    assert not (ROOT / "orchestrator" / ".railway-watch.json").exists()


# ── the check itself ────────────────────────────────────────────────────────

STAGE_AFTER_PRODUCTION = """\
FROM python:3.11-slim AS base
FROM base AS production
CMD ["serve"]
FROM production AS debug
RUN pip install debugpy
"""
UNNAMED_STAGE_AFTER_PRODUCTION = """\
FROM python:3.11-slim AS production
FROM production
"""
NO_PRODUCTION_STAGE = """\
FROM python:3.11-slim AS base
FROM base AS runtime
"""
SINGLE_UNNAMED_STAGE = "FROM python:3.11-slim\nCMD [\"serve\"]\n"
UNREADABLE_FROM = "FROM python:3.11-slim AS production\nFROM base AS\n"
PRODUCTION_LAST = """\
# syntax=docker/dockerfile:1
ARG BASE=python:3.11-slim
FROM --platform=$BUILDPLATFORM node:24-alpine as node
FROM ${BASE} as base
RUN echo one \\
    two
# FROM base AS debug   (a comment is not a stage)
FROM \\
    base \\
    AS Production
CMD ["serve"]
"""


@pytest.mark.parametrize(
    ("fixture", "expected"),
    [
        (STAGE_AFTER_PRODUCTION, "['debug'] come after production"),
        (UNNAMED_STAGE_AFTER_PRODUCTION, "['<unnamed>'] come after production"),
        (NO_PRODUCTION_STAGE, "no stage is named production"),
        (SINGLE_UNNAMED_STAGE, "no stage is named production"),
    ],
    ids=["stage-after-production", "unnamed-after", "no-production", "single-unnamed"],
)
def test_the_check_fails_when_production_is_not_last(fixture, expected):
    problem = production_last_problem(stage_names(fixture))
    assert problem is not None and expected in problem, problem


def test_the_check_refuses_a_from_line_it_cannot_read():
    with pytest.raises(ValueError, match="cannot read"):
        stage_names(UNREADABLE_FROM)


def test_the_check_reads_flags_lower_case_continuations_and_comments():
    assert stage_names(PRODUCTION_LAST) == ["node", "base", "production"]
    assert production_last_problem(stage_names(PRODUCTION_LAST)) is None
