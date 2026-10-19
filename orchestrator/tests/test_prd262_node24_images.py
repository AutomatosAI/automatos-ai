"""PRD-262 US-007: Node 24 in every image and CI lane, from the official images.

The built images are checked in CI: test.yml's prod-image-parity job looks inside
the frontend and worker images, and media-render.yml inside the media-render image.
This suite runs in the required orchestrator job and pins the definitions a later
edit could quietly undo:

- the frontend: the base stage on node:24-alpine3.24 with no apk build tools; the
  runtime on the same Alpine release with only the node binary copied in (no npm,
  yarn, corepack, C headers or curl); a busybox wget health check on /favicon.ico;
- the worker and media-render: Node from a stage named `node`, pinned as tag@digest,
  never NodeSource; npm and corepack copied, the C headers not; production last;
- the worker's pnpm pinned exactly, its npm cache cleaned in the same RUN;
- every Dockerfile that runs Node takes it through a named stage, never a bare
  `COPY --from=<image>`, because an updater reads FROM lines only;
- every setup-node step in the workflows on 24, and the CI checks of the built images.

Pure file reads: no Docker, no network.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"
FRONTEND = (ROOT / "frontend" / "Dockerfile").read_text()
WORKER = (ROOT / "services" / "workspace-worker" / "Dockerfile").read_text()
MEDIA_RENDER = (ROOT / "services" / "media-render" / "Dockerfile").read_text()
DEBIAN_NODE_IMAGES = {
    "services/workspace-worker/Dockerfile": WORKER,
    "services/media-render/Dockerfile": MEDIA_RENDER,
}
NODE_DOCKERFILES = (
    "frontend/Dockerfile",
    "services/workspace-worker/Dockerfile",
    "services/media-render/Dockerfile",
    ".oss-scanner/Dockerfile",
)
DIGEST = r"@sha256:[0-9a-f]{64}"
BUILD_TOOLS = {"python3", "make", "g++", "curl"}

_FROM = re.compile(r"^FROM\s+(\S+)(?:\s+AS\s+(\S+))?\s*$", re.M | re.I)
_APK_ADD = re.compile(r"\bapk[ \t]+add((?:[ \t]+[^\s&;|]+)*)")


def _stages(dockerfile: str) -> list[tuple[str, str]]:
    """(image, stage name) for every FROM line, in order; '' for an unnamed stage."""
    return [(image, name.lower()) for image, name in _FROM.findall(dockerfile)]


def _stage_images(dockerfile: str) -> dict[str, str]:
    return {name: image for image, name in _stages(dockerfile) if name}


def _instructions(text: str) -> list[str]:
    """Instructions with continuation lines joined; comment lines dropped."""
    code = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    return [line.strip() for line in code.replace("\\\n", " ").splitlines() if line.strip()]


def _stage_body(dockerfile: str, name: str) -> list[str]:
    """The instructions of stage `name`, up to the next FROM line."""
    match = re.search(rf"^FROM\s+\S+\s+AS\s+{re.escape(name)}\s*$", dockerfile, re.M | re.I)
    assert match, f"no stage named {name}"
    rest = dockerfile[match.end():]
    following = re.search(r"^FROM\s", rest, re.M)
    return _instructions(rest[: following.start()] if following else rest)


def _apk_packages(instructions: list[str]) -> list[str]:
    return [
        word
        for args in _APK_ADD.findall("\n".join(instructions))
        for word in args.split()
        if not word.startswith("-")
    ]


def _copies_from(instructions: list[str], source: str) -> list[str]:
    """The source paths of every `COPY --from=<source>`."""
    pattern = re.compile(rf"^COPY\s+(?:--\S+\s+)*--from={re.escape(source)}\s+(?:--\S+\s+)*(\S+)\s", re.I)
    return [match.group(1) for match in map(pattern.match, instructions) if match]


# ── frontend ────────────────────────────────────────────────────────────────


def test_the_frontend_builds_on_node_24_alpine_3_24_with_no_build_tools():
    images = _stage_images(FRONTEND)
    assert re.fullmatch(rf"node:24-alpine3\.24{DIGEST}", images["base"]), images["base"]
    assert images["development"] == "base" and images["builder"] == "base"
    assert not BUILD_TOOLS & set(_apk_packages(_stage_body(FRONTEND, "base")))


def test_the_frontend_runtime_is_alpine_plus_only_the_node_binary():
    image, name = _stages(FRONTEND)[-1]
    assert name == "production", "production must stay the last stage (Railway builds no target)"
    # The musl node binary is built for one Alpine release: the runtime runs that release.
    alpine = re.match(r"node:\d+-alpine(\d+\.\d+)@", _stage_images(FRONTEND)["base"])
    assert alpine, "the base stage names no Alpine release"
    assert re.fullmatch(rf"alpine:{re.escape(alpine.group(1))}{DIGEST}", image), image
    body = _stage_body(FRONTEND, "production")
    assert _copies_from(body, "base") == ["/usr/local/bin/node"]
    assert _apk_packages(body) == ["libstdc++"]
    code = "\n".join(body)
    for absent in ("npm", "yarn", "corepack", "curl"):
        assert not re.search(rf"\b{absent}\b", code), absent
    assert "/usr/local/include" not in code


def test_the_frontend_health_check_is_busybox_wget_on_the_favicon():
    health = [line for line in _stage_body(FRONTEND, "production") if line.startswith("HEALTHCHECK")]
    assert len(health) == 1
    assert re.search(r"\bCMD wget\b[^\n]*/favicon\.ico", health[0]), health[0]
    assert (ROOT / "frontend" / "public" / "favicon.ico").is_file()


# ── worker and media-render ─────────────────────────────────────────────────


@pytest.mark.parametrize("path", sorted(DEBIAN_NODE_IMAGES))
def test_the_debian_images_take_node_24_from_the_official_image(path):
    dockerfile = DEBIAN_NODE_IMAGES[path]
    images = _stage_images(dockerfile)
    assert re.fullmatch(rf"node:24-trixie-slim{DIGEST}", images.get("node", "")), images
    assert _stages(dockerfile)[-1][1] == "production", "production must stay the last stage"
    copied = sorted(_copies_from(_instructions(dockerfile), "node"))
    assert copied == [
        "/usr/local/bin/node",
        "/usr/local/lib/node_modules/corepack",
        "/usr/local/lib/node_modules/npm",
    ]
    code = "\n".join(_instructions(dockerfile)).lower()
    for gone in ("nodesource", "/usr/local/include", "gnupg", "python3.13"):
        assert gone not in code, gone


def test_the_worker_pins_pnpm_exactly_and_cleans_the_npm_cache_in_the_same_run():
    runs = [line for line in _instructions(WORKER) if line.startswith("RUN") and "pnpm" in line]
    assert len(runs) == 1, runs
    # PRD-262 §9-Q9: pinned exactly to the current 12.x.
    assert re.search(r"\bnpm install -g\b[^\n]*\bpnpm@12\.\d+\.\d+(?=\s|$)", runs[0]), runs[0]
    assert "npm cache clean --force" in runs[0]


@pytest.mark.parametrize("path", NODE_DOCKERFILES)
def test_node_comes_through_a_named_stage_pinned_as_tag_and_digest(path):
    dockerfile = (ROOT / path).read_text()
    names = set(_stage_images(dockerfile))
    sources = re.findall(r"^COPY\b[^\n]*?--from=(\S+)", dockerfile, re.M | re.I)
    assert [source for source in sources if source.lower() not in names] == [], "bare COPY --from=<image>"
    node_images = [image for image, _ in _stages(dockerfile) if image.startswith("node:")]
    assert node_images, f"{path} takes no Node stage"
    for image in node_images:
        assert re.fullmatch(rf"node:24-[a-z0-9.-]+{DIGEST}", image), image


# ── CI ──────────────────────────────────────────────────────────────────────


def _setup_node_versions() -> dict[str, str]:
    """`workflow:job` -> node-version, for every actions/setup-node step."""
    versions = {}
    for path in sorted(WORKFLOWS.glob("*.y*ml")):
        jobs = (yaml.safe_load(path.read_text()) or {}).get("jobs") or {}
        versions.update({
            f"{path.name}:{name}": str((step.get("with") or {}).get("node-version"))
            for name, job in jobs.items()
            for step in job.get("steps") or []
            if str(step.get("uses", "")).startswith("actions/setup-node@")
        })
    return versions


def test_every_setup_node_step_runs_node_24():
    versions = _setup_node_versions()
    assert versions, "no setup-node step found in .github/workflows"
    assert {where: v for where, v in versions.items() if v != "24"} == {}


def _job_script(workflow: str, job: str) -> str:
    steps = yaml.safe_load((WORKFLOWS / workflow).read_text())["jobs"][job]["steps"]
    return "\n".join(step.get("run", "") for step in steps)


def test_ci_checks_node_inside_the_built_images():
    parity = _job_script("test.yml", "prod-image-parity")
    for needle in (
        "docker run --rm parity-fe node -v",
        "for name in npm curl yarn corepack",
        "in_image 'node -v'",
        "in_image 'pnpm --version'",
        "for pkg in python3.13 nodejs gnupg",
    ):
        assert needle in parity, needle
    media_render = _job_script("media-render.yml", "media-render")
    assert '--entrypoint node "$IMAGE" -v' in media_render
    assert "for pkg in python3.13 nodejs gnupg" in media_render
