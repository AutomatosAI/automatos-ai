"""PRD-251 Wave 1, US-101: the media-render image definition and the GPL boundary.

The media-render CI job builds the image and renders the fixture inside it.
This suite runs in the required orchestrator job and pins what a later edit
could quietly lose:

- the GPL boundary: phonemizer and espeak-ng (GPL-3.0, pulled in by kokoro-onnx)
  are never imported by, or a dependency of, the orchestrator (PRD-251 D3, Traps);
- Hyperframes pinned exactly at 0.8.62 on Node 24 (PRD-262 US-007), from the js
  stage's lock (US-053), with telemetry, global skill installs and update checks
  switched off before npm ci and in the runtime;
- the Kokoro files hash-checked at build in the kokoro stage, and the espeak data
  moved to a short real path (espeak-ng truncates its data path at about 160
  characters);
- the CI job itself: a top-level sibling job, with no `needs:`.

Pure file reads: no Docker, no database.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
MEDIA_RENDER = ROOT / "services" / "media-render"
DOCKERFILE = (MEDIA_RENDER / "Dockerfile").read_text()
JS_PACKAGE = json.loads((MEDIA_RENDER / "js" / "package.json").read_text())
JS_LOCK = json.loads((MEDIA_RENDER / "js" / "package-lock.json").read_text())
HYPERFRAMES_PIN = "0.8.62"

GPL_MODULES = ("phonemizer", "espeakng_loader", "kokoro_onnx", "misaki")
GPL_DISTRIBUTIONS = ("phonemizer", "espeakng-loader", "espeakng_loader", "kokoro-onnx", "kokoro_onnx", "misaki")
_GPL_IMPORT = re.compile(
    r"^\s*(?:import|from)\s+(" + "|".join(GPL_MODULES) + r")(?![A-Za-z0-9_])", re.M
)
HYPERFRAMES_SWITCHES = (
    "HYPERFRAMES_NO_TELEMETRY",
    "DO_NOT_TRACK",
    "HYPERFRAMES_SKIP_SKILLS",
    "HYPERFRAMES_NO_UPDATE_CHECK",
)


def test_the_orchestrator_never_imports_the_gpl_voice_stack():
    offenders = [
        f"{path.relative_to(ROOT)}: {match.group(0).strip()}"
        for path in (ROOT / "orchestrator").rglob("*.py")
        for match in _GPL_IMPORT.finditer(path.read_text(encoding="utf-8", errors="replace"))
    ]
    assert not offenders, f"GPL-3.0 voice stack imported by the orchestrator: {offenders}"


def test_the_orchestrator_does_not_depend_on_the_gpl_voice_stack():
    manifests = sorted((ROOT / "orchestrator").glob("requirements*.txt"))
    assert manifests, "no orchestrator requirements file found"
    pyproject = ROOT / "orchestrator" / "pyproject.toml"
    if pyproject.exists():
        manifests.append(pyproject)
    pattern = re.compile(r"^\s*\"?(" + "|".join(map(re.escape, GPL_DISTRIBUTIONS)) + r")(?![A-Za-z0-9_-])", re.I | re.M)
    offenders = [f"{path.name}: {m.group(1)}" for path in manifests for m in pattern.finditer(path.read_text())]
    assert not offenders, f"the orchestrator depends on the GPL voice stack: {offenders}"


def _stage(name: str) -> str:
    """The text of stage ``name``: its FROM line up to the next FROM line."""
    match = re.search(rf"^FROM \S+ AS {re.escape(name)}$", DOCKERFILE, re.M)
    assert match, f"no stage named {name}"
    following = re.search(r"^FROM ", DOCKERFILE[match.end() :], re.M)
    return DOCKERFILE[match.start() : match.end() + following.start()] if following else DOCKERFILE[match.start() :]


def _instructions(text: str) -> str:
    """The instructions, not the comments."""
    return "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))


def test_the_gpl_voice_stack_lives_in_media_render():
    for name in ("requirements.in", "requirements.lock"):
        requirements = (MEDIA_RENDER / name).read_text()
        for package in ("kokoro-onnx", "espeakng-loader", "phonemizer"):
            assert re.search(rf"^{re.escape(package)}==", requirements, re.M), (name, package)


def test_hyperframes_is_pinned_exactly_on_node_24():
    # PRD-262 US-053: the exact pin, no caret, in js/package.json and its lock.
    assert JS_PACKAGE["dependencies"]["hyperframes"] == HYPERFRAMES_PIN
    assert JS_LOCK["packages"][""]["dependencies"]["hyperframes"] == HYPERFRAMES_PIN
    assert JS_LOCK["packages"]["node_modules/hyperframes"]["version"] == HYPERFRAMES_PIN
    assert "hyperframes@" not in _instructions(DOCKERFILE), "Hyperframes is installed from the lock only"
    # PRD-262 US-007: Node comes from the official image's named stage, not NodeSource.
    assert re.search(r"^FROM node:24-trixie-slim@sha256:[0-9a-f]{64} AS node$", DOCKERFILE, re.M)
    assert re.search(r"^FROM node AS js$", DOCKERFILE, re.M)
    # PRD-262 #22: the binary from the stripped stage built on that one.
    assert re.search(r"^FROM node AS node-stripped$", DOCKERFILE, re.M)
    assert "COPY --link --from=node-stripped /node-layer/ /" in _stage("runtime")
    # The instructions, not the comments: the comment that says NodeSource is gone names it.
    assert "nodesource" not in _instructions(DOCKERFILE).lower()


def test_the_image_switches_off_telemetry_skills_and_update_checks():
    js = _instructions(_stage("js"))
    runtime = _instructions(_stage("runtime"))
    for stage in (js, runtime):
        env_block = stage[stage.index("ENV HYPERFRAMES_NO_TELEMETRY") :]
        env_block = env_block[: env_block.index("\n", env_block.index("CONTAINER=true"))]
        for name in HYPERFRAMES_SWITCHES:
            assert re.search(rf"\b{name}=1\b", env_block), name
    # In the js stage (which chrome builds FROM) before npm ci and before any
    # hyperframes call, so no install step phones home; in the runtime, which runs
    # the CLI at every render.
    assert js.index("ENV HYPERFRAMES_NO_TELEMETRY") < js.index("npm ci")
    assert "hyperframes" not in js[: js.index("ENV HYPERFRAMES_NO_TELEMETRY")]
    assert re.search(r"^FROM js AS chrome$", DOCKERFILE, re.M)
    assert "hyperframes " not in _instructions(DOCKERFILE[: DOCKERFILE.index(_stage("js"))])


def test_the_kokoro_files_are_hash_checked_at_build():
    sums = (MEDIA_RENDER / "kokoro" / "SHA256SUMS").read_text().split("\n")
    entries = dict(reversed(line.split("  ", 1)) for line in sums if line.strip())
    assert set(entries) == {"kokoro-v1.0.onnx", "voices-v1.0.bin"}
    assert all(re.fullmatch(r"[0-9a-f]{64}", digest) for digest in entries.values())
    # PRD-262 US-053: the kokoro stage takes them from the assets and checks them.
    kokoro = _instructions(_stage("kokoro"))
    copied = re.search(r"^COPY --link --from=assets ((?:/kokoro/\S+ )+)/opt/kokoro/$", kokoro, re.M)
    assert copied, kokoro
    assert sorted(path.rsplit("/", 1)[1] for path in copied.group(1).split()) == sorted(entries)
    assert kokoro.index("COPY --link kokoro/SHA256SUMS /opt/kokoro/SHA256SUMS") < kokoro.index(
        "RUN cd /opt/kokoro && sha256sum -c SHA256SUMS"
    )
    assert "COPY --link --from=kokoro /opt/kokoro /opt/kokoro" in _stage("runtime")


def test_the_espeak_data_is_moved_to_a_short_real_path():
    py = _stage("py")
    # PRD-262 US-051: moved, not copied (the copy stored it twice), in the RUN that
    # installs the lock (every line from the install to the mv is continued), so the
    # package's own copy never reaches a layer.
    assert re.search(
        r"uv pip install[^\n]*\\\n(?:[^\n]*\\\n)*?[ \t]*mv [^\n]*espeakng_loader[^\n]* /opt/espeak; \\\n", py
    )
    assert "cp -r" not in py
    assert "test ! -L /opt/espeak" in py
    # phonemizer resolves symlinks, so a link to the long site-packages path fails.
    assert not re.search(r"ln -s[^\n]*/opt/espeak", DOCKERFILE)
    assert "COPY --link --from=py /opt/espeak /opt/espeak" in _stage("runtime")


def test_chrome_is_preinstalled_at_build_not_fetched_per_render():
    # PRD-262 US-053: in the chrome stage, which builds FROM js (both fetches run on Node).
    chrome = _instructions(_stage("chrome"))
    assert chrome.startswith("FROM js AS chrome")
    assert "HOME=/tmp/hf hyperframes browser ensure" in chrome
    # arm64: the headless shell of the playwright-core the lock pins, never one npx resolves.
    assert "playwright-core install chromium-headless-shell" in chrome and "npx" not in chrome
    assert JS_PACKAGE["devDependencies"]["playwright-core"] == JS_LOCK["packages"]["node_modules/playwright-core"]["version"]
    runtime = _stage("runtime")
    assert "COPY --link --from=chrome /opt/chrome /opt/chrome" in runtime
    assert "HYPERFRAMES_BROWSER_PATH=/opt/chrome/chrome-headless-shell" in runtime
    # Its libraries are the runtime's, so the runtime checks them.
    # PRD-262 #34, #47: with the libraries beside it, which it loads at runtime.
    assert "ldd /opt/chrome/chrome-headless-shell /opt/chrome/*.so* | grep 'not found'" in runtime


def test_the_media_render_job_runs_on_demand_only():
    """7 Oct (Gerard): the media-render lanes live in their own workflow, run from the Actions
    tab when the Socials feature changes, never on a pull request or a push."""
    workflow = yaml.safe_load((ROOT / ".github" / "workflows" / "media-render.yml").read_text())
    assert workflow.get("on", workflow.get(True)) == {"workflow_dispatch": None}
    job = workflow["jobs"].get("media-render")
    assert job is not None, "the media-render job is missing from media-render.yml"
    assert job["name"] == "media-render — image builds and renders the fixture"
    assert "needs" not in job and "if" not in job
    tests = yaml.safe_load((ROOT / ".github" / "workflows" / "test.yml").read_text())
    assert not [name for name in tests["jobs"] if name.startswith("media-render")]
    commands = "\n".join(step.get("run", "") for step in job["steps"])
    assert "docker build -t \"$IMAGE\" services/media-render/" in commands
    assert "assert_output.py" in commands and "--fps 30" in commands and "--audio-codec aac" in commands


def test_the_media_render_job_renders_the_fixture_through_the_api():
    """US-102: the fixture bundle goes through POST /render behind the internal
    token, and the job log carries ebur128's loudness, asserted at -14 +/- 1 LUFS."""
    workflow = yaml.safe_load((ROOT / ".github" / "workflows" / "media-render.yml").read_text())
    commands = "\n".join(step.get("run", "") for step in workflow["jobs"]["media-render"]["steps"])
    assert '"$IMAGE" fixture-bundle' in commands
    assert "X-Internal-Token: $TOKEN" in commands and "http://127.0.0.1:8090/render" in commands
    assert '[ "$code" != 401 ]' in commands, "the job proves the API refuses a request without the token"
    assert "ebur128=peak=true" in commands and "--lufs -14 --lufs-tolerance 1" in commands
    assert "--shm-size=1g" in commands
