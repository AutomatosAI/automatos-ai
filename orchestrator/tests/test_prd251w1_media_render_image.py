"""PRD-251 Wave 1, US-101: the media-render image definition and the GPL boundary.

The media-render CI job builds the image and renders the fixture inside it.
This suite runs in the required orchestrator job and pins what a later edit
could quietly lose:

- the GPL boundary: phonemizer and espeak-ng (GPL-3.0, pulled in by kokoro-onnx)
  are never imported by, or a dependency of, the orchestrator (PRD-251 D3, Traps);
- Hyperframes pinned exactly at 0.8.62 on Node 22, with telemetry, global skill
  installs and update checks switched off in the image;
- the Kokoro files hash-checked at build, and the espeak data copied to a short
  real path (espeak-ng truncates its data path at about 160 characters);
- the CI job itself: a top-level sibling job, with no `needs:`.

Pure file reads: no Docker, no database.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
MEDIA_RENDER = ROOT / "services" / "media-render"
DOCKERFILE = (MEDIA_RENDER / "Dockerfile").read_text()

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


def test_the_gpl_voice_stack_lives_in_media_render():
    requirements = (MEDIA_RENDER / "requirements.txt").read_text()
    for name in ("kokoro-onnx", "espeakng-loader", "phonemizer"):
        assert re.search(rf"^{re.escape(name)}==", requirements, re.M), name


def test_hyperframes_is_pinned_exactly_on_node_22():
    assert re.findall(r"hyperframes@([0-9][^\s\"']*)", DOCKERFILE) == ["0.8.62"]
    assert "deb.nodesource.com/setup_22.x" in DOCKERFILE
    assert "setup_20.x" not in DOCKERFILE


def test_the_image_switches_off_telemetry_skills_and_update_checks():
    env_block = DOCKERFILE[DOCKERFILE.index("ENV HYPERFRAMES_NO_TELEMETRY") :]
    env_block = env_block[: env_block.index("\n\n")]
    for name in HYPERFRAMES_SWITCHES:
        assert re.search(rf"\b{name}=1\b", env_block), name
    # ...and before the CLI is installed, so no install step phones home.
    assert DOCKERFILE.index("ENV HYPERFRAMES_NO_TELEMETRY") < DOCKERFILE.index("npm install -g")


def test_the_kokoro_files_are_hash_checked_at_build():
    sums = (MEDIA_RENDER / "kokoro" / "SHA256SUMS").read_text().split("\n")
    entries = dict(reversed(line.split("  ", 1)) for line in sums if line.strip())
    assert set(entries) == {"kokoro-v1.0.onnx", "voices-v1.0.bin"}
    assert all(re.fullmatch(r"[0-9a-f]{64}", digest) for digest in entries.values())
    assert "sha256sum -c SHA256SUMS" in DOCKERFILE


def test_the_espeak_data_is_copied_to_a_short_real_path():
    assert re.search(r"cp -r .*espeakng_loader.* /opt/espeak;", DOCKERFILE)
    assert "test ! -L /opt/espeak" in DOCKERFILE
    # phonemizer resolves symlinks, so a link to the long site-packages path fails.
    assert not re.search(r"ln -s[^\n]*/opt/espeak", DOCKERFILE)


def test_chrome_is_preinstalled_at_build_not_fetched_per_render():
    assert "hyperframes browser ensure" in DOCKERFILE
    assert "ENV HYPERFRAMES_BROWSER_PATH=/opt/chrome/chrome-headless-shell" in DOCKERFILE


def test_the_media_render_job_is_a_top_level_sibling_with_no_needs():
    workflow = yaml.safe_load((ROOT / ".github" / "workflows" / "test.yml").read_text())
    job = workflow["jobs"].get("media-render")
    assert job is not None, "the media-render job is missing from test.yml"
    assert job["name"] == "media-render — image builds and renders the fixture"
    assert "needs" not in job
    commands = "\n".join(step.get("run", "") for step in job["steps"])
    assert "docker build -t \"$IMAGE\" services/media-render/" in commands
    assert "assert_output.py" in commands and "--fps 30" in commands and "--audio-codec aac" in commands


def test_the_media_render_job_renders_the_fixture_through_the_api():
    """US-102: the fixture bundle goes through POST /render behind the internal
    token, and the job log carries ebur128's loudness, asserted at -14 +/- 1 LUFS."""
    workflow = yaml.safe_load((ROOT / ".github" / "workflows" / "test.yml").read_text())
    commands = "\n".join(step.get("run", "") for step in workflow["jobs"]["media-render"]["steps"])
    assert '"$IMAGE" fixture-bundle' in commands
    assert "X-Internal-Token: $TOKEN" in commands and "http://127.0.0.1:8090/render" in commands
    assert '[ "$code" != 401 ]' in commands, "the job proves the API refuses a request without the token"
    assert "ebur128=peak=true" in commands and "--lufs -14 --lufs-tolerance 1" in commands
    assert "--shm-size=1g" in commands
