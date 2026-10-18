"""PRD-262 US-053: media-render builds in parallel stages from locked dependencies.

media-render.yml builds the image and proves it (the proof run); this suite runs
in the required orchestrator job and pins the definition a later edit could undo:

- the stages: assets, kokoro, js, chrome (FROM js), py and music (FROM py), each
  `COPY --link`ed into the runtime, production last, no BuildKit cache mount
  (Railway refuses one without its service id, PRD-262 §10), and the syntax line
  first because of COPY --link;
- the media assets switch: MEDIA_ASSETS defaults to `fetch`, a stage that lays the
  files out as the media assets image does, and the kokoro and music stages read
  only the `assets` stage, so pointing it at the published image changes nothing
  else (the image itself is published by images.yml's media-assets job);
- the locks: js/package-lock.json (Hyperframes' caret ranges pinned, every package
  with its integrity) and requirements.lock (every pin hashed, the input's pins
  kept), installed with `npm ci` and `uv pip install --require-hashes --no-deps`;
- versions.py reads the js tree, Node's compile cache is seeded as the runtime
  user, and the Chrome launch at build is called a smoke test, which it is;
- Dependabot: pip for services/media-render, npm for services/media-render/js;
- the proof run's steps: the build time against 93 s (report only), the Kokoro
  layer kept across a Hyperframes bump, the background Kokoro load finished before
  the first POST, the first voice step under 2.394 s, check_seconds reported.

Pure file reads: no Docker, no network.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
MEDIA_RENDER = ROOT / "services" / "media-render"
DOCKERFILE = (MEDIA_RENDER / "Dockerfile").read_text()
PACKAGE = json.loads((MEDIA_RENDER / "js" / "package.json").read_text())
LOCK = json.loads((MEDIA_RENDER / "js" / "package-lock.json").read_text())
REQUIREMENTS_IN = (MEDIA_RENDER / "requirements.in").read_text()
REQUIREMENTS_LOCK = (MEDIA_RENDER / "requirements.lock").read_text()
ASSETS = json.loads((MEDIA_RENDER / "kokoro" / "assets.json").read_text())
SHA256SUMS = dict(
    reversed(line.split("  ", 1)) for line in (MEDIA_RENDER / "kokoro" / "SHA256SUMS").read_text().splitlines() if line.strip()
)
DEPENDABOT = yaml.safe_load((ROOT / ".github" / "dependabot.yml").read_text())
WORKFLOW = yaml.safe_load((ROOT / ".github" / "workflows" / "media-render.yml").read_text())

PINNED_PYTHON = re.compile(r"python:3\.12-slim@sha256:[0-9a-f]{64}")
STAGES = ("node", "uv", "assets-fetch", "assets", "kokoro", "js", "chrome", "py", "music", "runtime", "test", "production")
COPIED_IN = {
    "kokoro": "/opt/kokoro",
    "js": "/opt/media-render/js",
    "chrome": "/opt/chrome",
    "py": "/opt/venv",
    "music": "/opt/media-render/music",
}
KOKORO_RELEASES = "https://github.com/thewh1teagle/kokoro-onnx/releases/download"
EXACT = re.compile(r"^\d+\.\d+\.\d+$")
_FROM = re.compile(r"^FROM\s+(\S+)\s+AS\s+(\S+)$", re.M)


def _instructions(text: str) -> str:
    """The instructions, continuation lines joined; not the comments."""
    code = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    return re.sub(r"[ \t]*\\[ \t]*\n\s*", " ", code)


def _stage(name: str) -> str:
    """Stage ``name``'s instructions, from its FROM line to the next."""
    match = re.search(rf"^FROM \S+ AS {re.escape(name)}$", DOCKERFILE, re.M)
    assert match, f"no stage named {name}"
    following = re.search(r"^FROM ", DOCKERFILE[match.end() :], re.M)
    end = match.end() + following.start() if following else len(DOCKERFILE)
    return _instructions(DOCKERFILE[match.start() : end])


def _bases() -> dict[str, str]:
    return {name: image for image, name in _FROM.findall(DOCKERFILE)}


def _steps(job: str) -> list[dict]:
    return WORKFLOW["jobs"][job]["steps"]


def _step(job: str, needle: str) -> str:
    (run,) = [step.get("run", "") for step in _steps(job) if needle in step.get("run", "")]
    return run


# ── the stages ──────────────────────────────────────────────────────────────


def test_the_stages_are_named_in_order_and_production_is_last():
    assert [name for _, name in _FROM.findall(DOCKERFILE)] == list(STAGES)
    bases = _bases()
    assert bases["js"] == "node" and bases["chrome"] == "js" and bases["music"] == "py"
    assert bases["test"] == "runtime" and bases["production"] == "runtime"
    for name in ("assets-fetch", "kokoro", "py", "runtime"):
        assert PINNED_PYTHON.fullmatch(bases[name]), (name, bases[name])


def test_the_runtime_copy_links_each_stage():
    runtime = _stage("runtime")
    for stage, path in COPIED_IN.items():
        assert re.search(rf"^COPY --link --from={stage} {re.escape(path)} {re.escape(path)}$", runtime, re.M), stage
    # Each part is built once, in its stage: no install or download in the runtime,
    # which carries no curl or unzip at all (PRD-262 US-051).
    for absent in ("npm ", "pip install", "uv pip", "curl", "unzip", "hyperframes browser", "music_build.py"):
        assert absent not in runtime, absent


def test_no_buildkit_cache_mount_and_the_syntax_line_comes_first():
    # Railway fails a cache mount without a hard-coded service id (PRD-262 §10).
    assert "--mount=type=cache" not in DOCKERFILE
    # COPY --link needs the Dockerfile 1.4 syntax; a parser directive is only read on line 1.
    assert DOCKERFILE.splitlines()[0] == "# syntax=docker/dockerfile:1"


# ── the media assets switch ─────────────────────────────────────────────────


def test_the_assets_stage_is_picked_by_media_assets_whose_default_is_fetch():
    preamble = _instructions(DOCKERFILE[: DOCKERFILE.index("\nFROM ")])
    assert re.findall(r"^ARG (\w+)=(\S+)$", preamble, re.M) == [("MEDIA_ASSETS", "fetch")]
    assert _bases()["assets"] == "assets-${MEDIA_ASSETS}"
    assert PINNED_PYTHON.fullmatch(_bases()["assets-fetch"])
    # The kokoro and music stages read the assets stage alone, so switching it to the
    # published image changes nothing else.
    sources = re.findall(r"--from=(\S+)", "\n".join(_stage(name) for name in ("kokoro", "music")))
    assert sources and set(sources) == {"assets"}
    assert "--from=assets-" not in DOCKERFILE


def _fetched_kokoro_files() -> dict[str, str]:
    """``<release>/<file>`` -> the URL path the assets-fetch stage downloads it from."""
    fetch = _stage("assets-fetch")
    assert f"base={KOKORO_RELEASES};" in fetch
    pairs = re.findall(r'curl -fsSL --retry 3 -o /kokoro/(\S+) "\$base/(\S+)"', fetch)
    assert pairs, fetch
    return dict(pairs)


def test_the_fetch_stage_lays_the_kokoro_files_out_as_the_media_assets_image_does():
    fetched = _fetched_kokoro_files()
    assert all(path == url for path, url in fetched.items()), "each file sits under its release, as in the image"
    copied = re.search(r"^COPY --link --from=assets ((?:/kokoro/\S+ )+)/opt/kokoro/$", _stage("kokoro"), re.M)
    assert copied
    assert sorted(path.removeprefix("/kokoro/") for path in copied.group(1).split()) == sorted(fetched)
    in_image = {entry["path"]: entry for entry in ASSETS["files"]}
    for path in fetched:
        # The media assets image carries it at the same path, from the same URL, with
        # the sha256 the kokoro stage checks.
        assert in_image[path]["url"] == f"{KOKORO_RELEASES}/{path}"
        assert in_image[path]["sha256"] == SHA256SUMS[path.rsplit("/", 1)[1]]
    assert sorted(path.rsplit("/", 1)[1] for path in fetched) == sorted(SHA256SUMS)


def test_the_fetch_stage_fetches_and_checks_every_track_into_music():
    fetch = _stage("assets-fetch")
    assert "COPY music/manifest.json media_render/music_build.py /build/music/" in fetch
    assert "RUN python /build/music/music_build.py /build/music/manifest.json /music --fetch-only" in fetch
    music = _stage("music")
    assert "COPY --link --from=assets /music /opt/media-render/assets/music" in music
    assert "--tracks /opt/media-render/assets/music" in music


# ── the js lock ─────────────────────────────────────────────────────────────


def test_package_json_pins_every_package_exactly():
    assert set(PACKAGE["dependencies"]) == {"hyperframes", "gsap"}
    assert set(PACKAGE["devDependencies"]) == {"playwright-core"}
    for section in ("dependencies", "devDependencies"):
        for name, version in PACKAGE[section].items():
            assert EXACT.match(version), f"{name} {version}: a range, not an exact pin"


def test_the_lock_pins_the_whole_tree_with_integrity():
    assert LOCK["lockfileVersion"] == 3
    root = LOCK["packages"][""]
    assert root["dependencies"] == PACKAGE["dependencies"]
    assert root["devDependencies"] == PACKAGE["devDependencies"]
    for name in (*PACKAGE["dependencies"], *PACKAGE["devDependencies"]):
        assert LOCK["packages"][f"node_modules/{name}"]["version"] == (PACKAGE["dependencies"] | PACKAGE["devDependencies"])[name]
    assert LOCK["packages"]["node_modules/playwright-core"].get("dev") is True
    entries = {path: entry for path, entry in LOCK["packages"].items() if path}
    assert len(entries) > 50, "Hyperframes' own dependencies are in the lock"
    for path, entry in entries.items():
        assert entry["resolved"].startswith("https://registry.npmjs.org/"), path
        assert entry["integrity"].startswith("sha512-"), path


@pytest.mark.parametrize("package", ["@img/sharp-linux-x64", "@img/sharp-linux-arm64", "@esbuild/linux-x64", "@esbuild/linux-arm64"])
def test_the_lock_carries_the_native_packages_of_both_architectures(package):
    # npm ci on either builder finds its platform's optional package in the lock.
    assert f"node_modules/{package}" in LOCK["packages"]


def test_the_lock_records_each_sharp_builds_libc_so_npm_ci_skips_the_musl_ones():
    # npm ci checks a package's libc only when the lock records it. npm 10 never
    # does; the image's npm 11.19 does, and without it npm ci on glibc also installs
    # sharp's musl builds, about 19 MB per arch.
    linux = {path: entry for path, entry in LOCK["packages"].items() if re.search(r"/@img/sharp-(?:libvips-)?linux", path)}
    assert any("linuxmusl" in path for path in linux) and any("linux-x64" in path for path in linux)
    for path, entry in linux.items():
        expected = ["musl"] if "linuxmusl" in path else ["glibc"]
        assert entry.get("libc") == expected, f"{path}: refresh the lock with npm 11.19 or later, the image's npm"


def test_the_js_stage_installs_exactly_the_lock():
    js = _stage("js")
    assert "COPY js/package.json js/package-lock.json ./" in js
    assert "RUN npm ci --omit=dev --no-fund --no-audit" in js
    assert "npm install" not in _instructions(DOCKERFILE)
    assert "cp node_modules/gsap/dist/gsap.min.js /opt/media-render/vendor/gsap.min.js" in js
    assert "COPY --link --from=js /opt/media-render/vendor /opt/media-render/vendor" in _stage("runtime")


# ── the Python lock ─────────────────────────────────────────────────────────


def _lock_pins() -> dict[str, str]:
    blocks = re.split(r"\n(?=[A-Za-z0-9])", REQUIREMENTS_LOCK)
    pins = {}
    for block in blocks:
        match = re.match(r"^([A-Za-z0-9._-]+)==(\S+)", block)
        if match:
            assert "--hash=sha256:" in block, f"{match.group(1)} has no hash"
            pins[match.group(1).lower()] = match.group(2)
    return pins


def test_the_python_lock_hashes_every_pin_and_keeps_the_inputs_pins():
    pins = _lock_pins()
    assert len(pins) > len(re.findall(r"^\S+==", REQUIREMENTS_IN, re.M)), "the transitive pins are in the lock"
    direct = dict(re.findall(r"^([A-Za-z0-9._-]+)==(\S+)$", REQUIREMENTS_IN, re.M))
    assert direct, "requirements.in pins nothing"
    # A Dependabot bump of requirements.in fails here until the lock is compiled again.
    assert {name: pins.get(name.lower()) for name in direct} == direct
    assert "uv pip compile services/media-render/requirements.in" in REQUIREMENTS_IN
    assert "--generate-hashes" in REQUIREMENTS_LOCK.splitlines()[1]
    assert not (MEDIA_RENDER / "requirements.txt").exists(), "requirements.in replaced it"


def test_the_py_stage_installs_the_lock_with_hashes_and_nothing_else():
    py = _stage("py")
    assert "COPY --from=uv /uv /usr/local/bin/uv" in py
    assert re.search(r"^FROM ghcr\.io/astral-sh/uv:[\d.]+@sha256:[0-9a-f]{64} AS uv$", DOCKERFILE, re.M)
    assert re.search(
        r"uv pip install --python /opt/venv/bin/python --no-cache --compile-bytecode --require-hashes --no-deps"
        r" -r /opt/media-render/requirements\.lock",
        py,
    )
    assert "pip install --no-cache-dir" not in DOCKERFILE
    assert "/opt/venv/bin" in re.search(r"^ENV PATH=(\S+)", _stage("runtime"), re.M).group(1)


# ── the runtime ─────────────────────────────────────────────────────────────


def test_versions_reads_the_js_tree_and_hyperframes_is_on_path():
    versions = (MEDIA_RENDER / "media_render" / "versions.py").read_text()
    assert "npm" not in versions.replace("npm ci", "")
    runtime = _stage("runtime")
    assert "--node-modules /opt/media-render/js/node_modules" in runtime
    path = re.search(r"^ENV PATH=(\S+)", runtime, re.M).group(1).split(":")
    assert "/opt/media-render/js/node_modules/.bin" in path


def test_nodes_compile_cache_is_seeded_as_the_runtime_user():
    runtime = _stage("runtime")
    created = runtime.index("install -d -o render -g render /opt/media-render/node-cache")
    user = runtime.index("USER render")
    env = runtime.index("NODE_COMPILE_CACHE=/opt/media-render/node-cache")
    seed = runtime.index("RUN hyperframes --version")
    # Set only after the user switch, so nothing that ran as root wrote into it.
    assert created < user < env < seed
    assert "NODE_COMPILE_CACHE" not in runtime[:user]


def test_the_chrome_launch_at_build_is_called_a_smoke_test():
    launch = DOCKERFILE.index("RUN timeout 120 /opt/chrome/chrome-headless-shell")
    comment = DOCKERFILE[DOCKERFILE.rindex("\n\n", 0, launch) : launch]
    assert "Smoke test" in comment and "warm" not in comment
    assert "pre-warm" not in DOCKERFILE.lower()


def test_the_test_stage_installs_pytest_into_the_same_venv():
    test = _stage("test")
    assert "uv pip install --python /opt/venv/bin/python --no-cache -r /app/requirements-test.txt" in test


# ── Dependabot ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(("ecosystem", "directory"), [("pip", "/services/media-render"), ("npm", "/services/media-render/js")])
def test_dependabot_updates_the_media_render_locks_weekly_and_grouped(ecosystem, directory):
    entries = [
        entry for entry in DEPENDABOT["updates"]
        if entry["package-ecosystem"] == ecosystem and entry.get("directory") == directory
    ]
    assert len(entries) == 1, (ecosystem, directory)
    (entry,) = entries
    assert entry["schedule"]["interval"] == "weekly"
    assert any("*" in group["patterns"] for group in entry["groups"].values())


# ── the proof run ───────────────────────────────────────────────────────────


def test_the_build_time_is_reported_against_93_seconds_and_never_fails():
    build = _step("media-render", 'docker build -t "$IMAGE" services/media-render/')
    assert 'echo "image build: $(( $(date +%s) - start )) s (93 s in run 37612086159' in build
    assert "report only" in build and "exit 1" not in build


def test_a_hyperframes_bump_keeps_the_kokoro_layer():
    step = _step("media-render", "image_layer_digest.py")
    assert 'cp -r services/media-render "$pin"' in step
    bump = re.search(r"npm install --package-lock-only --ignore-scripts --no-fund --no-audit --save-exact hyperframes@(\S+)\)", step)
    assert bump and bump.group(1) != PACKAGE["dependencies"]["hyperframes"]
    assert 'docker build -t media-render:pin "$pin"' in step
    assert (
        'python3 scripts/ci/image_layer_digest.py "$IMAGE" media-render:pin --same /opt/kokoro --differ /opt/media-render/js'
        in step
    )
    names = [step.get("name", "") for step in _steps("media-render")]
    builds = [i for i, step in enumerate(_steps("media-render")) if 'docker build -t "$IMAGE"' in step.get("run", "")]
    assert builds and builds[0] < names.index("Two builds that differ only in the Hyperframes pin share the Kokoro layer")


def test_the_first_post_waits_for_the_background_kokoro_load_and_the_voice_step_is_timed():
    api = _step("media-render", "media-render-api")
    waited = api.index("kokoro: the voice model loaded in the background")
    assert waited < api.index("-X POST http://127.0.0.1:8090/render") < api.index("--data-binary @")
    assert "the background Kokoro load did not finish before the first POST" in api
    asserted = _step("media-render", "--max-voice-seconds")
    assert "--duration 3.0" in asserted and "--max-voice-seconds 2.394" in asserted
    assert 'print("check_seconds:", t.get("check_seconds"), "s (5.009 s in run 37612086159; report only)")' in asserted
    log = (MEDIA_RENDER / "media_render" / "server.py").read_text()
    assert '"kokoro: the voice model loaded in the background in %.2f s"' in log


def test_the_compile_cache_is_checked_in_the_image():
    step = _step("media-render", "NODE_COMPILE_CACHE")
    assert 'test "$(id -u)" = 1000' in step and 'test "$files" -gt 0' in step
    assert '! -readable' in step and 'test "$unreadable" -eq 0' in step
