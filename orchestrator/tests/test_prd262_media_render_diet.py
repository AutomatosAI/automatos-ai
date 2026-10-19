"""PRD-262 US-051: the renderer drops what renders never load.

media-render.yml builds the image and proves it (the proof run: the in-image
suite, services/media-render/tests/test_image.py, and the 1.30 GB bar). This
suite runs in the required orchestrator job and pins the definition a later edit
could undo:

- the runtime's apt RUN ends by purging the packages only mesa-libgallium needs
  (libllvm19, libz3-4 and the rest of that closure on trixie), keeps libgbm1 and
  Debian's ffmpeg, and fails the build if ffmpeg or ffprobe lost a library;
- the chrome stage installs binutils first, strips the arm64 headless shell, fails
  the build on a .symtab, and keeps only en-US.pak among the locales;
- the py stage moves the espeak data (no second copy) and deletes onnxruntime's
  unlinked library and numpy's tests in the RUN that installs the lock, then
  imports onnxruntime;
- no curl or unzip in the runtime: the HEALTHCHECK (with --start-interval=2s) and
  the compose probe are Python's urllib;
- the proof run: the in-image checks, Docker's health status, and the 1.30 GB bar,
  checked last.

Pure file reads: no Docker, no network.
"""
from __future__ import annotations

import re

import yaml

from tests.test_prd262_media_render_stages import DOCKERFILE, MEDIA_RENDER, ROOT, _stage, _step, _steps

# The packages only mesa-libgallium needs, from the Depends/Pre-Depends closure of
# the runtime's apt line over trixie main, updates and security, on top of
# python:3.12-slim's own packages (18 Oct 2026). libdrm-intel1 and libpciaccess0
# are amd64's alone.
MESA_ONLY = {
    "mesa-libgallium", "libllvm19", "libz3-4", "libelf1t64", "libedit2", "libdrm-amdgpu1",
    "libsensors5", "libsensors-config", "libxcb-randr0", "libxcb-sync1",
}
MESA_ONLY_AMD64 = {"libdrm-intel1", "libpciaccess0"}
SIZE_STEP = "The image is at most 1.30 GB on disk (amd64)"


def _run_with(stage: str, needle: str) -> str:
    """The one RUN instruction of ``stage`` that contains ``needle``, continuations joined."""
    (run,) = [line for line in _stage(stage).splitlines() if line.startswith("RUN ") and needle in line]
    return run


# ── the runtime: Mesa's LLVM out, curl and unzip out ────────────────────────


def test_the_apt_run_ends_by_purging_what_only_mesa_libgallium_needs():
    run = _run_with("runtime", "apt-get install")
    installed = re.search(r"apt-get install -y --no-install-recommends ([^;]+);", run).group(1).split()
    # Chrome links libgbm.so.1, and Debian's ffmpeg keeps its security updates.
    assert {"libgbm1", "ffmpeg"} <= set(installed)
    listed = set(re.search(r'mesa_only="([a-z0-9. -]+)"', run).group(1).split())
    amd64 = re.search(r'if \[ "\$\(dpkg --print-architecture\)" = amd64 \]; then mesa_only="\$mesa_only ([a-z0-9. -]+)"', run)
    assert listed == MESA_ONLY and amd64 and set(amd64.group(1).split()) == MESA_ONLY_AMD64
    assert {"mesa-libgallium", "libllvm19", "libz3-4"} <= listed
    assert not (set(installed) & (MESA_ONLY | MESA_ONLY_AMD64)), "a package the apt line asks for is purged"
    # In the same RUN (a later layer would save nothing), before the font cache.
    purge = run.index("dpkg --purge --force-depends $mesa_only;")
    assert run.index("apt-get install") < purge < run.index("fc-cache -f")
    # The purge proves itself: ffmpeg and ffprobe still find every library.
    check = run.index("if ldd /usr/bin/ffmpeg /usr/bin/ffprobe | grep 'not found'; then")
    assert purge < check and "exit 1" in run[check:]


def test_nothing_after_the_purge_runs_apt():
    # dpkg now records libgbm1's dependency on mesa-libgallium as unmet, and apt
    # refuses to work on such a system.
    runtime = _stage("runtime")
    assert [line for line in runtime.splitlines() if "apt-get" in line] == [_run_with("runtime", "apt-get install")]
    for stage in ("test", "production"):
        assert "apt-get" not in _stage(stage), stage


def test_the_runtime_carries_no_curl_or_unzip():
    runtime = _stage("runtime")
    assert "curl" not in runtime and "unzip" not in runtime
    # unzip is the amd64 chrome stage's alone (@puppeteer/browsers calls it).
    assert "unzip" in _stage("chrome")


def test_the_healthcheck_is_pythons_urllib_probing_every_2_s_while_it_starts():
    (check,) = [line for line in _stage("runtime").splitlines() if line.startswith("HEALTHCHECK ")]
    assert "--start-interval=2s" in check and "--interval=30s" in check
    assert 'CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen(' in check
    assert "'http://127.0.0.1:%s/health' % os.environ['MEDIA_RENDER_PORT']" in check
    # --start-interval needs the Dockerfile syntax line, which is read on line 1 only.
    assert DOCKERFILE.splitlines()[0] == "# syntax=docker/dockerfile:1"


def test_the_compose_probe_is_pythons_urllib():
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    probe = compose["services"]["media-render"]["healthcheck"]["test"]
    assert probe[:3] == ["CMD", "python", "-c"]
    assert "urllib.request.urlopen('http://127.0.0.1:8090/health', timeout=4)" in probe[3]
    assert "curl" not in " ".join(probe)


# ── chrome: the arm64 shell stripped, one locale ────────────────────────────


def test_the_chrome_stage_installs_binutils_first_and_strips_the_arm64_shell():
    run = _run_with("chrome", "playwright-core install")
    binutils = run.index("apt-get install -y --no-install-recommends binutils;")
    arm64 = run.index('if [ "${TARGETARCH:-amd64}" = "arm64" ]; then')
    amd64 = run.index("else", arm64)
    assert binutils < arm64
    assert 'strip --strip-unneeded "$BIN";' in run[arm64:amd64]
    # On either arch, after the branches: a .symtab left in the shell or a library
    # beside it (PRD-262 #38) fails the build.
    assert 'fi; test -x "$BIN"; for elf in "$BIN" $(find "$(dirname "$BIN")" -maxdepth 1 -type f' in run
    assert 'do if readelf -SW "$elf" | grep -F .symtab; then' in run
    check = run.index("readelf -SW")
    assert "exit 1" in run[check : run.index("mv ", check)]
    # strip and readelf stay in the builder stages; the runtime only names the
    # node-stripped stage it copies node from (PRD-262 #22).
    runtime = _stage("runtime").replace("--from=node-stripped ", "--from=<node> ")
    assert "binutils" not in runtime and "strip" not in runtime and "readelf" not in runtime


def test_the_chrome_stage_keeps_only_en_us_pak():
    run = _run_with("chrome", "playwright-core install")
    trim = run.index("find /opt/chrome/locales -mindepth 1 ! -name en-US.pak -delete;")
    assert run.index('mv "$(dirname "$BIN")" /opt/chrome;') < trim
    assert 'test "$(ls /opt/chrome/locales)" = en-US.pak;' in run[trim:]
    # The arm64 headless shell has no locales directory.
    assert "if [ -d /opt/chrome/locales ]; then" in run


# ── py: one espeak copy, no unlinked onnxruntime, no numpy tests ────────────


def test_the_py_stage_trims_the_venv_in_the_run_that_installs_the_lock():
    run = _run_with("py", "uv pip install")
    install = run.index("uv pip install")
    move = run.index(
        "mv \"$(/opt/venv/bin/python -c 'import espeakng_loader; print(espeakng_loader.get_data_path())')\" /opt/espeak;"
    )
    onnxruntime = run.index('rm "$site"/onnxruntime/capi/libonnxruntime.so.*;')
    numpy = run.index('find "$site/numpy" -type d -name tests -prune -exec rm -rf {} +;')
    probe = run.index("/opt/venv/bin/python -c 'import onnxruntime as o; print(o.get_available_providers())'")
    # PRD-262 #18: the deletions come first, so the strip after them skips what goes.
    assert install < run.index("site=/opt/venv/lib/python3.12/site-packages;") < onnxruntime < move < probe
    assert install < numpy < move
    assert "cp -r" not in _stage("py")
    # libonnxruntime_providers_shared.so stays: the pattern is the unlinked library's.
    assert "libonnxruntime_providers" not in run


def test_kokoro_takes_the_espeak_path_from_config_not_from_the_package():
    # The package's own copy is gone, so nothing may ask espeakng_loader for it.
    kokoro_tts = (MEDIA_RENDER / "media_render" / "kokoro_tts.py").read_text()
    assert "data_path=espeak_data_path" in kokoro_tts
    assert "get_data_path" not in kokoro_tts
    config = (MEDIA_RENDER / "media_render" / "config.py").read_text()
    assert '"espeak_data_path": env.text("MEDIA_RENDER_ESPEAK_DATA_PATH", "/opt/espeak")' in config


# ── the proof run ───────────────────────────────────────────────────────────


def test_the_in_image_suite_carries_the_image_checks():
    suite = (MEDIA_RENDER / "tests" / "test_image.py").read_text()
    for needle in (
        'LINKED = ("/usr/bin/ffmpeg", "/usr/bin/ffprobe", "/opt/chrome/chrome-headless-shell")',
        'subprocess.run(["ldd", binary]',
        'MESA_ONLY = ("mesa-libgallium", "libllvm19")',
        'subprocess.run(["dpkg", "-s", package]',
        'assert kept == ["en-US.pak"]',
        'probe = "import onnxruntime as o; print(o.get_available_providers())"',
        'for tool in ("curl", "unzip"):',
    ):
        assert needle in suite, needle
    # The media-render job runs that suite inside the image.
    assert '--entrypoint python "$TEST_IMAGE" -m pytest' in _step("media-render", "--target test")


def test_the_proof_run_fails_above_1_30_gb_on_disk():
    # The bar is media-render's amd64 on_disk_max_mb, the entry images.yml checks too
    # (PRD-262 US-054; tests/test_prd262_media_render_images.py holds it at 1,300 MB).
    reads = "budget=$(jq -er '.budgets[\"media-render\"].amd64.on_disk_max_mb * 1000000 | floor' .github/image-budgets.json)"
    step = _step("media-render", reads)
    assert 'docker image ls "$IMAGE"' in step
    assert "size=$(docker image inspect -f '{{.Size}}' \"$IMAGE\")" in step
    assert 'if [ "$arch" != amd64 ]' in step
    gate = step.index('if [ "$size" -gt "$budget" ]; then')
    assert "exit 1" in step[gate:]
    # Last and always: an image over the bar still gets every render proved, and a
    # failed render still gets its size reported.
    last = _steps("media-render")[-1]
    assert last["name"] == SIZE_STEP and last["if"] == "always()"


def test_the_proof_run_waits_for_dockers_health_status():
    step = _step("media-render", "media-render-health")
    assert "docker inspect -f '{{.State.Health.Status}}' media-render-health" in step
    assert re.search(r"for i in \$\(seq 1 20\); do\n\s+health=", step)
    assert 'if [ "$health" != healthy ]; then' in step
    assert step.index("docker rm -f media-render-health") < step.index('if [ "$health" != healthy ]; then')
