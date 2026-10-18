"""Facts about the BUILT image (US-101). These read the real container: the CI
job runs this suite inside the media-render image, never on a bare runner.
"""

from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from media_render import boot
from media_render.config import ESPEAK_DATA_PATH_LIMIT, HYPERFRAMES_OFF_SWITCHES, load_settings
from media_render.versions import read_versions

HYPERFRAMES_PIN = "0.8.62"
# PRD-262 US-053: the js stage's locked tree, and the py stage's venv.
JS_TREE = Path("/opt/media-render/js/node_modules")
VENV = "/opt/venv"
# PRD-262 US-051: what renders never load. One package per check: dpkg -s fails on
# the first package it does not know, so a pair would hide the second.
MESA_ONLY = ("mesa-libgallium", "libllvm19")
LINKED = ("/usr/bin/ffmpeg", "/usr/bin/ffprobe", "/opt/chrome/chrome-headless-shell")
CHROME_LOCALES = Path("/opt/chrome/locales")


def test_the_hyperframes_switches_are_set_in_the_image():
    for name in HYPERFRAMES_OFF_SWITCHES:
        assert os.environ.get(name) == "1", name
    # Defence in depth: the CLI's background self-update would move the pin.
    assert os.environ.get("HYPERFRAMES_NO_AUTO_INSTALL") == "1"


def test_the_image_passes_its_own_boot_assertions():
    assert boot.boot_problems(load_settings(), os.environ) == []


def test_the_espeak_data_sits_at_a_short_real_path():
    path = load_settings().espeak_data_path
    assert not os.path.islink(path)
    assert os.path.realpath(path) == path
    assert len(path) < ESPEAK_DATA_PATH_LIMIT
    assert Path(path, "phontab").is_file()


def test_the_espeak_data_was_moved_not_copied():
    # PRD-262 US-051: one copy, at the short path; none left in the package.
    package = Path(importlib.util.find_spec("espeakng_loader").origin).parent
    assert not (package / "espeak-ng-data").exists()


def test_hyperframes_is_pinned_exactly_and_runs_on_node_24():
    versions = read_versions(load_settings().versions_path)
    assert versions["hyperframes"] == HYPERFRAMES_PIN
    assert versions["node"].startswith("v24.")
    live = subprocess.run(["hyperframes", "--version"], capture_output=True, text=True, timeout=60)
    assert live.returncode == 0 and HYPERFRAMES_PIN in live.stdout


def test_hyperframes_runs_from_the_js_stages_locked_tree_and_the_runtime_has_no_npm():
    assert Path(shutil.which("hyperframes") or "").parent == JS_TREE / ".bin"
    assert (JS_TREE / "hyperframes" / "package.json").is_file() and (JS_TREE / "gsap" / "package.json").is_file()
    # playwright-core is a dev dependency: only the arm64 chrome stage installs it.
    assert not (JS_TREE / "playwright-core").exists()
    for tool in ("npm", "npx", "corepack"):
        assert shutil.which(tool) is None, tool


def test_the_python_packages_come_from_the_locked_venv():
    assert sys.prefix == VENV
    assert Path(shutil.which("python") or "").parent == Path(VENV, "bin")


def test_nodes_compile_cache_is_seeded_and_readable_by_the_runtime_user():
    cache = os.environ.get("NODE_COMPILE_CACHE", "")
    assert cache, "NODE_COMPILE_CACHE is not set"
    files = [path for path in Path(cache).rglob("*") if path.is_file()]
    assert files, f"{cache} is empty: the build did not seed it"
    assert os.getuid() == 1000
    assert [str(path) for path in Path(cache).rglob("*") if not os.access(path, os.R_OK)] == []


def test_chrome_headless_shell_is_preinstalled_and_is_the_one_hyperframes_uses():
    settings = load_settings()
    assert os.environ.get("HYPERFRAMES_BROWSER_PATH") == settings.browser_path
    assert os.access(settings.browser_path, os.X_OK)
    assert Path(settings.browser_path).name == "chrome-headless-shell"


def test_ffmpeg_ffprobe_and_tini_are_installed():
    for tool in ("ffmpeg", "ffprobe", "tini"):
        assert shutil.which(tool), tool


def test_gsap_and_the_kokoro_files_are_in_the_image():
    settings = load_settings()
    assert Path(settings.gsap_path).stat().st_size > 10_000
    for path in (settings.kokoro_model_path, settings.kokoro_voices_path):
        assert Path(path).stat().st_size > 1_000_000, path


def test_the_service_runs_as_an_unprivileged_user():
    assert os.getuid() != 0


# ── PRD-262 US-051: what renders never load ─────────────────────────────────


@pytest.mark.parametrize("binary", LINKED)
def test_ffmpeg_ffprobe_and_the_headless_shell_find_every_library(binary):
    ldd = subprocess.run(["ldd", binary], capture_output=True, text=True, timeout=60)
    assert ldd.returncode == 0, ldd.stderr
    assert "not found" not in ldd.stdout, ldd.stdout


@pytest.mark.parametrize("package", MESA_ONLY)
def test_mesa_and_llvm_are_purged(package):
    status = subprocess.run(["dpkg", "-s", package], capture_output=True, text=True, timeout=60)
    assert status.returncode != 0, f"{package} is installed:\n{status.stdout}"


def test_libgbm_stays_and_no_llvm_library_is_left():
    # Chrome links libgbm.so.1, which needs only libdrm, libexpat, libm and libc.
    assert list(Path("/usr/lib").glob("*/libgbm.so.1"))
    assert not list(Path("/usr/lib").glob("*/libLLVM*"))


def test_chrome_keeps_only_the_en_us_locale():
    kept = sorted(os.listdir(CHROME_LOCALES)) if CHROME_LOCALES.is_dir() else []
    if platform.machine() == "x86_64":
        assert kept == ["en-US.pak"]
    # The arm64 headless shell (Playwright's) ships none; whatever it ships is trimmed.
    assert kept in ([], ["en-US.pak"]), kept


def test_onnxruntime_loads_without_its_unlinked_library():
    probe = "import onnxruntime as o; print(o.get_available_providers())"
    run = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stderr
    assert "CPUExecutionProvider" in run.stdout
    capi = Path(importlib.util.find_spec("onnxruntime").origin).parent / "capi"
    assert not list(capi.glob("libonnxruntime.so.*"))
    assert list(capi.glob("onnxruntime_pybind11_state*.so"))


def test_numpy_ships_no_tests():
    numpy = Path(importlib.util.find_spec("numpy").origin).parent
    assert [str(path) for path in numpy.rglob("tests") if path.is_dir()] == []


def test_the_runtime_has_no_curl_or_unzip():
    # Downloads happen in the builder stages; the HEALTHCHECK is Python's urllib.
    for tool in ("curl", "unzip"):
        assert shutil.which(tool) is None, tool
