"""Facts about the BUILT image (US-101). These read the real container: the CI
job runs this suite inside the media-render image, never on a bare runner.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from media_render import boot
from media_render.config import ESPEAK_DATA_PATH_LIMIT, HYPERFRAMES_OFF_SWITCHES, load_settings
from media_render.versions import read_versions

HYPERFRAMES_PIN = "0.8.62"


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


def test_hyperframes_is_pinned_exactly_and_runs_on_node_22():
    versions = read_versions(load_settings().versions_path)
    assert versions["hyperframes"] == HYPERFRAMES_PIN
    assert versions["node"].startswith("v22.")
    live = subprocess.run(["hyperframes", "--version"], capture_output=True, text=True, timeout=60)
    assert live.returncode == 0 and HYPERFRAMES_PIN in live.stdout


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
