"""Facts about the built image: speech only, as the runtime user, the model checked at build."""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def test_it_runs_as_an_unprivileged_user():
    assert os.getuid() == 1000


def test_it_carries_speech_only_no_node_browser_or_video_tools():
    for tool in ("node", "npm", "ffmpeg", "ffprobe", "chrome-headless-shell", "curl", "exiftool"):
        assert shutil.which(tool) is None, f"the voice image carries {tool}"
    assert shutil.which("lame") and shutil.which("tini")


def test_the_model_and_voices_are_in_place_and_the_checksums_stayed_in_the_build():
    kokoro = Path("/opt/kokoro")
    assert sorted(path.name for path in kokoro.iterdir()) == ["kokoro-v1.0.onnx", "voices-v1.0.bin"]
    assert (kokoro / "kokoro-v1.0.onnx").stat().st_size == 325532387
    assert (kokoro / "voices-v1.0.bin").stat().st_size == 28214398


def test_the_espeak_data_is_a_short_real_path_with_every_language():
    espeak = Path("/opt/espeak")
    assert not espeak.is_symlink() and (espeak / "phontab").is_file()
    for dictionary in ("en_dict", "es_dict", "fr_dict", "hi_dict", "it_dict", "pt_dict", "cmn_dict"):
        assert (espeak / dictionary).is_file(), dictionary


def test_the_venv_lost_what_the_service_never_loads():
    site = Path("/opt/venv/lib/python3.12/site-packages")
    for gone in ("onnxruntime/transformers", "onnxruntime/quantization", "onnxruntime/tools", "numpy/f2py"):
        assert not (site / gone).exists(), gone
    assert not list((site / "onnxruntime" / "capi").glob("libonnxruntime.so.*"))
    assert not (site / "c2pa").exists(), "the voice image signs nothing: no C2PA SDK"
