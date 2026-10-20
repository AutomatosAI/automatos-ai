"""PRD-262 #18 (19 Oct breakdown): media-render's venv holds what the service runs, stripped.

About 25.6 MB: onnxruntime's transformers, quantization and tools (never imported:
the service loads onnxruntime.capi alone), protobuf and flatbuffers (only those
import them, so they stay out of the lock), numpy's f2py, espeakng_loader's two
extra copies of libespeak-ng.so (links now), and every shared library's symbol
table. This suite pins the lock, the py stage's RUN (the trim before the espeak
build, which then speaks through the stripped library, and the probes that load
every native part before the layer is kept) and the in-image suite
(services/media-render/tests/test_image_venv.py).

Pure file reads: no Docker, no network.
"""
from __future__ import annotations

import re

from tests.test_prd262_media_render_diet import _run_with
from tests.test_prd262_media_render_stages import MEDIA_RENDER, _stage

REQUIREMENTS_IN = (MEDIA_RENDER / "requirements.in").read_text()
REQUIREMENTS_LOCK = (MEDIA_RENDER / "requirements.lock").read_text()
NO_EMIT = "--no-emit-package protobuf --no-emit-package flatbuffers"
ONNXRUNTIME_UNUSED = ("transformers", "quantization", "tools")


def test_protobuf_and_flatbuffers_stay_out_of_the_lock():
    # The lock's header, and the command requirements.in says to regenerate it with.
    assert NO_EMIT in REQUIREMENTS_LOCK.splitlines()[1]
    assert NO_EMIT in REQUIREMENTS_IN
    for name in ("protobuf", "flatbuffers"):
        assert not re.search(rf"^{name}==", REQUIREMENTS_LOCK, re.M), f"{name} is back in the lock: compile it again"
    assert re.search(r"^onnxruntime==", REQUIREMENTS_LOCK, re.M)


def test_the_py_stage_trims_and_strips_before_the_espeak_build_and_the_probes():
    run = _run_with("py", "uv pip install")
    binutils = run.index("apt-get install -y --no-install-recommends binutils;")
    install = run.index("uv pip install")
    trim = run.index(
        'rm -r "$site"/onnxruntime/transformers "$site"/onnxruntime/quantization "$site"/onnxruntime/tools'
        ' "$site"/numpy/f2py /opt/venv/bin/f2py;'
    )
    links = run.index(
        'for lib in "$site"/espeakng_loader/libespeak-ng.so.*; do cmp "$lib" "$site"/espeakng_loader/libespeak-ng.so;'
        ' ln -sf libespeak-ng.so "$lib"; done;'
    )
    strip = run.index("find \"$site\" -type f \\( -name '*.so' -o -name '*.so.*' \\) -exec strip --strip-unneeded {} +;")
    espeak = run.index("/opt/venv/bin/python /opt/media-render/build/espeak_build.py")
    native = run.index(
        "/opt/venv/bin/python -c 'import ctypes, aiohttp, numpy, soundfile, espeakng_loader;"
        " ctypes.CDLL(espeakng_loader.get_library_path()).espeak_Initialize'"
    )
    onnxruntime = run.index("/opt/venv/bin/python -c 'import onnxruntime as o; print(o.get_available_providers())'")
    assert binutils < install < trim < links < strip < espeak < native < onnxruntime
    assert all(name in run[trim:links] for name in ONNXRUNTIME_UNUSED)
    # binutils stays in the builder stage.
    assert "binutils" not in _stage("runtime")


def test_c2pas_static_library_goes_before_the_strip_and_the_signer_loads_after_it():
    # PRD-264 EUAI-06b (D14): c2pa-python signs every marked file, so the trim keeps
    # its shared library (stripped with the rest) and the probe loads it and cryptography.
    run = _run_with("py", "uv pip install")
    install = run.index("uv pip install")
    static = run.index('rm "$site"/c2pa/libs/libc2pa_c.rlib "$site"/c2pa/libs/libc2pa_c.d;')
    strip = run.index("find \"$site\" -type f \\( -name '*.so' -o -name '*.so.*' \\) -exec strip --strip-unneeded {} +;")
    probe = run.index("/opt/venv/bin/python -c 'import c2pa, cryptography.x509; print(\"c2pa\", c2pa.sdk_version())'")
    assert install < static < strip < probe
    assert "libc2pa_c.so" not in run
    assert re.search(r"^c2pa-python==", REQUIREMENTS_LOCK, re.M) and re.search(r"^cryptography==", REQUIREMENTS_LOCK, re.M)


def test_the_in_image_suite_checks_the_venv():
    suite = (MEDIA_RENDER / "tests" / "test_image_venv.py").read_text()
    for needle in (
        '@pytest.mark.parametrize("distribution", ["protobuf", "flatbuffers"])',
        "def test_espeak_ng_ships_one_library_and_links_the_other_names_to_it():",
        "def test_no_shared_library_in_the_venv_carries_a_symbol_table():",
        "def test_every_compiled_extension_in_the_venv_still_imports():",
    ):
        assert needle in suite, needle
