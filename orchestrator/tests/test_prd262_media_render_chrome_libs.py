"""PRD-262 #38 (19 Oct breakdown): Chrome's own libraries ship without symbol tables.

US-051 stripped the arm64 headless shell; the libraries beside it (SwiftShader,
ANGLE's GLESv2 and EGL, the Vulkan loader: about 9.8 MB of .symtab) were left. The
chrome stage now strips them in the arm64 branch, and the .symtab check after the
branches covers the shell and its libraries on both arches (the amd64 CfT build
ships them stripped: 152.0.7977.30, read 19 Oct). The in-image suite reads the
image (services/media-render/tests/test_image_chrome_libs.py).

Pure file reads: no Docker, no network.
"""
from __future__ import annotations

from tests.test_prd262_media_render_diet import _run_with
from tests.test_prd262_media_render_stages import MEDIA_RENDER

LIBRARIES = "find \"$(dirname \"$BIN\")\" -maxdepth 1 -type f \\( -name '*.so' -o -name '*.so.*' \\)"


def test_the_arm64_shells_libraries_are_stripped_and_checked_on_both_arches():
    run = _run_with("chrome", "playwright-core install")
    arm64 = run.index('if [ "${TARGETARCH:-amd64}" = "arm64" ]; then')
    amd64 = run.index("else", arm64)
    assert f"{LIBRARIES} -exec strip --strip-unneeded {{}} +;" in run[arm64:amd64]
    check = run.index(f'for elf in "$BIN" $({LIBRARIES}); do')
    readelf = run.index('if readelf -SW "$elf" | grep -F .symtab; then')
    assert run.index("fi;", amd64) < check < readelf < run.index("mv ")


def test_the_in_image_suite_reads_the_libraries():
    suite = (MEDIA_RENDER / "tests" / "test_image_chrome_libs.py").read_text()
    assert "libraries = shared_libraries(shell.parent)" in suite
    assert 'if ".symtab" in section_names(path)] == []' in suite
