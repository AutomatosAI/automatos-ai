"""PRD-262 US-019: the API image's system packages are the ones the code calls.

CI runs it inside a built API image, with the image's own Python, as the image's user:

    docker run --rm -i --entrypoint python IMAGE - < scripts/ci/image_system_tools.py

It checks what US-019 changed in the base stage's apt RUN (orchestrator/Dockerfile):

- packages: ghostscript, libgs10 and libgdk-pixbuf-2.0-0 are not installed, and
  fonts-urw-base35, libpangoft2-1.0-0 and libharfbuzz-subset0 are (dpkg's own records);
- OCR: tesseract's orientation model is gone and its English model is there, and an OCR
  fixture (two lines drawn in DejaVu Sans) comes back as its text through
  ``pytesseract.image_to_string`` with the default page mode, the one OCR call the code
  makes (modules/rag/ingestion/multimodal/processors.py);
- git: its server-side and mail binaries are gone, git-core's copy of git is a link to
  /usr/bin/git, the builtins in git-core reach /usr/bin/git through it, and the HTTPS
  helper is there.

test.yml's prod-image-parity runs it. ``! command -v gs``, an HTTPS clone, the Helvetica
render (scripts/ci/image_pdf_fixture.py) and the clone tests in the test image
(scripts/ci/image_test_image.py) are steps of their own there. Standard library apart
from the image's Pillow and pytesseract. Exit 0 when every check passes, 1 otherwise.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import List

ABSENT_PACKAGES = ("ghostscript", "libgs10", "libgdk-pixbuf-2.0-0")
NAMED_PACKAGES = ("fonts-urw-base35", "libpangoft2-1.0-0", "libharfbuzz-subset0")
TESSDATA_ROOT = Path("/usr/share/tesseract-ocr")
OSD_MODEL = "osd.traineddata"
ENGLISH_MODEL = "eng.traineddata"
OCR_FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
OCR_LINES = ("Automatos reads this page", "Invoice 4827 paid in full")
OCR_FONT_PX = 48
OCR_LINE_PX = 90
OCR_MARGIN_PX = 40
OCR_WIDTH_PX = 1400
GIT = Path("/usr/bin/git")
GIT_CORE = Path("/usr/lib/git-core")
# The files orchestrator/Dockerfile deletes; tests/test_prd262_system_packages.py holds the two lists equal.
GIT_REMOVED = (
    "/usr/bin/git-shell",
    "/usr/bin/scalar",
    "/usr/lib/git-core/git-daemon",
    "/usr/lib/git-core/git-http-backend",
    "/usr/lib/git-core/git-http-push",
    "/usr/lib/git-core/git-imap-send",
    "/usr/lib/git-core/git-shell",
    "/usr/lib/git-core/scalar",
)
# Builtins a clone and a pull run from git-core, and the helper an HTTPS remote needs.
GIT_BUILTINS = ("git-clone", "git-upload-pack", "git-fetch")
GIT_HTTPS_HELPER = "git-remote-https"
TIMEOUT_S = 120


def _run(cmd: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_S, check=False)


def dpkg_status(package: str) -> str:
    """dpkg's status word for ``package`` (``installed``, ``config-files``…), ``not-installed`` when unknown."""
    done = _run(["dpkg-query", "--show", "--showformat=${db:Status-Status}", package])
    return done.stdout.strip() if done.returncode == 0 and done.stdout.strip() else "not-installed"


def package_problems() -> List[str]:
    problems = [f"the Debian package {name} is installed" for name in ABSENT_PACKAGES
                if dpkg_status(name) == "installed"]
    problems += [f"the Debian package {name} is not installed" for name in NAMED_PACKAGES
                 if dpkg_status(name) != "installed"]
    return problems


def tesseract_problems() -> List[str]:
    models = {path.name: path for path in TESSDATA_ROOT.rglob("*.traineddata")}
    problems = [f"{models[OSD_MODEL]} is still in the image"] if OSD_MODEL in models else []
    if ENGLISH_MODEL not in models:
        problems.append(f"no {ENGLISH_MODEL} under {TESSDATA_ROOT}")
    return problems


def ocr_fixture():
    """The OCR fixture: ``OCR_LINES`` in black DejaVu Sans on white, one line under the other."""
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.truetype(str(OCR_FONT), OCR_FONT_PX)
    image = Image.new("RGB", (OCR_WIDTH_PX, 2 * OCR_MARGIN_PX + OCR_LINE_PX * len(OCR_LINES)), "white")
    draw = ImageDraw.Draw(image)
    for row, line in enumerate(OCR_LINES):
        draw.text((OCR_MARGIN_PX, OCR_MARGIN_PX + OCR_LINE_PX * row), line, fill="black", font=font)
    return image


def missing_lines(text: str) -> List[str]:
    """The fixture's lines that ``text`` (tesseract's output) does not hold, its spacing normalised. Pure."""
    found = " ".join(text.split())
    return [line for line in OCR_LINES if line not in found]


def ocr_problems() -> List[str]:
    import pytesseract

    text = pytesseract.image_to_string(ocr_fixture())
    missing = missing_lines(text)
    return [f"OCR of the fixture missed {missing!r}; tesseract read {' '.join(text.split())!r}"] if missing else []


def git_problems() -> List[str]:
    problems = [f"{path} is still in the image" for path in GIT_REMOVED if os.path.lexists(path)]
    git = GIT.resolve()
    core_copy = GIT_CORE / "git"
    if not core_copy.is_symlink() or core_copy.resolve() != git:
        problems.append(f"{core_copy} is not a link to {GIT}")
    problems += [f"{GIT_CORE / name} does not reach {GIT}" for name in GIT_BUILTINS
                 if (GIT_CORE / name).resolve() != git]
    if not (GIT_CORE / GIT_HTTPS_HELPER).exists():
        problems.append(f"{GIT_CORE / GIT_HTTPS_HELPER} is missing: git can't clone over HTTPS")
    done = _run([str(core_copy), "--version"])
    if done.returncode != 0:
        problems.append(f"{core_copy} --version failed: {done.stderr.strip()}")
    return problems


CHECKS = (("packages", package_problems), ("tesseract", tesseract_problems),
          ("OCR fixture", ocr_problems), ("git", git_problems))


def main() -> int:
    failed = False
    for name, check in CHECKS:
        problems = check()
        for problem in problems:
            print(f"::error title=System packages ({name})::{problem}")
        if not problems:
            print(f"{name}: ok")
        failed = failed or bool(problems)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
