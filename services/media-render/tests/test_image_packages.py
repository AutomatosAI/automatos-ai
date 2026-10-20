"""PRD-262 #34, #47: Debian packages no render reads are not installed. Runs inside the image.

#47: libcups2t64 and libxshmfence1 are off the apt line (no headless shell links
them), and libxshmfence1 goes with Mesa. #34: shared-mime-info, xkb-data and
libx11-data are purged. ffmpeg, ffprobe and the shell still find every library
(test_image.py), and the renders prove the rest.
"""

from __future__ import annotations

import subprocess

import pytest

# One package per check: dpkg -s fails on the first package it does not know.
NOT_INSTALLED = ("libcups2t64", "libxshmfence1", "shared-mime-info", "xkb-data", "libx11-data")


@pytest.mark.parametrize("package", NOT_INSTALLED)
def test_packages_no_render_reads_are_not_installed(package):
    status = subprocess.run(["dpkg", "-s", package], capture_output=True, text=True, timeout=60)
    assert status.returncode != 0, f"{package} is installed:\n{status.stdout}"
