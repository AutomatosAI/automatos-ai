"""PRD-262 #34 and #47 (19 Oct breakdown): Debian packages no render reads leave the runtime.

#47: libcups2t64 (with the avahi libraries it pulls in) and libxshmfence1 come off
the apt line: neither headless shell links them (DT_NEEDED: Playwright's arm64
build, and the amd64 CfT 152.0.7977.30, read 19 Oct), and libxshmfence1, then only
mesa-libgallium's, goes with Mesa. libatk-bridge2.0-0t64 stays: the amd64 shell
links libatk-bridge-2.0.so.0.
#34: shared-mime-info, xkb-data and libx11-data (about 12.7 MB of data no render
reads) are purged the way Mesa is, before the ffmpeg check and the font cache.
The renders and the in-image suite (services/media-render/tests/test_image_packages.py)
prove it.

Pure file reads: no Docker, no network.
"""
from __future__ import annotations

import re

from tests.test_prd262_media_render_diet import _run_with
from tests.test_prd262_media_render_stages import MEDIA_RENDER

NOT_ON_THE_APT_LINE = ("libcups2t64", "libxshmfence1")
UNREAD_DATA = ("shared-mime-info", "xkb-data", "libx11-data")


def test_the_apt_line_leaves_off_what_no_headless_shell_links():
    run = _run_with("runtime", "apt-get install")
    installed = re.search(r"apt-get install -y --no-install-recommends ([^;]+);", run).group(1).split()
    assert not set(NOT_ON_THE_APT_LINE) & set(installed)
    assert "libatk-bridge2.0-0t64" in installed
    assert re.search(r'mesa_only="[^"]*\blibxshmfence1\b', run)


def test_the_x_and_mime_data_is_purged_before_the_ffmpeg_check():
    run = _run_with("runtime", "apt-get install")
    installed = re.search(r"apt-get install -y --no-install-recommends ([^;]+);", run).group(1).split()
    listed = re.search(r'unread_data="([a-z0-9. -]+)"', run).group(1).split()
    assert tuple(listed) == UNREAD_DATA and not set(listed) & set(installed)
    purge = run.index("dpkg --purge --force-depends $unread_data;")
    assert run.index("dpkg --purge --force-depends $mesa_only;") < purge
    assert purge < run.index("if ldd /usr/bin/ffmpeg /usr/bin/ffprobe | grep 'not found'; then") < run.index("fc-cache -f")


def test_the_in_image_suite_checks_each_package():
    suite = (MEDIA_RENDER / "tests" / "test_image_packages.py").read_text()
    assert 'NOT_INSTALLED = ("libcups2t64", "libxshmfence1", "shared-mime-info", "xkb-data", "libx11-data")' in suite
