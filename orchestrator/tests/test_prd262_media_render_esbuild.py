"""PRD-262 §8 (scan row): media-render's esbuild is built with a patched Go.

images.yml run 39399292631 (main, 330439eef) blocked media-render on two fixable
Criticals in the Go standard library of esbuild's binary, go1.23.12, at
/opt/media-render/js/node_modules/esbuild/bin/esbuild and
@esbuild/linux-<arch>/bin/esbuild: GO-2026-4337 (crypto/tls, fixed in 1.24.13,
1.25.7, 1.26) and GO-2026-6612 (net/http, fixed in 1.26.9). Hyperframes requires
esbuild ^0.25.12, and no esbuild release is built with Go 1.26.9 or later yet, so
the Dockerfile's esbuild stage builds the locked version from source on Go's
patched 1.26 line and the js stage swaps it in. This suite pins that definition:

- the stage's Go image is pinned by digest at 1.26.9 or later, and it builds the
  esbuild version js/package-lock.json locks, with esbuild's release flags;
- the js stage puts the binary at npm's platform path, hard-links
  esbuild/bin/esbuild to it, and fails the build unless the JavaScript API runs it.

The in-image suite reads the Go version back from each binary and runs it
(services/media-render/tests/test_image_esbuild.py). Pure file reads: no Docker.
"""
from __future__ import annotations

import re

from tests.test_prd262_media_render_stages import LOCK, _bases, _stage

# GO-2026-6612 is fixed in Go 1.26.9; GO-2026-4337 already in 1.26.0.
PATCHED_GO = (1, 26, 9)
GO_IMAGE = re.compile(r"golang:(\d+)\.(\d+)\.(\d+)-trixie@sha256:[0-9a-f]{64}")
INSTALL = re.compile(r"go install (?P<flags>.*?) github\.com/evanw/esbuild/cmd/esbuild@v(?P<version>\S+)")


def _locked_esbuild() -> str:
    return LOCK["packages"]["node_modules/esbuild"]["version"]


def test_the_go_image_is_pinned_and_patched():
    match = GO_IMAGE.fullmatch(_bases()["esbuild"])
    assert match, _bases()["esbuild"]
    assert tuple(int(part) for part in match.groups()) >= PATCHED_GO


def test_the_stage_builds_the_locked_esbuild_with_its_release_flags():
    stage = _stage("esbuild")
    install = INSTALL.search(stage)
    assert install, stage
    assert install.group("version") == _locked_esbuild(), "a lock bump moves the source build with it"
    for flag in ("-trimpath", "-buildvcs=false", "-ldflags='-s -w -buildid='"):
        assert flag in install.group("flags"), flag
    for env in ("CGO_ENABLED=0", "GOTOOLCHAIN=local", "GOBIN=/out"):
        assert env in stage, env
    assert "/out/esbuild --version" in stage


def test_the_js_stage_swaps_in_the_binary_and_proves_it_runs():
    js = _stage("js")
    assert "COPY --from=esbuild /out/esbuild /tmp/esbuild" in js
    steps = [
        "RUN npm ci --omit=dev --no-fund --no-audit",
        'native="node_modules/@esbuild/linux-$(node -p process.arch)/bin/esbuild"',
        'cp --remove-destination /tmp/esbuild "$native"',
        'ln -f "$native" node_modules/esbuild/bin/esbuild',
        """node -e 'require("esbuild").transformSync("let a = 1")'""",
    ]
    at = [js.index(step) for step in steps]
    assert at == sorted(at), "npm ci, then the swap, the link and the run, in that order"
    # Only the js tree reaches the runtime: the source build's toolchain never does.
    assert "--from=esbuild" not in _stage("runtime")
