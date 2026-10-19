"""PRD-262 #22 (19 Oct breakdown): media-render runs a stripped node binary.

The node-stripped stage strips the official image's binary (about 20 MB of .symtab,
.strtab and .debug sections; .dynsym stays, so native addons still resolve Node-API).
Its text is the worker's, byte for byte: the same RUN on the same base makes the
same layer, and the runtime copies the binary as the first layer on the shared
python:3.12-slim base, so a host with both images keeps one node binary
(scripts/ci/image_layer_digest.py compares the two). The in-image suite proves the
binary is stripped and sharp's addon loads (services/media-render/tests/test_image_node.py).

Pure file reads: no Docker, no network.
"""
from __future__ import annotations

import re
import subprocess

from tests.test_prd262_media_render_stages import DOCKERFILE, MEDIA_RENDER, ROOT, _stage

# The stage both Debian images carry byte for byte, so their node layers match.
NODE_STRIPPED_STAGE = """# Node, stripped. The same stage, byte for byte, in services/workspace-worker/Dockerfile
# and services/media-render/Dockerfile, so the two images share one node layer on a host.
FROM node AS node-stripped
RUN apt-get update \\
 && apt-get install -y --no-install-recommends binutils \\
 && strip --strip-unneeded --preserve-dates /usr/local/bin/node \\
 && rm -rf /var/lib/apt/lists/*
"""
NODE_COPY = "COPY --link --from=node-stripped /usr/local/bin/node /usr/local/bin/node"


def _tracked_dockerfiles() -> list[str]:
    proc = subprocess.run(["git", "ls-files", "*Dockerfile*"], cwd=str(ROOT), capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stderr
    return [path for path in proc.stdout.splitlines() if path.endswith("Dockerfile")]


def test_the_node_stripped_stage_follows_the_node_stage_word_for_word():
    node = re.search(r"^FROM node:24-trixie-slim@sha256:[0-9a-f]{64} AS node\n\n", DOCKERFILE, re.M)
    assert node, "no pinned node stage"
    assert DOCKERFILE[node.end() :].startswith(NODE_STRIPPED_STAGE)


def test_every_dockerfile_with_a_node_stripped_stage_carries_the_same_text():
    # The worker carries it too (its own lane): any Dockerfile that has the stage
    # must have this text, or the two node layers differ.
    carriers = [path for path in _tracked_dockerfiles() if "AS node-stripped" in (ROOT / path).read_text()]
    assert "services/media-render/Dockerfile" in carriers
    for path in carriers:
        assert NODE_STRIPPED_STAGE in (ROOT / path).read_text(), path


def test_the_runtime_takes_node_from_the_stripped_stage_as_its_first_layer():
    runtime = [line for line in _stage("runtime").splitlines() if line.strip()]
    assert runtime[1] == NODE_COPY, "the node binary is the first instruction on the base"
    assert not [line for line in runtime if re.match(r"COPY .*--from=node /usr/local/bin/node", line)]


def test_the_in_image_suite_proves_the_binary_stripped_and_an_addon_loading():
    suite = (MEDIA_RENDER / "tests" / "test_image_node.py").read_text()
    assert 'assert ".symtab" not in sections and ".dynsym" in sections' in suite
    assert 'run = subprocess.run(["node", "-e", SHARP_VERSIONS, str(JS_TREE / "sharp")]' in suite
