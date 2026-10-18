"""PRD-262 US-053: scripts/ci/image_layer_digest.py finds the layer a COPY made.

media-render.yml builds the renderer twice, the second time with only the
Hyperframes pin changed, and this script must show the Kokoro layer kept its
digest while the js layer did not. These tests feed it ``docker save`` archives
built in memory, in the layout Docker 25 writes (``manifest.json`` naming the
config blob) and the older one (``<id>.json``).

Pure in-memory work: no Docker, no network.
"""
from __future__ import annotations

import importlib.util
import io
import json
import sys
import tarfile
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _REPO / "scripts" / "ci" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # before exec, as the scripts register their siblings
    spec.loader.exec_module(module)
    return module


digest = _load("image_layer_digest")

KOKORO = "sha256:" + "a" * 64
JS_OLD = "sha256:" + "b" * 64
JS_NEW = "sha256:" + "c" * 64
BASE = "sha256:" + "d" * 64


def _config(js: str) -> dict:
    """A media-render config: metadata entries make no layer; each COPY and RUN makes one."""
    return {
        "history": [
            {"created_by": "# debian.sh --arch 'amd64' out/ 'trixie' '@1760918400'"},
            {"created_by": "WORKDIR /app", "empty_layer": True},
            {"created_by": "ENV HYPERFRAMES_NO_TELEMETRY=1 DO_NOT_TRACK=1", "empty_layer": True},
            {"created_by": "COPY /opt/kokoro /opt/kokoro # buildkit"},
            {"created_by": "COPY /opt/media-render/js /opt/media-render/js # buildkit"},
        ],
        "rootfs": {"type": "layers", "diff_ids": [BASE, KOKORO, js]},
    }


def _archive(config: dict, *, oci: bool = True, big_layer: bool = False) -> io.BytesIO:
    """A ``docker save`` stream: the config blob, a layer, then manifest.json last, as Docker writes it."""
    config_name = ("blobs/sha256/" + "e" * 64) if oci else ("e" * 64 + ".json")
    members = {config_name: json.dumps(config).encode()}
    members["blobs/sha256/" + "f" * 64] = b"\0" * ((digest.SMALL_MEMBER_BYTES + 1) if big_layer else 512)
    members["manifest.json"] = json.dumps([{"Config": config_name, "RepoTags": ["media-render:ci"], "Layers": []}]).encode()
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    buffer.seek(0)
    return buffer


@pytest.mark.parametrize("oci", [True, False])
def test_the_config_is_read_from_either_save_layout_in_one_pass(oci):
    config = _config(JS_OLD)
    assert digest.read_config(_archive(config, oci=oci, big_layer=True)) == config


def test_an_archive_without_a_manifest_is_refused():
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w"):
        pass
    buffer.seek(0)
    with pytest.raises(digest.LayerError, match="no manifest.json"):
        digest.read_config(buffer)


def test_each_copy_destination_finds_the_layer_it_made():
    config = _config(JS_OLD)
    assert digest.layer_by_destination(config, "/opt/kokoro") == KOKORO
    assert digest.layer_by_destination(config, "/opt/media-render/js/") == JS_OLD


@pytest.mark.parametrize(("instruction", "destination"), [
    ("COPY /opt/kokoro /opt/kokoro", "/opt/kokoro"),
    ("COPY --chown=1000:1000 . .", "."),
    ("COPY a b /opt/kokoro/", "/opt/kokoro"),
    ("RUN cp -r /a /opt/kokoro", None),
    ("COPY", None),
])
def test_the_destination_is_a_copys_last_word(instruction, destination):
    assert digest.copy_destination(instruction) == destination


def test_a_destination_no_copy_wrote_or_two_copies_wrote_is_refused():
    config = _config(JS_OLD)
    with pytest.raises(digest.LayerError, match="0 layers"):
        digest.layer_by_destination(config, "/opt/chrome")
    twice = _config(JS_OLD)
    twice["history"].append({"created_by": "COPY /opt/kokoro /opt/kokoro # buildkit"})
    twice["rootfs"]["diff_ids"].append(KOKORO)
    with pytest.raises(digest.LayerError, match="2 layers"):
        digest.layer_by_destination(twice, "/opt/kokoro")


def test_history_that_does_not_match_the_layers_is_refused():
    config = _config(JS_OLD)
    config["rootfs"]["diff_ids"].pop()
    with pytest.raises(digest.LayerError, match="history entries"):
        digest.layer_by_destination(config, "/opt/kokoro")


def test_a_hyperframes_bump_keeps_the_kokoro_layer_and_moves_the_js_layer():
    lines = digest.compare(_config(JS_OLD), _config(JS_NEW), ["/opt/kokoro"], ["/opt/media-render/js"])
    assert [line.split()[0] for line in lines] == ["PASS", "PASS"]


def test_a_kokoro_layer_that_moved_or_a_js_layer_that_did_not_fails():
    moved = _config(JS_NEW)
    moved["rootfs"]["diff_ids"][1] = "sha256:" + "9" * 64
    lines = digest.compare(_config(JS_OLD), moved, ["/opt/kokoro"], [])
    assert lines[0].startswith("FAIL")
    lines = digest.compare(_config(JS_OLD), _config(JS_OLD), [], ["/opt/media-render/js"])
    assert lines[0].startswith("FAIL")


def test_main_exits_one_on_a_failed_check_and_zero_when_every_check_passes(monkeypatch, capsys):
    configs = {"a": _config(JS_OLD), "b": _config(JS_NEW)}
    monkeypatch.setattr(digest, "saved_config", configs.__getitem__)
    assert digest.main(["a", "b", "--same", "/opt/kokoro", "--differ", "/opt/media-render/js"]) == 0
    assert digest.main(["a", "b", "--same", "/opt/media-render/js"]) == 1
    assert digest.main(["a", "b", "--same", "/opt/chrome"]) == 1
    out = capsys.readouterr().out
    assert "PASS  /opt/kokoro" in out and "FAIL  /opt/media-render/js" in out and "FAIL  0 layers" in out


def test_main_needs_a_check_to_make():
    with pytest.raises(SystemExit):
        digest.main(["a", "b"])
