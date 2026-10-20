"""The signed Content Credentials manifest (PRD-264 EUAI-06b, EU AI Act Art. 50(2)), file by file.

Inside the image, with its own ffmpeg, exiftool and c2pa-python: one file of every format
media-render hands out (MP4, PNG, JPEG, WAV) is made, XMP-marked (EUAI-06a), signed, and
read back with c2pa. The manifest says ``c2pa.created`` with the same digital source type
as the XMP, names "Automatos AI media-render/<version>", carries the XMP's instance id and
no identifier; the XMP mark survives the signing untouched. A JPEG made from a still this
service signed keeps the still's manifest as an ingredient; one made from an image another
generator signed (a person's own, FR-10) drops it. When signing fails the file keeps its XMP mark,
unchanged, and the failure is logged with the job and counted. The signing certificate is
a throwaway one made at test time (conftest: a per-install one in the scratch dir).
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import pytest
from aiohttp.test_utils import TestClient, TestServer

from c2pa_checks import active_manifest, assert_signed, ingredient_labels, store_of
from helpers import TOKEN
from media_render import content_credentials, provenance, signing_keys
from media_render.content_credentials import ContentSigner
from media_render.jpeg import to_jpeg
from media_render.provenance import COMPOSITE, TRAINED
from media_render.server import MANIFEST_HEADER, SHA256_HEADER, SOURCE_TYPE_HEADER, TOKEN_HEADER, create_app
from media_render.signing_keys import SigningKeyError
from media_render.tts import Speaker
from media_render.versions import SERVICE_VERSION
from provenance_checks import assert_only_the_mark, ffmpeg, packet_of, seconds_of, source_type_of
from test_server import fake_synthesize

AUTH = {TOKEN_HEADER: TOKEN}
JOB = "render 0123456789abcdef0123456789abcdef"
# Another tool's claim generator, on a stand-in manifest signed with the test certificate.
FOREIGN_GENERATOR = "Another Tool"


def _mp4(settings, path: Path) -> Path:
    ffmpeg(settings, "-f", "lavfi", "-i", "testsrc=size=64x64:rate=10", "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000",
           "-t", "1", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path))
    return path


def _png(settings, path: Path) -> Path:
    ffmpeg(settings, "-f", "lavfi", "-i", "color=c=0x14171c:s=32x16", "-frames:v", "1", str(path))
    return path


def _jpeg(settings, path: Path) -> Path:
    path.write_bytes(to_jpeg(_png(settings, path.with_suffix(".png")).read_bytes(), ffmpeg_bin=settings.ffmpeg_bin, timeout_seconds=30))
    return path


def _wav(settings, path: Path) -> Path:
    ffmpeg(settings, "-f", "lavfi", "-i", "sine=frequency=220:sample_rate=48000", "-t", "0.5", "-ac", "2", "-c:a", "pcm_s16le", str(path))
    return path


def _mark(settings, path: Path, kind: str) -> str:
    return provenance.mark(path, kind, exiftool_bin=settings.exiftool_bin, timeout_seconds=60)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _broken_signer(settings) -> ContentSigner:
    def no_material(_settings):
        raise SigningKeyError("simulated: no signing certificate")

    return ContentSigner(settings, load=no_material)


FORMATS = [(_mp4, "render.mp4"), (_png, "render.png"), (_jpeg, "image.jpg"), (_wav, "mix.wav")]


# ── each format, signed and read back ─────────────────────────────────────


@pytest.mark.parametrize("make, name", FORMATS)
@pytest.mark.parametrize("kind", [TRAINED, COMPOSITE])
def test_every_format_carries_a_signed_manifest_created_as_its_xmp_says(settings, tmp_path, make, name, kind):
    path = make(settings, tmp_path / name)
    instance_id = _mark(settings, path, kind)
    signer = ContentSigner(settings)
    record = signer.sign(path, kind, job=JOB, instance_id=instance_id)
    assert record.manifest_id is not None, "the file was not signed"
    assert record.digital_source_type == kind and record.sha256 == _sha256(path), "the record is the shipped file's"
    store = assert_signed(path, kind, instance_id=instance_id, absent=(str(tmp_path), JOB, settings.work_dir))
    assert store["active_manifest"] == record.manifest_id
    # The XMP mark (EUAI-06a) is still there, and still alone, after the signing.
    assert source_type_of(settings, path) == kind
    assert_only_the_mark(packet_of(settings, path), kind, absent=(str(tmp_path),))
    assert signer.counts() == {"signed": 1, "failed": 0}
    assert not list(tmp_path.glob("*" + content_credentials.SIGNING_SUFFIX)), "the signing copy was left behind"


@pytest.mark.parametrize("make, name", [(_mp4, "render.mp4"), (_wav, "mix.wav")])
def test_a_signed_file_still_plays(settings, tmp_path, make, name):
    path = make(settings, tmp_path / name)
    before = seconds_of(settings, path)
    ContentSigner(settings).sign(path, TRAINED, job=JOB, instance_id=_mark(settings, path, TRAINED))
    assert abs(seconds_of(settings, path) - before) < 0.05


@pytest.mark.parametrize("make, name", FORMATS)
def test_two_signed_copies_of_one_file_differ_in_hash_and_manifest(settings, tmp_path, make, name):
    """FR-16: identical input never gives two files one SHA-256, signed or not."""
    first = make(settings, tmp_path / name)
    second = tmp_path / f"again-{name}"
    shutil.copyfile(first, second)
    signer = ContentSigner(settings)
    records = [signer.sign(path, COMPOSITE, job=JOB, instance_id=_mark(settings, path, COMPOSITE)) for path in (first, second)]
    assert records[0].sha256 != records[1].sha256
    assert records[0].manifest_id != records[1].manifest_id and None not in {r.manifest_id for r in records}


# ── ingredients: an input's manifest is kept ──────────────────────────────


def test_an_input_that_carries_a_manifest_is_kept_as_an_ingredient(settings, tmp_path):
    signer = ContentSigner(settings)
    footage = _mp4(settings, tmp_path / "footage.mp4")
    footage_record = signer.sign(footage, TRAINED, job=JOB, instance_id=_mark(settings, footage, TRAINED))
    plain = _png(settings, tmp_path / "upload.png")
    found = content_credentials.ingredients_in([footage, plain, tmp_path / "missing.mp4"])
    assert [ingredient.path for ingredient in found] == [footage], "only an input with a manifest is an ingredient"
    output = _mp4(settings, tmp_path / "render.mp4")
    signer.sign(output, COMPOSITE, job=JOB, instance_id=_mark(settings, output, COMPOSITE), ingredients=found)
    store = assert_signed(output, COMPOSITE, absent=(str(tmp_path), "footage.mp4"))
    manifest = active_manifest(store)
    assert ingredient_labels(manifest) == [footage_record.manifest_id]
    assert footage_record.manifest_id in store["manifests"], "the input's own signed manifest is kept in the store"
    assert [ingredient["relationship"] for ingredient in manifest["ingredients"]] == ["componentOf"]


def _signed_elsewhere(settings, path: Path, kind: str) -> Dict[str, Any]:
    """``path`` signed the way another tool or a phone would sign it: its own claim generator
    (the test certificate stands in for theirs). The store it then carries."""
    definition = {
        "claim_generator_info": [{"name": FOREIGN_GENERATOR, "version": "1.0"}],
        "assertions": [{"label": content_credentials.ACTIONS_LABEL, "data": {"actions": [
            {"action": content_credentials.CREATED, "digitalSourceType": kind},
        ]}}],
    }
    staged = path.with_name(f"signing-{path.name}")
    mime = content_credentials.mime_type(path)
    content_credentials.embed(path, staged, mime, definition, (), signing_keys.load_material(settings))
    staged.replace(path)
    return store_of(path)


def _converted(settings, signer: ContentSigner, source: bytes, out: Path):
    """``source`` made into a JPEG the way POST /jpeg makes it: its AI mark carried, then signed."""
    kwargs = {"exiftool_bin": settings.exiftool_bin, "timeout_seconds": 60}
    carried = provenance.carry_mark(source, to_jpeg(source, ffmpeg_bin=settings.ffmpeg_bin, timeout_seconds=30), **kwargs)
    assert carried.kind is not None and carried.instance_id
    data, record = content_credentials.sign_converted(
        signer, carried.data, ".jpg", carried.kind, job="POST /jpeg", instance_id=carried.instance_id, source=source
    )
    out.write_bytes(data)
    assert record.manifest_id is not None and record.sha256 == _sha256(out)
    return carried, record


def test_a_jpeg_made_from_a_still_this_service_signed_keeps_the_stills_manifest(settings, tmp_path):
    still = _png(settings, tmp_path / "render.png")
    signer = ContentSigner(settings)
    still_record = signer.sign(still, COMPOSITE, job=JOB, instance_id=_mark(settings, still, COMPOSITE))
    out = tmp_path / "out.jpg"
    carried, _record = _converted(settings, signer, still.read_bytes(), out)
    assert carried.kind == COMPOSITE
    store = assert_signed(out, COMPOSITE, instance_id=carried.instance_id, absent=(str(tmp_path),))
    assert ingredient_labels(active_manifest(store)) == [still_record.manifest_id]
    assert source_type_of(settings, out) == COMPOSITE


def test_a_jpeg_made_from_an_image_another_generator_signed_drops_that_manifest(settings, tmp_path):
    """FR-10: a person's own image with an AI mark can carry another tool's or a phone's
    manifest (a place, EXIF data, who made it). The JPEG is signed fresh, with no ingredient,
    and nothing of that manifest reaches it."""
    image = _png(settings, tmp_path / "own-image.png")
    _mark(settings, image, TRAINED)
    foreign = _signed_elsewhere(settings, image, TRAINED)
    assert not content_credentials.signed_here(foreign), "the stand-in reads as this service's own"
    out = tmp_path / "out.jpg"
    carried, record = _converted(settings, ContentSigner(settings), image.read_bytes(), out)
    assert carried.kind == TRAINED
    absent = (str(tmp_path), FOREIGN_GENERATOR, foreign["active_manifest"])
    store = assert_signed(out, TRAINED, instance_id=carried.instance_id, absent=absent)
    assert ingredient_labels(active_manifest(store)) == []
    assert list(store["manifests"]) == [record.manifest_id], "the other generator's manifest was kept in the store"
    assert source_type_of(settings, out) == TRAINED


def test_only_a_manifest_this_service_signed_counts_as_its_own():
    def store(manifest: Dict[str, Any]) -> Dict[str, Any]:
        return {"active_manifest": "urn:c2pa:active", "manifests": {"urn:c2pa:active": manifest, "urn:c2pa:other": {}}}

    ours = {"name": content_credentials.CLAIM_GENERATOR_NAME, "version": SERVICE_VERSION}
    assert content_credentials.signed_here(store({"claim_generator_info": [ours]}))
    assert content_credentials.signed_here(store({"claim_generator": f"{content_credentials.claim_generator()} c2pa-rs/0.70"}))
    # The claim generator is the first entry: this service named second is another tool's claim.
    assert not content_credentials.signed_here(store({"claim_generator_info": [{"name": FOREIGN_GENERATOR}, ours]}))
    assert not content_credentials.signed_here(store({"claim_generator": f"{FOREIGN_GENERATOR}/2.1"}))
    assert not content_credentials.signed_here(store({}))
    assert not content_credentials.signed_here({})


def test_post_jpeg_answers_a_signed_jpeg_and_its_record_in_the_headers(settings, tmp_path):
    png = _png(settings, tmp_path / "render.png")
    ContentSigner(settings).sign(png, TRAINED, job=JOB, instance_id=_mark(settings, png, TRAINED))

    async def go():
        app = create_app(settings, pipeline=SimpleNamespace(), speaker=SimpleNamespace(), library={})
        async with TestClient(TestServer(app)) as client:
            response = await client.post("/jpeg", data=png.read_bytes(), headers=AUTH)
            assert response.status == 200 and response.content_type == "image/jpeg"
            named = {name: response.headers.get(name) for name in (SHA256_HEADER, SOURCE_TYPE_HEADER, MANIFEST_HEADER)}
            return named, await response.read()

    headers, body = asyncio.run(go())
    out = tmp_path / "answer.jpg"
    out.write_bytes(body)
    assert headers[SHA256_HEADER] == _sha256(out) and headers[SOURCE_TYPE_HEADER] == TRAINED
    assert store_of(out)["active_manifest"] == headers[MANIFEST_HEADER]
    assert_signed(out, TRAINED, absent=(str(tmp_path),))


# ── POST /tts: each WAV it hands out is signed ────────────────────────────


def test_post_tts_signs_every_wav_and_names_its_hash_and_manifest(settings, tmp_path):
    async def go():
        speaker = Speaker(settings, synthesize=fake_synthesize)
        app = create_app(settings, pipeline=SimpleNamespace(), speaker=speaker, library={})
        async with TestClient(TestServer(app)) as client:
            lines = [{"id": "l01", "text": "Three weeks to go."}, {"id": "l02", "text": "Three weeks to go."}]
            response = await client.post("/tts", data=json.dumps({"lines": lines, "include_audio": True}), headers=AUTH)
            assert response.status == 200
            health = await (await client.get("/health")).json()
            return await response.json(), health

    body, health = asyncio.run(go())
    for line in body["lines"]:
        wav = tmp_path / f"{line['id']}.wav"
        wav.write_bytes(base64.b64decode(line["audio_base64"]))
        assert line["sha256"] == _sha256(wav) and line["digital_source_type"] == TRAINED
        assert store_of(wav)["active_manifest"] == line["manifest_id"]
        assert_signed(wav, TRAINED, absent=(str(tmp_path), settings.work_dir))
    assert body["lines"][0]["sha256"] != body["lines"][1]["sha256"], "two identical lines share a hash"
    assert health["content_credentials"] == {"signed": 2, "failed": 0}


# ── a signing failure: the file keeps its XMP mark ────────────────────────


@pytest.mark.parametrize("make, name", FORMATS)
def test_a_signing_failure_leaves_the_file_marked_and_is_logged_and_counted(settings, tmp_path, caplog, make, name):
    path = make(settings, tmp_path / name)
    _mark(settings, path, TRAINED)
    marked = path.read_bytes()
    signer = _broken_signer(settings)
    with caplog.at_level(logging.ERROR, logger="media_render.content_credentials"):
        record = signer.sign(path, TRAINED, job=JOB)
    assert record.manifest_id is None and record.sha256 == _sha256(path)
    assert path.read_bytes() == marked, "a failed signing changed the file"
    assert source_type_of(settings, path) == TRAINED, "the file lost its XMP mark"
    assert content_credentials.read_store(path, content_credentials.mime_type(path)) is None
    assert signer.counts() == {"signed": 0, "failed": 1}
    logged = [entry.getMessage() for entry in caplog.records if entry.name == "media_render.content_credentials"]
    assert len(logged) == 1 and JOB in logged[0] and name in logged[0], logged
    assert not list(tmp_path.glob("*" + content_credentials.SIGNING_SUFFIX))


def test_post_tts_still_hands_out_marked_wavs_when_signing_fails(settings, tmp_path):
    async def go():
        speaker = Speaker(settings, synthesize=fake_synthesize)
        signer = _broken_signer(settings)
        app = create_app(settings, pipeline=SimpleNamespace(), speaker=speaker, library={}, signer=signer)
        async with TestClient(TestServer(app)) as client:
            lines = [{"id": "l01", "text": "Three weeks to go."}]
            response = await client.post("/tts", data=json.dumps({"lines": lines, "include_audio": True}), headers=AUTH)
            assert response.status == 200
            health = await (await client.get("/health")).json()
            return await response.json(), health

    body, health = asyncio.run(go())
    (line,) = body["lines"]
    wav = tmp_path / "l01.wav"
    wav.write_bytes(base64.b64decode(line["audio_base64"]))
    assert line["manifest_id"] is None and line["sha256"] == _sha256(wav)
    assert source_type_of(settings, wav) == TRAINED
    assert health["content_credentials"] == {"signed": 0, "failed": 1}


# ── the manifest's definition ─────────────────────────────────────────────


def test_the_manifest_definition_holds_the_action_the_generator_and_nothing_else():
    definition = content_credentials.manifest_definition(TRAINED, "xmp.iid:00000000-0000-4000-8000-000000000000")
    assert set(definition) == {"claim_generator_info", "assertions", "instance_id"}
    (assertion,) = definition["assertions"]
    assert assertion == {"label": "c2pa.actions", "data": {"actions": [{"action": "c2pa.created", "digitalSourceType": TRAINED}]}}
    assert content_credentials.claim_generator().startswith("Automatos AI media-render/")
    assert "instance_id" not in content_credentials.manifest_definition(COMPOSITE, None)
