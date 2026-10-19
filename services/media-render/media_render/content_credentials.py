"""A signed Content Credentials (C2PA) manifest on every file this service marks AI-made
(PRD-264 EUAI-06b, EU AI Act Art. 50(2)).

Every file provenance.py gives the IPTC "AI-generated" XMP mark is then signed with
c2pa-python (D14), the Content Authenticity Initiative's SDK, which signs and reads MP4,
PNG, JPEG and WAV. Signing comes after the XMP mark: the manifest's hash covers the file as
it ships, so nothing writes into the file after it.

**The manifest.** One ``c2pa.actions`` assertion with one ``c2pa.created`` action whose
``digitalSourceType`` is the same IPTC term as the XMP mark, and the claim generator
"Automatos AI media-render/<version>" (versions.SERVICE_VERSION). Its instance id is the
XMP's own random ``xmpMM:InstanceID`` (FR-16). Nothing else goes in: no title, no user,
workspace, prompt or file path (Guidelines ¶94); thumbnails are off, and ingredients are
named ``input-NN`` whatever their files are called.

**Ingredients.** An input that already carries a C2PA manifest (AI footage or a still from
a provider that signs, the mix inside the video, the PNG a JPEG is made from) is kept as a
``componentOf`` ingredient, so its own signed history stays readable (FR-9).

**When signing fails** the file still ships, with its XMP mark: the failure is logged with
the job, counted (``counts``, served by /health), and the file's record carries no manifest
id. A file is signed into a copy beside it and moved into place only once the copy reads
back, so a failed signing never leaves a half-written file. A render never ships unmarked.

**The record.** Each marked file's SHA-256 (of the bytes that ship) and its manifest id go
into the job's outputs, for the orchestrator's provenance rows (EUAI-10). Neither is logged
(FR-17).
"""

from __future__ import annotations

import hashlib
import json
import logging
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Tuple

from c2pa import Builder, C2paError, C2paSignerInfo, Context, Reader, Signer

from . import signing_keys
from .config import Settings
from .signing_keys import SigningMaterial
from .versions import SERVICE_VERSION

logger = logging.getLogger(__name__)

CLAIM_GENERATOR_NAME = "Automatos AI media-render"
ACTIONS_LABEL = "c2pa.actions"
CREATED = "c2pa.created"
INGREDIENT_RELATIONSHIP = "componentOf"
INGREDIENT_TITLE = "input-{:02d}"
MIME_TYPES = {
    ".mp4": "video/mp4", ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp",
    ".wav": "audio/wav",
}
# A converted file's source arrives as bytes (POST /jpeg): its type, by its first bytes.
MAGIC = ((b"\x89PNG\r\n\x1a\n", ".png"), (b"\xff\xd8\xff", ".jpg"))
WEBP_RIFF, WEBP_FORM, WEBP_FORM_AT = b"RIFF", b"WEBP", slice(8, 12)
SOURCE_STEM = "source"
CONVERTED_STEM = "converted"
VISUAL_PREFIXES = ("video/", "image/")
AUDIO_PREFIX = "audio/"
# No network while reading or signing (no remote manifests, no OCSP), and no thumbnails:
# a manifest holds what this module puts in it and nothing else.
CONTEXT_SETTINGS = {
    "verify": {"remote_manifest_fetch": False, "ocsp_fetch": False},
    "builder": {"thumbnail": {"enabled": False}},
}
SIGNING_SUFFIX = ".signing"
HASH_CHUNK_BYTES = 1 << 20


class SigningError(RuntimeError):
    """c2pa signed nothing it could read back."""


@dataclass(frozen=True)
class Ingredient:
    """An input file that carries a C2PA manifest, and its MIME type."""

    path: Path
    mime: str


@dataclass(frozen=True)
class MarkRecord:
    """A marked file: its IPTC term, the SHA-256 of the bytes that ship, its manifest id
    (``None`` when signing failed and the file ships with its XMP mark alone)."""

    digital_source_type: str
    sha256: str
    manifest_id: Optional[str]

    def fields(self) -> Dict[str, Any]:
        """What a job's output (or a /tts line) says about its mark."""
        return {"digital_source_type": self.digital_source_type, "sha256": self.sha256, "manifest_id": self.manifest_id}


def claim_generator() -> str:
    return f"{CLAIM_GENERATOR_NAME}/{SERVICE_VERSION}"


def manifest_definition(kind: str, instance_id: Optional[str]) -> Dict[str, Any]:
    """The manifest c2pa signs: one ``c2pa.created`` action with the IPTC term, the claim generator."""
    definition: Dict[str, Any] = {
        "claim_generator_info": [{"name": CLAIM_GENERATOR_NAME, "version": SERVICE_VERSION}],
        "assertions": [{"label": ACTIONS_LABEL, "data": {"actions": [{"action": CREATED, "digitalSourceType": kind}]}}],
    }
    if instance_id:
        definition["instance_id"] = instance_id
    return definition


def mime_type(path: Path) -> Optional[str]:
    """The MIME type c2pa signs ``path`` as, by its extension; ``None`` for a format this service does not sign."""
    return MIME_TYPES.get(path.suffix.lower())


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_store(path: Path, mime: str) -> Optional[Dict[str, Any]]:
    """The file's manifest store as c2pa reads (and validates) it; ``None`` when it carries none.

    Raises c2pa's ``C2paError`` when the file cannot be read as ``mime``. Blocking."""
    with Context.from_dict(CONTEXT_SETTINGS) as context, path.open("rb") as stream:
        reader = Reader.try_create(mime, stream, context=context)
        if reader is None:
            return None
        with reader:
            return json.loads(reader.json())


def _carries_manifest(path: Path, mime: str) -> bool:
    try:
        return read_store(path, mime) is not None
    except (C2paError, OSError, ValueError) as exc:  # an input c2pa cannot read is no ingredient, and says so
        logger.warning("c2pa: input %s could not be read for a manifest, so it is not kept as an ingredient: %s", path.name, exc)
        return False


def ingredients_in(paths: Iterable[Path], prefixes: Tuple[str, ...] = VISUAL_PREFIXES + (AUDIO_PREFIX,)) -> Tuple[Ingredient, ...]:
    """The files among ``paths`` whose type starts with one of ``prefixes`` and that carry a manifest. Blocking."""
    found = []
    for path in paths:
        mime = mime_type(path)
        if mime is not None and mime.startswith(prefixes) and path.is_file() and _carries_manifest(path, mime):
            found.append(Ingredient(path, mime))
    return tuple(found)


def _signer_info(material: SigningMaterial) -> C2paSignerInfo:
    return C2paSignerInfo(alg=material.alg, sign_cert=material.chain_pem, private_key=material.key_pem, ta_url=material.tsa_url)


def embed(
    source: Path, target: Path, mime: str, definition: Dict[str, Any], ingredients: Sequence[Ingredient], material: SigningMaterial
) -> None:
    """Sign ``source`` into ``target`` with ``definition`` and the ``ingredients``. Blocking."""
    with Context.from_dict(CONTEXT_SETTINGS) as context, Signer.from_info(_signer_info(material)) as signer:
        with Builder(definition, context) as builder:
            for index, ingredient in enumerate(ingredients, start=1):
                title = {"title": INGREDIENT_TITLE.format(index), "relationship": INGREDIENT_RELATIONSHIP}
                with ingredient.path.open("rb") as stream:
                    builder.add_ingredient(title, ingredient.mime, stream)
            with source.open("rb") as original, target.open("w+b") as signed:
                builder.sign(signer, mime, original, signed)


def active_manifest_id(path: Path, mime: str) -> str:
    store = read_store(path, mime)
    label = (store or {}).get("active_manifest")
    if not isinstance(label, str) or not label:
        raise SigningError(f"{path.name} reads back with no active manifest")
    return label


class ContentSigner:
    """Signs marked files, one signing material per process, loaded (or made) on first use."""

    def __init__(self, settings: Settings, load: Callable[[Settings], SigningMaterial] = signing_keys.load_material) -> None:
        self._settings = settings
        self._load = load
        self._lock = threading.Lock()
        self._material: Optional[SigningMaterial] = None
        self._signed = 0
        self._failed = 0

    def counts(self) -> Dict[str, int]:
        """How many files this process signed, and how many shipped with their XMP mark alone."""
        with self._lock:
            return {"signed": self._signed, "failed": self._failed}

    def _material_now(self) -> SigningMaterial:
        with self._lock:
            if self._material is None:
                self._material = self._load(self._settings)
            return self._material

    def _count(self, signed: bool) -> None:
        with self._lock:
            if signed:
                self._signed += 1
            else:
                self._failed += 1

    def _sign(self, path: Path, kind: str, instance_id: Optional[str], ingredients: Sequence[Ingredient]) -> str:
        mime = mime_type(path)
        if mime is None:
            raise SigningError(f"{path.name} is not a format this service signs")
        staged = path.with_name(path.name + SIGNING_SUFFIX)
        try:
            embed(path, staged, mime, manifest_definition(kind, instance_id), ingredients, self._material_now())
            manifest_id = active_manifest_id(staged, mime)
            staged.replace(path)
        finally:
            staged.unlink(missing_ok=True)
        return manifest_id

    def sign(
        self, path: Path, kind: str, *, job: str, instance_id: Optional[str] = None, ingredients: Sequence[Ingredient] = ()
    ) -> MarkRecord:
        """Sign ``path`` (already XMP-marked ``kind``) in place, and answer its record. A failure
        is logged with ``job`` and counted, and the file keeps its XMP mark. Blocking."""
        manifest_id: Optional[str] = None
        try:
            manifest_id = self._sign(path, kind, instance_id, ingredients)
        except Exception:  # c2pa, the key material, the disk: whatever it was, the file ships with its XMP mark
            logger.exception("%s: %s could not be signed (C2PA); it ships with its XMP mark alone", job, path.name)
        self._count(manifest_id is not None)
        return MarkRecord(kind, sha256_of(path), manifest_id)


def sniffed_suffix(data: bytes) -> Optional[str]:
    """The extension of an image given as bytes: PNG, JPEG or WebP; ``None`` for anything else."""
    if data[:4] == WEBP_RIFF and data[WEBP_FORM_AT] == WEBP_FORM:
        return ".webp"
    return next((suffix for magic, suffix in MAGIC if data.startswith(magic)), None)


def sign_converted(
    signer: ContentSigner, data: bytes, suffix: str, kind: str, *, job: str, instance_id: Optional[str], source: bytes
) -> Tuple[bytes, MarkRecord]:
    """``data`` (a file converted from ``source`` and XMP-marked ``kind``) signed, keeping the
    manifest ``source`` carried as an ingredient; and its record. Blocking."""
    with tempfile.TemporaryDirectory(prefix="media-render-c2pa-") as scratch:
        target = Path(scratch) / f"{CONVERTED_STEM}{suffix}"
        target.write_bytes(data)
        source_suffix = sniffed_suffix(source)
        ingredients: Tuple[Ingredient, ...] = ()
        if source_suffix is not None:
            original = Path(scratch) / f"{SOURCE_STEM}{source_suffix}"
            original.write_bytes(source)
            ingredients = ingredients_in([original])
        record = signer.sign(target, kind, job=job, instance_id=instance_id, ingredients=ingredients)
        return target.read_bytes(), record
