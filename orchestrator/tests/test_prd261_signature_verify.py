"""PRD-261 C-05a — the signing keys file and the Ed25519 verifier (``services/themes/signature_verify.py``).

Pins (plan §2.2.6, §2.6 C-05a):

* the shared vector (``tests/fixtures/themes/signing-vector.json``) holds RFC 8032 section 7.1's
  Ed25519 cases and they verify with ``cryptography``; its canonical cases are what
  ``signing.canonical_text`` writes, and ``payload_for`` rebuilds its signing payload;
* ``keyId`` is ``ed25519:`` and the first 16 hex of sha256(SPKI DER);
* with the keys path pointed at an RFC public key: a payload the RFC seed signs here is
  ``verified`` with that keyId; a flipped file hash, a flipped signature byte, a retired key, a
  revoked key are ``invalid``; no signature is ``unsigned``;
* a keys file with a malformed key, a duplicate keyId or an unknown status fails with a plain
  message; the shipped (empty) keys file loads and nothing verifies;
* no private key anywhere under ``services/themes`` or ``tests/fixtures/themes``, and no new
  dependency.

The only keys used are the RFC's published seeds, held in memory here; nothing is written but
the keys files of ``tmp_path``.
"""
from __future__ import annotations

import ast
import base64
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey  # noqa: E402

from core.models.themes import ThemePackage  # noqa: E402
from services.themes import signature_verify as sv  # noqa: E402
from services.themes import signing  # noqa: E402

VECTOR_PATH = _ORCH / "tests" / "fixtures" / "themes" / "signing-vector.json"
# The kit's tests/fixtures/signing-vector.json at automatos-play #9 (C-05m) has this sha256 too:
# the two files are the same bytes (seam check; from W-A8 on, A-19f4's lane step compares them in CI).
VECTOR_SHA256 = "62474a90ac82819304d605d418fc1e9b6d29c71def94854807efe116280277f3"
VECTOR = json.loads(VECTOR_PATH.read_text(encoding="utf-8"))
RFC = VECTOR["ed25519"]["cases"]
SIGNER, OTHER = RFC[0], RFC[1]
PAYLOAD_CASE = next(case for case in VECTOR["canonical"]["cases"] if case["name"] == "signing payload")
SPKI_PREFIX = bytes.fromhex("302a300506032b6570032100")  # an Ed25519 public key's SPKI DER, before the key
X25519_SPKI_PREFIX = bytes.fromhex("302a300506032b656e032100")  # the same shape for an X25519 key
ADDED = "2026-10-20"
NO_PRIVATE_KEY_ROOTS = (_ORCH / "services" / "themes", _ORCH / "tests" / "fixtures" / "themes")
PRIVATE_KEY_MARKER = b"PRIVATE KEY"


def _seed(case):
    return Ed25519PrivateKey.from_private_bytes(bytes.fromhex(case["seed"]))


def _key_entry(case, status="active", **changes):
    return {"keyId": case["keyId"], "spki": case["spki"], "status": status, "added": ADDED, **changes}


def _keys_file(tmp_path, *entries, name="signing-keys.json"):
    path = tmp_path / name
    path.write_text(json.dumps({"schema": 1, "keys": list(entries)}), encoding="utf-8")
    return path


@pytest.fixture
def keys_at(monkeypatch, tmp_path):
    """Point the verifier's keys path at a file holding the given entries."""

    def point(*entries):
        monkeypatch.setattr(sv, "KEYS_PATH", _keys_file(tmp_path, *entries))

    return point


def _payload():
    value = PAYLOAD_CASE["value"]
    return sv.payload_for(value["id"], value["version"], value["files"])


def _signed(payload, case=SIGNER):
    return base64.b64encode(_seed(case).sign(sv.canonical_bytes(payload))).decode("ascii")


# --- the shared vector ---------------------------------------------------------------------


def test_the_vector_is_the_kits_bytes_and_cites_rfc_8032():
    assert hashlib.sha256(VECTOR_PATH.read_bytes()).hexdigest() == VECTOR_SHA256
    assert "RFC 8032 section 7.1" in VECTOR["ed25519"]["source"]
    assert [case["name"] for case in RFC] == ["TEST 1", "TEST 2", "TEST 3", "TEST 1024", "TEST SHA(abc)"]


@pytest.mark.parametrize("case", RFC, ids=[case["name"] for case in RFC])
def test_the_rfc_8032_cases_verify_with_cryptography(case):
    public, message, signature = (bytes.fromhex(case[field]) for field in ("public", "message", "signature"))
    Ed25519PublicKey.from_public_bytes(public).verify(signature, message)  # raises InvalidSignature on a bad case
    seed = _seed(case)
    assert seed.public_key().public_bytes_raw() == public
    assert seed.sign(message) == signature  # Ed25519 is deterministic: the seed gives the RFC's signature
    der = base64.b64decode(case["spki"])
    assert der == SPKI_PREFIX + public
    assert sv.key_id_of(der) == case["keyId"] == "ed25519:" + hashlib.sha256(der).hexdigest()[:16]
    (key,) = sv.parse_keys(json.dumps({"schema": 1, "keys": [_key_entry(case)]}))
    key.public_key.verify(signature, message)
    assert (key.key_id, key.status) == (case["keyId"], "active")


def test_payload_for_rebuilds_the_vectors_payload_and_its_canonical_text():
    value = PAYLOAD_CASE["value"]
    payload = _payload()
    assert payload == value and list(payload) == ["id", "version", "digest", "files"]
    assert list(payload["files"]) == sorted(value["files"])
    assert signing.canonical_text(payload) == PAYLOAD_CASE["text"]
    assert sv.canonical_bytes(payload) == PAYLOAD_CASE["text"].encode("ascii")
    assert hashlib.sha256(sv.canonical_bytes(payload)).hexdigest() == PAYLOAD_CASE["sha256"]


@pytest.mark.parametrize("case", VECTOR["canonical"]["cases"], ids=lambda case: case["name"])
def test_signing_canonical_text_writes_each_canonical_case(case):
    assert signing.canonical_text(case["value"]) == case["text"]
    assert hashlib.sha256(case["text"].encode("ascii")).hexdigest() == case["sha256"]


# --- verdicts ------------------------------------------------------------------------------


def test_a_payload_the_rfc_seed_signed_is_verified_with_its_key_id(keys_at):
    keys_at(_key_entry(SIGNER), _key_entry(OTHER))
    payload = _payload()
    verdict = sv.verify(payload, _signed(payload))
    assert verdict == sv.Verdict("verified", SIGNER["keyId"]) and verdict.verified
    assert verdict.view() == {"state": "verified", "keyId": SIGNER["keyId"]}
    assert sv.verify(payload, _signed(payload, OTHER)) == sv.Verdict("verified", OTHER["keyId"])


def test_a_flipped_file_hash_is_invalid(keys_at):
    keys_at(_key_entry(SIGNER))
    payload = _payload()
    signature = _signed(payload)
    path = sorted(payload["files"])[0]
    flipped_sha = ("0" if payload["files"][path][0] != "0" else "1") + payload["files"][path][1:]
    files = {**payload["files"], path: flipped_sha}
    assert sv.verify(sv.payload_for(payload["id"], payload["version"], files), signature) == sv.Verdict("invalid")
    assert sv.verify({**payload, "files": files}, signature) == sv.Verdict("invalid")  # the old digest kept
    assert sv.verify({**payload, "version": "1.0.1"}, signature) == sv.Verdict("invalid")


def test_a_flipped_signature_byte_is_invalid(keys_at):
    keys_at(_key_entry(SIGNER))
    payload = _payload()
    raw = bytearray(base64.b64decode(_signed(payload)))
    raw[0] ^= 0x01
    assert sv.verify(payload, base64.b64encode(bytes(raw)).decode("ascii")) == sv.Verdict("invalid")


@pytest.mark.parametrize("status", ["retired", "revoked"])
def test_a_retired_or_revoked_key_is_invalid_and_named(keys_at, status):
    keys_at(_key_entry(SIGNER, status=status))
    payload = _payload()
    verdict = sv.verify(payload, _signed(payload))
    assert verdict == sv.Verdict("invalid", SIGNER["keyId"]) and not verdict.verified


def test_a_signature_from_a_key_not_in_the_file_is_invalid(keys_at):
    keys_at(_key_entry(SIGNER))
    payload = _payload()
    assert sv.verify(payload, _signed(payload, OTHER)) == sv.Verdict("invalid")


@pytest.mark.parametrize("text", ["not base64!", base64.b64encode(b"x" * 63).decode("ascii"), "QUJD"])
def test_a_signature_that_is_not_64_bytes_of_base64_is_invalid(keys_at, text):
    keys_at(_key_entry(SIGNER))
    assert sv.verify(_payload(), text) == sv.Verdict("invalid")


@pytest.mark.parametrize("signature", [None, ""])
def test_no_signature_is_unsigned(keys_at, signature):
    keys_at(_key_entry(SIGNER))
    assert sv.verify(_payload(), signature) == sv.UNSIGNED == sv.Verdict("unsigned")


def _package(payload, signature):
    return ThemePackage(
        theme_id=payload["id"], version=payload["version"], digest=payload["digest"], signature=signature,
        file_manifest={path: {"type": "text/plain", "size": 1, "sha256": sha} for path, sha in payload["files"].items()},
    )


def test_a_stored_version_is_checked_over_the_payload_its_file_list_gives(keys_at):
    keys_at(_key_entry(SIGNER))
    payload = _payload()
    signature = _signed(payload)
    assert signing.signing_payload(_package(payload, signature)) == payload
    assert sv.package_verdict(_package(payload, signature)) == sv.Verdict("verified", SIGNER["keyId"])
    assert sv.package_verdict(_package(payload, None)) == sv.UNSIGNED
    tampered = _package(payload, signature)
    tampered.digest = "ab" * 32  # the file hashes no longer add up to the digest
    assert sv.package_verdict(tampered) == sv.Verdict("invalid")


# --- the keys file -------------------------------------------------------------------------


def _x25519_spki():
    return base64.b64encode(X25519_SPKI_PREFIX + bytes(range(32))).decode("ascii")


BAD_KEYS = {
    "not json": ("{schema", sv.NOT_JSON),
    "another schema": ({"schema": 2, "keys": []}, sv.NOT_SCHEMA),
    "schema true": ({"schema": True, "keys": []}, sv.NOT_SCHEMA),
    "keys not a list": ({"schema": 1, "keys": {}}, sv.NOT_SCHEMA),
    "not an object": ([], sv.NOT_SCHEMA),
    "a missing field": ({"schema": 1, "keys": [{"keyId": SIGNER["keyId"], "spki": SIGNER["spki"], "status": "active"}]},
                        sv.KEY_SHAPE.format(number=1)),
    "an extra field": ({"schema": 1, "keys": [_key_entry(SIGNER, note="x")]}, sv.KEY_SHAPE.format(number=1)),
    "an unknown status": ({"schema": 1, "keys": [_key_entry(SIGNER, status="expired")]}, sv.KEY_STATUS.format(number=1)),
    "a status that is not text": ({"schema": 1, "keys": [_key_entry(SIGNER, status=["active"])]},
                                  sv.KEY_STATUS.format(number=1)),
    "a bad added date": ({"schema": 1, "keys": [_key_entry(SIGNER, added="20 Oct 2026")]}, sv.KEY_ADDED.format(number=1)),
    "spki not base64": ({"schema": 1, "keys": [_key_entry(SIGNER, spki="not base64!")]}, sv.KEY_SPKI.format(number=1)),
    "spki not a key": ({"schema": 1, "keys": [_key_entry(SIGNER, spki=base64.b64encode(b"garbage").decode())]},
                       sv.KEY_SPKI.format(number=1)),
    "spki not ed25519": ({"schema": 1, "keys": [_key_entry(SIGNER, spki=_x25519_spki())]}, sv.KEY_SPKI.format(number=1)),
    "a keyId of another key": ({"schema": 1, "keys": [_key_entry(SIGNER), _key_entry(OTHER, keyId=SIGNER["keyId"])]},
                               sv.KEY_ID_MISMATCH.format(number=2)),
    "a duplicate keyId": ({"schema": 1, "keys": [_key_entry(SIGNER), _key_entry(SIGNER, status="retired")]},
                          sv.KEY_TWICE.format(key_id=SIGNER["keyId"])),
}


@pytest.mark.parametrize("name", sorted(BAD_KEYS))
def test_a_keys_file_that_breaks_the_contract_fails_with_a_plain_message(tmp_path, monkeypatch, name):
    content, message = BAD_KEYS[name]
    text = content if isinstance(content, str) else json.dumps(content)
    with pytest.raises(sv.SigningKeysError) as refused:
        sv.parse_keys(text)
    assert str(refused.value) == message
    for echoed in ("expired", "20 Oct", "garbage", "not base64!"):
        assert echoed not in str(refused.value)
    path = tmp_path / "keys.json"
    path.write_text(text, encoding="utf-8")
    monkeypatch.setattr(sv, "KEYS_PATH", path)
    with pytest.raises(sv.SigningKeysError):
        sv.verify(_payload(), _signed(_payload()))


def test_a_keys_file_that_cannot_be_read_fails_with_a_plain_message(tmp_path):
    with pytest.raises(sv.SigningKeysError, match="can't be read"):
        sv.load_keys(tmp_path / "missing.json")


def test_the_shipped_keys_file_loads_and_nothing_verifies():
    assert sv.KEYS_PATH == _ORCH / "services" / "themes" / "data" / "signing-keys.json"
    assert json.loads(sv.KEYS_PATH.read_text(encoding="utf-8")) == {"schema": 1, "keys": []}
    assert sv.load_keys() == ()
    payload = _payload()
    for case in RFC:
        assert sv.verify(payload, _signed(payload, case)) == sv.Verdict("invalid")


# --- no private key, no new dependency -----------------------------------------------------


def test_no_private_key_under_services_themes_or_the_theme_fixtures():
    found = [
        str(path.relative_to(_ORCH)) for root in NO_PRIVATE_KEY_ROOTS for path in sorted(root.rglob("*"))
        if path.is_file() and PRIVATE_KEY_MARKER in path.read_bytes()
    ]
    assert found == []


def _imported_roots(source):
    roots = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def test_the_verifier_uses_cryptography_already_required_and_loads_public_keys_only():
    source = (_ORCH / "services" / "themes" / "signature_verify.py").read_text(encoding="utf-8")
    stdlib = {"__future__", "base64", "binascii", "hashlib", "json", "re", "dataclasses", "pathlib", "typing"}
    assert _imported_roots(source) <= stdlib | {"cryptography", "core", "services"}
    requirements = (_ORCH / "requirements.in").read_text(encoding="utf-8")
    assert re.search(r"^cryptography\b", requirements, re.MULTILINE)
    imports = [line for line in source.splitlines() if re.match(r"\s*(import|from)\s", line)]
    assert not [line for line in imports if re.search(r"private|pem|pkcs", line, re.IGNORECASE)]
