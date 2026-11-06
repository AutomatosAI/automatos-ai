"""PRD-273 §5.1 / §8-Q5 — the optional passphrase: AES-256-GCM under a scrypt-derived key.

``cryptography`` (already in ``orchestrator/requirements.in``) does both; no new package.

* **The key.** scrypt over the passphrase with a random 16-byte salt per file, its cost
  (``SCRYPT_N``, ``SCRYPT_R``, ``SCRYPT_P``) recorded in the manifest so a later reader can
  derive the same key. A ``check`` value (``KEY_CHECK_TEXT`` sealed under the key) lets an
  import say "wrong passphrase" before it reads any part.
* **An entry.** Its plaintext is deflated, then sealed: ``nonce (12 bytes) || ciphertext ||
  tag (16 bytes)``, a fresh random nonce per entry and the entry's name as associated data,
  so an entry cannot be renamed or swapped into another place unnoticed. The zip stores it
  uncompressed (ciphertext does not compress). Sealing streams, so a large file never sits
  in memory.

Every entry but ``manifest.json`` is sealed; the manifest stays readable so the import page
can show what the file holds before the passphrase is typed. ``open_entry`` is the reader the
tests use and the import (wave 2) will.
"""
from __future__ import annotations

import base64
import os
import zlib
from dataclasses import dataclass
from typing import Any, BinaryIO, Dict

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

CIPHER = "AES-256-GCM"
KDF = "scrypt"
COMPRESSION = "deflate"
KEY_BYTES = 32
SALT_BYTES = 16
NONCE_BYTES = 12
TAG_BYTES = 16
# scrypt's cost: 2^15 x 8 x 1 takes about 32 MiB and a fraction of a second, once per file.
SCRYPT_N = 2 ** 15
SCRYPT_R = 8
SCRYPT_P = 1
KEY_CHECK_TEXT = b"automatos.workspace/v1 key check"
KEY_CHECK_AAD = b"manifest.json#check"
DEFLATE_LEVEL = 6


class WrongPassphrase(ValueError):
    """The passphrase does not open this file, or an entry was changed after it was sealed."""


@dataclass(frozen=True)
class FileKey:
    """A file's key and the manifest block that lets a reader derive it again."""

    key: bytes
    salt: bytes
    n: int = SCRYPT_N
    r: int = SCRYPT_R
    p: int = SCRYPT_P

    def manifest_block(self) -> Dict[str, Any]:
        """The manifest's ``encryption`` block: cipher, KDF and its cost, salt and the check value."""
        nonce = os.urandom(NONCE_BYTES)
        check = nonce + AESGCM(self.key).encrypt(nonce, KEY_CHECK_TEXT, KEY_CHECK_AAD)
        return {
            "cipher": CIPHER, "kdf": KDF, "compression": COMPRESSION,
            "salt": _b64(self.salt), "n": self.n, "r": self.r, "p": self.p, "check": _b64(check),
        }


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _derive(passphrase: str, salt: bytes, n: int, r: int, p: int) -> bytes:
    return Scrypt(salt=salt, length=KEY_BYTES, n=n, r=r, p=p).derive(passphrase.encode("utf-8"))


def new_file_key(passphrase: str) -> FileKey:
    """A fresh key for one export: a new random salt, scrypt at the module's cost."""
    salt = os.urandom(SALT_BYTES)
    return FileKey(key=_derive(passphrase, salt, SCRYPT_N, SCRYPT_R, SCRYPT_P), salt=salt)


def key_from_manifest(passphrase: str, block: Dict[str, Any]) -> FileKey:
    """The key a manifest's ``encryption`` block describes; WrongPassphrase when the check fails."""
    salt = base64.b64decode(block["salt"])
    key = FileKey(key=_derive(passphrase, salt, int(block["n"]), int(block["r"]), int(block["p"])),
                  salt=salt, n=int(block["n"]), r=int(block["r"]), p=int(block["p"]))
    check = base64.b64decode(block["check"])
    try:
        AESGCM(key.key).decrypt(check[:NONCE_BYTES], check[NONCE_BYTES:], KEY_CHECK_AAD)
    except InvalidTag as wrong:
        raise WrongPassphrase("The passphrase does not open this file.") from wrong
    return key


class SealingStream:
    """Writes one entry's plaintext as deflate, then AES-256-GCM, into ``out``."""

    def __init__(self, key: FileKey, name: str, out: BinaryIO) -> None:
        nonce = os.urandom(NONCE_BYTES)
        self._encryptor = Cipher(algorithms.AES(key.key), modes.GCM(nonce)).encryptor()
        self._encryptor.authenticate_additional_data(name.encode("utf-8"))
        self._deflate = zlib.compressobj(DEFLATE_LEVEL)
        self._out = out
        out.write(nonce)

    def write(self, data: bytes) -> None:
        """Seal more plaintext."""
        self._out.write(self._encryptor.update(self._deflate.compress(data)))

    def close(self) -> None:
        """Flush the deflate stream, finish the cipher and append the tag."""
        self._out.write(self._encryptor.update(self._deflate.flush()))
        self._out.write(self._encryptor.finalize())
        self._out.write(self._encryptor.tag)


def open_entry(sealed: bytes, key: FileKey, name: str) -> bytes:
    """The plaintext of a sealed entry named ``name``; WrongPassphrase when it does not open."""
    if len(sealed) < NONCE_BYTES + TAG_BYTES:
        raise WrongPassphrase("The entry is too short to be sealed.")
    nonce, body, tag = sealed[:NONCE_BYTES], sealed[NONCE_BYTES:-TAG_BYTES], sealed[-TAG_BYTES:]
    decryptor = Cipher(algorithms.AES(key.key), modes.GCM(nonce, tag)).decryptor()
    decryptor.authenticate_additional_data(name.encode("utf-8"))
    try:
        deflated = decryptor.update(body) + decryptor.finalize()
    except InvalidTag as wrong:
        raise WrongPassphrase("The entry does not open with this passphrase.") from wrong
    return zlib.decompress(deflated)


__all__ = [
    "CIPHER", "FileKey", "KDF", "SealingStream", "WrongPassphrase", "key_from_manifest", "new_file_key",
    "open_entry",
]
