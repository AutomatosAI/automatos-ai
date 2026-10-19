"""Where the Content Credentials signing certificate and key come from (PRD-264 EUAI-06b, D15).

**Hosted.** The certificate chain (the signing certificate first, then the CA certificates
that issued it) and its private key are settings: ``MEDIA_RENDER_C2PA_SIGNING_CERT_PEM``
and ``MEDIA_RENDER_C2PA_SIGNING_KEY_PEM``, Railway secrets, read in config.py. Until the
C2PA trust-list certificate arrives they hold a self-signed chain; the trust-list one is
swapped in by changing those two settings alone, with no code change. Boot refuses a chain
and a key that do not belong together (``configured_problems``).

**Local edition** (Art. 2(12): the open-source exemption does not cover Article 50, so local
renders are signed too). With no chain configured, this install makes its own on first use,
in ``MEDIA_RENDER_C2PA_DATA_DIR`` (the compose file's ``media_render_data`` volume, never the
image): a root CA, made here and thrown away once it has issued the signing certificate, and
the signing certificate, an ES256 end-entity certificate shaped the way c2pa checks one
(digitalSignature, emailProtection, an authority key identifier, not a CA). The directory is
``0700`` and the key ``0600``. It is made again when it is missing, half written or about to
expire. Verifiers show its manifests as signed by an unknown signer: machine-readable and
detectable, not trusted (D15).

The certificates name Automatos AI and nothing else: no host, user or workspace.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional, Tuple

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

from .config import C2PA_CERT_ENV, C2PA_KEY_ENV, Settings

logger = logging.getLogger(__name__)

ORGANIZATION = "Automatos AI"
SIGNER_NAME = "Automatos AI media-render"
ROOT_NAME = "Automatos AI media-render install root"
LOCAL_ALG = "es256"  # the per-install key is P-256, so c2pa signs it ES256
CHAIN_FILE = "signing-chain.pem"
KEY_FILE = "signing-key.pem"
DIR_MODE = 0o700
KEY_MODE = 0o600
CHAIN_MODE = 0o644
STAGED_SUFFIX = ".new"
# Valid from a day back, so a clock a little behind still accepts it.
BACKDATE = timedelta(days=1)
# A per-install certificate this close to its end is made again before it signs.
RENEW_BEFORE = timedelta(days=30)

_LOCK = threading.Lock()  # one install's material is made once, whichever signer asks first


class SigningKeyError(RuntimeError):
    """The signing certificate or key is missing, unreadable, or does not belong together."""


@dataclass(frozen=True)
class SigningMaterial:
    """What c2pa signs with: the PEM chain, its key (never in a repr), the algorithm, the TSA."""

    chain_pem: bytes
    key_pem: bytes = field(repr=False)
    alg: str
    tsa_url: Optional[str]
    per_install: bool


def _name(common_name: str) -> x509.Name:
    return x509.Name(
        [x509.NameAttribute(NameOID.ORGANIZATION_NAME, ORGANIZATION), x509.NameAttribute(NameOID.COMMON_NAME, common_name)]
    )


def _key_usage(*, signs_certificates: bool) -> x509.KeyUsage:
    return x509.KeyUsage(
        digital_signature=not signs_certificates,
        content_commitment=False,
        key_encipherment=False,
        data_encipherment=False,
        key_agreement=False,
        key_cert_sign=signs_certificates,
        crl_sign=signs_certificates,
        encipher_only=False,
        decipher_only=False,
    )


def _certificate(subject: str, issuer: str, public_key, now: datetime, days: int) -> x509.CertificateBuilder:
    return (
        x509.CertificateBuilder()
        .subject_name(_name(subject))
        .issuer_name(_name(issuer))
        .public_key(public_key)
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - BACKDATE)
        .not_valid_after(now + timedelta(days=days))
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(public_key), critical=False)
    )


def generate_chain(now: datetime, days: int) -> Tuple[bytes, bytes]:
    """A fresh chain (the signing certificate, then the root that issued it) and the signing
    key, PEM (the key PKCS#8). The root's own key is never kept."""
    root_key = ec.generate_private_key(ec.SECP256R1())
    root = (
        _certificate(ROOT_NAME, ROOT_NAME, root_key.public_key(), now, days)
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(_key_usage(signs_certificates=True), critical=True)
        .sign(root_key, hashes.SHA256())
    )
    key = ec.generate_private_key(ec.SECP256R1())
    leaf = (
        _certificate(SIGNER_NAME, ROOT_NAME, key.public_key(), now, days)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(_key_usage(signs_certificates=False), critical=True)
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.EMAIL_PROTECTION]), critical=False)
        .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(root_key.public_key()), critical=False)
        .sign(root_key, hashes.SHA256())
    )
    chain = leaf.public_bytes(serialization.Encoding.PEM) + root.public_bytes(serialization.Encoding.PEM)
    key_pem = key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
    )
    return chain, key_pem


def check_pair(chain_pem: bytes, key_pem: bytes, now: datetime) -> None:
    """Refuse a chain whose first certificate is not the key's, or is not valid ``now``.

    The messages never quote the PEM: it is a secret."""
    try:
        certificates = x509.load_pem_x509_certificates(chain_pem)
        key = serialization.load_pem_private_key(key_pem, password=None)
    except (ValueError, TypeError) as exc:
        raise SigningKeyError(f"the signing certificate chain or key cannot be read ({type(exc).__name__})") from None
    spki = serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo
    if certificates[0].public_key().public_bytes(*spki) != key.public_key().public_bytes(*spki):
        raise SigningKeyError("the signing key is not the key of the chain's first certificate")
    leaf = certificates[0]
    if not leaf.not_valid_before_utc <= now <= leaf.not_valid_after_utc:
        raise SigningKeyError(f"the signing certificate is valid only from {leaf.not_valid_before_utc} to {leaf.not_valid_after_utc}")


def configured_problems(settings: Settings, now: Optional[datetime] = None) -> List[str]:
    """Boot (boot.py): a configured chain and key that c2pa could not sign with. None configured: none."""
    if not settings.c2pa_cert_pem:
        return []
    try:
        check_pair(settings.c2pa_cert_pem.encode("ascii"), settings.c2pa_key_pem.encode("ascii"), now or _now())
    except (SigningKeyError, UnicodeEncodeError) as exc:
        return [f"{C2PA_CERT_ENV} / {C2PA_KEY_ENV}: {exc}"]
    return []


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _write(path: Path, data: bytes, mode: int) -> None:
    """Write ``data`` beside ``path``, with ``mode`` from the start, then move it into place."""
    staged = path.with_name(path.name + STAGED_SUFFIX)
    descriptor = os.open(staged, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
    try:
        os.fchmod(descriptor, mode)
        os.write(descriptor, data)
    finally:
        os.close(descriptor)
    os.replace(staged, path)


def _held(directory: Path, now: datetime) -> Optional[Tuple[bytes, bytes]]:
    """This install's chain and key, when both are there and still good for a while."""
    chain, key = directory / CHAIN_FILE, directory / KEY_FILE
    if not (chain.is_file() and key.is_file()):
        return None
    held = chain.read_bytes(), key.read_bytes()
    try:
        check_pair(*held, now + RENEW_BEFORE)
    except SigningKeyError as exc:
        logger.warning("c2pa: this install's signing certificate is made again: %s", exc)
        return None
    return held


def per_install_material(directory: Path, days: int, now: Optional[datetime] = None) -> Tuple[bytes, bytes]:
    """This install's chain and key, made in ``directory`` on first use. Blocking."""
    moment = now or _now()
    with _LOCK:
        held = _held(directory, moment)
        if held is not None:
            return held
        directory.mkdir(mode=DIR_MODE, parents=True, exist_ok=True)
        os.chmod(directory, DIR_MODE)
        chain, key = generate_chain(moment, days)
        _write(directory / KEY_FILE, key, KEY_MODE)
        _write(directory / CHAIN_FILE, chain, CHAIN_MODE)
        logger.info("c2pa: made this install's signing certificate in %s (self-signed: verifiers show an unknown signer)", directory)
        return chain, key


def load_material(settings: Settings) -> SigningMaterial:
    """The configured chain and key, or this install's own. Blocking."""
    tsa = settings.c2pa_tsa_url or None
    if settings.c2pa_cert_pem:
        return SigningMaterial(
            settings.c2pa_cert_pem.encode("ascii"), settings.c2pa_key_pem.encode("ascii"), settings.c2pa_alg, tsa, False
        )
    if settings.is_production:
        logger.warning(
            "c2pa: %s is not set, so production signs with a per-install certificate (D15: set the hosted chain and key)",
            C2PA_CERT_ENV,
        )
    chain, key = per_install_material(Path(settings.c2pa_data_dir), settings.c2pa_local_cert_days)
    return SigningMaterial(chain, key, LOCAL_ALG, tsa, True)
