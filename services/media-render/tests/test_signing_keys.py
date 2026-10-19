"""Where the C2PA signing certificate and key come from (PRD-264 EUAI-06b, D15).

Hosted: a configured chain and key, swapped by configuration alone; boot refuses a pair
that does not belong together. Local edition: none configured, so the install makes its own
on first use in its data directory (0700, the key 0600), keeps it, and makes it again when
it is missing, half written or about to expire. Every certificate here is made at test
time and thrown away: nothing secret is committed.
"""

from __future__ import annotations

import stat
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from cryptography import x509
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

from c2pa_checks import assert_signed
from media_render import boot, provenance, signing_keys
from media_render.config import C2PA_CERT_ENV, C2PA_KEY_ENV, ConfigError, load_settings
from media_render.content_credentials import ContentSigner
from media_render.provenance import TRAINED
from provenance_checks import ffmpeg

NOW = datetime.now(timezone.utc)


def _pair(days: int = 365):
    return signing_keys.generate_chain(NOW, days)


def _configured(settings, chain: bytes, key: bytes):
    return replace(settings, c2pa_cert_pem=chain.decode("ascii"), c2pa_key_pem=key.decode("ascii"))


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


# ── the certificate c2pa is given ─────────────────────────────────────────


def test_the_made_chain_is_an_es256_signing_certificate_under_its_own_root():
    chain, key = _pair()
    leaf, root = x509.load_pem_x509_certificates(chain)
    assert leaf.issuer == root.subject and root.issuer == root.subject
    assert not leaf.extensions.get_extension_for_class(x509.BasicConstraints).value.ca
    assert root.extensions.get_extension_for_class(x509.BasicConstraints).value.ca
    usage = leaf.extensions.get_extension_for_class(x509.KeyUsage).value
    assert usage.digital_signature and not usage.key_cert_sign
    assert list(leaf.extensions.get_extension_for_class(x509.ExtendedKeyUsage).value) == [ExtendedKeyUsageOID.EMAIL_PROTECTION]
    leaf.extensions.get_extension_for_class(x509.AuthorityKeyIdentifier)  # c2pa requires one on the signing certificate
    names = {attribute.value for attribute in leaf.subject} | {attribute.value for attribute in root.subject}
    assert names == {signing_keys.ORGANIZATION, signing_keys.SIGNER_NAME, signing_keys.ROOT_NAME}, "the names say Automatos AI alone"
    assert leaf.subject.get_attributes_for_oid(NameOID.COMMON_NAME)[0].value == signing_keys.SIGNER_NAME
    assert key.splitlines()[0].endswith(b"BEGIN PRIVATE KEY" + b"-" * 5), "a PKCS#8 key"
    signing_keys.check_pair(chain, key, NOW)


def test_a_key_that_is_not_the_chains_or_a_lapsed_chain_is_refused():
    chain, _ = _pair()
    _, other_key = _pair()
    with pytest.raises(signing_keys.SigningKeyError, match="not the key"):
        signing_keys.check_pair(chain, other_key, NOW)
    chain, key = _pair()
    with pytest.raises(signing_keys.SigningKeyError, match="valid only"):
        signing_keys.check_pair(chain, key, NOW + timedelta(days=400))
    with pytest.raises(signing_keys.SigningKeyError, match="cannot be read"):
        signing_keys.check_pair(b"not a certificate", key, NOW)


# ── hosted: configured, swapped by configuration alone ────────────────────


def test_a_configured_chain_is_used_and_nothing_is_made(settings):
    chain, key = _pair()
    material = signing_keys.load_material(_configured(settings, chain, key))
    assert (material.chain_pem, material.key_pem, material.per_install) == (chain, key, False)
    assert not Path(settings.c2pa_data_dir).exists(), "a configured install made a certificate of its own"
    assert "PRIVATE KEY" not in repr(material) and "PRIVATE KEY" not in repr(_configured(settings, chain, key))


def test_a_configured_chain_signs(settings, tmp_path):
    png = tmp_path / "render.png"
    ffmpeg(settings, "-f", "lavfi", "-i", "color=c=0x14171c:s=32x16", "-frames:v", "1", str(png))
    instance_id = provenance.mark(png, TRAINED, exiftool_bin=settings.exiftool_bin, timeout_seconds=60)
    chain, key = _pair()
    record = ContentSigner(_configured(settings, chain, key)).sign(png, TRAINED, job="render test", instance_id=instance_id)
    assert record.manifest_id is not None
    assert_signed(png, TRAINED, instance_id=instance_id)


def test_boot_refuses_a_configured_pair_that_cannot_sign(settings):
    chain, key = _pair()
    _, other_key = _pair()
    assert boot.signing_problems(settings) == [], "none configured: the install makes its own"
    assert boot.signing_problems(_configured(settings, chain, key)) == []
    (problem,) = boot.signing_problems(_configured(settings, chain, other_key))
    assert problem.startswith(f"{C2PA_CERT_ENV} / {C2PA_KEY_ENV}:") and "BEGIN" not in problem
    assert problem in boot.boot_problems(_configured(settings, chain, other_key), {})


def test_the_settings_take_the_pair_together_and_unfold_a_one_line_pem():
    chain, key = _pair()
    with pytest.raises(ConfigError, match="set together"):
        load_settings({C2PA_CERT_ENV: chain.decode()})
    folded = {C2PA_CERT_ENV: chain.decode().replace("\n", "\\n"), C2PA_KEY_ENV: key.decode().replace("\n", "\\n")}
    loaded = load_settings(folded)
    assert loaded.c2pa_cert_pem.encode() == chain and loaded.c2pa_key_pem.encode() == key
    assert "PRIVATE KEY" not in repr(loaded)
    defaults = load_settings({})
    assert (defaults.c2pa_cert_pem, defaults.c2pa_alg, defaults.c2pa_tsa_url) == ("", "es256", "")
    assert defaults.c2pa_data_dir == "/var/lib/media-render/c2pa"


@pytest.mark.parametrize(
    "name, raw, message",
    [
        ("MEDIA_RENDER_C2PA_SIGNING_ALG", "rs256", "expected one of es256"),
        ("MEDIA_RENDER_C2PA_TSA_URL", "ftp://tsa.example", "timestamp authority"),
        ("MEDIA_RENDER_C2PA_LOCAL_CERT_DAYS", "30", "at least 60"),
    ],
)
def test_a_signing_setting_the_service_cannot_use_is_refused(name, raw, message):
    with pytest.raises(ConfigError, match=message):
        load_settings({name: raw})


# ── local edition: this install's own, made on first use ──────────────────


def test_an_install_without_a_certificate_makes_one_on_first_use_and_keeps_it(settings):
    directory = Path(settings.c2pa_data_dir)
    assert not directory.exists()
    first = signing_keys.load_material(settings)
    assert first.per_install and first.alg == signing_keys.LOCAL_ALG
    assert _mode(directory) == signing_keys.DIR_MODE
    assert _mode(directory / signing_keys.KEY_FILE) == signing_keys.KEY_MODE
    assert (directory / signing_keys.CHAIN_FILE).read_bytes() == first.chain_pem
    again = signing_keys.load_material(settings)
    assert (again.chain_pem, again.key_pem) == (first.chain_pem, first.key_pem), "the install made a second certificate"
    assert not list(directory.glob("*" + signing_keys.STAGED_SUFFIX))


def test_a_half_written_or_expiring_certificate_is_made_again(settings):
    directory = Path(settings.c2pa_data_dir)
    first = signing_keys.per_install_material(directory, 3650)
    (directory / signing_keys.CHAIN_FILE).unlink()
    remade = signing_keys.per_install_material(directory, 3650)
    assert remade[1] != first[1], "a key without its chain was kept"
    expiring = signing_keys.per_install_material(directory, 3650, now=NOW + timedelta(days=3640))
    assert expiring != remade, "a certificate ten days from its end was kept"
    leaf = x509.load_pem_x509_certificates(expiring[0])[0]
    assert leaf.not_valid_after_utc > NOW + timedelta(days=3650)
