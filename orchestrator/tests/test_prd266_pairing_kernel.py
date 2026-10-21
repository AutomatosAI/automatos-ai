"""PRD-266 US-003a — the one pairing kernel (``core/auth/pairing.py``).

* the code alphabet has no look-alikes (0/O/1/I); a code is 8 of its
  characters shown as ``XXXX-XXXX``; normalising gives the canonical (issued)
  form, so a code typed in lower case, with spaces or without its hyphen pairs a
  CLI host, a phone and a browser alike (US-005b, lead 20 Oct);
* ``hash_secret`` is the SHA-256 hex CLI hosts have always stored;
* ``mint_token`` keeps its prefix and returns ``hash_secret`` of the token;
* ``secrets_match`` compares through ``hmac.compare_digest`` and refuses an
  empty or missing value on either side before comparing anything;
* the rotation overlap is open strictly before ``valid_until``;
* ``services/cli_host_service.py`` holds no copy of any of it (AGENTS.md
  "Replace, don't shim"): it imports the kernel.
"""
from __future__ import annotations

import ast
import hashlib
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

from core.auth import pairing  # noqa: E402

CLI_HOST_SERVICE = _ORCH / "services" / "cli_host_service.py"
KERNEL_FUNCTIONS = {
    "hash_secret", "normalize_pairing_code", "new_pairing_code", "_new_pairing_code",
    "mint_token", "secrets_match", "within_rotation_overlap",
}
KERNEL_CONSTANTS = {"PAIRING_ALPHABET", "_PAIRING_ALPHABET", "PAIRING_CODE_LENGTH", "PAIRING_CODE_TTL_SECONDS"}
CODE_SHAPE = re.compile(r"^[A-Z2-9]{4}-[A-Z2-9]{4}$")
SAMPLES = 500


# ── the code ────────────────────────────────────────────────────────────────

def test_the_alphabet_has_no_look_alikes():
    for look_alike in "0O1I":
        assert look_alike not in pairing.PAIRING_ALPHABET
    assert len(set(pairing.PAIRING_ALPHABET)) == len(pairing.PAIRING_ALPHABET) == 32


def test_a_code_is_eight_alphabet_characters_shown_as_two_groups():
    assert pairing.PAIRING_CODE_LENGTH == 8
    assert pairing.PAIRING_CODE_TTL_SECONDS == 600
    codes = {pairing.new_pairing_code() for _ in range(SAMPLES)}
    for code in codes:
        assert CODE_SHAPE.match(code), code
        assert set(code.replace("-", "")) <= set(pairing.PAIRING_ALPHABET), code
    assert len(codes) > SAMPLES // 2, "codes must be random, not a fixed sequence"


def test_normalize_upper_cases_and_strips_spaces():
    assert pairing.normalize_pairing_code("  abcd efgh ") == "ABCD-EFGH"
    assert pairing.normalize_pairing_code("abcd-efgh") == "ABCD-EFGH"
    assert pairing.normalize_pairing_code(None) == ""
    assert pairing.normalize_pairing_code("   ") == ""
    code = pairing.new_pairing_code()
    assert pairing.normalize_pairing_code(code.lower()) == code, "a code typed in lower case is the same code"


@pytest.mark.parametrize("typed", ["abcdefgh", "ABCD EFGH", "abcd-efgh", "  AbCd-EfGh ", "ab cd-ef gh", "ABCD--EFGH"])
def test_a_typed_code_is_canonicalised_to_its_issued_form(typed):
    assert pairing.normalize_pairing_code(typed) == "ABCD-EFGH"
    assert pairing.hash_secret(pairing.normalize_pairing_code(typed)) == pairing.hash_secret("ABCD-EFGH")


def test_an_issued_code_is_its_own_canonical_form_so_stored_hashes_keep_matching():
    for code in {pairing.new_pairing_code() for _ in range(SAMPLES)}:
        assert pairing.normalize_pairing_code(code) == code
        assert pairing.hash_secret(pairing.normalize_pairing_code(code)) == pairing.hash_secret(code)


@pytest.mark.parametrize("typed, canonical", [("abc", "ABC"), ("abcd-efg", "ABCDEFG"), ("abcd-efghj", "ABCDEFGHJ"), ("-", "")])
def test_a_code_of_any_other_length_stays_ungrouped(typed, canonical):
    assert pairing.normalize_pairing_code(typed) == canonical


# ── hashing and minting ─────────────────────────────────────────────────────

def test_hash_secret_is_the_sha256_hex_cli_hosts_have_always_stored():
    assert pairing.hash_secret("a-token") == hashlib.sha256(b"a-token").hexdigest()
    assert pairing.hash_secret("") == hashlib.sha256(b"").hexdigest()
    assert pairing.hash_secret(None) == hashlib.sha256(b"").hexdigest()


def test_mint_token_keeps_its_prefix_and_returns_the_tokens_hash():
    token, digest = pairing.mint_token("amd_", 32)
    assert token.startswith("amd_")
    assert digest == pairing.hash_secret(token)
    assert len(token) == len("amd_") + 43, "token_urlsafe(32) is 43 characters"
    assert pairing.mint_token("amd_", 32)[0] != token, "every mint is a new secret"


def test_the_empty_prefix_gives_todays_cli_host_token():
    token, digest = pairing.mint_token("", 32)
    assert re.fullmatch(r"[A-Za-z0-9_-]{43}", token), token
    assert digest == hashlib.sha256(token.encode("utf-8")).hexdigest()


# ── comparing ───────────────────────────────────────────────────────────────

def test_secrets_match_compares_through_hmac_compare_digest(monkeypatch):
    real = pairing.hmac.compare_digest
    calls = []

    def spy(a, b):
        calls.append((a, b))
        return real(a, b)

    monkeypatch.setattr(pairing.hmac, "compare_digest", spy)
    token, digest = pairing.mint_token("amd_", 32)
    assert pairing.secrets_match(digest, token) is True
    assert calls == [(digest.encode("utf-8"), digest.encode("utf-8"))]
    assert pairing.secrets_match(digest, token + "x") is False
    assert len(calls) == 2, "a wrong token is refused by the constant-time compare, not before it"


@pytest.mark.parametrize(
    "stored, presented",
    [
        (None, "a-token"),
        ("", "a-token"),
        (pairing.hash_secret("a-token"), None),
        (pairing.hash_secret("a-token"), ""),
        (pairing.hash_secret(""), ""),
        (None, None),
        ("", ""),
    ],
)
def test_secrets_match_refuses_an_empty_or_missing_side(monkeypatch, stored, presented):
    def never(*_a, **_k):  # pragma: no cover - must not be reached
        raise AssertionError("an empty side must be refused before any compare")

    monkeypatch.setattr(pairing.hmac, "compare_digest", never)
    assert pairing.secrets_match(stored, presented) is False


# ── the rotation overlap ────────────────────────────────────────────────────

def test_the_rotation_overlap_is_open_strictly_before_valid_until():
    valid_until = datetime(2026, 10, 20, 12, 0, tzinfo=timezone.utc)
    assert pairing.within_rotation_overlap(valid_until, valid_until - timedelta(microseconds=1)) is True
    assert pairing.within_rotation_overlap(valid_until, valid_until) is False
    assert pairing.within_rotation_overlap(valid_until, valid_until + timedelta(seconds=1)) is False


def test_the_rotation_overlap_without_a_window_or_with_naive_times():
    assert pairing.within_rotation_overlap(None) is False
    naive_until = datetime(2026, 10, 20, 12, 0)
    aware_before = datetime(2026, 10, 20, 11, 59, tzinfo=timezone.utc)
    assert pairing.within_rotation_overlap(naive_until, aware_before) is True, "naive is read as UTC"
    assert pairing.within_rotation_overlap(naive_until, naive_until) is False


def test_the_rotation_overlap_defaults_to_now():
    now = datetime.now(timezone.utc)
    assert pairing.within_rotation_overlap(now + timedelta(minutes=10)) is True
    assert pairing.within_rotation_overlap(now - timedelta(seconds=1)) is False


# ── the CLI host lane holds no copy ─────────────────────────────────────────

def _module_shape(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    defs = {n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    assigned = {
        t.id
        for n in tree.body if isinstance(n, (ast.Assign, ast.AnnAssign))
        for t in (n.targets if isinstance(n, ast.Assign) else [n.target]) if isinstance(t, ast.Name)
    }
    imports_from = {n.module for n in tree.body if isinstance(n, ast.ImportFrom)}
    attrs = {
        f"{n.value.id}.{n.attr}"
        for n in ast.walk(tree) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
    }
    return defs, assigned, imports_from, attrs


def test_cli_host_service_holds_no_copy_of_the_kernel():
    defs, assigned, imports_from, attrs = _module_shape(CLI_HOST_SERVICE)
    assert not defs & KERNEL_FUNCTIONS, f"copies left: {sorted(defs & KERNEL_FUNCTIONS)}"
    assert not assigned & KERNEL_CONSTANTS, f"copies left: {sorted(assigned & KERNEL_CONSTANTS)}"
    assert "core.auth.pairing" in imports_from
    inline = attrs & {"hashlib.sha256", "hmac.compare_digest", "secrets.compare_digest"}
    assert not inline, f"the kernel's primitives are re-implemented inline: {sorted(inline)}"


def test_the_names_cli_host_callers_read_are_the_kernels():
    from services import cli_host_service as svc

    assert svc.hash_secret is pairing.hash_secret
    assert svc.normalize_pairing_code is pairing.normalize_pairing_code
    assert svc.PAIRING_CODE_TTL_SECONDS == pairing.PAIRING_CODE_TTL_SECONDS
    assert svc.HOST_TOKEN_BYTES == 32, "a CLI host's token length is unchanged"


class _FakeQuery:
    """Stands in for ``db.query(...).filter(...)``: the filters are ignored."""

    def __init__(self, rows):
        self._rows = list(rows)

    def query(self, *_a, **_k):
        return self

    def filter(self, *_a, **_k):
        return self

    def first(self):
        return self._rows[0] if self._rows else None

    def all(self):
        return list(self._rows)


def test_a_cli_host_token_still_resolves_through_the_kernel():
    from services import cli_host_service as svc

    token, digest = pairing.mint_token("", svc.HOST_TOKEN_BYTES)
    host = SimpleNamespace(token_hash=digest)
    assert svc.resolve_host_by_token(_FakeQuery([host]), token) is host
    stale = SimpleNamespace(token_hash=pairing.hash_secret("another-token"))
    assert svc.resolve_host_by_token(_FakeQuery([stale]), token) is None, "the row must match in constant time too"
    assert svc.resolve_host_by_token(_FakeQuery([SimpleNamespace(token_hash=None)]), token) is None


def test_the_session_token_scan_still_matches_by_the_kernel():
    from services import cli_host_service as svc

    ref = {}
    token = svc.mint_session_token(ref)
    assert ref[svc.SESSION_TOKEN_HASH_KEY] == pairing.hash_secret(token)
    live = SimpleNamespace(runtime_ref=ref, lease_until=datetime.now(timezone.utc) + timedelta(minutes=5))
    assert svc._scan_for_session_token(_FakeQuery([live]), token) is live
    assert svc._scan_for_session_token(_FakeQuery([live]), token + "x") is None
    lapsed = SimpleNamespace(runtime_ref=ref, lease_until=datetime.now(timezone.utc) - timedelta(seconds=1))
    assert svc._scan_for_session_token(_FakeQuery([lapsed]), token) is None, "the live-lease rule still holds"


# ── the canonical form on every path that pairs (Postgres, CI) ─────────────

TYPED_FORMS = (str.lower, lambda code: code.replace("-", ""), lambda code: code.replace("-", " ").lower())
WEB_URL = "https://studio.prd266-kernel.test"
REQUEST_BASE = "http://192.0.2.20:8000/"


def _typed_variants(code: str):
    return [form(code) for form in TYPED_FORMS]


@pytest.fixture
def local_install(monkeypatch):
    """A local install that takes paired devices, its web app at ``WEB_URL``."""
    from config import HOSTED_PUBLIC_API_HOST, config

    monkeypatch.setattr(config, "AUTH_EDITION", "local")
    monkeypatch.setattr(config, "LOCAL_REMOTE_ACCESS", "paired")
    monkeypatch.setattr(config, "FRONTEND_URL", WEB_URL)
    monkeypatch.setattr(config, "PUBLIC_API_HOST", HOSTED_PUBLIC_API_HOST)
    return config


def _seed_person(db) -> int:
    import uuid

    from sqlalchemy import text

    tag = uuid.uuid4().hex[:12]
    return db.execute(
        text("INSERT INTO users (username, email, is_active) VALUES (:u, :e, true) RETURNING id"),
        {"u": f"prd266-kernel-{tag}", "e": f"prd266-kernel-{tag}@t.test"},
    ).scalar()


@pytest.mark.parametrize("variant", range(len(TYPED_FORMS)))
def test_a_cli_host_pairs_with_the_code_typed_without_its_hyphen(db_session, seed_workspace, variant):
    import uuid

    from services import cli_host_service as svc

    host, code, _expires = svc.create_pairing_code(db_session, uuid.UUID(seed_workspace()), "laptop")
    paired = svc.pair_host(db_session, _typed_variants(code)[variant], "laptop", {})
    assert paired is not None and paired[0].id == host.id


@pytest.mark.parametrize("platform", ["ios", "web"])
@pytest.mark.parametrize("variant", range(len(TYPED_FORMS)))
def test_a_phone_and_a_browser_pair_with_the_code_typed_without_its_hyphen(
    db_session, seed_workspace, local_install, platform, variant,
):
    """The phone's code field and the web 'Pair this browser' field send what was typed."""
    import uuid

    from services.mobile.pairing import create_pairing_code, pair_device

    issued = create_pairing_code(
        db_session, uuid.UUID(seed_workspace()), _seed_person(db_session), request_base_url=REQUEST_BASE)
    typed = _typed_variants(issued.code)[variant]
    paired = pair_device(db_session, typed, f"{platform} test", platform, "1.0.0", f"prd266-kernel-{platform}-{variant}")
    assert paired is not None and paired.token.startswith("amd_")
