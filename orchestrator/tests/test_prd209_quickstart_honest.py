"""PRD-209 S8 + PRD-233 S5 — the install docs tell the truth (against the merged reality).

The old QUICKSTART claimed "That's it! No `.env` file needed" while compose hard-fails
on three ``${VAR:?}``-required secrets, and it invented default passwords compose no
longer sets. These guards are pure (read the docs + the compose file, no services)
and lock the docs to the compose reality US-001..008 made true:

* every ``${VAR:?}``-required variable in docker-compose.yml is named in QUICKSTART.md
  AND in the self-hosting guide (a reader can't miss a secret compose will refuse to
  start without) — the same checker, shared, as PRD-233 S5 asks;
* the dishonest "no .env needed" claim is gone from both QUICKSTART.md and README.md;
* the honest Composio limitation line is present (the integrations need a key —
  PRD-233 owns first-class local setup);
* the self-hosting guide names the local edition's dials: the worker host-access
  directory, the bring-your-own Composio key, the object store's credential name
  and the edition flag.
"""
from __future__ import annotations

import pathlib
import re

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_COMPOSE = _REPO_ROOT / "docker-compose.yml"
_QUICKSTART = _REPO_ROOT / "QUICKSTART.md"
_README = _REPO_ROOT / "README.md"
_SELF_HOSTING = _REPO_ROOT / "docs" / "getting-started" / "self-hosting.md"

# ${VAR:?message} — a variable compose treats as REQUIRED (errors if unset/empty).
_REQUIRED_VAR = re.compile(r"\$\{([A-Z_][A-Z0-9_]*):\?")

# PRD-233 S5: the local-edition dials the guide must document by name.
_SELF_HOSTING_REQUIRED_MENTIONS = (
    "AUTOMATOS_WORKSPACE_DIR",  # S1 — the worker's host-access directory
    "COMPOSIO_API_KEY",  # S2 — bring-your-own Composio key (env-only)
    "S3_ACCESS_KEY_ID",  # S4 — the local object store's credential name (not AWS_*)
    "AUTH_EDITION",  # PRD-175 — the one edition flag
)


def _required_secrets() -> set[str]:
    return set(_REQUIRED_VAR.findall(_COMPOSE.read_text(encoding="utf-8")))


def _missing_required_secrets(doc: pathlib.Path) -> list[str]:
    """The ``${VAR:?}`` secrets ``doc`` does not name — the shared checker."""
    text = doc.read_text(encoding="utf-8")
    required = _required_secrets()
    assert required, "no ${VAR:?} required secrets parsed from docker-compose.yml — parser drift"
    return sorted(v for v in required if v not in text)


def test_compose_has_exactly_the_three_required_secrets():
    # Non-vacuity + a pin: if a fourth required secret is added to compose, this fails
    # so the doc (and this guard) are updated together — the secret can't go undocumented.
    assert _required_secrets() == {"POSTGRES_PASSWORD", "REDIS_PASSWORD", "API_KEY"}


def test_quickstart_documents_every_required_secret():
    missing = _missing_required_secrets(_QUICKSTART)
    assert not missing, (
        f"QUICKSTART.md does not name required compose secret(s) {missing}; compose refuses "
        "to start without them, so the quickstart must tell the reader to set them."
    )


def test_self_hosting_guide_exists():
    # PRD-233 S5: the real guide supersedes the scattered compose pages, which point at it.
    assert _SELF_HOSTING.is_file(), f"{_SELF_HOSTING} is missing — PRD-233 S5's guide"


def test_self_hosting_guide_documents_every_required_secret():
    missing = _missing_required_secrets(_SELF_HOSTING)
    assert not missing, (
        f"docs/getting-started/self-hosting.md does not name required compose secret(s) "
        f"{missing}; compose refuses to start without them, so the guide must tell the "
        "reader to set them."
    )


def test_self_hosting_guide_names_the_local_edition_dials():
    text = _SELF_HOSTING.read_text(encoding="utf-8")
    missing = [name for name in _SELF_HOSTING_REQUIRED_MENTIONS if name not in text]
    assert not missing, (
        f"docs/getting-started/self-hosting.md does not mention {missing} — the worker "
        "host-access dial, the BYO Composio key, the object store's S3_* credential name "
        "and the edition flag are the local edition's dials and must be documented."
    )


def test_self_hosting_guide_is_linked_from_quickstart_and_readme():
    # The short path and the README both hand the reader on to the full guide.
    for path in (_QUICKSTART, _README):
        text = path.read_text(encoding="utf-8")
        assert "docs/getting-started/self-hosting.md" in text, (
            f"{path.name} must link docs/getting-started/self-hosting.md (the full guide)"
        )


def test_no_no_env_needed_claim_in_quickstart_or_readme():
    # The specific dishonest claim the old QUICKSTART made, plus close variants.
    bad = re.compile(r"no\s+`?\.env`?\s+(?:file\s+)?needed", re.IGNORECASE)
    for path in (_QUICKSTART, _README):
        text = path.read_text(encoding="utf-8")
        assert not bad.search(text), (
            f"{path.name} still claims no .env is needed — compose hard-fails on three "
            "required secrets, so a .env (or equivalent) IS needed."
        )


def test_quickstart_names_an_optional_llm_key():
    # BYOK: the honest quickstart tells the reader AI features need a key they bring.
    text = _QUICKSTART.read_text(encoding="utf-8")
    assert re.search(r"OPENAI_API_KEY|ANTHROPIC_API_KEY|OpenRouter", text), (
        "QUICKSTART.md must mention the optional (bring-your-own) LLM key for AI features"
    )


def test_quickstart_has_honest_composio_limitation_line():
    text = _QUICKSTART.read_text(encoding="utf-8")
    assert re.search(r"composio", text, re.IGNORECASE), (
        "QUICKSTART.md must state the Composio-integrations limitation honestly"
    )
    assert "PRD-233" in text, "the Composio limitation line must point to PRD-233 (owns the fix)"


def test_readme_quickstart_references_required_secrets_or_quickstart():
    text = _README.read_text(encoding="utf-8")
    # Consistency: the README quickstart either names the required secrets or points to
    # QUICKSTART.md for them — it must not leave a reader thinking `docker compose up` alone works.
    assert ("POSTGRES_PASSWORD" in text) or ("QUICKSTART.md" in text), (
        "README quickstart must name the required secrets or link QUICKSTART.md"
    )
