"""PRD-209 S8 — QUICKSTART tells the truth (against the merged reality).

The old QUICKSTART claimed "That's it! No `.env` file needed" while compose hard-fails
on three ``${VAR:?}``-required secrets, and it invented default passwords compose no
longer sets. These guards are pure (read the two docs + the compose file, no services)
and lock the docs to the compose reality US-001..008 made true:

* every ``${VAR:?}``-required variable in docker-compose.yml is named in QUICKSTART.md
  (a reader can't miss a secret compose will refuse to start without);
* the dishonest "no .env needed" claim is gone from both QUICKSTART.md and README.md;
* the honest Composio limitation line is present (the 1,000+ integrations need a key —
  PRD-233 owns first-class local setup).
"""
from __future__ import annotations

import pathlib
import re

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_COMPOSE = _REPO_ROOT / "docker-compose.yml"
_QUICKSTART = _REPO_ROOT / "QUICKSTART.md"
_README = _REPO_ROOT / "README.md"

# ${VAR:?message} — a variable compose treats as REQUIRED (errors if unset/empty).
_REQUIRED_VAR = re.compile(r"\$\{([A-Z_][A-Z0-9_]*):\?")


def _required_secrets() -> set[str]:
    return set(_REQUIRED_VAR.findall(_COMPOSE.read_text(encoding="utf-8")))


def test_compose_has_exactly_the_three_required_secrets():
    # Non-vacuity + a pin: if a fourth required secret is added to compose, this fails
    # so the doc (and this guard) are updated together — the secret can't go undocumented.
    assert _required_secrets() == {"POSTGRES_PASSWORD", "REDIS_PASSWORD", "API_KEY"}


def test_quickstart_documents_every_required_secret():
    quickstart = _QUICKSTART.read_text(encoding="utf-8")
    required = _required_secrets()
    assert required, "no ${VAR:?} required secrets parsed from docker-compose.yml — parser drift"
    missing = sorted(v for v in required if v not in quickstart)
    assert not missing, (
        f"QUICKSTART.md does not name required compose secret(s) {missing}; compose refuses "
        "to start without them, so the quickstart must tell the reader to set them."
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
