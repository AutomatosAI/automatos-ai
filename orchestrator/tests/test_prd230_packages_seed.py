"""PRD-230 US-008 — the two Shopify package seeds.

PURE: validates the seed DATA (``PACKAGES``) against the REAL marketplace
inventory (``SHOPIFY_AGENTS``), runs the shared matcher over the actual seed
rows, and proves the upsert is idempotent with a fake session (no Postgres). The
"every member ref resolves" guarantee is structural — the seed builds members
from the roster, so a drifted slug fails at import — and re-asserted here.
"""
from __future__ import annotations

from core.models.marketplace_packages import MEMBER_TYPES, MarketplacePackage
from core.seeds.seed_packages import (
    DEVELOPMENT_PACKAGE,
    MANAGEMENT_PACKAGE,
    PACKAGES,
    seed_packages,
)
from core.seeds.seed_shopify_agents import SHOPIFY_AGENTS
from services.marketplace_packages import match_by_signals

_ROSTER_SLUGS = {a["slug"] for a in SHOPIFY_AGENTS}
_ROSTER_NAMES = {a["name"] for a in SHOPIFY_AGENTS}


def _instances() -> list[MarketplacePackage]:
    return [MarketplacePackage(**dict(p)) for p in PACKAGES]


# --------------------------------------------------------------------------- #
# Contents cited against real inventory (AC: every member ref resolves)
# --------------------------------------------------------------------------- #


def test_every_agent_member_ref_resolves_to_real_inventory():
    for pkg in PACKAGES:
        for m in pkg["members"]:
            assert m["type"] in MEMBER_TYPES, m
            if m["type"] == "agent":
                assert m["ref"] in _ROSTER_SLUGS, f"unresolved agent ref: {m['ref']}"
                # The cited name matches the real roster (no invented artifacts).
                assert m["name"] in _ROSTER_NAMES, f"invented name: {m['name']}"


def test_no_invented_member_types_and_management_is_run_the_store():
    mgmt_refs = {m["ref"] for m in MANAGEMENT_PACKAGE["members"]}
    dev_refs = {m["ref"] for m in DEVELOPMENT_PACKAGE["members"]}
    # D5 split: the two rosters are disjoint and each cites known slugs.
    assert not (mgmt_refs & dev_refs)
    assert mgmt_refs == {
        "shopify-ops", "shopify-support", "shopify-inventory-watchdog",
        "shopify-business-analyst",
    }
    assert dev_refs == {
        "shopify-app-dev", "shopify-storefront-dev", "shopify-extension-dev",
    }


def test_no_customer_data_in_seed():
    """Public repo — the seed carries generic curation text only, no PII / emails."""
    import json

    blob = json.dumps(PACKAGES).lower()
    assert "@" not in blob, "no email addresses in package seed"
    for needle in ("password", "secret", "api_key", "token", "gerard"):
        assert needle not in blob, f"suspicious token in seed: {needle}"


# --------------------------------------------------------------------------- #
# Showcase + matching (AC: both showcased; matcher returns them for shopify)
# --------------------------------------------------------------------------- #


def test_both_packages_showcased_and_shopify_tagged():
    for pkg in PACKAGES:
        assert pkg["showcase"] is True
        tags = {t.lower() for t in pkg["vertical_tags"]}
        assert {"shopify", "ecommerce"} <= tags


def test_matcher_returns_both_for_shopify_signals():
    matches = match_by_signals(
        {"platforms": ["shopify"], "urls": ["acme.myshopify.com"]}, _instances()
    )
    slugs = [m.package.slug for m in matches]
    assert "shopify-management" in slugs
    assert "shopify-development" in slugs


def test_management_outranks_development_on_run_the_store_signals():
    matches = match_by_signals(
        {"platforms": ["shopify"], "text": "run my store orders inventory customers revenue"},
        _instances(),
    )
    assert matches, "expected at least one match"
    assert matches[0].package.slug == "shopify-management"


def test_development_outranks_management_on_dev_signals():
    matches = match_by_signals(
        {"platforms": ["shopify"], "text": "build a theme liquid app extension checkout code"},
        _instances(),
    )
    assert matches
    assert matches[0].package.slug == "shopify-development"


# --------------------------------------------------------------------------- #
# Setup manifest — the Shopify two-step + D7 guide + reports
# --------------------------------------------------------------------------- #


def test_management_carries_weekly_numbers_report_and_shopify_two_step():
    manifest = MANAGEMENT_PACKAGE["setup_manifest"]
    report_names = {r["name"] for r in manifest["report_templates"]}
    assert "weekly-numbers" in report_names

    connects = manifest["required_connects"]
    shopify = next(c for c in connects if c["app_name"] == "SHOPIFY")
    # The honest two-step: connect now → app install → Site in Settings → sync.
    note = shopify["note"].lower()
    assert "site" in note and "settings" in note and "sync" in note


def test_guide_steps_are_the_three_step_flow():
    for pkg in PACKAGES:
        steps = pkg["setup_manifest"]["guide_steps"]
        assert [s["step"] for s in steps] == [1, 2, 3]  # D7 three steps
        # Step 2 is the guided connect (never auto-connect — FR-4).
        assert "connect" in steps[1]["description"].lower()


def test_development_connects_github_then_optional_shopify():
    connects = DEVELOPMENT_PACKAGE["setup_manifest"]["required_connects"]
    apps = [c["app_name"] for c in connects]
    assert "GITHUB" in apps
    assert apps[0] == "GITHUB"  # primary connect for a dev team


# --------------------------------------------------------------------------- #
# Idempotency (AC: registers both idempotently, no dupes)
# --------------------------------------------------------------------------- #


class _FakeQuery:
    def __init__(self, store: dict):
        self._store = store
        self._slug = None

    def filter_by(self, **kw):
        self._slug = kw.get("slug")
        return self

    def one_or_none(self):
        return self._store.get(self._slug)


class _FakeSession:
    """Just enough of a Session for seed_packages: query/filter_by/one_or_none,
    add, commit — keyed by slug so a second seed run finds the existing row."""

    def __init__(self):
        self.store: dict = {}
        self.commits = 0

    def query(self, _model):
        return _FakeQuery(self.store)

    def add(self, obj):
        self.store[obj.slug] = obj

    def commit(self):
        self.commits += 1

    def rollback(self):  # pragma: no cover - not hit on the happy path
        pass

    def close(self):  # pragma: no cover
        pass


def test_seed_is_idempotent_no_duplicates():
    db = _FakeSession()

    created, updated = seed_packages(db)
    assert (created, updated) == (2, 0)
    assert set(db.store) == {"shopify-management", "shopify-development"}

    # Second run: every package already present → updated, nothing created.
    created2, updated2 = seed_packages(db)
    assert (created2, updated2) == (0, 2)
    assert len(db.store) == 2  # no dupes
