"""PRD-230 US-003 — marketplace_packages model + the pure signal matcher.

PURE tests: the matcher takes plain objects, so no Postgres is needed. The model
and the single-head migration are asserted structurally (import + source grep) —
the real-Postgres schema check is CI's job.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from services.marketplace_packages import PackageMatch, match_by_signals

REPO = Path(__file__).resolve().parents[1]


def _pkg(slug, *, platforms=None, url_patterns=None, vocabulary=None,
         vertical_tags=None, showcase=False):
    return SimpleNamespace(
        slug=slug,
        name=slug.replace("-", " ").title(),
        showcase=showcase,
        vertical_tags=vertical_tags or [],
        matching={
            "platforms": platforms or [],
            "url_patterns": url_patterns or [],
            "vocabulary": vocabulary or [],
        },
    )


SHOPIFY_MGMT = _pkg(
    "shopify-management",
    platforms=["shopify"],
    url_patterns=["myshopify.com"],
    vocabulary=["store", "inventory", "orders", "customers"],
    vertical_tags=["shopify", "ecommerce"],
    showcase=True,
)
SHOPIFY_DEV = _pkg(
    "shopify-development",
    platforms=["shopify"],
    url_patterns=["myshopify.com"],
    vocabulary=["theme", "liquid", "storefront"],
    vertical_tags=["shopify", "ecommerce"],
    showcase=True,
)
SUPPORT = _pkg(
    "support-desk",
    platforms=["zendesk"],
    vocabulary=["ticket", "helpdesk"],
    vertical_tags=["support"],
)
ALL = [SHOPIFY_MGMT, SHOPIFY_DEV, SUPPORT]


# --------------------------------------------------------------------------- #
# The canonical AC: shopify signals → shopify packages ranked
# --------------------------------------------------------------------------- #


def test_shopify_signals_rank_shopify_packages_first():
    signals = {
        "platforms": ["shopify"],
        "urls": ["acme.myshopify.com"],
        "text": "we run an online store and want inventory and orders handled",
    }
    matches = match_by_signals(signals, ALL)
    slugs = [m.package.slug for m in matches]
    # Both shopify packages match; the non-shopify one is excluded entirely.
    assert "shopify-management" in slugs
    assert "shopify-development" in slugs
    assert "support-desk" not in slugs
    # Shopify packages outrank anything else.
    assert slugs[0].startswith("shopify")


def test_management_outranks_dev_on_run_the_store_vocabulary():
    # A store-runner's words (inventory/orders/customers) lift Management above Dev.
    signals = {"platforms": ["shopify"], "text": "manage inventory, orders and customers"}
    matches = match_by_signals(signals, [SHOPIFY_DEV, SHOPIFY_MGMT])
    assert matches[0].package.slug == "shopify-management"
    assert matches[0].score > matches[1].score


def test_no_signal_match_returns_empty():
    assert match_by_signals({"text": "quantum astrophysics research lab"}, ALL) == []


def test_matcher_is_pure_and_deterministic():
    signals = {"platforms": ["shopify"]}
    a = [m.package.slug for m in match_by_signals(signals, ALL)]
    b = [m.package.slug for m in match_by_signals(signals, list(reversed(ALL)))]
    assert a == b  # order is signal-driven, not input-order driven


def test_plain_string_signal_is_accepted_as_free_text():
    matches = match_by_signals("looking for a shopify store setup", ALL)
    assert matches and matches[0].package.slug.startswith("shopify")


def test_url_pattern_alone_matches():
    matches = match_by_signals({"urls": ["https://acme.myshopify.com/admin"]}, ALL)
    assert {m.package.slug for m in matches} == {"shopify-management", "shopify-development"}


def test_reasons_explain_the_match():
    m = match_by_signals({"platforms": ["shopify"]}, [SHOPIFY_MGMT])[0]
    assert isinstance(m, PackageMatch)
    assert any(r.startswith("platform:shopify") for r in m.reasons)


def test_showcase_breaks_score_ties_before_slug():
    # Two equal-scoring packages, one showcased → showcased first.
    a = _pkg("z-showcased", platforms=["x"], showcase=True)
    b = _pkg("a-plain", platforms=["x"], showcase=False)
    matches = match_by_signals({"platforms": ["x"]}, [b, a])
    assert [m.package.slug for m in matches] == ["z-showcased", "a-plain"]


# --------------------------------------------------------------------------- #
# Model + migration structure (real-Postgres schema check is CI's job)
# --------------------------------------------------------------------------- #


def test_model_columns_and_defaults():
    from core.models.marketplace_packages import MEMBER_TYPES, MarketplacePackage

    cols = {c.name for c in MarketplacePackage.__table__.columns}
    assert {
        "id", "slug", "name", "description", "vertical_tags", "matching",
        "members", "setup_manifest", "showcase", "created_at", "updated_at",
    } <= cols
    assert MarketplacePackage.__tablename__ == "marketplace_packages"
    assert MarketplacePackage.__table__.c.slug.unique is True
    assert set(MEMBER_TYPES) == {"agent", "tool", "skill", "plugin", "playbook", "llm"}


def test_to_dict_shape():
    from core.models.marketplace_packages import MarketplacePackage

    p = MarketplacePackage(
        slug="s", name="S", description="d",
        vertical_tags=["shopify"], matching={"platforms": ["shopify"]},
        members=[{"type": "agent", "ref": "a"}], setup_manifest={"questions": []},
        showcase=True,
    )
    d = p.to_dict()
    assert d["slug"] == "s" and d["showcase"] is True
    assert d["members"] == [{"type": "agent", "ref": "a"}]
    assert d["vertical_tags"] == ["shopify"]


def test_exactly_one_new_migration_creates_the_table():
    versions = REPO / "alembic" / "versions"
    creators = [
        f for f in versions.glob("*.py")
        if "marketplace_packages" in f.read_text() and "CREATE TABLE" in f.read_text()
    ]
    assert len(creators) == 1, f"expected exactly one creating migration, got {creators}"
    src = creators[0].read_text()
    assert 'down_revision = "prd225_s1_asks_on_grants"' in src  # chained onto the single head
    assert "IF NOT EXISTS" in src  # idempotent / additive
    assert "gen_random_uuid()" in src
