"""PRD-230 US-007 — marketplace Packages read/install HTTP surface.

PURE: the router imports with no DB dial (``create_engine`` is lazy), so these
lock the three package routes' paths/methods, the install route's permission
dependency, and that ``PackageOut`` validates the model's ``to_dict()`` shape
1:1. The install route's *behaviour* (D6 one-package-during-onboarding, D9
over-quota, the returned manifest) is covered by the US-006 tool tests it
delegates to — here we only guard the wiring so the UI contract can't drift.
"""
from __future__ import annotations

import uuid


def _package_routes() -> set:
    from api.marketplace import router

    return {
        (r.path, tuple(sorted(m for m in r.methods if m in {"GET", "POST"})))
        for r in router.routes
        if "/packages" in getattr(r, "path", "")
    }


def test_package_routes_registered_with_expected_methods():
    routes = _package_routes()
    assert ("/api/marketplace/packages", ("GET",)) in routes
    assert ("/api/marketplace/packages/{slug}", ("GET",)) in routes
    assert ("/api/marketplace/packages/{slug}/install", ("POST",)) in routes


def test_install_route_carries_a_permission_dependency():
    """The install route reuses the marketplace install permission (agents:create)
    — a browser click must be authorised exactly like the agent-install route."""
    from api.marketplace import router

    install = next(
        r
        for r in router.routes
        if getattr(r, "path", "") == "/api/marketplace/packages/{slug}/install"
    )
    assert install.dependencies, "install route must carry a permission dependency"


def test_package_out_validates_model_to_dict_one_to_one():
    from api.marketplace import PackageOut
    from core.models.marketplace_packages import MarketplacePackage

    pkg = MarketplacePackage(
        slug="shopify-management",
        name="Shopify Management",
        description="Run the store",
        vertical_tags=["shopify", "ecommerce"],
        matching={"platforms": ["shopify"]},
        members=[{"type": "agent", "ref": "store-manager"}],
        setup_manifest={"required_connects": [{"app_name": "SHOPIFY"}]},
        showcase=True,
    )
    pkg.id = uuid.uuid4()

    out = PackageOut(**pkg.to_dict())  # the exact payload the route returns
    assert out.slug == "shopify-management"
    assert out.showcase is True
    assert out.members == [{"type": "agent", "ref": "store-manager"}]
    assert out.setup_manifest["required_connects"][0]["app_name"] == "SHOPIFY"
    assert out.vertical_tags == ["shopify", "ecommerce"]
