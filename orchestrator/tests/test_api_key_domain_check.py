"""Regression tests for widget API-key origin matching.

Covers the scheme-mismatch bug where a request origin (bare host, as returned
by ``_extract_origin``) could never match an allow-list entry stored with a
scheme (``https://*.myshopify.com``), causing every Bearer-key blog/chat
request from a real storefront to 403.
"""

from types import SimpleNamespace

import pytest

from api.shopify import _build_allowed_domains
from core.services.api_key_service import ApiKeyService


def _key(domains):
    return SimpleNamespace(allowed_domains=domains)


class TestCheckDomainSchemeInsensitive:
    def test_bare_host_matches_scheme_prefixed_wildcard(self):
        # The actual prod failure: bare host vs https://*.myshopify.com.
        key = _key(["https://*.myshopify.com"])
        assert ApiKeyService.check_domain(key, "besafe-ltd.myshopify.com") is True

    def test_bare_host_matches_exact_scheme_prefixed_entry(self):
        key = _key(["https://besafe-ltd.myshopify.com"])
        assert ApiKeyService.check_domain(key, "besafe-ltd.myshopify.com") is True

    def test_trailing_slash_in_pattern_is_ignored(self):
        key = _key(["https://www.inbuilduk.com/"])
        assert ApiKeyService.check_domain(key, "www.inbuilduk.com") is True

    def test_scheme_prefixed_origin_also_matches(self):
        key = _key(["https://*.myshopify.com"])
        assert ApiKeyService.check_domain(key, "https://x.myshopify.com") is True

    def test_non_matching_origin_is_rejected(self):
        key = _key(["https://*.myshopify.com"])
        assert ApiKeyService.check_domain(key, "evil.example.com") is False

    def test_empty_allow_list_permits_all(self):
        assert ApiKeyService.check_domain(_key([]), "anything.example.com") is True
        assert ApiKeyService.check_domain(_key(None), "anything.example.com") is True


class TestBuildAllowedDomains:
    def test_myshopify_only_when_no_custom_domain(self):
        domains = _build_allowed_domains("shop.myshopify.com", {})
        assert domains == [
            "https://shop.myshopify.com",
            "https://*.shop.myshopify.com",
            "https://*.myshopify.com",
        ]

    def test_custom_www_domain_adds_apex_and_wildcard(self):
        domains = _build_allowed_domains(
            "inbuild-uk.myshopify.com", {"domain": "https://www.inbuilduk.com"}
        )
        assert "https://www.inbuilduk.com" in domains
        assert "https://inbuilduk.com" in domains
        assert "https://*.inbuilduk.com" in domains

    def test_custom_domain_origins_pass_check_domain(self):
        domains = _build_allowed_domains(
            "inbuild-uk.myshopify.com", {"domain": "https://www.inbuilduk.com"}
        )
        key = _key(domains)
        for origin in ("www.inbuilduk.com", "inbuilduk.com", "shop.inbuilduk.com"):
            assert ApiKeyService.check_domain(key, origin) is True

    def test_no_duplicate_entries(self):
        domains = _build_allowed_domains(
            "inbuild-uk.myshopify.com", {"domain": "https://www.inbuilduk.com"}
        )
        assert len(domains) == len(set(domains))

    def test_myshopify_primary_domain_does_not_duplicate(self):
        # primaryDomain == the myshopify domain → no custom entries added.
        domains = _build_allowed_domains(
            "shop.myshopify.com", {"domain": "https://shop.myshopify.com"}
        )
        assert domains == [
            "https://shop.myshopify.com",
            "https://*.shop.myshopify.com",
            "https://*.myshopify.com",
        ]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
