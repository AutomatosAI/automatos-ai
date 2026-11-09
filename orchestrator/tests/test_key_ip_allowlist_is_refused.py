"""An SDK key's IP allow-list is refused until something enforces it.

``allowed_ips`` was accepted by ``POST /api/api-keys``, stored and returned, but no key
check read it: a key whose owner limited it to their own addresses worked from anywhere.
Until the allow-list is enforced, a non-empty one is refused (422), and the key service
writes none.
"""
from __future__ import annotations

import inspect

import pytest
from pydantic import ValidationError

from api.api_keys import IP_ALLOWLIST_REFUSAL, ApiKeyCreateRequest


@pytest.mark.parametrize("ips", [["203.0.113.7"], ["10.0.0.0/8", "198.51.100.1"]])
def test_a_key_with_an_ip_allowlist_is_refused(ips):
    with pytest.raises(ValidationError) as refused:
        ApiKeyCreateRequest(name="Acme billing backend", key_type="server", allowed_ips=ips)
    assert IP_ALLOWLIST_REFUSAL in str(refused.value)


@pytest.mark.parametrize("ips", [None, []])
def test_a_key_without_one_is_created_as_before(ips):
    body = ApiKeyCreateRequest(name="Acme billing backend", key_type="server", allowed_ips=ips)
    assert not body.allowed_ips


def test_the_key_service_writes_no_ip_allowlist():
    from core.services.api_key_service import ApiKeyService

    assert "allowed_ips" not in inspect.signature(ApiKeyService.create_api_key).parameters


def test_the_create_route_validates_its_body_with_this_model():
    """The refusal is the route's own body validation (FastAPI answers 422)."""
    from api import api_keys

    assert inspect.signature(api_keys.create_api_key).parameters["body"].annotation is ApiKeyCreateRequest
