"""POST /api/widgets/auth: a session token never carries a permission its key does not grant.

The exchange's subset check skipped keys that list no permissions ("an empty key
permissions list means unrestricted"), so a server key granting nothing could ask for
any permissions, "*" included, and the signed token carried them. And a key listing
"*" was refused an explicit subset ("chat"), because a plain set difference does not
read the wildcard. Each requested permission is now judged by the platform's one
authority, ``modules.policy.roles.has_permission`` (PRD-195 S1): empty = deny, "*" = all.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from fastapi import HTTPException

from api.widgets.session import SessionTokenRequest, _exchange_session_token_permissions

ORCH = Path(__file__).resolve().parent.parent


def _effective(key_permissions, requested):
    key = NS(permissions=key_permissions)
    request = SessionTokenRequest(api_key="ak_srv_test", permissions=requested)
    return _exchange_session_token_permissions(key, request)


@pytest.mark.parametrize("key_permissions", [[], None])
@pytest.mark.parametrize("requested", [["*"], ["chat"], ["chat", "documents:read"]])
def test_a_key_that_grants_nothing_cannot_mint_a_token_that_grants_something(key_permissions, requested):
    with pytest.raises(HTTPException) as refused:
        _effective(key_permissions, requested)
    assert refused.value.status_code == 403


def test_a_key_is_never_widened_by_the_request():
    with pytest.raises(HTTPException) as refused:
        _effective(["chat"], ["chat", "documents:read"])
    assert refused.value.status_code == 403 and "documents:read" in refused.value.detail


def test_a_subset_of_the_key_is_granted():
    assert _effective(["chat", "documents:read"], ["chat"]) == ["chat"]


def test_a_wildcard_key_grants_an_explicit_subset():
    assert _effective(["*"], ["chat"]) == ["chat"]


def test_asking_for_nothing_in_particular_gets_the_keys_own():
    assert _effective(["chat"], None) == ["chat"]
    assert _effective([], None) == []          # nothing, which every route then refuses


def test_the_exchange_has_no_unrestricted_arm_left():
    source = (ORCH / "api" / "widgets" / "session.py").read_text(encoding="utf-8")
    assert "means unrestricted" not in source and "has_permission" in source
