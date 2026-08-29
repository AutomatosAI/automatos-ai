"""BudStacks vertical provisioning — unit tests.

Covers the three behaviours the BudStacks vertical adds to the generic
provisioning plane (PRD-183 S5 seam, extended):

* the required, normalized origin allowlist (``VerticalConfigError`` on empty);
* ``reuse_existing_key`` — a re-provision against a workspace holding an active
  public key must NOT rotate it (``api_key: None`` / ``key_minted: False``),
  while a fresh workspace still mints;
* the per-vertical internal-key verifier's fail-closed triad
  (unset ⇒ 503 dark, wrong ⇒ 401, correct ⇒ pass).

No real DB: the provision flow is exercised through a minimal fake Session,
with ``_create_widget_key`` monkeypatched to count mints — the same seams the
PRD-183 S5 suite patches.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

import integrations.budstacks  # noqa: F401 — registers the provisioner
from integrations.budstacks.provision import provisioner as budstacks_provisioner
from integrations.provisioning import (
    PROVISIONER_REGISTRY,
    VerticalConfigError,
    provision_vertical,
)


# ── Registration + declaration ──────────────────────────────────────────


def test_budstacks_provisioner_registered():
    assert PROVISIONER_REGISTRY.get("budstacks") is budstacks_provisioner
    assert budstacks_provisioner.reuse_existing_key is True
    assert budstacks_provisioner.key_permissions == ["chat"]
    assert budstacks_provisioner.external_id_key == "budstacks_tenant_id"


# ── allowed_domains ─────────────────────────────────────────────────────


def test_allowed_domains_normalizes_and_dedupes():
    domains = budstacks_provisioner.allowed_domains(
        "tenant-1",
        {
            "domains": [
                "healingbuds.budstacks.io",
                "https://healingbuds.co.za/",
                "www.healingbuds.co.za",
                "healingbuds.budstacks.io",  # duplicate
            ]
        },
    )
    assert domains == [
        "https://healingbuds.budstacks.io",
        "https://healingbuds.co.za",
        "https://www.healingbuds.co.za",
    ]


@pytest.mark.parametrize("metadata", [{}, {"domains": []}, {"domains": [42, ""]}])
def test_allowed_domains_required(metadata):
    with pytest.raises(VerticalConfigError):
        budstacks_provisioner.allowed_domains("tenant-1", metadata)


# ── provision flow: reuse_existing_key ──────────────────────────────────


class _FakeQuery:
    def __init__(self, first_result=None, count_result=0, all_result=None):
        self._first = first_result
        self._count = count_result
        self._all = all_result or []

    def filter(self, *args, **kwargs):
        return self

    def order_by(self, *args, **kwargs):
        return self

    def first(self):
        return self._first

    def count(self):
        return self._count

    def all(self):
        return self._all


class _FakeSession:
    """Answers the three model queries the provision flow makes, in a fixed
    per-model mapping; add/flush/commit are no-ops."""

    def __init__(self, results_by_model):
        self._results = results_by_model
        self.added = []

    def query(self, model):
        return self._results.get(model.__name__, _FakeQuery())

    def add(self, obj):
        self.added.append(obj)

    def flush(self):
        pass

    def commit(self):
        pass


def _mint_counter(calls):
    def _fake_create_widget_key(**kwargs):
        calls.append(kwargs)
        return {"key": f"ak_pub_test{len(calls)}"}

    return _fake_create_widget_key


def test_reprovision_does_not_rotate_existing_key(monkeypatch):
    from core.models.workspaces import Workspace

    existing_workspace = Workspace(name="HealingBuds", slug="healingbuds")
    existing_workspace.id = "11111111-1111-1111-1111-111111111111"

    class _ExistingKey:  # only identity matters — presence short-circuits mint
        pass

    db = _FakeSession(
        {
            "Workspace": _FakeQuery(first_result=existing_workspace),
            "Agent": _FakeQuery(count_result=1),  # roster already seeded
            "SdkApiKey": _FakeQuery(first_result=_ExistingKey()),
        }
    )

    calls: list = []
    monkeypatch.setattr(
        "integrations.provisioning._create_widget_key", _mint_counter(calls)
    )

    result = provision_vertical(
        db=db,
        vertical="budstacks",
        external_id="tenant-1",
        name="HealingBuds",
        metadata={"domains": ["healingbuds.budstacks.io"]},
    )

    assert calls == []  # no rotation
    assert result["api_key"] is None
    assert result["key_minted"] is False
    assert result["is_new"] is False


def test_fresh_workspace_still_mints(monkeypatch):
    db = _FakeSession(
        {
            "Workspace": _FakeQuery(first_result=None),
            "Agent": _FakeQuery(count_result=1),  # skip roster seeding
            "SdkApiKey": _FakeQuery(first_result=None),
        }
    )

    calls: list = []
    monkeypatch.setattr(
        "integrations.provisioning._create_widget_key", _mint_counter(calls)
    )

    result = provision_vertical(
        db=db,
        vertical="budstacks",
        external_id="tenant-2",
        name="LekkerWeed",
        metadata={"domains": ["lekkerweed.budstacks.io"]},
    )

    assert len(calls) == 1
    assert calls[0]["allowed_domains"] == ["https://lekkerweed.budstacks.io"]
    assert result["api_key"] == "ak_pub_test1"
    assert result["key_minted"] is True
    assert result["is_new"] is True


def test_missing_domains_rejects_before_any_key_decision(monkeypatch):
    from core.models.workspaces import Workspace

    existing_workspace = Workspace(name="HealingBuds", slug="healingbuds")
    existing_workspace.id = "11111111-1111-1111-1111-111111111111"

    db = _FakeSession(
        {
            "Workspace": _FakeQuery(first_result=existing_workspace),
            "Agent": _FakeQuery(count_result=1),
        }
    )

    calls: list = []
    monkeypatch.setattr(
        "integrations.provisioning._create_widget_key", _mint_counter(calls)
    )

    with pytest.raises(VerticalConfigError):
        provision_vertical(
            db=db,
            vertical="budstacks",
            external_id="tenant-1",
            name="HealingBuds",
            metadata={},
        )
    assert calls == []


# ── per-vertical internal key verifier ──────────────────────────────────


def test_vertical_key_verifier_triad(monkeypatch):
    from config import config as app_config
    from api.verticals import _verify_vertical_internal_key

    # Unset ⇒ 503 (dark, never open)
    monkeypatch.setattr(app_config, "BUDSTACKS_INTERNAL_API_KEY", "", raising=False)
    with pytest.raises(HTTPException) as exc:
        _verify_vertical_internal_key("budstacks", authorization="Bearer whatever")
    assert exc.value.status_code == 503

    # Wrong ⇒ 401
    monkeypatch.setattr(
        app_config, "BUDSTACKS_INTERNAL_API_KEY", "s3cret", raising=False
    )
    with pytest.raises(HTTPException) as exc:
        _verify_vertical_internal_key("budstacks", authorization="Bearer nope")
    assert exc.value.status_code == 401

    # Correct ⇒ passes
    assert (
        _verify_vertical_internal_key("budstacks", authorization="Bearer s3cret")
        is None
    )

    # Unknown vertical ⇒ 503 (no key configured for it)
    with pytest.raises(HTTPException) as exc:
        _verify_vertical_internal_key("nonexistent", authorization="Bearer s3cret")
    assert exc.value.status_code == 503
