"""Harbourline fixes — workspace base-model seeding + key-availability routing.

2026-08-29 (Gerard's design): every workspace starts with working models; the
factory routes vendor models via OpenRouter when no vendor key exists; the
trial is a spend cap (pinning deleted — covered in the trial test files).
"""
from __future__ import annotations

import pytest


# --------------------------------------------------------------------------- #
# _openrouter_model_id — vendor id -> OpenRouter form (pure)
# --------------------------------------------------------------------------- #

def _factory():
    from modules.agents.factory.agent_factory import AgentFactory
    return AgentFactory(db_session=None)


def test_vendor_model_gains_openrouter_prefix():
    f = _factory()
    assert f._openrouter_model_id("openai", "gpt-4o-mini") == "openai/gpt-4o-mini"
    assert f._openrouter_model_id("anthropic", "claude-haiku-4-5") == "anthropic/claude-haiku-4-5"


def test_prefixed_model_untouched():
    f = _factory()
    assert f._openrouter_model_id("openai", "openai/gpt-4o-mini") == "openai/gpt-4o-mini"


def test_unknown_vendor_passes_through():
    f = _factory()
    assert f._openrouter_model_id("huggingface", "some-model") == "some-model"


# --------------------------------------------------------------------------- #
# seed_workspace_models — selection, idempotency, primary (fake session)
# --------------------------------------------------------------------------- #

class _Model:
    def __init__(self, id, model_id, default=False, featured=False, pop=0):
        self.id = id
        self.model_id = model_id
        self.is_default = default
        self.is_featured = featured
        self.popularity_score = pop


class _Q:
    """Chainable fake for the service's two query shapes."""
    def __init__(self, rows):
        self._rows = rows

    def filter(self, *_a, **_k):
        return self

    def order_by(self, *_a):
        return self

    def limit(self, _n):
        return self

    def all(self):
        return list(self._rows)


class _DB:
    def __init__(self, models, existing_ids=(), workspace=None):
        self._models = models
        self._existing = [type("R", (), {"model_id": i})() for i in existing_ids]
        self._ws = workspace
        self.added = []

    def query(self, target):
        from core.models.core import LLMModel, WorkspaceModel
        if target is LLMModel:
            class _BaseQ(_Q):
                def __init__(qself):
                    super().__init__([])
                def filter(qself, *_a, **_k):
                    return qself
                def all(qself):
                    return []
            # The service filters defaults first, then featured — emulate by
            # returning defaults for the first .all(), featured for the next.
            db = self

            class _ModelQ:
                def __init__(qself):
                    qself._mode = None
                def filter(qself, *args, **_k):
                    return qself
                def order_by(qself, *_a):
                    return qself
                def limit(qself, _n):
                    return qself
                def all(qself):
                    calls = getattr(db, "_model_calls", 0)
                    db._model_calls = calls + 1
                    if calls == 0:
                        return [m for m in db._models if m.is_default]
                    return [m for m in db._models if m.is_featured]
            return _ModelQ()
        if target is WorkspaceModel.model_id or getattr(target, "key", "") == "model_id":
            return _Q(self._existing)
        return _Q(self._existing)

    def add(self, obj):
        self.added.append(obj)

    def get(self, _id):  # pragma: no cover
        return self._ws


class _WS:
    def __init__(self):
        self.id = "ws-1"
        self.settings = None


def _run_seed(db, ws):
    import services.workspace_model_seeding as m

    real_get = db.query
    # patch Workspace lookup inside the service
    class _WSQ:
        def get(self, _id):
            return ws
    orig_query = db.query
    def query(target):
        from core.models.workspaces import Workspace
        if target is Workspace:
            return _WSQ()
        return orig_query(target)
    db.query = query
    return m.seed_workspace_models(db, ws.id)


def test_seeds_defaults_then_featured_capped_and_sets_primary():
    from core.models.core import WorkspaceModel

    models = [
        _Model(1, "openai/gpt-4o-mini", default=True),
        _Model(2, "anthropic/claude-haiku-4-5", featured=True, pop=90),
        _Model(3, "deepseek/deepseek-chat", featured=True, pop=80),
        _Model(4, "meta-llama/llama-3-70b", featured=True, pop=70),
        _Model(5, "qwen/qwen-2", featured=True, pop=60),
    ]
    db = _DB(models)
    ws = _WS()
    primary = _run_seed(db, ws)

    added_models = [a for a in db.added if isinstance(a, WorkspaceModel)]
    assert len(added_models) == 4  # capped
    assert all(a.source == "default" and a.approval_status == "approved" for a in added_models)
    assert primary == "openai/gpt-4o-mini"  # first default pick
    assert ws.settings["orchestrator"]["model"] == "openai/gpt-4o-mini"


def test_existing_rows_not_duplicated_and_primary_kept():
    from core.models.core import WorkspaceModel

    models = [_Model(1, "openai/gpt-4o-mini", default=True)]
    db = _DB(models, existing_ids=(1,))
    ws = _WS()
    ws.settings = {"orchestrator": {"model": "user-chosen"}}
    primary = _run_seed(db, ws)

    assert [a for a in db.added if isinstance(a, WorkspaceModel)] == []  # idempotent
    assert primary == "user-chosen"  # user's choice never overwritten


def test_no_base_models_degrades_without_raising():
    db = _DB([])
    ws = _WS()
    assert _run_seed(db, ws) is None  # logged, degraded, never raises
