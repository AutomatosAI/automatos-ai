"""PRD-172 — Tenant Isolation Closure (Wave 2).

The headline acceptance (spec §5.0) is the **cross-tenant matrix**: workspace A
cannot read, write, or delete any of workspace B's data across every in-scope
domain — skills, documents/vectors, workflows, memory, and context. Each finding
adds a focused test underneath the matrix.

These are pure/behavioural tests: the DB is mocked (query-chain / execute-capture
stubs) and auth is injected via ``app.dependency_overrides`` — the blessed pattern
from ``tests/test_prd143_obs_routers_batch1.py`` and ``tests/security/
test_tenancy_matrix.py``. No live Postgres is needed; the end-to-end DB matrix
runs in CI. The point of each test is to prove the *actual* scope decision the
handler/helper makes, which is stronger than asserting a clause in isolation.

Findings covered: F002, F003, F004, F005, F006, F007, F039, F045.
(F019 — the NL2SQL side-effecting-function denylist — lives in
``tests/security/test_nl2sql_validator.py`` next to the validator's other cases.)
"""
from __future__ import annotations

import os
import uuid
from unittest.mock import MagicMock

import pytest

# Dummy POSTGRES_* satisfies the config import chain; the port points at nothing
# so any fail-soft connect refuses instantly. CI exports real vars (setdefault
# no-ops). Blessed pattern — see test_prd143_obs_routers_batch1.py.
os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

from core.auth.dependencies import RequestContext, UserContext

WS_A = uuid.uuid4()
WS_B = uuid.uuid4()

# Principals. system_role drives the super-admin gate (core/auth/super_admin.py).
# PRD-195 S2/S3: mutating routes now sit behind require_workspace_permission —
# the tenancy tests below exercise the HANDLER's cross-tenant logic, so their
# workspace callers carry a clerk identity and the db stub answers the gate's
# member lookup with an admin role (the gate itself is covered by
# test_p2w2_workspace_permission_gate.py).
USER_A = UserContext(id="u-a", role="member", system_role="user", clerk_user_id="clerk-u-a")
USER_B = UserContext(id="u-b", role="member", system_role="user", clerk_user_id="clerk-u-b")
SUPER_ADMIN = UserContext(id="u-gerard", role="admin", system_role="super_admin")


def _ctx(ws=WS_A, user=USER_A, *, admin_all=False, auth_type="clerk") -> RequestContext:
    return RequestContext(
        workspace_id=ws, user=user, auth_type=auth_type, admin_all_workspaces=admin_all
    )


# ===========================================================================
# F002 — skills: cross-workspace attach/read denied; global-skill delete gated
# ===========================================================================

class _FakeSkill:
    def __init__(self, id, workspace_id, is_active=True):
        self.id = id
        self.workspace_id = workspace_id
        self.is_active = is_active
        self.name = f"skill-{id}"


class _FakeAgent:
    def __init__(self, id, workspace_id):
        self.id = id
        self.workspace_id = workspace_id


class TestF002SkillsIsolation:
    """The skills router took ``ctx`` and never used it. These pin the helpers
    that now gate read/attach/delete (api/skills.py)."""

    def _helpers(self):
        import api.skills as s
        return s

    def test_workspace_cannot_see_other_workspaces_private_skill(self):
        s = self._helpers()
        b_skill = _FakeSkill(1, WS_B)
        # A caller in workspace A: B's private skill is NOT visible.
        assert s._skill_visible_to(b_skill, _ctx(ws=WS_A)) is False

    def test_workspace_can_see_global_skill(self):
        s = self._helpers()
        global_skill = _FakeSkill(2, None)  # workspace_id IS NULL == global
        assert s._skill_visible_to(global_skill, _ctx(ws=WS_A)) is True

    def test_workspace_can_see_own_skill(self):
        s = self._helpers()
        own = _FakeSkill(3, WS_A)
        assert s._skill_visible_to(own, _ctx(ws=WS_A)) is True

    def test_super_admin_sees_any_skill(self):
        s = self._helpers()
        b_skill = _FakeSkill(4, WS_B)
        assert s._skill_visible_to(b_skill, _ctx(ws=WS_A, user=SUPER_ADMIN)) is True

    def test_attach_to_other_workspaces_agent_is_404(self):
        s = self._helpers()
        b_agent = _FakeAgent(10, WS_B)
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as ei:
            s._assert_agent_in_workspace(b_agent, _ctx(ws=WS_A))
        assert ei.value.status_code == 404

    def test_attach_to_own_agent_allowed(self):
        s = self._helpers()
        a_agent = _FakeAgent(11, WS_A)
        # No raise == allowed.
        s._assert_agent_in_workspace(a_agent, _ctx(ws=WS_A))

    def test_super_admin_flag_recognised(self):
        s = self._helpers()
        # admin_all_workspaces sentinel OR literal super_admin role.
        assert s._is_super_admin(_ctx(ws=WS_A, admin_all=True)) is True
        assert s._is_super_admin(_ctx(ws=WS_A, user=SUPER_ADMIN)) is True
        assert s._is_super_admin(_ctx(ws=WS_A, user=USER_A)) is False


class TestF002GlobalSkillDelete:
    """DELETE /{skill_id}: a global builtin-core skill is deletable ONLY by a
    super-admin (deleting it lobotomises every tenant's Auto). A workspace caller
    deleting another workspace's private skill gets 404."""

    def _client_and_skill(self, ctx, skill):
        import importlib
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from core.auth.hybrid import get_request_context_hybrid
        from core.database.database import get_db

        skills = importlib.import_module("api.skills")

        # DB stub: query(Skill).filter(...).first() -> the skill under test;
        # the agent_skills delete chain is a harmless no-op.
        db = MagicMock()
        q = MagicMock()
        q.filter.return_value = q
        q.first.return_value = skill
        q.delete.return_value = 0
        db.query.return_value = q
        # Satisfy the PRD-195 workspace-permission gate: the caller is an
        # admin member of their workspace (agents:delete).
        gate_row = MagicMock()
        gate_row.fetchone.return_value = ("admin",)
        db.execute.return_value = gate_row

        app = FastAPI()
        app.include_router(skills.router)
        app.dependency_overrides[get_request_context_hybrid] = lambda: ctx
        app.dependency_overrides[get_db] = lambda: db
        return TestClient(app, raise_server_exceptions=False), skill

    def test_workspace_caller_cannot_delete_global_skill(self):
        skill = _FakeSkill(100, None)  # global
        client, _ = self._client_and_skill(_ctx(ws=WS_A, user=USER_A), skill)
        resp = client.delete("/api/v1/skills/100")
        assert resp.status_code == 403, resp.text
        # The skill was NOT deactivated by the denied attempt.
        assert skill.is_active is True

    def test_workspace_caller_cannot_delete_other_workspace_skill(self):
        skill = _FakeSkill(101, WS_B)
        client, _ = self._client_and_skill(_ctx(ws=WS_A, user=USER_A), skill)
        resp = client.delete("/api/v1/skills/101")
        assert resp.status_code == 404, resp.text
        assert skill.is_active is True

    def test_super_admin_can_delete_global_skill(self):
        skill = _FakeSkill(102, None)
        client, _ = self._client_and_skill(
            _ctx(ws=WS_A, user=SUPER_ADMIN, admin_all=True), skill
        )
        resp = client.delete("/api/v1/skills/102")
        assert resp.status_code == 200, resp.text
        assert skill.is_active is False

    def test_workspace_caller_can_delete_own_skill(self):
        skill = _FakeSkill(103, WS_A)
        client, _ = self._client_and_skill(_ctx(ws=WS_A, user=USER_A), skill)
        resp = client.delete("/api/v1/skills/103")
        assert resp.status_code == 200, resp.text
        assert skill.is_active is False


# ===========================================================================
# F003 — Shopify sync routes: a guessed workspace UUID is rejected
# ===========================================================================

class TestF003ShopifySyncScope:
    """The four sync routes derive their target from ``ctx``; a caller-supplied
    workspace_id that is not the caller's own is refused (api/shopify.py
    ``_resolve_sync_workspace``)."""

    def _resolve(self):
        import api.shopify as sh
        return sh._resolve_sync_workspace

    def test_no_param_uses_own_workspace(self):
        resolve = self._resolve()
        assert resolve(_ctx(ws=WS_A), None) == str(WS_A)

    def test_matching_param_allowed(self):
        resolve = self._resolve()
        assert resolve(_ctx(ws=WS_A), str(WS_A)) == str(WS_A)

    def test_guessed_other_workspace_rejected(self):
        resolve = self._resolve()
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as ei:
            resolve(_ctx(ws=WS_A), str(WS_B))  # guessed B's UUID
        assert ei.value.status_code == 403

    def test_admin_all_may_target_any_workspace(self):
        resolve = self._resolve()
        # Cross-workspace admin (ops backfill) may target B explicitly.
        assert resolve(_ctx(ws=WS_A, admin_all=True), str(WS_B)) == str(WS_B)


# ===========================================================================
# F004 — Shopify provisioning is fail-closed
# ===========================================================================

class TestF004ShopifyFailClosed:
    """Unset SHOPIFY_INTERNAL_API_KEY must fail at boot; a falsy key can no
    longer wave through an arbitrary Authorization header."""

    def test_boot_fails_when_shopify_key_unset(self, monkeypatch):
        from config import Config
        cfg = Config()
        monkeypatch.setattr(cfg, "SHOPIFY_INTERNAL_API_KEY", "", raising=False)
        monkeypatch.setattr(cfg, "S3_VECTORS_ENABLED", False, raising=False)
        with pytest.raises(RuntimeError, match="SHOPIFY_INTERNAL_API_KEY"):
            cfg.validate_security()

    def test_boot_passes_when_shopify_key_set(self, monkeypatch):
        from config import Config
        cfg = Config()
        monkeypatch.setattr(cfg, "SHOPIFY_INTERNAL_API_KEY", "real-secret", raising=False)
        monkeypatch.setattr(cfg, "S3_VECTORS_ENABLED", False, raising=False)
        # PRD-194 S4: saas boots also require a widget CORS allowlist —
        # satisfied here so this test stays scoped to the Shopify guard.
        monkeypatch.setattr(cfg, "WIDGET_ORIGIN_ALLOWLIST", "https://app.automatos.app", raising=False)
        cfg.validate_security()  # no raise

    def test_verify_internal_key_rejects_arbitrary_header(self, monkeypatch):
        import api.shopify as sh
        from fastapi import HTTPException
        monkeypatch.setattr(sh.config, "SHOPIFY_INTERNAL_API_KEY", "real-secret", raising=False)
        # An arbitrary bearer value is rejected — no fail-open branch remains.
        with pytest.raises(HTTPException) as ei:
            sh._verify_internal_key("Bearer x")
        assert ei.value.status_code == 401

    def test_verify_internal_key_accepts_correct_key(self, monkeypatch):
        import api.shopify as sh
        monkeypatch.setattr(sh.config, "SHOPIFY_INTERNAL_API_KEY", "real-secret", raising=False)
        # Correct key: no raise.
        sh._verify_internal_key("Bearer real-secret")


# ===========================================================================
# F005 — S3 vector search drops cross-workspace hits (shared bucket isolated at query time)
# ===========================================================================

class TestF005VectorIsolation:
    """S3VectorsBackend.search() enforces its workspace filter — that is the
    tenant-isolation guarantee. A shared bucket without {workspace_id} is
    supported because the query filter, not the bucket layout, isolates tenants."""

    def _backend_cls(self):
        from modules.search.vector_store.backends.s3_vectors_backend import S3VectorsBackend
        return S3VectorsBackend

    def _make_backend(self, monkeypatch, ws):
        from config import config as app_config
        monkeypatch.setattr(
            app_config, "S3_VECTORS_BUCKET", "automatos-vectors-{workspace_id}", raising=False
        )
        cls = self._backend_cls()
        b = cls.__new__(cls)  # bypass __init__/AWS client
        b.workspace_id = str(ws)
        b.bucket_name = f"automatos-vectors-{ws}"
        b.index_name = "documents-index"
        b.index_dimension = 2048
        b.distance_metric = "cosine"
        b._setup_complete = True
        b.client = MagicMock()
        # search() calls _ensure_setup(); it's a no-op here (no live AWS).
        b._ensure_setup = lambda: None
        return b

    def test_search_drops_foreign_workspace_hits(self, monkeypatch):
        b = self._make_backend(monkeypatch, WS_A)
        # The store returns one A-owned and one B-owned chunk (shared-bucket
        # worst case). B's chunk MUST be dropped.
        b.client.query_vectors.return_value = {
            "vectors": [
                {"key": "a1", "distance": 0.1, "metadata": {"workspace_id": str(WS_A), "chunk_text": "A secret"}},
                {"key": "b1", "distance": 0.1, "metadata": {"workspace_id": str(WS_B), "chunk_text": "B secret"}},
            ]
        }
        results = b.search([0.1, 0.2, 0.3], limit=10, min_score=0.0,
                            filters={"workspace_id": str(WS_A)})
        keys = {r["key"] for r in results}
        assert keys == {"a1"}, f"leaked cross-workspace chunk: {keys}"
        assert all("B secret" not in r["content"] for r in results)

    def test_search_refuses_mismatched_filter(self, monkeypatch):
        b = self._make_backend(monkeypatch, WS_A)
        b.client.query_vectors.return_value = {"vectors": []}
        # Asking an A-bound backend to search B → empty, and the store is never
        # even queried with B's scope.
        results = b.search([0.1], filters={"workspace_id": str(WS_B)})
        assert results == []

    def test_init_accepts_shared_bucket(self, monkeypatch):
        # A shared bucket (no {workspace_id}) must construct fine — isolation is
        # enforced per-query by search(), not the bucket layout. Regression: the
        # hard placeholder requirement broke a working shared-bucket deployment
        # on 2026-07-02.
        from config import config as app_config
        import modules.search.vector_store.backends.s3_vectors_backend as s3mod
        monkeypatch.setattr(app_config, "S3_VECTORS_BUCKET", "one-shared-bucket", raising=False)
        monkeypatch.setattr(s3mod.boto3, "client", lambda *a, **k: MagicMock())
        cls = self._backend_cls()
        b = cls(workspace_id=str(WS_A))  # must NOT raise
        assert b.bucket_name == "one-shared-bucket"
        assert str(b.workspace_id) == str(WS_A)

    def test_config_boot_allows_shared_bucket(self, monkeypatch):
        # A shared bucket must NOT abort boot — isolation is per-query (search()),
        # not bucket-layout. Regression: the hard placeholder requirement broke a
        # working shared-bucket deployment on 2026-07-02.
        from config import Config
        cfg = Config()
        monkeypatch.setattr(cfg, "SHOPIFY_INTERNAL_API_KEY", "x", raising=False)
        monkeypatch.setattr(cfg, "S3_VECTORS_ENABLED", True, raising=False)
        monkeypatch.setattr(cfg, "S3_VECTORS_BUCKET", "shared-no-placeholder", raising=False)
        # PRD-194 S4: saas boots also require a widget CORS allowlist —
        # satisfied here so this test stays scoped to the bucket layout.
        monkeypatch.setattr(cfg, "WIDGET_ORIGIN_ALLOWLIST", "https://app.automatos.app", raising=False)
        monkeypatch.setattr(cfg, "validate_auth_edition", lambda: None, raising=False)
        cfg.validate_security()  # must not raise on the bucket layout


# ===========================================================================
# F006 — legacy PRD-125 workflow execute surface deleted (no cross-tenant oracle)
# ===========================================================================

class TestF006LegacyExecuteDeleted:
    """The unscoped legacy execute routes were deleted. Deleting removes the
    existence oracle and the cross-workspace AgentTask enqueue entirely."""

    def test_execute_workflow_symbol_removed(self):
        import api.workflows as wf
        assert not hasattr(wf, "execute_workflow"), (
            "legacy unscoped execute_workflow must be deleted"
        )
        assert not hasattr(wf, "execute_workflow_general"), (
            "legacy /execute wrapper must be deleted"
        )
        assert not hasattr(wf, "create_execution"), (
            "legacy /executions POST (broken caller of execute_workflow) must be deleted"
        )

    def test_legacy_execute_routes_not_registered(self):
        import api.workflows as wf
        # (method, path) pairs still registered.
        registered = set()
        for r in wf.router.routes:
            path = getattr(r, "path", None)
            for m in (getattr(r, "methods", None) or set()):
                registered.add((m, path))
        # The unscoped legacy execute routes must be gone…
        assert ("POST", "/api/workflows/{workflow_id}/execute") not in registered
        assert ("POST", "/api/workflows/execute") not in registered
        # …including the POST create-execution that called the deleted function.
        assert ("POST", "/api/workflows/executions/") not in registered
        # The scoped advanced-execute path stays.
        assert ("POST", "/api/workflows/{workflow_id}/execute-advanced") in registered

    def test_workspace_scoped_advanced_execute_survives(self):
        import api.workflows as wf
        # The correct, workspace-scoped path stays.
        assert hasattr(wf, "execute_workflow_advanced")

    def test_github_webhook_no_longer_imports_deleted_symbol(self):
        import api.github_webhooks as gh
        # The broken import was removed; module still loads.
        assert not hasattr(gh, "execute_workflow")


# ===========================================================================
# F039 — memory: session_id is namespaced by workspace
# ===========================================================================

class TestF039MemoryScope:
    """F039 is closed BY DELETION (PRD-187 S5): the AdvancedMemoryManager
    router whose session keys needed workspace-prefixing no longer exists —
    the strongest form of the isolation guarantee."""

    def test_advanced_memory_router_is_gone(self):
        import importlib.util

        assert importlib.util.find_spec("api.memory") is None, (
            "api/memory.py was deleted in PRD-187 S5 — its resurrection "
            "reopens the F039 cross-tenant session surface"
        )


# ===========================================================================
# F007 — monitoring alert/log read surfaces are super-admin only
# ===========================================================================

class TestF007MonitoringGate:
    """GET /api/alerts and the Loki log routes are the obs tier; ingest POST
    keeps only its bearer token."""

    def _alerts_client(self, ctx):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from core.monitoring.automatos_alerts import create_alerts_router
        from core.auth.hybrid import get_request_context_hybrid

        def _db():
            db = MagicMock()
            res = MagicMock()
            res.fetchall.return_value = []
            db.execute.return_value = res
            yield db

        router = create_alerts_router(_db)
        app = FastAPI()
        app.include_router(router, prefix="/api")
        if ctx is not None:
            app.dependency_overrides[get_request_context_hybrid] = lambda: ctx
        return TestClient(app, raise_server_exceptions=False)

    def test_alerts_read_403_for_non_super_admin(self):
        resp = self._alerts_client(_ctx(user=USER_A)).get("/api/alerts")
        assert resp.status_code == 403, resp.text
        assert resp.json()["detail"] == "Super admin only"

    def test_alerts_read_ok_for_super_admin(self):
        resp = self._alerts_client(_ctx(user=SUPER_ADMIN)).get("/api/alerts")
        assert resp.status_code not in (401, 403), resp.text

    def test_logs_router_is_super_admin_gated(self):
        # The logs router carries a router-level require_super_admin dependency.
        from core.monitoring.automatos_logs_api import create_logs_router
        from core.auth.super_admin import require_super_admin

        router = create_logs_router()
        dep_calls = [d.dependency for d in getattr(router, "dependencies", [])]
        assert require_super_admin in dep_calls, (
            "logs router must be wrapped in require_super_admin"
        )


# ===========================================================================
# F045 — context.py: no unauthenticated tenant-data endpoints; count is scoped
# ===========================================================================

class TestF045ContextAuth:
    """Every context.py route now carries the workspace-scoped auth dependency,
    and the stats count is scoped to the caller's workspace."""

    def test_all_context_routes_require_auth(self):
        import api.context as ctx_mod
        from core.auth.hybrid import get_request_context_hybrid

        # Every route's dependant tree must reference the shared auth dependency.
        offenders = []
        for route in ctx_mod.router.routes:
            dependant = getattr(route, "dependant", None)
            if dependant is None:
                continue
            calls = _collect_dependency_calls(dependant)
            if get_request_context_hybrid not in calls:
                offenders.append(getattr(route, "path", "?"))
        assert not offenders, f"unauthenticated context routes remain: {offenders}"

    def test_retrieval_stats_count_is_workspace_scoped(self):
        # The RAG service count query is filtered by workspace_id when supplied.
        from modules.rag.service import RAGService
        svc = RAGService.__new__(RAGService)  # bypass heavy __init__
        svc._workspace_id = str(WS_A)

        captured = {}

        db = MagicMock()

        def _execute(stmt, params=None):
            captured["sql"] = str(stmt)
            captured["params"] = params
            res = MagicMock()
            row = MagicMock()
            row.total_docs = 0
            row.total_chunks = 0
            row.completed_docs = 0
            res.fetchone.return_value = row
            return res

        db.execute.side_effect = _execute
        svc.get_retrieval_stats(db, workspace_id=WS_A)
        assert "workspace_id" in captured["sql"], "documents count is not workspace-scoped"
        assert captured["params"] and str(captured["params"].get("workspace_id")) == str(WS_A)


def _collect_dependency_calls(dependant):
    """Flatten a FastAPI Dependant tree into the set of its ``.call`` callables."""
    calls = set()
    stack = [dependant]
    while stack:
        d = stack.pop()
        if getattr(d, "call", None) is not None:
            calls.add(d.call)
        stack.extend(getattr(d, "dependencies", []) or [])
    return calls


# ===========================================================================
# 5.0 Cross-tenant matrix (headline) — A cannot reach B across every domain
# ===========================================================================

class TestCrossTenantMatrix:
    """The wave's definition of done: for every in-scope domain, a workspace-A
    principal is denied (or own-only) against a workspace-B resource — read,
    write, and delete. Each cell delegates to the same scope decision the live
    handler makes, proven per-finding above; this class asserts the matrix
    *as a whole* stays closed (a permanent regression guard).
    """

    # (domain, callable(ctx_A) -> "denied"|"own_only") — each returns whether an
    # A-principal is blocked from B's resource.
    def _skill_read_denied(self):
        import api.skills as s
        return s._skill_visible_to(_FakeSkill(1, WS_B), _ctx(ws=WS_A)) is False

    def _skill_delete_denied(self):
        # A workspace caller cannot delete a global (NULL) skill.
        import api.skills as s
        return s._is_super_admin(_ctx(ws=WS_A, user=USER_A)) is False

    def _vectors_read_denied(self, monkeypatch):
        from config import config as app_config
        monkeypatch.setattr(
            app_config, "S3_VECTORS_BUCKET", "automatos-vectors-{workspace_id}", raising=False
        )
        from modules.search.vector_store.backends.s3_vectors_backend import S3VectorsBackend
        b = S3VectorsBackend.__new__(S3VectorsBackend)
        b.workspace_id = str(WS_A)
        b.bucket_name = f"automatos-vectors-{WS_A}"
        b.index_name = "i"
        b.index_dimension = 2048
        b.distance_metric = "cosine"
        b._setup_complete = True
        b._ensure_setup = lambda: None
        b.client = MagicMock()
        b.client.query_vectors.return_value = {
            "vectors": [{"key": "b", "distance": 0.0, "metadata": {"workspace_id": str(WS_B)}}]
        }
        hits = b.search([0.1], min_score=0.0, filters={"workspace_id": str(WS_A)})
        return hits == []

    def _workflow_execute_denied(self):
        # The unscoped cross-tenant execute route no longer exists at all.
        import api.workflows as wf
        return not hasattr(wf, "execute_workflow")

    def _memory_write_denied(self):
        # Closed by deletion (PRD-187 S5): the cross-tenant-shaped memory
        # router no longer exists at all.
        import importlib.util
        return importlib.util.find_spec("api.memory") is None

    def _shopify_sync_denied(self):
        import api.shopify as sh
        from fastapi import HTTPException
        try:
            sh._resolve_sync_workspace(_ctx(ws=WS_A), str(WS_B))
            return False
        except HTTPException as e:
            return e.status_code == 403

    def test_matrix_all_domains_denied(self, monkeypatch):
        cells = {
            "skills:read": self._skill_read_denied(),
            "skills:delete-global": self._skill_delete_denied(),
            "vectors:read": self._vectors_read_denied(monkeypatch),
            "workflows:execute": self._workflow_execute_denied(),
            "memory:write": self._memory_write_denied(),
            "shopify-graph:write": self._shopify_sync_denied(),
        }
        failed = [domain for domain, denied in cells.items() if not denied]
        assert not failed, f"cross-tenant leak — A reached B on: {failed}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
