"""PRD-142 Wave 3 · WS-J · W3-S9 — NL2SQL validation path + tenant + heartbeat.

The NL2SQL primitive's BRAIN §3.x contract says: *generated SQL is always
validated/rewritten read-only before execution; no unvalidated query reaches
the database; credentials are never logged*. The §H DoD adds: *failure path
tested, tenant-isolated, observable via heartbeat finding*.

Today's gaps (verified 2026-06-06 against ``modules/nl2sql/service.py``):

* The validator IS called before execution (line 332 vs line 369) — but no
  test pins the order, so a future refactor could swap them. W3-S9 PINS
  it under the Wave 2 net.
* The execution path uses ``validated_sql`` (line 369) — but no test
  asserts that the *generated* SQL isn't accidentally executed instead.
* ``_get_source(source_id)`` (line 105) queries ``DatabaseKnowledgeSource``
  by ``id`` only — NO workspace filter. The API route
  ``POST /api/knowledge/sources/database/{source_id}/query`` takes
  ``ctx = Depends(get_request_context_hybrid)`` but never passes
  ``ctx.workspace_id`` to ``smart_query``, leaving a cross-tenant
  read-side leak: workspace A can pass workspace B's source_id and the
  service will decrypt B's creds + execute against B's database.
* The NL2SQL path emits NO primitive heartbeat finding — the W3-S2 tile
  reads ``unknown`` for ``nl2sql`` no matter how the live query path is
  behaving.

These tests pin the W3-S9 hardening contract:

1. **Validator gates execution (failure-path):** the source of
   ``query_database`` lays validator BEFORE execute (textual order pin),
   and execution always uses ``validated_sql``, never ``generated_sql``.
2. **Credentials never logged:** static grep over the NL2SQL package
   asserts that no ``logger.*`` format string surfaces the connection
   string, the password, or the raw ``credentials`` dict.
3. **Workspace isolation (A4):** ``_get_source(source_id, workspace_id=X)``
   adds the ``workspace_id == X`` filter to the SQLAlchemy query — and
   a cross-workspace request raises (not "best-effort allow").
4. **API route propagates workspace_id:** the route passes
   ``workspace_id=str(ctx.workspace_id)`` into ``smart_query`` so the
   filter has a real value to enforce.
5. **Heartbeat (W3-S1 wiring):** a tiny stateless helper
   ``_emit_nl2sql_primitive`` calls ``emit_primitive_finding`` with
   primitive='nl2sql' and the correct status — green on a clean turn,
   down on a validation or execution failure.
6. **No-workspace skip:** the helper emits NOTHING when ``workspace_id``
   is falsy (A4: honest gap over fabricated default).
7. **Best-effort emit:** a failed heartbeat write NEVER breaks the
   NL2SQL caller.

The tests deliberately operate at the *unit* level via source-text
inspection + targeted importlib loads — full integration of the service
would drag sqlparse, the LLM provider, the credential decryption chain,
and the audit service into the unit suite. Mirrors the W3-S6 / W3-S7 /
W3-S8 patterns.
"""
from __future__ import annotations

import ast
import importlib.util
import re
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Paths to the surfaces we pin without importing them through the heavy
# ``modules.nl2sql`` package __init__ (which eagerly loads sqlparse, RAG,
# benchmarks, intelligence, etc).
# ---------------------------------------------------------------------------

ORCH_ROOT = Path(__file__).resolve().parent.parent
if str(ORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(ORCH_ROOT))

NL2SQL_DIR = ORCH_ROOT / "modules" / "nl2sql"
SERVICE_PY = NL2SQL_DIR / "service.py"
NL2SQL_SERVICE_PY = NL2SQL_DIR / "query" / "nl2sql_service.py"
VALIDATOR_PY = NL2SQL_DIR / "query" / "validator.py"
PRIMITIVE_HEARTBEAT_PY = NL2SQL_DIR / "primitive_heartbeat.py"
API_ROUTE_PY = ORCH_ROOT / "api" / "database_knowledge.py"


# ---------------------------------------------------------------------------
# Validator import (lightweight — pure-stdlib module).
# ---------------------------------------------------------------------------


def _load_validator():
    spec = importlib.util.spec_from_file_location(
        "nl2sql_validator_w3s9", str(VALIDATOR_PY)
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


VAL_MOD = _load_validator()


# ===========================================================================
# 1. VALIDATOR CONTRACT — the failure-path is visible (not silent).
# ===========================================================================


class TestValidatorBlocksUnsafeSQL:
    """The validator is the gate. Pin the failure-path errors so a later
    refactor can't widen the deny list silently or downgrade a hard error
    to a logged warning."""

    def test_validator_raises_on_delete(self):
        v = VAL_MOD.SQLValidator(max_limit=100)
        with pytest.raises(VAL_MOD.SQLValidationError):
            v.validate_and_rewrite("DELETE FROM users WHERE id = 1")

    def test_validator_raises_on_subquery_insert(self):
        v = VAL_MOD.SQLValidator(max_limit=100)
        # The "must start with SELECT" check passes; the deny-keyword scan
        # over the full statement catches the hidden INSERT.
        with pytest.raises(VAL_MOD.SQLValidationError, match="forbidden keyword"):
            v.validate_and_rewrite(
                "SELECT * FROM (INSERT INTO users (name) VALUES ('x') "
                "RETURNING *) AS t"
            )

    def test_validator_raises_on_cte_union_escape(self):
        v = VAL_MOD.SQLValidator(max_limit=100)
        with pytest.raises(VAL_MOD.SQLValidationError):
            v.validate_and_rewrite(
                "WITH x AS (SELECT * FROM users) SELECT * FROM x"
            )

    def test_validator_caps_limit_for_read_only_safety(self):
        v = VAL_MOD.SQLValidator(max_limit=100)
        sql, _ = v.validate_and_rewrite("SELECT * FROM users LIMIT 99999")
        assert "LIMIT 100" in sql, (
            f"validator must cap excessive LIMIT to its read-only ceiling; "
            f"got {sql!r}"
        )

    def test_validator_injects_limit_when_missing(self):
        v = VAL_MOD.SQLValidator(max_limit=100)
        sql, reasons = v.validate_and_rewrite("SELECT * FROM users")
        assert "LIMIT 100" in sql
        assert any("injected" in r for r in reasons), (
            f"validator must explain when it injected a LIMIT; reasons={reasons!r}"
        )


# ===========================================================================
# 2. CALL-ORDER PIN — validator runs BEFORE execute in query_database; the
#    executor sees validated_sql, never generated_sql.
# ===========================================================================


class TestValidatorRunsBeforeExecute:
    """A future refactor could move the executor above the validator, or
    pass ``generated_sql`` instead of ``validated_sql`` to ``conn.execute``
    — both would silently re-introduce the §H "no unvalidated query"
    failure. Pin both via source-text + AST inspection of the
    ``query_database`` method body."""

    @pytest.fixture(scope="class")
    def query_db_source(self) -> str:
        return SERVICE_PY.read_text()

    @pytest.fixture(scope="class")
    def query_db_body(self, query_db_source: str) -> str:
        """Extract the ``query_database`` method body via AST so we can
        reason about *its* statement order, not other methods'."""
        tree = ast.parse(query_db_source)
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "query_database":
                # Reproduce the body by splitting the source at the line
                # range of the function. Using ast.get_source_segment is
                # simplest and stable on 3.11+.
                segment = ast.get_source_segment(query_db_source, node)
                assert segment is not None, "query_database source not extractable"
                return segment
        raise AssertionError("query_database method not found in service.py")

    def test_validate_call_appears_in_query_database(self, query_db_body: str):
        assert "validator.validate_and_rewrite(" in query_db_body, (
            "query_database must call validator.validate_and_rewrite"
        )

    def test_execute_call_appears_in_query_database(self, query_db_body: str):
        assert "conn.execute(text(" in query_db_body or "conn.execute(\n" in query_db_body or "execute(text(validated_sql" in query_db_body, (
            "query_database must execute the validated SQL through the connection"
        )

    def test_validate_appears_before_execute(self, query_db_body: str):
        idx_validate = query_db_body.find("validate_and_rewrite(")
        idx_execute = query_db_body.find("conn.execute(")
        assert idx_validate >= 0, "validate_and_rewrite call not found"
        assert idx_execute >= 0, "conn.execute call not found"
        assert idx_validate < idx_execute, (
            "validator MUST run before conn.execute — found validate at "
            f"col {idx_validate}, execute at col {idx_execute}"
        )

    def test_executor_uses_validated_sql_not_generated_sql(self, query_db_body: str):
        # The exact line we pin: ``conn.execute(text(validated_sql))``.
        # If a future refactor passes ``generated_sql`` (the raw LLM
        # output) instead, this test fails — and the §H "no unvalidated
        # query reaches the database" line is back.
        assert re.search(
            r"conn\.execute\s*\(\s*text\s*\(\s*validated_sql\b",
            query_db_body,
        ), (
            "conn.execute must take text(validated_sql), never the raw "
            "generated_sql — that would bypass the validator gate"
        )
        # And the inverse: no execute on generated_sql in this method.
        assert not re.search(
            r"conn\.execute\s*\(\s*text\s*\(\s*generated_sql\b",
            query_db_body,
        ), "generated_sql must NEVER be passed to conn.execute"

    def test_validation_failure_returns_visible_error(self, query_db_body: str):
        # On SQLValidationError, the failure path appends to ``attempted_sqls``
        # AND returns ``{"success": False, "error": <msg>}`` — never silently
        # succeeds. Pin both shape elements.
        assert "Validation failed" in query_db_body, (
            "validation failure must surface a visible 'Validation failed' "
            "string in the returned error"
        )
        assert '"success": False' in query_db_body, (
            "validation failure must return success=False (not a silent retry)"
        )


# ===========================================================================
# 3. CREDENTIALS NEVER LOGGED — static grep over the NL2SQL package.
# ===========================================================================


class TestCredentialsNeverLogged:
    """A logger format string that mentions ``conn_str``, ``password``, or
    the raw ``credentials`` dict is the §H "creds never logged" leak. Pin
    that the NL2SQL package never accidentally adds one — this is a
    static guarantee, not a runtime check."""

    _LOG_CALL_RE = re.compile(
        r"logger\.(?:debug|info|warning|error|critical|exception)\s*\([^)]*\)",
        re.DOTALL,
    )
    _FORBIDDEN_TOKENS = (
        "conn_str",
        "connection_string",
        "password",
        "{credentials",
        "credentials!r",
        "credentials[",
        # The decrypted dict's keys we DON'T want surfaced in any logger format:
        "credentials.get",
    )

    def _scan_file(self, path: Path) -> list[tuple[str, str, str]]:
        """Return (file, log_call_snippet, offending_token) triples."""
        if not path.exists():
            return []
        src = path.read_text()
        out: list[tuple[str, str, str]] = []
        for m in self._LOG_CALL_RE.finditer(src):
            snippet = m.group(0)
            for tok in self._FORBIDDEN_TOKENS:
                if tok in snippet:
                    out.append((str(path.name), snippet, tok))
        return out

    def test_service_py_does_not_log_credentials(self):
        offenders = self._scan_file(SERVICE_PY)
        assert offenders == [], (
            "modules/nl2sql/service.py logger.* calls must not surface "
            f"credentials; offenders={offenders!r}"
        )

    def test_nl2sql_service_py_does_not_log_credentials(self):
        offenders = self._scan_file(NL2SQL_SERVICE_PY)
        assert offenders == [], (
            "modules/nl2sql/query/nl2sql_service.py logger.* calls must "
            f"not surface credentials; offenders={offenders!r}"
        )

    def test_validator_py_does_not_log_credentials(self):
        offenders = self._scan_file(VALIDATOR_PY)
        assert offenders == [], (
            "modules/nl2sql/query/validator.py logger.* calls must not "
            f"surface credentials; offenders={offenders!r}"
        )

    def test_no_print_of_credentials_or_conn_str(self):
        # The bare ``print(...)`` escape hatch is what's normally banned
        # by lint — pin it explicitly for the NL2SQL path.
        for path in (SERVICE_PY, NL2SQL_SERVICE_PY, VALIDATOR_PY):
            src = path.read_text()
            for tok in ("conn_str", "password", "credentials"):
                # Allow ``credentials = ...`` assignments / param names / etc.
                # — only block ``print(`` formatting that names them.
                prints = re.findall(
                    rf"print\s*\([^)]*\b{re.escape(tok)}\b[^)]*\)", src
                )
                assert prints == [], (
                    f"{path.name}: print() must not surface {tok!r}; "
                    f"matches={prints!r}"
                )


# ===========================================================================
# 4. WORKSPACE ISOLATION — _get_source filters by workspace_id when given.
# ===========================================================================


class _FakeSource:
    def __init__(self, *, id: int, workspace_id: str, name: str = "src"):
        self.id = id
        self.workspace_id = workspace_id
        self.name = name


class _RecordingFilter:
    """Records filter expressions so we can assert the workspace clause
    was applied — without standing up a real SQLAlchemy engine."""

    def __init__(self, store: "list[Any]", rows: "list[_FakeSource]"):
        self._store = store
        self._rows = rows

    def filter(self, *exprs):
        for expr in exprs:
            self._store.append(repr(expr))
        return self

    def first(self):
        return self._rows[0] if self._rows else None


class _RecordingSession:
    def __init__(self, rows: "list[_FakeSource]"):
        self.filter_calls: list[str] = []
        self._rows = rows
        self.closed = False

    def query(self, _model):
        return _RecordingFilter(self.filter_calls, self._rows)

    def close(self):
        self.closed = True


def _load_service_module():
    """Load ``modules.nl2sql.service`` with the heavy deps stubbed out.
    Returns the loaded module so we can probe ``DatabaseKnowledgeService``
    directly without going through the package __init__."""
    # Parent packages — path-only so leaf imports succeed without firing
    # the heavy __init__.
    for _pkg in (
        "modules",
        "modules.nl2sql",
        "modules.nl2sql.query",
        "modules.nl2sql.training",
        "modules.nl2sql.intelligence",
        "modules.nl2sql.benchmarks",
        "modules.nl2sql.schema",
        "modules.rag",
        "modules.search",
        "modules.search.services",
        "modules.tools",
        "modules.tools.services",
    ):
        if _pkg not in sys.modules:
            stub = types.ModuleType(_pkg)
            stub.__path__ = [str(ORCH_ROOT / _pkg.replace(".", "/"))]
            sys.modules[_pkg] = stub

    # sqlparse — stub. service.py only uses sqlparse for the unused
    # ``_validate_sql`` / ``_extract_tables_from_sql`` helpers; the
    # ``query_database`` path under test does not touch it at runtime.
    if "sqlparse" not in sys.modules:
        sp = types.ModuleType("sqlparse")
        sp.parse = lambda *_a, **_k: [MagicMock()]
        sp.sql = types.ModuleType("sqlparse.sql")
        sp.sql.Identifier = MagicMock
        sys.modules["sqlparse"] = sp
        sys.modules["sqlparse.sql"] = sp.sql

    # Heavy sibling modules service.py imports at the top.
    stub_targets = {
        "core.credentials.resolver": {"CredentialResolver": MagicMock},
        "core.llm": {"LLMProvider": MagicMock, "create_llm_manager": MagicMock},
        "modules.rag": {"RAGService": MagicMock},
        "modules.search.services.context_engineering_service": {
            "ContextEngineeringService": MagicMock,
        },
        "core.services.audit_service": {"AuditService": MagicMock},
        "modules.nl2sql.query.nl2sql_service": {
            "NaturalLanguageToSQLService": MagicMock,
        },
        "modules.nl2sql.query.validator": {
            "SQLValidator": MagicMock,
            "SQLValidationError": Exception,
        },
        "modules.tools.services.pandas_ai_service": {
            "get_pandasai_service": MagicMock,
        },
    }
    for name, attrs in stub_targets.items():
        mod = sys.modules.get(name)
        if mod is None:
            mod = types.ModuleType(name)
            sys.modules[name] = mod
        # Add (or overwrite) the symbols the service file imports — even if
        # the module was previously path-stubbed, we still need its
        # attributes for ``from X import Y`` to resolve.
        for k, v in attrs.items():
            setattr(mod, k, v)

    # core.models.database_knowledge.DatabaseKnowledgeSource — used inside
    # _get_source for the query model class. The column attributes need
    # __eq__ implementations that return a marker string so the recording
    # session can see WHICH column was filtered on (a plain MagicMock
    # returns False on ==, which loses the column name).
    class _FakeColumn:
        def __init__(self, name: str):
            self._name = name

        def __eq__(self, other):  # type: ignore[override]
            return f"{self._name} == {other!r}"

        def __hash__(self):
            return hash(self._name)

    if "core.models.database_knowledge" not in sys.modules:
        mod = types.ModuleType("core.models.database_knowledge")
        mod.DatabaseKnowledgeSource = type(
            "DatabaseKnowledgeSource",
            (),
            {
                "id": _FakeColumn("id"),
                "workspace_id": _FakeColumn("workspace_id"),
            },
        )
        # Other names imported elsewhere but not exercised here.
        for n in (
            "DatabaseKnowledgeSourceCreate",
            "SemanticMetricCreate",
            "SemanticDimensionCreate",
            "DatabaseQueryRequest",
            "QueryTemplateExecute",
            "DatabaseQueryAudit",
        ):
            setattr(mod, n, MagicMock())
        sys.modules["core.models.database_knowledge"] = mod

    # core.database.database — try to import the real module so we don't
    # shadow it for other tests. The _get_source method does the
    # ``from core.database.database import SessionLocal`` *lazily* inside
    # the call, so importing the real module once + monkeypatching
    # SessionLocal at test time gives a clean fake without polluting
    # sys.modules for subsequent tests (e.g. test_primitive_health_endpoint
    # which legitimately needs ``get_db``).
    try:
        import core.database.database  # noqa: F401
    except Exception:
        # Dev env may lack a dep — fall back to a stub that at least
        # has both names the rest of the suite needs.
        mod = types.ModuleType("core.database.database")
        mod.SessionLocal = MagicMock()
        mod.get_db = MagicMock()
        sys.modules["core.database.database"] = mod

    spec = importlib.util.spec_from_file_location(
        "modules.nl2sql.service", str(SERVICE_PY)
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["modules.nl2sql.service"] = mod
    spec.loader.exec_module(mod)
    return mod


SERVICE_MOD = _load_service_module()


class TestGetSourceFiltersByWorkspace:
    """``_get_source(source_id, workspace_id=X)`` must add the
    ``workspace_id == X`` filter to the query, and reject when the
    source belongs to a different workspace. Without this filter, the
    NL2SQL ``query_database`` path lets workspace A use workspace B's
    source_id — and decrypt B's credentials. Cross-tenant leak."""

    def _bare_service(self) -> "Any":
        svc = SERVICE_MOD.DatabaseKnowledgeService.__new__(
            SERVICE_MOD.DatabaseKnowledgeService
        )
        svc.schema_cache = {}
        svc.query_cache = {}
        return svc

    @pytest.mark.asyncio
    async def test_get_source_filters_by_workspace_when_provided(self, monkeypatch):
        svc = self._bare_service()
        target = _FakeSource(id=42, workspace_id="ws-A")
        sess = _RecordingSession([target])
        sl = MagicMock(return_value=sess)
        monkeypatch.setitem(
            sys.modules["core.database.database"].__dict__,
            "SessionLocal",
            sl,
        )

        out = await svc._get_source("42", workspace_id="ws-A")

        assert out is target, "expected the matching source to come back"
        # The filter chain MUST have been called with a workspace_id clause —
        # we record the repr of each filter expr; one of them mentions
        # ``workspace_id``.
        joined = " | ".join(sess.filter_calls)
        assert "workspace_id" in joined, (
            f"_get_source(workspace_id=...) must add a workspace_id filter; "
            f"recorded filter calls: {sess.filter_calls!r}"
        )

    @pytest.mark.asyncio
    async def test_get_source_rejects_cross_workspace_source(self, monkeypatch):
        """Source belongs to ws-B but the caller asks as ws-A: the filter
        eliminates the row, ``first()`` returns None, and the method
        raises (NOT 'return the wrong workspace's source anyway')."""
        svc = self._bare_service()
        # The recording session returns NO rows because the filter would
        # exclude them — we model the filter outcome directly.
        sess = _RecordingSession([])
        sl = MagicMock(return_value=sess)
        monkeypatch.setitem(
            sys.modules["core.database.database"].__dict__,
            "SessionLocal",
            sl,
        )

        with pytest.raises(Exception):
            await svc._get_source("42", workspace_id="ws-A")

    @pytest.mark.asyncio
    async def test_get_source_without_workspace_keeps_legacy_shape(
        self, monkeypatch
    ):
        """Backwards-compatible: callers that don't pass workspace_id
        still get the existing behaviour (id-only filter). The API
        route is the only call site that NEEDS the workspace filter;
        in-process callers that already enforce isolation upstream
        (the scheduling handler) keep their narrower interface."""
        svc = self._bare_service()
        target = _FakeSource(id=42, workspace_id="ws-X")
        sess = _RecordingSession([target])
        sl = MagicMock(return_value=sess)
        monkeypatch.setitem(
            sys.modules["core.database.database"].__dict__,
            "SessionLocal",
            sl,
        )

        out = await svc._get_source("42")

        assert out is target
        # No workspace_id clause was added (legacy path).
        joined = " | ".join(sess.filter_calls)
        assert "workspace_id" not in joined, (
            "legacy _get_source(source_id) shape must NOT silently add a "
            "workspace filter — that would break other call paths; the "
            "isolation lives at the API route boundary"
        )


# ===========================================================================
# 5. API ROUTE WIRING — the route passes workspace_id through.
# ===========================================================================


class TestAPIRoutePropagatesWorkspace:
    """The /api/knowledge/sources/database/{source_id}/query route owns
    workspace boundary enforcement: it must hand ``ctx.workspace_id`` to
    the service's ``smart_query`` so ``_get_source`` can filter. Pin via
    source-text — the route body grows over time and a future refactor
    could drop the kwarg silently."""

    @pytest.fixture(scope="class")
    def route_src(self) -> str:
        return API_ROUTE_PY.read_text()

    def test_query_route_calls_smart_query(self, route_src: str):
        # Locate the ``query_database`` route handler.
        assert "smart_query" in route_src, (
            "API route must delegate to service.smart_query"
        )

    def test_query_route_passes_workspace_id_kwarg(self, route_src: str):
        # The smart_query call inside ``query_database`` route must
        # pass ``workspace_id=...`` derived from ``ctx.workspace_id``.
        # Pin the kwarg name + the ctx-derived value to keep the boundary
        # explicit.
        tree = ast.parse(route_src)
        found = False
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "query_database":
                seg = ast.get_source_segment(route_src, node) or ""
                if "smart_query(" in seg:
                    assert "workspace_id=" in seg, (
                        "API route /query MUST pass workspace_id= into "
                        "smart_query; otherwise the cross-tenant filter "
                        "has no value to enforce"
                    )
                    assert "ctx.workspace_id" in seg, (
                        "API route /query MUST derive workspace_id from "
                        "ctx.workspace_id (the authenticated boundary), "
                        "not a request body / query-string value"
                    )
                    found = True
        assert found, "query_database route handler not found in api/database_knowledge.py"


# ===========================================================================
# 6. HEARTBEAT — W3-S1 wiring for nl2sql primitive.
# ===========================================================================


def _load_nl2sql_heartbeat(monkeypatch):
    """Load ``modules.nl2sql.primitive_heartbeat`` after the test has had
    a chance to swap ``services.heartbeat_service.emit_primitive_finding``
    for a fake. Same pattern as W3-S6 / W3-S8."""
    # Ensure parent packages are path-stubbed.
    for _pkg in ("modules", "modules.nl2sql"):
        if _pkg not in sys.modules:
            stub = types.ModuleType(_pkg)
            stub.__path__ = [str(ORCH_ROOT / _pkg.replace(".", "/"))]
            sys.modules[_pkg] = stub

    # Drop a previously cached version (each test rebinds the fake module).
    sys.modules.pop("modules.nl2sql.primitive_heartbeat", None)

    spec = importlib.util.spec_from_file_location(
        "modules.nl2sql.primitive_heartbeat", str(PRIMITIVE_HEARTBEAT_PY)
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["modules.nl2sql.primitive_heartbeat"] = mod
    spec.loader.exec_module(mod)
    return mod


class TestEmitNL2SQLHeartbeat:
    """The W3-S1 helper guarantees ``nl2sql`` primitive emits route through
    the same shared writer as ``chat`` / ``memory`` / ``rag``. Pin the
    contract on the nl2sql wrapper module."""

    def test_emit_green_on_success(self, monkeypatch):
        calls: list[tuple] = []

        def _fake_emit(workspace_id, primitive, status, detail=""):
            calls.append((workspace_id, primitive, status, detail))
            return True

        fake_mod = types.ModuleType("services.heartbeat_service")
        fake_mod.emit_primitive_finding = _fake_emit
        monkeypatch.setitem(sys.modules, "services.heartbeat_service", fake_mod)

        mod = _load_nl2sql_heartbeat(monkeypatch)
        mod._emit_nl2sql_primitive(
            "ws-nl2sql-1", success=True, detail="3 rows in 42ms"
        )

        assert calls == [("ws-nl2sql-1", "nl2sql", "green", "3 rows in 42ms")]

    def test_emit_down_on_validation_failure(self, monkeypatch):
        calls: list[tuple] = []

        def _fake_emit(workspace_id, primitive, status, detail=""):
            calls.append((workspace_id, primitive, status, detail))
            return True

        fake_mod = types.ModuleType("services.heartbeat_service")
        fake_mod.emit_primitive_finding = _fake_emit
        monkeypatch.setitem(sys.modules, "services.heartbeat_service", fake_mod)

        mod = _load_nl2sql_heartbeat(monkeypatch)
        mod._emit_nl2sql_primitive(
            "ws-nl2sql-1",
            success=False,
            detail="Validation failed: forbidden keyword",
        )

        assert calls == [(
            "ws-nl2sql-1",
            "nl2sql",
            "down",
            "Validation failed: forbidden keyword",
        )]

    def test_emit_skips_when_no_workspace(self, monkeypatch):
        calls: list[tuple] = []

        def _fake_emit(*a, **k):
            calls.append((a, k))
            return True

        fake_mod = types.ModuleType("services.heartbeat_service")
        fake_mod.emit_primitive_finding = _fake_emit
        monkeypatch.setitem(sys.modules, "services.heartbeat_service", fake_mod)

        mod = _load_nl2sql_heartbeat(monkeypatch)
        mod._emit_nl2sql_primitive(None, success=True, detail="anything")
        mod._emit_nl2sql_primitive("", success=False, detail="anything")

        assert calls == [], (
            "no workspace_id MUST mean no emit; A4 — fabricated default is "
            "worse than an honest 'unknown' on the tile"
        )

    def test_emit_swallows_writer_failure(self, monkeypatch):
        def _raising_emit(*_a, **_k):
            raise RuntimeError("heartbeat DB unreachable")

        fake_mod = types.ModuleType("services.heartbeat_service")
        fake_mod.emit_primitive_finding = _raising_emit
        monkeypatch.setitem(sys.modules, "services.heartbeat_service", fake_mod)

        mod = _load_nl2sql_heartbeat(monkeypatch)
        # MUST NOT raise — best-effort emit per §H 'observable'.
        mod._emit_nl2sql_primitive(
            "ws-nl2sql-1", success=True, detail="ok"
        )

    def test_emit_uses_canonical_primitive_name(self, monkeypatch):
        """The W3-S1 writer rejects unknown primitive names. Pin that the
        wrapper uses exactly 'nl2sql' (lowercase, no synonym)."""
        captured: dict = {}

        def _fake_emit(workspace_id, primitive, status, detail=""):
            captured["primitive"] = primitive
            return True

        fake_mod = types.ModuleType("services.heartbeat_service")
        fake_mod.emit_primitive_finding = _fake_emit
        monkeypatch.setitem(sys.modules, "services.heartbeat_service", fake_mod)

        mod = _load_nl2sql_heartbeat(monkeypatch)
        mod._emit_nl2sql_primitive("ws-1", success=True)

        assert captured.get("primitive") == "nl2sql", (
            "primitive name must be the canonical lowercase 'nl2sql'; got "
            f"{captured!r}"
        )


# ===========================================================================
# 7. SERVICE WIRE-UP — query_database CALLS the helper on both branches.
# ===========================================================================


class TestServiceWiresHeartbeat:
    """A heartbeat helper that no one calls is dead code. Pin via source
    inspection that ``service.py`` imports the helper and invokes it on
    BOTH the success and failure branches of ``query_database``."""

    @pytest.fixture(scope="class")
    def svc_src(self) -> str:
        return SERVICE_PY.read_text()

    def test_service_imports_helper(self, svc_src: str):
        assert (
            "from .primitive_heartbeat import" in svc_src
            or "from modules.nl2sql.primitive_heartbeat import" in svc_src
        ), (
            "service.py must import _emit_nl2sql_primitive so the "
            "query_database path emits the W3-S1 finding"
        )

    def test_service_calls_helper_on_success(self, svc_src: str):
        tree = ast.parse(svc_src)
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "query_database":
                body = ast.get_source_segment(svc_src, node) or ""
                # The helper is called somewhere in this method — at least
                # once with success=True for the green path.
                assert "_emit_nl2sql_primitive" in body, (
                    "query_database must call _emit_nl2sql_primitive — the "
                    "W3-S2 tile depends on it"
                )
                assert "success=True" in body, (
                    "query_database must emit success=True on the green path"
                )
                assert "success=False" in body, (
                    "query_database must emit success=False on the failure "
                    "path — otherwise a tile that lit green stays green "
                    "through a brownout"
                )
                return
        raise AssertionError("query_database method not found in service.py")
