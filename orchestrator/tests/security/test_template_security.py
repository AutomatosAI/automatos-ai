"""PRD-156 S4 — document template IDOR + SSTI + WeasyPrint SSRF.

  * IDOR: DocumentTemplate get/update/delete are workspace-scoped — the query
    filters by workspace_id and every CRUD caller passes the caller's workspace.
  * SSTI: rendering uses jinja2 ``SandboxedEnvironment`` — ``__globals__``-class
    payloads raise SecurityError (inert) instead of reaching Python internals.
  * SSRF: the WeasyPrint ``url_fetcher`` refuses file:// and internal/non-public
    hosts (10.x / 127.x / 169.254.x metadata) from user templates.

Unit / structural — no DB and no weasyprint needed (the block paths raise before
any network/weasyprint import).
"""
from __future__ import annotations

import importlib.util as _ilu
import pathlib
import sys as _sys
from unittest.mock import MagicMock

import pytest

ORCH = pathlib.Path(__file__).resolve().parents[2]


def _camelot_unlocatable() -> bool:  # pragma: no cover - env-dependent
    try:
        return _ilu.find_spec("camelot") is None
    except ValueError:
        return False


if _camelot_unlocatable():  # pragma: no cover - env-dependent
    import types as _types

    _sys.modules.setdefault("camelot", _types.ModuleType("camelot"))


# --- SSTI: SandboxedEnvironment renders dangerous payloads inert ---------------

def test_sandbox_blocks_globals_access():
    from jinja2.exceptions import SecurityError
    from jinja2.sandbox import SandboxedEnvironment

    env = SandboxedEnvironment(autoescape=True)
    with pytest.raises(SecurityError):
        env.from_string("{{ ''.__class__.__init__.__globals__ }}").render()


def test_generation_service_uses_sandboxed_env():
    txt = (ORCH / "modules/documents/generation_service.py").read_text()
    assert "SandboxedEnvironment(autoescape=True)" in txt
    assert "jinja2.Environment(" not in txt  # the unsandboxed env is gone


# --- SSRF: url_fetcher blocks file:// and internal targets --------------------

@pytest.mark.parametrize(
    "bad_url",
    [
        "file:///etc/passwd",
        "http://169.254.169.254/latest/meta-data/",  # cloud metadata
        "http://10.0.0.5/internal",                   # private
        "http://127.0.0.1:8000/admin",                # loopback
        "ftp://example.com/x",                        # non-http(s) scheme
    ],
)
def test_url_fetcher_refuses_file_and_internal(bad_url):
    from modules.documents.generation_service import _safe_url_fetcher

    with pytest.raises(ValueError):
        _safe_url_fetcher(bad_url)


# --- IDOR: DocumentTemplate CRUD is workspace-scoped --------------------------

def test_get_template_query_filters_by_workspace():
    from modules.documents.template_service import DocumentTemplateService

    db = MagicMock()
    DocumentTemplateService(db).get_template("tid", "ws-A")
    filter_args = db.query.return_value.filter.call_args.args
    clauses = " ".join(str(a) for a in filter_args)
    assert "workspace_id" in clauses, "get_template does not filter by workspace_id (IDOR)"


def test_crud_signatures_require_workspace():
    import inspect

    from modules.documents.template_service import DocumentTemplateService

    for name in ("get_template", "update_template", "delete_template"):
        params = inspect.signature(getattr(DocumentTemplateService, name)).parameters
        assert "workspace_id" in params, f"{name} missing workspace_id param (IDOR)"


def test_all_crud_endpoints_pass_workspace():
    txt = (ORCH / "api/document_generation.py").read_text()
    assert "get_template(template_id, ctx.workspace_id)" in txt
    assert "update_template(template_id, ctx.workspace_id" in txt
    assert "delete_template(template_id, ctx.workspace_id)" in txt
