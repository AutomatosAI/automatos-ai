"""Scaffold a platform-tool skeleton from a FastAPI router endpoint (PRD-143 S9).

Scaffold-then-curate (PRD-143 FR-7): given a router module and 'METHOD /path',
introspect the live FastAPI route (params, body model, docstring) and emit an
ActionDefinition skeleton plus a workspace-scoped handler skeleton for HUMAN
curation. This script NEVER writes into modules/tools/discovery/ and NEVER
registers anything in platform_actions.py — registration stays a manual,
reviewed step (the 3-file pattern's third file is always edited by hand).

Endpoints on routers locked by require_super_admin (the S6/S7 obs perimeter)
are detected by dependency IDENTITY — not a hardcoded module list — and emit
super_admin_only=True with an explicit review flag.

Usage:
    python scripts/scaffold_platform_tool.py api/agents.py 'POST /api/agents'
    python scripts/scaffold_platform_tool.py api/agents.py 'POST /api/agents' --out-dir gen/
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import pprint
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_ORCH_ROOT = Path(__file__).resolve().parent.parent
if str(_ORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_ORCH_ROOT))

_DISCOVERY_DIR = _ORCH_ROOT / "modules" / "tools" / "discovery"

_VERB_PERMISSION = {
    "GET": "read",
    "HEAD": "read",
    "OPTIONS": "read",
    "POST": "write",
    "PUT": "write",
    "PATCH": "write",
    "DELETE": "destructive",
}

_MAX_DESCRIPTION_CHARS = 280
_MAX_CANDIDATES_IN_ERROR = 12


def verb_to_permission_level(method: str) -> str:
    """Guess permission_level from the HTTP verb.

    Unknown verbs map to 'destructive' — fail-closed: maximum review scrutiny.
    """
    return _VERB_PERMISSION.get(method.upper(), "destructive")


def derive_action_name(method: str, path: str) -> str:
    """'POST /api/agents' → 'platform_create_agents' (curation reviews the name)."""
    segments = [s for s in path.strip("/").split("/") if s]
    literals = [s for s in segments if not (s.startswith("{") and s.endswith("}"))]
    while literals and literals[0] in ("api", "v1", "v2"):
        literals = literals[1:]
    resource = "_".join(s.replace("-", "_") for s in literals) or "root"
    method = method.upper()
    if method == "GET" and len(literals) == 1 and len(segments) - 1 <= len(literals):
        verb = "list" if not any(s.startswith("{") for s in segments) else "get"
    elif method in ("GET", "HEAD", "OPTIONS"):
        verb = "get"
    elif method == "POST":
        verb = "create"
    elif method in ("PUT", "PATCH"):
        verb = "update"
    elif method == "DELETE":
        verb = "delete"
    else:
        verb = method.lower()
    return f"platform_{verb}_{resource}"


@dataclass
class ScaffoldResult:
    name: str
    super_admin_only: bool
    action_filename: str
    handler_filename: str
    action_source: str
    handler_source: str
    action_path: Optional[Path] = None
    handler_path: Optional[Path] = None


def _assert_safe_out_dir(out_dir: Path) -> Path:
    resolved = Path(out_dir).resolve()
    if resolved == _DISCOVERY_DIR or resolved.is_relative_to(_DISCOVERY_DIR):
        raise ValueError(
            "refusing to write into modules/tools/discovery/ — scaffolds are "
            "staging-only; curate and register by hand (PRD-143 FR-7)"
        )
    return resolved


def _import_router_module(module_path: str):
    name = module_path.replace("\\", "/").strip()
    if name.endswith(".py"):
        name = name[:-3]
    return importlib.import_module(name.strip("/").replace("/", "."))


def _norm_path(path: str) -> str:
    return path.rstrip("/") or "/"


def _find_route(module, method: str, path: str) -> Tuple[Any, Any]:
    """Locate the APIRoute for 'METHOD path' across the module's routers."""
    from fastapi import APIRouter
    from fastapi.routing import APIRoute

    method = method.upper()
    want = _norm_path(path)
    candidates: List[Tuple[Any, Any]] = [
        (obj, route)
        for obj in vars(module).values()
        if isinstance(obj, APIRouter)
        for route in obj.routes
        if isinstance(route, APIRoute)
    ]
    for router, route in candidates:
        if method in (route.methods or ()) and _norm_path(route.path) == want:
            return router, route
    available = sorted(
        f"{m} {route.path}" for _, route in candidates for m in (route.methods or ())
    )
    shown = ", ".join(available[:_MAX_CANDIDATES_IN_ERROR]) or "none"
    raise ValueError(
        f"no route matching '{method} {path}' in {module.__name__}; available: {shown}"
    )


def _is_super_admin_locked(router, route) -> bool:
    """su detection by IDENTITY of the require_super_admin dependency (S6/S7)."""
    from core.auth.super_admin import require_super_admin

    for dep in getattr(router, "dependencies", None) or []:
        if getattr(dep, "dependency", None) is require_super_admin:
            return True
    dependant = getattr(route, "dependant", None)
    for sub in getattr(dependant, "dependencies", None) or []:
        if getattr(sub, "call", None) is require_super_admin:
            return True
    return False


def _collapse_nullable(schema: Dict[str, Any]) -> Dict[str, Any]:
    """anyOf [T, null] → T (pydantic v2 Optional fields), for skeleton readability."""
    any_of = schema.get("anyOf")
    if isinstance(any_of, list) and len(any_of) == 2:
        non_null = [s for s in any_of if isinstance(s, dict) and s.get("type") != "null"]
        if len(non_null) == 1:
            merged = dict(non_null[0])
            for key, value in schema.items():
                if key != "anyOf" and key not in merged:
                    merged[key] = value
            return merged
    return schema


def _resolve_refs(schema: Any, components: Dict[str, Any], depth: int = 0) -> Dict[str, Any]:
    """Inline #/components/schemas refs (shallow, depth-capped for skeletons)."""
    if not isinstance(schema, dict):
        return {"type": "object"}
    if depth > 4:
        return {"type": "object"}
    if "$ref" in schema:
        target = components.get(str(schema["$ref"]).rsplit("/", 1)[-1])
        return _resolve_refs(target, components, depth + 1) if target else {"type": "object"}
    if "allOf" in schema and isinstance(schema["allOf"], list):
        merged: Dict[str, Any] = {}
        for part in schema["allOf"]:
            merged.update(_resolve_refs(part, components, depth + 1))
        merged.update({k: v for k, v in schema.items() if k != "allOf"})
        schema = merged
    out: Dict[str, Any] = {}
    for key, value in schema.items():
        if key == "properties" and isinstance(value, dict):
            out[key] = {
                pk: _resolve_refs(pv, components, depth + 1) for pk, pv in value.items()
            }
        elif key in ("items", "additionalProperties") and isinstance(value, dict):
            out[key] = _resolve_refs(value, components, depth + 1)
        elif key in ("anyOf", "oneOf") and isinstance(value, list):
            out[key] = [_resolve_refs(v, components, depth + 1) for v in value]
        else:
            out[key] = value
    return _collapse_nullable(out)


def _build_parameters(route, method: str) -> Dict[str, Any]:
    """OpenAI-function parameters schema from FastAPI's own OpenAPI generation.

    workspace_id never enters the schema — the executor injects workspace
    scoping server-side; an LLM-supplied value would invite cross-tenant input.
    """
    from fastapi.openapi.utils import get_openapi

    spec = get_openapi(title="scaffold", version="0", routes=[route])
    components = (spec.get("components") or {}).get("schemas") or {}
    path_item = next(iter((spec.get("paths") or {}).values()), None) or {}
    op = path_item.get(method.lower()) or {}

    properties: Dict[str, Any] = {}
    required: List[str] = []

    content = (op.get("requestBody") or {}).get("content") or {}
    body_schema = (content.get("application/json") or next(iter(content.values()), {})).get("schema")
    if body_schema:
        body = _resolve_refs(body_schema, components)
        for pname, pschema in (body.get("properties") or {}).items():
            if pname != "workspace_id":
                properties[pname] = pschema
        required.extend(r for r in (body.get("required") or []) if r in properties)

    for param in op.get("parameters") or []:
        pname = param.get("name")
        if param.get("in") not in ("path", "query") or not pname or pname == "workspace_id":
            continue
        pschema = _resolve_refs(param.get("schema") or {}, components)
        if param.get("description") and "description" not in pschema:
            pschema["description"] = param["description"]
        properties[pname] = pschema
        if param.get("required") and pname not in required:
            required.append(pname)

    return {"type": "object", "properties": properties, "required": required}


def _describe(route, method: str) -> str:
    doc = inspect.getdoc(route.endpoint) or ""
    first_para = " ".join(doc.split("\n\n")[0].split()) if doc.strip() else ""
    description = first_para or f"{method.upper()} {route.path} — describe for tool selection."
    return description[:_MAX_DESCRIPTION_CHARS]


def _source_ref(route) -> str:
    try:
        src = Path(inspect.getsourcefile(route.endpoint)).resolve()
        line = inspect.getsourcelines(route.endpoint)[1]
        try:
            src = src.relative_to(_ORCH_ROOT)
        except ValueError:
            pass
        return f"{src}:{line}"
    except (TypeError, OSError):
        return "source unavailable"


def _render_action(
    *,
    name: str,
    method: str,
    path: str,
    description: str,
    category: str,
    parameters: Dict[str, Any],
    permission_level: str,
    su_locked: bool,
    source_ref: str,
) -> str:
    params_block = textwrap.indent(
        pprint.pformat(parameters, width=80, sort_dicts=False), " " * 8
    ).lstrip()
    requires_confirmation = permission_level == "destructive"
    if su_locked:
        tier_note = (
            "REVIEW: super_admin_only=True — source router is su-locked "
            "(require_super_admin, obs tier S6/S7)"
        )
        su_line = (
            "super_admin_only=True,  # CURATE: REVIEW — source router is "
            "su-locked (obs tier)"
        )
    else:
        tier_note = "operator (super_admin_only=False) — confirm not obs/oversight"
        su_line = "super_admin_only=False,"
    return f'''"""Scaffolded ActionDefinition — {method} {path} ({source_ref}).

Generated by scripts/scaffold_platform_tool.py (PRD-143 S9). STAGING ONLY:
after review, move into modules/tools/discovery/actions_<domain>.py and
register in platform_actions.py BY HAND (FR-7 — no auto-registration).
"""
from .action_registry import ActionDefinition, ActionRegistry

# CURATE: review EVERY item before registering (PRD-143 FR-7):
# CURATE: [ ] name             — '{name}' guessed from {method} {path}
# CURATE: [ ] description      — seeded from the route docstring; rewrite for LLM tool selection
# CURATE: [ ] params           — verify properties/required against the endpoint contract
# CURATE: [ ] tier             — {tier_note}
# CURATE: [ ] permission_level — '{permission_level}' guessed from HTTP verb {method}; confirm
# CURATE: registration in platform_actions.py is MANUAL — nothing imports this file.


def register_scaffolded_actions(registry: ActionRegistry) -> None:
    """Register {name} once curated — called from platform_actions.py by hand."""
    registry.register(ActionDefinition(
        name="{name}",
        description=(
            {description!r}
        ),
        category="{category}",
        parameters={params_block},
        permission_level="{permission_level}",  # CURATE: guessed from {method} — confirm
        requires_confirmation={requires_confirmation},
        workspace_scoped=True,
        {su_line}
        tags={[category]!r},
    ))
'''


def _render_handler(
    *,
    name: str,
    method: str,
    path: str,
    description: str,
    module_path: str,
    source_ref: str,
    endpoint_name: str,
) -> str:
    handler_fn = name.removeprefix("platform_")
    doc_line = description.replace('"""', "'''")
    return f'''"""Scaffolded handler — {method} {path} ({source_ref}).

Generated by scripts/scaffold_platform_tool.py (PRD-143 S9). STAGING ONLY.

# CURATE: implement by REUSING the service layer the source route uses
# CURATE: ({source_ref} — {endpoint_name}); never reimplement business logic.
# CURATE: keep it workspace-scoped — filter every query by workspace_id.
"""
from __future__ import annotations

import logging
from typing import Any, Dict
from uuid import UUID

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


async def {handler_fn}(
    db: Session, workspace_id: UUID, params: Dict[str, Any]
) -> Dict[str, Any]:
    """{doc_line}"""
    # CURATE: reuse the same service/DB calls as {source_ref}. Return
    # {{"success": True, "data": ...}} on success, {{"success": False, "error": str}} on failure.
    raise NotImplementedError(
        "scaffold for {name}: implement against the {module_path} service layer, "
        "then register by hand"
    )
'''


def scaffold(
    module_path: str, endpoint: str, out_dir: Optional[Path] = None
) -> ScaffoldResult:
    """Emit the action + handler skeleton pair for one router endpoint."""
    method, _, path = endpoint.strip().partition(" ")
    path = path.strip()
    if not method or not path.startswith("/"):
        raise ValueError("endpoint must be 'METHOD /path', e.g. 'POST /api/agents'")
    resolved_out = _assert_safe_out_dir(out_dir) if out_dir is not None else None

    module = _import_router_module(module_path)
    router, route = _find_route(module, method, path)
    method = method.upper()

    name = derive_action_name(method, route.path)
    su_locked = _is_super_admin_locked(router, route)
    description = _describe(route, method)
    source_ref = _source_ref(route)
    literals = [
        s for s in route.path.strip("/").split("/")
        if s and not s.startswith("{") and s not in ("api", "v1", "v2")
    ]
    category = literals[0].replace("-", "_") if literals else "platform"

    action_source = _render_action(
        name=name,
        method=method,
        path=route.path,
        description=description,
        category=category,
        parameters=_build_parameters(route, method),
        permission_level=verb_to_permission_level(method),
        su_locked=su_locked,
        source_ref=source_ref,
    )
    handler_source = _render_handler(
        name=name,
        method=method,
        path=route.path,
        description=description,
        module_path=module_path,
        source_ref=source_ref,
        endpoint_name=getattr(route.endpoint, "__name__", "endpoint"),
    )

    base = name.removeprefix("platform_")
    result = ScaffoldResult(
        name=name,
        super_admin_only=su_locked,
        action_filename=f"actions_{base}_scaffold.py",
        handler_filename=f"handlers_{base}_scaffold.py",
        action_source=action_source,
        handler_source=handler_source,
    )
    if resolved_out is not None:
        resolved_out.mkdir(parents=True, exist_ok=True)
        result.action_path = resolved_out / result.action_filename
        result.handler_path = resolved_out / result.handler_filename
        result.action_path.write_text(action_source)
        result.handler_path.write_text(handler_source)
    return result


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Scaffold an ActionDefinition + handler skeleton from a FastAPI route "
        "(scaffold-then-curate, PRD-143 FR-7 — never auto-registers).",
    )
    parser.add_argument(
        "router_module", help="router module relative to orchestrator/, e.g. api/agents.py"
    )
    parser.add_argument("endpoint", help="'METHOD /path', e.g. 'POST /api/agents'")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="staging dir for the two skeleton files (default: stdout). "
        "modules/tools/discovery/ is refused.",
    )
    args = parser.parse_args(argv)
    try:
        result = scaffold(
            args.router_module,
            args.endpoint,
            Path(args.out_dir) if args.out_dir else None,
        )
    except (ValueError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if result.action_path:
        print(f"wrote {result.action_path}")
        print(f"wrote {result.handler_path}")
    else:
        print(f"# ==== {result.action_filename} ====")
        print(result.action_source)
        print(f"# ==== {result.handler_filename} ====")
        print(result.handler_source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
