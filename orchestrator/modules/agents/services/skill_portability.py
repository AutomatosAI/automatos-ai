"""
Spec-conformant skill import / export (PRD-202 S1)
==================================================

Aligns the in-house skill schema with the **Agent Skills open standard** so
that:

* any standard-conformant folder — a ``SKILL.md`` with ``name`` + ``description``
  frontmatter, optional ``scripts/`` and resource files — **imports** cleanly to
  a ``Skill`` row (+ ``SkillFile`` index rows), with the frontmatter
  ``description`` persisted as the **L1 trigger text**; and
* any ``Skill`` row **exports** back to that same folder shape (portable out —
  an external runner would accept it).

The DB stays the source of truth (the standard has no tenancy — that is
Automatos's addition). Provenance is written in the canonical ``scheme:ref``
form via :mod:`skill_source_scheme`. Every import runs the 2-stage security
scanner (S4) — static always, the LLM stage ON for third-party/external
sources — and auto-blocks on a critical finding.

Reuses (not re-implements): ``parse_yaml_frontmatter`` (the frontmatter reader
the loader already ships), the scanner's ``scan_skill_content`` (quick_scan +
optional llm_security_scan — no new scanner, dossier E), and the canonical
provenance scheme.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from modules.agents.services.skill_source_scheme import (
    canonicalize_skill_source,
    is_external_source,
)

logger = logging.getLogger(__name__)

# Files whose parent dir is one of these are L3 resources (load on demand).
_L3_SCRIPT_DIR = "scripts"
_L3_RESOURCE_DIRS = ("scripts", "examples", "data", "resources", "references", "assets")

# Frontmatter keys that are skill-schema columns, not free metadata.
_STRUCTURED_KEYS = {"name", "description", "version", "tags", "category", "type", "skill_type"}


# ---------------------------------------------------------------------------
# Scan verdict (S4 rides this path) — thin wrapper over the existing scanner.
# ---------------------------------------------------------------------------

async def scan_skill_for_import(
    content: str,
    filename: str,
    *,
    run_llm_scan: bool,
) -> Dict[str, Any]:
    """Run the 2-stage scanner on skill content and return a verdict dict.

    Static ``quick_scan`` always runs; the LLM stage runs only when
    ``run_llm_scan`` (third-party/external sources). A critical static finding
    auto-blocks (mirrors ``scan_plugin``'s critical-static short-circuit) — the
    LLM stage is not consulted because the verdict is already terminal.

    Returns ``{"verdict": "passed"|"blocked"|"review", "findings": [...],
    "llm_stage_run": bool}``.
    """
    from core.services.plugin_security_scanner import quick_scan

    static_findings = quick_scan(content, filename=filename)
    critical = [f for f in static_findings if f.severity == "critical"]
    findings_payload = [
        {"type": f.type, "severity": f.severity, "line": f.line, "description": f.description}
        for f in static_findings
    ]

    if critical:
        return {"verdict": "blocked", "findings": findings_payload, "llm_stage_run": False}

    if not run_llm_scan:
        # Trusted hot path (builtin / workspace-authored): static-only.
        return {"verdict": "passed", "findings": findings_payload, "llm_stage_run": False}

    # External source: LLM stage ON (dossier ClawHub incident D.3).
    from core.services.plugin_security_scanner import llm_security_scan

    llm = await llm_security_scan({filename: content})
    if llm.status == "failed":
        verdict = "review"
    elif llm.risk_score >= 70:
        verdict = "blocked"
    elif llm.risk_score >= 20:
        verdict = "review"
    else:
        verdict = "passed"
    return {"verdict": verdict, "findings": findings_payload, "llm_stage_run": True}


# ---------------------------------------------------------------------------
# Import — standard folder -> Skill (+ SkillFile) rows
# ---------------------------------------------------------------------------

def _find_skill_md(folder: Path) -> Optional[Path]:
    """Locate the SKILL.md at the folder root (case-insensitive)."""
    if not folder.exists() or not folder.is_dir():
        return None
    for child in folder.iterdir():
        if child.is_file() and child.name.lower() == "skill.md":
            return child
    return None


def _classify_file(rel_path: Path) -> Optional[tuple[str, int]]:
    """Return (file_type, load_level) for a bundled file, or None to skip."""
    if rel_path.name.lower() == "skill.md":
        return ("core", 2)
    parent = rel_path.parent.name
    if parent == _L3_SCRIPT_DIR:
        return ("script", 3)
    if parent in _L3_RESOURCE_DIRS:
        return ("example" if parent in ("examples", "data") else "resource", 3)
    if rel_path.suffix.lower() == ".md":
        return ("resource", 3)
    return None


def _index_bundle_files(db, skill_id: int, skill_dir: Path) -> List[Dict[str, Any]]:
    """Create SkillFile index rows for every bundled file. Returns their specs."""
    from core.models.core import SkillFile

    db.query(SkillFile).filter(SkillFile.skill_id == skill_id).delete()

    indexed: List[Dict[str, Any]] = []
    for path in sorted(skill_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(skill_dir)
        classified = _classify_file(rel)
        if classified is None:
            continue
        file_type, load_level = classified
        try:
            raw = path.read_text(encoding="utf-8")
            size = len(raw.encode("utf-8"))
            tokens = len(raw) // 4
        except Exception:
            size, tokens = 0, 0
        db.add(SkillFile(
            skill_id=skill_id,
            file_path=str(rel),
            file_type=file_type,
            load_level=load_level,
            file_size_bytes=size,
            estimated_tokens=tokens,
        ))
        indexed.append({"file_path": str(rel), "file_type": file_type, "load_level": load_level})
    return indexed


async def import_standard_skill_folder(
    db,
    folder_path: str,
    *,
    source_scheme: str,
    source_ref: Optional[str] = None,
    workspace_id: Optional[Any] = None,
    actor: str = "import",
    run_llm_scan: Optional[bool] = None,
) -> Dict[str, Any]:
    """Import a spec-conformant skill folder into a ``Skill`` (+ ``SkillFile``) rows.

    Args:
        db: SQLAlchemy session.
        folder_path: Path to the skill folder (contains ``SKILL.md``).
        source_scheme: Canonical provenance scheme (git|plugin|builtin|workspace).
        source_ref: Origin id (git source name, plugin slug, ...).
        workspace_id: Owning workspace (None = marketplace/global).
        actor: Who initiated the import (audit).
        run_llm_scan: Force LLM scan on/off; default derives from the scheme
            (external → on).

    Returns a result dict with ``success``, ``skill_id``, ``name``, ``verdict``.
    A critical scanner finding blocks the import (no row is written).
    """
    from core.models.core import Skill
    from modules.agents.services.skill_loader import parse_yaml_frontmatter

    folder = Path(folder_path)
    skill_md = _find_skill_md(folder)
    if skill_md is None:
        return {"success": False, "error": f"No SKILL.md found in {folder_path}"}

    content = skill_md.read_text(encoding="utf-8")
    yaml_data, body = parse_yaml_frontmatter(content)
    yaml_data = yaml_data or {}

    name = (yaml_data.get("name") or folder.name or "").strip()
    description = (yaml_data.get("description") or "").strip()
    if not name:
        return {"success": False, "error": "SKILL.md frontmatter missing required 'name'"}
    if not description:
        return {"success": False, "error": "SKILL.md frontmatter missing required 'description' (the L1 trigger text)"}

    skill_source = canonicalize_skill_source(source_scheme, source_ref)

    # --- Security scan (S4 rides the import path) ---
    external = run_llm_scan if run_llm_scan is not None else is_external_source(skill_source)
    scan = await scan_skill_for_import(content, filename=f"{name}/SKILL.md", run_llm_scan=external)
    if scan["verdict"] == "blocked":
        logger.warning("[skill-import] BLOCKED '%s' — critical/high-risk findings", name)
        return {
            "success": False,
            "verdict": "blocked",
            "findings": scan["findings"],
            "error": f"Skill '{name}' blocked by the security scanner — fix the flagged patterns and re-import.",
        }

    # description IS the L1 trigger text — persist it on the column AND into the
    # skill_metadata JSONB (which load_skill_metadata reads as L1).
    skill_metadata = dict(yaml_data)
    skill_metadata["description"] = description
    skill_metadata["security_scan"] = {
        "verdict": scan["verdict"],
        "llm_stage_run": scan["llm_stage_run"],
        "scanned_by": actor,
    }

    existing = (
        db.query(Skill)
        .filter(
            Skill.name == name,
            Skill.skill_source == skill_source,
            Skill.workspace_id == workspace_id,
        )
        .first()
    )

    if existing:
        existing.description = description
        existing.prompt_template = body
        existing.skill_version = str(yaml_data.get("version", existing.skill_version or "1.0.0"))
        existing.tags = yaml_data.get("tags", existing.tags)
        existing.category = yaml_data.get("category") or existing.category or "general"
        existing.filesystem_path = str(folder)
        existing.skill_metadata = skill_metadata
        existing.is_active = True
        skill = existing
    else:
        skill = Skill(
            name=name,
            description=description,
            skill_type=yaml_data.get("skill_type") or yaml_data.get("type") or "technical",
            category=yaml_data.get("category") or "general",
            skill_version=str(yaml_data.get("version", "1.0.0")),
            skill_source=skill_source,
            prompt_template=body,
            filesystem_path=str(folder),
            tags=yaml_data.get("tags", []),
            skill_metadata=skill_metadata,
            workspace_id=workspace_id,
            is_active=True,
        )
        db.add(skill)
        db.flush()

    indexed = _index_bundle_files(db, skill.id, folder)
    db.commit()

    logger.info(
        "[skill-import] imported '%s' (id=%s, source=%s, %d bundled files, verdict=%s)",
        name, skill.id, skill_source, len(indexed), scan["verdict"],
    )
    return {
        "success": True,
        "skill_id": skill.id,
        "name": name,
        "skill_source": skill_source,
        "description": description,
        "verdict": scan["verdict"],
        "bundled_files": indexed,
    }


# ---------------------------------------------------------------------------
# Bundle read — for L3 worker materialization (S3)
# ---------------------------------------------------------------------------

def collect_skill_bundle(filesystem_path: Optional[str]) -> Dict[str, str]:
    """Return ``{relative_path: text_content}`` for a skill's bundled files.

    Walks the skill's on-disk directory (``filesystem_path`` — what
    ``get_skill_script_path`` resolves against) and returns every classifiable
    script/resource file's text, EXCLUDING ``SKILL.md`` (the L2 body is the DB's
    job, not the worker's). This is the bundle S3 materializes into the
    workspace worker so a script's sibling files resolve at run time.
    """
    bundle: Dict[str, str] = {}
    if not filesystem_path:
        return bundle
    root = Path(filesystem_path)
    if not root.exists() or not root.is_dir():
        return bundle
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if rel.name.lower() == "skill.md":
            continue
        if _classify_file(rel) is None:
            continue
        try:
            bundle[str(rel)] = path.read_text(encoding="utf-8")
        except Exception:
            logger.warning("[skill-bundle] could not read %s (skipped)", rel, exc_info=True)
    return bundle


# ---------------------------------------------------------------------------
# Export — Skill row -> standard folder
# ---------------------------------------------------------------------------

def _build_frontmatter(skill) -> Dict[str, Any]:
    """Assemble the standard frontmatter dict from a Skill row (name+description first)."""
    fm: Dict[str, Any] = {
        "name": getattr(skill, "name", None),
        "description": getattr(skill, "description", None) or "",
    }
    version = getattr(skill, "skill_version", None)
    if version:
        fm["version"] = version
    tags = getattr(skill, "tags", None)
    if tags:
        fm["tags"] = list(tags)
    category = getattr(skill, "category", None)
    if category:
        fm["category"] = category
    # Carry any extra author metadata that isn't a structured column, minus
    # internal bookkeeping we don't export.
    meta = getattr(skill, "skill_metadata", None)
    if isinstance(meta, dict):
        for k, v in meta.items():
            if k in _STRUCTURED_KEYS or k in ("security_scan", "forked_from_skill_id"):
                continue
            fm.setdefault(k, v)
    return fm


def export_skill_to_folder(db, skill, dest_dir: str) -> str:
    """Emit a standard skill folder from a ``Skill`` row.

    Writes ``<dest_dir>/<name>/SKILL.md`` (frontmatter with name + description +
    body) and copies any bundled ``scripts/``/resource files that live under the
    skill's ``filesystem_path``. Returns the emitted skill-folder path. An
    external Agent-Skills runner would accept the result.
    """
    name = getattr(skill, "name", None) or "skill"
    out_dir = Path(dest_dir) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    frontmatter = _build_frontmatter(skill)
    fm_yaml = yaml.safe_dump(frontmatter, sort_keys=False, allow_unicode=True).strip()
    body = getattr(skill, "prompt_template", None) or ""
    skill_md = f"---\n{fm_yaml}\n---\n\n{body.strip()}\n"
    (out_dir / "SKILL.md").write_text(skill_md, encoding="utf-8")

    # Copy the L3 bundle (scripts + resources) from the on-disk source, if any.
    src_path = getattr(skill, "filesystem_path", None)
    copied = 0
    if src_path:
        src = Path(src_path)
        if src.exists() and src.is_dir():
            for path in sorted(src.rglob("*")):
                if not path.is_file():
                    continue
                rel = path.relative_to(src)
                if rel.name.lower() == "skill.md":
                    continue  # already emitted from the DB body (source of truth)
                if _classify_file(rel) is None:
                    continue
                target = out_dir / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                try:
                    target.write_bytes(path.read_bytes())
                    copied += 1
                except Exception:
                    logger.warning("[skill-export] could not copy %s", rel, exc_info=True)

    logger.info("[skill-export] exported '%s' to %s (%d bundled files)", name, out_dir, copied)
    return str(out_dir)
