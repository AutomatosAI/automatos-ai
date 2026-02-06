#!/usr/bin/env python3
"""
GitHub Plugin Harvester
=======================

Clones curated plugin repos, bridges their varied manifest formats into our
manifest.json standard, and feeds them through the upload+scan pipeline.
Auto-approves safe plugins (risk < 20).

Usage:
    python scripts/harvest_plugins.py [--dry-run]
"""

import argparse
import asyncio
import io
import json
import logging
import shutil
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Path setup — same pattern as other scripts in this directory
# ---------------------------------------------------------------------------
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.database.database import SessionLocal
from core.services.marketplace_s3 import MarketplaceS3Service
from core.services.plugin_security_scanner import PluginScanService
from core.services.plugin_upload_service import PluginUploadService
from core.models.marketplace_plugins import MarketplacePlugin

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("harvest_plugins")

# ---------------------------------------------------------------------------
# Curated repo definitions
# ---------------------------------------------------------------------------

CURATED_REPOS: List[Dict] = [
    {
        "url": "https://github.com/dhofheinz/open-plugins.git",
        "name": "dhofheinz/open-plugins",
        "plugins": [
            "10x-fullstack-engineer",
            "git-commit-assistant",
            "spec-refine",
            "afk-mode",
            "plugin-quickstart-generator",
            "marketplace-validator-plugin",
        ],
    },
    {
        "url": "https://github.com/the-answerai/alphaagent-team.git",
        "name": "the-answerai/alphaagent-team",
        "plugins": [
            "aai-architecture",
            "aai-core",
            "aai-dev-backend",
            "aai-dev-frontend",
            "aai-dev-fullstack",
            "aai-devops",
            "aai-docs",
            "aai-blog",
            "aai-quality",
            "aai-testing",
            "aai-pm-github",
            "aai-stack-nextjs",
            "aai-stack-react",
            "aai-stack-tailwind",
        ],
    },
    {
        "url": "https://github.com/DNYoussef/context-cascade.git",
        "name": "DNYoussef/context-cascade",
        "plugins": [
            "12fa-core",
            "12fa-three-loop",
            "12fa-security",
            "12fa-visual-docs",
        ],
    },
    {
        "url": "https://github.com/masuP9/a11y-specialist-skills.git",
        "name": "masuP9/a11y-specialist-skills",
        "plugins": [
            "a11y-specialist-skills",
        ],
    },
    # ----- Business / Marketing / Sales / HR plugins -----
    {
        "url": "https://github.com/Salesably/salesably-marketplace.git",
        "name": "Salesably/salesably-marketplace",
        "plugins": [
            "marketing-skills",
            "sales-skills",
        ],
    },
    {
        "url": "https://github.com/coreyhaines31/marketingskills.git",
        "name": "coreyhaines31/marketingskills",
        "plugins": [
            "marketing-skills",  # slug from marketplace.json — will be remapped
        ],
        "slug_remap": {"marketing-skills": "advanced-marketing-skills"},
    },
    {
        "url": "https://github.com/muratcankoylan/ralph-wiggum-marketer.git",
        "name": "muratcankoylan/ralph-wiggum-marketer",
        "plugins": [
            "ralph-wiggum-marketer",
        ],
    },
    {
        "url": "https://github.com/kivilaid/plugin-marketplace.git",
        "name": "kivilaid/plugin-marketplace",
        "plugins": [
            "seo-content-creation",
            "seo-technical-optimization",
            "seo-analysis-monitoring",
            "content-marketing",
            "customer-sales-automation",
            "business-analytics",
            "hr-legal-compliance",
            "payment-processing",
        ],
    },
]

# Files/dirs to skip when zipping plugin contents
SKIP_PATTERNS = {".git", "__pycache__", ".DS_Store", "node_modules", ".pytest_cache"}


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def clone_repo(url: str, dest: Path) -> bool:
    """Shallow-clone a repo. Returns True on success."""
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", url, str(dest)],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        logger.error("Failed to clone %s: %s", url, e)
        return False


# ---------------------------------------------------------------------------
# Plugin discovery — detect repo format and find plugin directories
# ---------------------------------------------------------------------------

class DiscoveredPlugin:
    """A plugin found in a cloned repo."""

    def __init__(
        self,
        slug: str,
        content_dir: Path,
        source_manifest: dict,
        format_type: str,
        repo_name: str,
    ):
        self.slug = slug
        self.content_dir = content_dir
        self.source_manifest = source_manifest
        self.format_type = format_type  # "plugin_json" | "marketplace_single" | "marketplace_multi"
        self.repo_name = repo_name


def _resolve_content_dir(repo_dir: Path, entry: dict) -> Path:
    """Resolve the plugin content directory from marketplace entry fields.

    Checks (in order): installation.path, source, falls back to repo root.
    """
    # installation.path (context-cascade style)
    install_path = entry.get("installation", {}).get("path", "")
    if install_path:
        resolved = (repo_dir / install_path).resolve()
        if resolved.is_dir():
            return resolved

    # source field (open-plugins / alphaagent-team style: "./plugins/name")
    source = entry.get("source", "")
    if source:
        resolved = (repo_dir / source).resolve()
        if resolved.is_dir():
            return resolved

    return repo_dir


def _extract_slug(entry: dict, fallback: str = "") -> str:
    """Extract slug from a marketplace entry, trying id → slug → name."""
    slug = entry.get("id") or entry.get("slug") or ""
    if not slug:
        slug = entry.get("name", fallback).lower().replace(" ", "-")
    return slug


def discover_plugins(repo_dir: Path, repo_name: str) -> List[DiscoveredPlugin]:
    """Auto-detect repo format and find all plugin directories."""
    found: List[DiscoveredPlugin] = []

    # ------------------------------------------------------------------
    # Check for .claude-plugin/marketplace.json first
    # ------------------------------------------------------------------
    marketplace_path = repo_dir / ".claude-plugin" / "marketplace.json"
    if marketplace_path.exists():
        try:
            marketplace = json.loads(marketplace_path.read_text("utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Bad marketplace.json in %s: %s", repo_name, e)
            return found

        plugins_array = marketplace.get("plugins")

        if plugins_array and isinstance(plugins_array, list):
            # Multi-plugin marketplace
            for entry in plugins_array:
                slug = _extract_slug(entry)
                content_dir = _resolve_content_dir(repo_dir, entry)
                found.append(DiscoveredPlugin(
                    slug=slug,
                    content_dir=content_dir,
                    source_manifest=entry,
                    format_type="marketplace_multi",
                    repo_name=repo_name,
                ))
        else:
            # Single-plugin marketplace (no plugins[] array)
            slug = _extract_slug(marketplace)
            found.append(DiscoveredPlugin(
                slug=slug,
                content_dir=_resolve_content_dir(repo_dir, marketplace),
                source_manifest=marketplace,
                format_type="marketplace_single",
                repo_name=repo_name,
            ))
        return found

    # ------------------------------------------------------------------
    # Fall back to plugins/*/plugin.json pattern
    # ------------------------------------------------------------------
    plugins_dir = repo_dir / "plugins"
    if plugins_dir.is_dir():
        for child in sorted(plugins_dir.iterdir()):
            if not child.is_dir():
                continue
            pj = child / "plugin.json"
            if pj.exists():
                try:
                    manifest = json.loads(pj.read_text("utf-8"))
                except (json.JSONDecodeError, OSError) as e:
                    logger.warning("Bad plugin.json in %s/%s: %s", repo_name, child.name, e)
                    continue
                slug = manifest.get("name", child.name).lower().replace(" ", "-")
                found.append(DiscoveredPlugin(
                    slug=slug,
                    content_dir=child,
                    source_manifest=manifest,
                    format_type="plugin_json",
                    repo_name=repo_name,
                ))

    # ------------------------------------------------------------------
    # Also check root-level plugin.json (single-plugin repo)
    # ------------------------------------------------------------------
    root_pj = repo_dir / "plugin.json"
    if root_pj.exists() and not found:
        try:
            manifest = json.loads(root_pj.read_text("utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
        else:
            slug = manifest.get("name", repo_dir.name).lower().replace(" ", "-")
            found.append(DiscoveredPlugin(
                slug=slug,
                content_dir=repo_dir,
                source_manifest=manifest,
                format_type="plugin_json",
                repo_name=repo_name,
            ))

    return found


# ---------------------------------------------------------------------------
# Manifest bridge — convert any source format to our manifest.json
# ---------------------------------------------------------------------------

def _count_content_items(plugin_dir: Path, subdir: str) -> List[str]:
    """List content items in a content subdirectory (skills/, commands/, etc.).

    Items can be:
    - Direct files (e.g. commands/review.md)
    - Subdirectories containing SKILL.md (e.g. skills/react-patterns/SKILL.md)
    """
    target = plugin_dir / subdir
    if not target.is_dir():
        return []
    items = []
    for entry in sorted(target.iterdir()):
        if entry.name in SKIP_PATTERNS:
            continue
        if entry.is_file():
            items.append(entry.name)
        elif entry.is_dir():
            # Skill directories contain SKILL.md — count the directory as a skill
            items.append(entry.name)
    return items


def _find_skill_files(plugin_dir: Path) -> List[str]:
    """Find skills: subdirs with SKILL.md, or files in skills/ dir."""
    skills = _count_content_items(plugin_dir, "skills")
    # Also check for root SKILL.md (common in single-plugin repos)
    skill_md = plugin_dir / "SKILL.md"
    if skill_md.exists() and "SKILL.md" not in skills:
        skills.append("SKILL.md")
    return skills


def bridge_to_manifest(source: dict, plugin_dir: Path, slug: str) -> dict:
    """Convert any source manifest format to our manifest.json standard."""
    if not slug:
        raise ValueError("Plugin slug cannot be empty")

    # Extract author — can be string or dict with "name" key
    author_raw = source.get("author", "")
    if isinstance(author_raw, dict):
        author = author_raw.get("name", "")
    else:
        author = str(author_raw)

    # Extract tags/keywords — always normalise to List[str]
    tags = source.get("tags", source.get("keywords", []))
    if tags is None:
        tags = []
    elif isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",") if t.strip()]
    else:
        tags = [str(t) for t in tags]

    # Discover content by scanning actual directories first
    skills = _find_skill_files(plugin_dir)
    commands = _count_content_items(plugin_dir, "commands")
    agents = _count_content_items(plugin_dir, "agents")
    hooks = _count_content_items(plugin_dir, "hooks")

    # For plugins with relative-path refs or component lists (context-cascade),
    # count from the source manifest if local dirs are empty
    if not skills and not commands:
        components = source.get("components", {})
        if components:
            skills = components.get("skills", []) or []
            commands = components.get("commands", []) or []
            agents = components.get("agents", []) or []
            hooks = components.get("hooks", []) or []
        else:
            # Also check if source manifest has direct arrays of path refs
            for key, lst_ref in [("skills", skills), ("commands", commands)]:
                refs = source.get(key, [])
                if isinstance(refs, list) and refs and not lst_ref:
                    if key == "skills":
                        skills = [Path(r).name for r in refs if isinstance(r, str)]
                    else:
                        commands = [Path(r).name for r in refs if isinstance(r, str)]

    return {
        "slug": slug,
        "name": source.get("name", slug),
        "version": source.get("version", "1.0.0"),
        "description": source.get("description", ""),
        "author": author,
        "license": source.get("license", ""),
        "tags": tags,
        "contents": {
            "skills": skills,
            "commands": commands,
            "agents": agents,
            "hooks": hooks,
        },
    }


# ---------------------------------------------------------------------------
# Zip builder
# ---------------------------------------------------------------------------

def _should_skip(path: Path) -> bool:
    """Check if a path component matches skip patterns."""
    return any(part in SKIP_PATTERNS for part in path.parts)


def _resolve_relative_refs(plugin_dir: Path, source_manifest: dict) -> List[Path]:
    """Resolve relative path references in plugin.json (context-cascade style).

    Some plugins reference files via relative paths like "../../commands/foo.md".
    This resolves those to absolute paths for inclusion in the zip.
    Returns list of (absolute_path, arcname_prefix) pairs.
    """
    extra_files: List[Path] = []
    for key in ("commands", "skills"):
        refs = source_manifest.get(key, [])
        if not isinstance(refs, list):
            # Could be a single string path like "../../agents"
            refs = [refs] if isinstance(refs, str) else []
        for ref in refs:
            if not isinstance(ref, str) or not ref.startswith(".."):
                continue
            resolved = (plugin_dir / ref).resolve()
            if resolved.is_file():
                extra_files.append(resolved)
            elif resolved.is_dir():
                for f in resolved.rglob("*"):
                    if f.is_file() and not _should_skip(f.relative_to(resolved)):
                        extra_files.append(f)
    return extra_files


def create_plugin_zip(plugin_dir: Path, manifest: dict, source_manifest: Optional[dict] = None) -> bytes:
    """Create in-memory zip with all plugin files + manifest.json at root.

    For plugins with relative-path references (context-cascade), also resolves
    and includes those external files under their content-type directories.
    """
    buf = io.BytesIO()
    added_arcnames: set = set()

    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # Inject our generated manifest.json
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))
        added_arcnames.add("manifest.json")

        # Walk plugin directory and add all files
        for file_path in sorted(plugin_dir.rglob("*")):
            if not file_path.is_file():
                continue
            rel = file_path.relative_to(plugin_dir)
            if _should_skip(rel):
                continue
            # Skip the original plugin.json — we've replaced it with manifest.json
            if rel.name == "plugin.json" and len(rel.parts) == 1:
                continue
            # Skip .claude-plugin directory contents
            if rel.parts and rel.parts[0] == ".claude-plugin":
                continue
            # Skip .git at any level
            if ".git" in rel.parts:
                continue
            arcname = str(rel)
            if arcname not in added_arcnames:
                zf.write(file_path, arcname)
                added_arcnames.add(arcname)

        # Resolve relative-path references (context-cascade style)
        if source_manifest:
            for key in ("commands", "skills", "agents"):
                refs = source_manifest.get(key, [])
                if isinstance(refs, str):
                    refs = [refs]
                if not isinstance(refs, list):
                    continue
                for ref in refs:
                    if not isinstance(ref, str) or not ref.startswith(".."):
                        continue
                    resolved = (plugin_dir / ref).resolve()
                    if resolved.is_file():
                        arcname = f"{key}/{resolved.name}"
                        if arcname not in added_arcnames:
                            zf.write(resolved, arcname)
                            added_arcnames.add(arcname)
                    elif resolved.is_dir():
                        for f in sorted(resolved.rglob("*")):
                            if not f.is_file():
                                continue
                            rel_to_resolved = f.relative_to(resolved)
                            if _should_skip(rel_to_resolved):
                                continue
                            arcname = f"{key}/{rel_to_resolved}"
                            if arcname not in added_arcnames:
                                zf.write(f, arcname)
                                added_arcnames.add(arcname)

    return buf.getvalue()


# ---------------------------------------------------------------------------
# Main harvest pipeline
# ---------------------------------------------------------------------------

async def harvest(dry_run: bool = False, repair_s3: bool = False) -> None:
    """Run the full harvest pipeline."""
    results: List[Dict] = []
    tmp_base = Path(tempfile.mkdtemp(prefix="harvest_plugins_"))

    try:
        # Set up services (only needed for real runs)
        db = None
        upload_service = None
        if not dry_run:
            db = SessionLocal()
            s3_service = MarketplaceS3Service()
            scan_service = PluginScanService(db)
            upload_service = PluginUploadService(db, s3_service, scan_service)

        for repo_def in CURATED_REPOS:
            repo_url = repo_def["url"]
            repo_name = repo_def["name"]
            curated_slugs = set(repo_def["plugins"])
            slug_remap = repo_def.get("slug_remap", {})

            logger.info("=" * 60)
            logger.info("Cloning %s", repo_name)
            logger.info("=" * 60)

            repo_dir = tmp_base / repo_name.replace("/", "_")
            if not clone_repo(repo_url, repo_dir):
                for slug in curated_slugs:
                    results.append({
                        "slug": slug,
                        "repo": repo_name,
                        "status": "CLONE_FAILED",
                        "verdict": "-",
                        "approval": "-",
                    })
                continue

            # Discover all plugins in this repo
            all_plugins = discover_plugins(repo_dir, repo_name)
            discovered_slugs = {p.slug for p in all_plugins}
            logger.info(
                "Discovered %d plugins: %s",
                len(all_plugins),
                ", ".join(sorted(discovered_slugs)),
            )

            # Warn about curated plugins not found
            missing = curated_slugs - discovered_slugs
            for slug in sorted(missing):
                logger.warning("Curated plugin '%s' not found in %s", slug, repo_name)
                results.append({
                    "slug": slug,
                    "repo": repo_name,
                    "status": "NOT_FOUND",
                    "verdict": "-",
                    "approval": "-",
                })

            # Process each curated plugin
            for plugin in all_plugins:
                if plugin.slug not in curated_slugs:
                    logger.info("Skipping non-curated plugin: %s", plugin.slug)
                    continue

                logger.info("-" * 40)

                # Apply slug remapping if defined (avoids cross-repo slug collisions)
                final_slug = slug_remap.get(plugin.slug, plugin.slug)
                if final_slug != plugin.slug:
                    logger.info("Remapping slug: %s → %s", plugin.slug, final_slug)
                    plugin.slug = final_slug

                logger.info("Processing: %s (format=%s)", plugin.slug, plugin.format_type)

                # Bridge manifest
                manifest = bridge_to_manifest(
                    plugin.source_manifest, plugin.content_dir, plugin.slug
                )
                logger.info(
                    "  Manifest: name=%s, version=%s, skills=%d, commands=%d",
                    manifest["name"],
                    manifest["version"],
                    len(manifest["contents"]["skills"]),
                    len(manifest["contents"]["commands"]),
                )

                if dry_run:
                    logger.info("  [DRY RUN] Would upload: %s", plugin.slug)
                    logger.info("  Manifest preview:\n%s", json.dumps(manifest, indent=2))
                    results.append({
                        "slug": plugin.slug,
                        "repo": repo_name,
                        "status": "DRY_RUN",
                        "verdict": "-",
                        "approval": "-",
                    })
                    continue

                # Check for existing plugin with same slug
                existing = db.query(MarketplacePlugin).filter(
                    MarketplacePlugin.slug == plugin.slug
                ).first()
                if existing and not repair_s3:
                    logger.info("  Plugin '%s' already exists (id=%s), skipping", plugin.slug, existing.id)
                    results.append({
                        "slug": plugin.slug,
                        "repo": repo_name,
                        "status": "ALREADY_EXISTS",
                        "verdict": existing.security_status,
                        "approval": existing.approval_status,
                    })
                    continue

                if existing and repair_s3:
                    # Re-upload S3 content only — no new DB record or scan
                    zip_bytes = create_plugin_zip(plugin.content_dir, manifest, plugin.source_manifest)
                    try:
                        await s3_service.extract_plugin(existing.slug, existing.version, zip_bytes)
                        logger.info("  Repaired S3 content for %s@%s", existing.slug, existing.version)
                        results.append({
                            "slug": plugin.slug,
                            "repo": repo_name,
                            "status": "S3_REPAIRED",
                            "verdict": existing.security_status,
                            "approval": existing.approval_status,
                        })
                    except Exception as e:
                        logger.error("  Failed to repair S3 for %s: %s", plugin.slug, e)
                        results.append({
                            "slug": plugin.slug,
                            "repo": repo_name,
                            "status": f"REPAIR_ERROR: {e}",
                            "verdict": "-",
                            "approval": "-",
                        })
                    continue

                # Build zip (pass source_manifest for relative-path resolution)
                zip_bytes = create_plugin_zip(plugin.content_dir, manifest, plugin.source_manifest)
                logger.info("  Zip size: %d bytes", len(zip_bytes))

                # Upload through the standard pipeline
                try:
                    # Build source URL based on format type
                    base_url = repo_def["url"].replace(".git", "")
                    if plugin.format_type == "plugin_json":
                        source_url = f"{base_url}/tree/main/plugins/{plugin.slug}"
                    elif plugin.format_type == "marketplace_multi":
                        install_path = plugin.source_manifest.get("installation", {}).get("path", "")
                        source_url = f"{base_url}/tree/main/{install_path}" if install_path else base_url
                    else:
                        # Derive path from content_dir relative to repo root
                        try:
                            rel = plugin.content_dir.relative_to(repo_dir)
                            source_url = f"{base_url}/tree/main/{rel}" if str(rel) != "." else base_url
                        except ValueError:
                            source_url = base_url

                    plugin_record = await upload_service.upload_plugin(
                        zip_bytes=zip_bytes,
                        source_type="github",
                        source_url=source_url,
                        uploaded_by="harvest-script",
                    )

                    # Auto-approve safe plugins
                    if plugin_record.security_status == "safe":
                        plugin_record.approval_status = "approved"
                        plugin_record.approved_by = "harvest-script"
                        plugin_record.approved_at = datetime.utcnow()
                        db.commit()
                        logger.info("  Auto-approved %s (safe)", plugin.slug)

                    results.append({
                        "slug": plugin.slug,
                        "repo": repo_name,
                        "status": "UPLOADED",
                        "verdict": plugin_record.security_status,
                        "approval": plugin_record.approval_status,
                    })

                except Exception as e:
                    logger.error("  Failed to upload %s: %s", plugin.slug, e)
                    if db:
                        try:
                            db.rollback()
                        except Exception:
                            pass
                    results.append({
                        "slug": plugin.slug,
                        "repo": repo_name,
                        "status": f"ERROR: {e}",
                        "verdict": "-",
                        "approval": "-",
                    })

        # ------------------------------------------------------------------
        # Print summary table
        # ------------------------------------------------------------------
        print("\n" + "=" * 80)
        print("HARVEST SUMMARY")
        print("=" * 80)
        print(f"{'Slug':<35} {'Repo':<30} {'Status':<15} {'Verdict':<15} {'Approval':<10}")
        print("-" * 80)

        uploaded = 0
        approved = 0
        errors = 0
        for r in results:
            print(f"{r['slug']:<35} {r['repo']:<30} {r['status']:<15} {r['verdict']:<15} {r['approval']:<10}")
            if r["status"] == "UPLOADED":
                uploaded += 1
            if r["approval"] == "approved":
                approved += 1
            if r["status"].startswith("ERROR"):
                errors += 1

        print("-" * 80)
        print(f"Total: {len(results)} | Uploaded: {uploaded} | Auto-approved: {approved} | Errors: {errors}")
        if dry_run:
            print("[DRY RUN MODE — no writes performed]")
        print("=" * 80)

    finally:
        # Clean up temp directory
        shutil.rmtree(tmp_base, ignore_errors=True)
        if db:
            db.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Harvest curated plugins from GitHub repos into the marketplace"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Clone and discover plugins but don't upload or scan",
    )
    parser.add_argument(
        "--repair-s3",
        action="store_true",
        help="Re-upload S3 content for existing plugins whose files are missing",
    )
    args = parser.parse_args()

    asyncio.run(harvest(dry_run=args.dry_run, repair_s3=args.repair_s3))


if __name__ == "__main__":
    main()
