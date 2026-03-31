"""
PRD-71: Skill Materializer
===========================

Converts approved plugin SKILL.md files into Skill database records.
Called after plugin approval to make skills available through the unified system.
"""

import logging
from typing import List, Optional

from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified

from core.models.core import Skill
from core.models.marketplace_plugins import MarketplacePlugin
from modules.agents.services.skill_loader import parse_yaml_frontmatter

logger = logging.getLogger(__name__)


class SkillMaterializer:
    """Materializes SKILL.md files from plugins into Skill DB records."""

    def __init__(self, db: Session):
        self.db = db

    async def materialize_plugin(self, plugin: MarketplacePlugin) -> List[int]:
        """
        Find all SKILL.md files in a plugin's S3 storage and create/update
        corresponding Skill records.

        Args:
            plugin: The approved MarketplacePlugin record

        Returns:
            List of materialized Skill IDs
        """
        from core.services.marketplace_s3 import MarketplaceS3Service

        s3 = MarketplaceS3Service()
        skill_ids: List[int] = []

        try:
            files = await s3.list_plugin_files(plugin.slug, plugin.version)
        except Exception as e:
            logger.error("Failed to list files for plugin %s: %s", plugin.slug, e)
            return skill_ids

        # Find all SKILL.md files
        skill_files = [f for f in files if f.lower().endswith("skill.md")]
        if not skill_files:
            logger.info("No SKILL.md files found in plugin %s", plugin.slug)
            return skill_ids

        for skill_path in skill_files:
            try:
                skill_id = await self._materialize_single(plugin, s3, skill_path)
                if skill_id is not None:
                    skill_ids.append(skill_id)
            except Exception as e:
                logger.error(
                    "Failed to materialize skill from %s in plugin %s: %s",
                    skill_path, plugin.slug, e,
                )

        # Store materialized skill IDs on the plugin record
        if skill_ids:
            plugin.materialized_skill_ids = skill_ids
            flag_modified(plugin, "materialized_skill_ids")
            self.db.commit()
            logger.info(
                "Materialized %d skill(s) from plugin %s: %s",
                len(skill_ids), plugin.slug, skill_ids,
            )

        return skill_ids

    async def _materialize_single(
        self,
        plugin: MarketplacePlugin,
        s3,
        skill_path: str,
    ) -> Optional[int]:
        """Materialize a single SKILL.md file into a Skill record."""
        content = await s3.get_file(skill_path)
        if not content or not content.strip():
            logger.warning("Empty SKILL.md at %s", skill_path)
            return None

        # PRD-71: Security scan before persisting — reject critical/high findings
        try:
            from core.services.plugin_security_scanner import quick_scan
            findings = quick_scan(content, filename=skill_path)
            critical_or_high = [f for f in findings if f.severity in ("critical", "high")]
            if critical_or_high:
                logger.error(
                    "Rejecting SKILL.md %s — security scan flagged %d issue(s): %s",
                    skill_path, len(critical_or_high),
                    [f.description for f in critical_or_high],
                )
                return None
        except Exception as e:
            logger.warning("Security scan failed for %s (non-blocking): %s", skill_path, e)

        yaml_data, markdown_body = parse_yaml_frontmatter(content)

        # Extract skill metadata from frontmatter
        name = (yaml_data or {}).get("name", "")
        if not name:
            # Derive name from path: plugins/my-plugin/1.0.0/skills/jira-admin/SKILL.md → jira-admin
            parts = skill_path.replace("\\", "/").split("/")
            # Find the directory containing SKILL.md
            for i, part in enumerate(parts):
                if part.lower() == "skill.md" and i > 0:
                    name = parts[i - 1]
                    break
            if not name:
                name = f"{plugin.slug}-skill"

        description = (yaml_data or {}).get("description", "")
        skill_type = (yaml_data or {}).get("type", "technical")
        category = (yaml_data or {}).get("category", "plugin")
        tools_schema = (yaml_data or {}).get("tools", None)
        if tools_schema and not isinstance(tools_schema, dict):
            # Normalize to {"tools": [...]} format
            tools_schema = {"tools": tools_schema} if isinstance(tools_schema, list) else None
        version = (yaml_data or {}).get("version", plugin.version)

        # Check for existing skill from same package_slug + name
        existing = (
            self.db.query(Skill)
            .filter(
                Skill.package_slug == plugin.slug,
                Skill.name == name,
            )
            .first()
        )

        if existing:
            # Update existing skill
            existing.prompt_template = markdown_body
            existing.description = description or existing.description
            existing.tools_schema = tools_schema
            existing.skill_version = version
            existing.is_active = True
            self.db.flush()
            self._invalidate_skill_cache(name)
            logger.info("Updated existing skill '%s' (id=%d) from plugin %s", name, existing.id, plugin.slug)
            return existing.id

        # Create new skill record
        skill = Skill(
            name=name,
            description=description,
            skill_type=skill_type,
            category=category,
            prompt_template=markdown_body,
            tools_schema=tools_schema,
            skill_version=version,
            package_slug=plugin.slug,
            workspace_id=None,  # PRD-71: marketplace/global skill
            is_active=True,
            skill_source=f"plugin:{plugin.slug}",
        )
        self.db.add(skill)
        self.db.flush()  # Get the ID
        self._invalidate_skill_cache(name)
        logger.info("Created new skill '%s' (id=%d) from plugin %s", name, skill.id, plugin.slug)
        return skill.id

    def _invalidate_skill_cache(self, skill_name: str) -> None:
        """Clear SkillLoader caches so updated content is served immediately."""
        try:
            from modules.agents.services.skill_loader import get_skill_loader
            loader = get_skill_loader(self.db)
            loader.metadata_cache.pop(skill_name, None)
            loader.core_content_cache.pop(skill_name, None)
            logger.debug("Invalidated cache for skill '%s'", skill_name)
        except Exception:
            pass  # SkillLoader not initialized yet — no cache to clear
