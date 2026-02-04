"""
Plugin Upload Service
=====================

Handles the full plugin upload flow: validate manifest, extract to S3,
run security scans, and create database records.
"""

import io
import json
import logging
import zipfile
from datetime import datetime
from typing import Optional, Dict

logger = logging.getLogger(__name__)


class PluginUploadService:
    """Orchestrates the admin plugin upload workflow.

    Flow:
        1. Extract zip to temp directory
        2. Validate manifest.json exists at root
        3. Parse manifest for plugin metadata
        4. Read all text files into dict for scanning
        5. Run PluginScanService.scan_plugin() with extracted files
        6. Upload to S3 via MarketplaceS3Service.extract_plugin()
        7. Create MarketplacePlugin database record
        8. Create PluginSyncHistory record
    """

    def __init__(self, db, s3_service, scan_service):
        """
        Args:
            db: SQLAlchemy Session instance.
            s3_service: MarketplaceS3Service instance.
            scan_service: PluginScanService instance.
        """
        self.db = db
        self.s3_service = s3_service
        self.scan_service = scan_service

    async def upload_plugin(
        self,
        zip_bytes: bytes,
        source_type: str,
        source_url: Optional[str],
        uploaded_by: str,
    ):
        """Handle the full plugin upload flow.

        Args:
            zip_bytes: Raw zip file bytes.
            source_type: Origin type ('upload', 'github', etc.).
            source_url: Optional URL if sourced externally.
            uploaded_by: Identifier of who uploaded (user ID or system).

        Returns:
            MarketplacePlugin database record.

        Raises:
            ValueError: If zip is invalid or manifest.json is missing/invalid.
        """
        from core.models.marketplace_plugins import (
            MarketplacePlugin,
            PluginSyncHistory,
        )

        # ------------------------------------------------------------------
        # 1. Extract zip and validate manifest
        # ------------------------------------------------------------------
        try:
            zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
        except zipfile.BadZipFile:
            raise ValueError("Uploaded file is not a valid zip archive")

        with zf:
            names = zf.namelist()

            # manifest.json must exist at the root of the zip
            if "manifest.json" not in names:
                raise ValueError(
                    "manifest.json not found at root of zip archive"
                )

            manifest_raw = zf.read("manifest.json").decode("utf-8")
            try:
                manifest = json.loads(manifest_raw)
            except json.JSONDecodeError as e:
                raise ValueError(f"manifest.json is not valid JSON: {e}")

            # ------------------------------------------------------------------
            # 2. Parse manifest metadata
            # ------------------------------------------------------------------
            slug = manifest.get("slug")
            name = manifest.get("name")
            version = manifest.get("version")

            if not slug or not name or not version:
                raise ValueError(
                    "manifest.json must contain 'slug', 'name', and 'version' fields"
                )

            description = manifest.get("description", "")
            long_description = manifest.get("long_description", "")
            original_author = manifest.get("author", "")
            license_val = manifest.get("license", "")
            tags = manifest.get("tags", [])
            recommended_models = manifest.get("recommended_models", [])
            token_estimate = manifest.get("token_estimate", 0)

            # Count content items from manifest
            contents = manifest.get("contents", {})
            skills_count = len(contents.get("skills", []))
            commands_count = len(contents.get("commands", []))
            agents_count = len(contents.get("agents", []))
            hooks_count = len(contents.get("hooks", []))

            # ------------------------------------------------------------------
            # 3. Read all text files into dict for scanning
            # ------------------------------------------------------------------
            plugin_files: Dict[str, str] = {}
            for member in names:
                if member.endswith("/"):
                    continue
                try:
                    raw = zf.read(member)
                    content = raw.decode("utf-8")
                    plugin_files[member] = content
                except (UnicodeDecodeError, KeyError):
                    # Skip binary files
                    continue

        # ------------------------------------------------------------------
        # 4. Run security scan
        # ------------------------------------------------------------------
        logger.info("Running security scan for %s@%s", slug, version)
        scan_record = await self.scan_service.scan_plugin(
            plugin_slug=slug,
            plugin_version=version,
            plugin_files=plugin_files,
            scanned_by=uploaded_by,
        )

        # Map scan verdict to security_status
        security_status = scan_record.overall_verdict  # safe, review_required, blocked

        # ------------------------------------------------------------------
        # 5. Upload to S3
        # ------------------------------------------------------------------
        logger.info("Uploading plugin %s@%s to S3", slug, version)
        s3_prefix = await self.s3_service.extract_plugin(
            slug=slug,
            version=version,
            zip_bytes=zip_bytes,
        )

        # ------------------------------------------------------------------
        # 6. Create MarketplacePlugin database record
        # ------------------------------------------------------------------
        plugin = MarketplacePlugin(
            slug=slug,
            name=name,
            version=version,
            s3_path=s3_prefix,
            description=description,
            long_description=long_description,
            tags=tags,
            skills_count=skills_count,
            commands_count=commands_count,
            agents_count=agents_count,
            hooks_count=hooks_count,
            source_type=source_type,
            source_url=source_url,
            original_author=original_author,
            license=license_val,
            token_estimate=token_estimate,
            recommended_models=recommended_models,
            security_scan_id=scan_record.id,
            security_status=security_status,
            approval_status="pending",
        )
        self.db.add(plugin)
        self.db.commit()
        self.db.refresh(plugin)

        # ------------------------------------------------------------------
        # 7. Create PluginSyncHistory record
        # ------------------------------------------------------------------
        history = PluginSyncHistory(
            action="upload",
            plugin_id=plugin.id,
            plugin_slug=slug,
            status="completed",
            completed_at=datetime.utcnow(),
            details={
                "version": version,
                "source_type": source_type,
                "source_url": source_url,
                "security_status": security_status,
                "scan_id": str(scan_record.id),
                "files_count": len(plugin_files),
                "skills_count": skills_count,
                "commands_count": commands_count,
            },
            performed_by=uploaded_by,
        )
        self.db.add(history)
        self.db.commit()

        logger.info(
            "Plugin %s@%s uploaded: id=%s, security=%s",
            slug,
            version,
            plugin.id,
            security_status,
        )
        return plugin
