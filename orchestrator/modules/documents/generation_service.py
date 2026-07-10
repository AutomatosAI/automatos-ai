"""
Document Generation Service (PRD-63).

Generates PDF (WeasyPrint), DOCX (python-docx-template), and XLSX (XlsxWriter)
documents from templates + data.

Files are generated locally then uploaded to S3 for persistent storage.
On Railway (ephemeral containers) the local file vanishes after the request,
so S3 is the source of truth for downloads.
"""

import asyncio
import logging
import mimetypes
import os
import re
from datetime import datetime
from typing import Optional
from uuid import UUID

import ipaddress
import socket
from urllib.parse import urlparse

import jinja2
from jinja2.sandbox import SandboxedEnvironment
import jsonschema
from sqlalchemy.orm import Session


def _safe_url_fetcher(url, *args, **kwargs):
    """WeasyPrint URL fetcher that blocks file:// and internal/non-public network
    targets from user-controlled templates (PRD-156 S4 — SSRF).

    Inline ``data:`` URIs (embedded chart images) are allowed; ``http(s)`` is
    allowed only to PUBLIC addresses; everything else — file://, and
    private/loopback/link-local hosts such as 10.x / 127.x / 169.254.x (the cloud
    metadata endpoint) — is refused.
    """
    parsed = urlparse(url)
    scheme = (parsed.scheme or "").lower()
    if scheme == "data":
        from weasyprint import default_url_fetcher
        return default_url_fetcher(url, *args, **kwargs)
    if scheme not in ("http", "https"):
        raise ValueError(f"Blocked non-http(s) URL scheme in template: {scheme!r}")
    host = parsed.hostname or ""
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        raise ValueError(f"Cannot resolve template URL host: {host!r}")
    for info in infos:
        if not ipaddress.ip_address(info[4][0]).is_global:
            raise ValueError(f"Blocked non-public address in template URL: {host!r}")
    from weasyprint import default_url_fetcher
    return default_url_fetcher(url, *args, **kwargs)

try:
    import boto3
    from botocore.config import Config as BotoConfig
    from botocore.exceptions import ClientError
except ImportError:
    boto3 = None

from config import config
from core.models.core import DocumentTemplate
from core.models.workspaces import Workspace
from modules.documents.models import GeneratedDocument, UnresolvedDeliverableError
from modules.documents.template_service import DocumentTemplateService
from modules.documents.brand_kit import get_brand_kit
from modules.documents.blocks import (
    blocks_from_legacy,
    collect_variable_paths,
    render_document_docx,
    render_document_html,
    validate_blocks,
)
from modules.documents.variables import VariableResolver

logger = logging.getLogger(__name__)

# Base directory for generated documents
GENERATED_DIR = config.DOCUMENT_STORAGE_DIR

# PRD-167 S2/S4: the hardcoded `_FALLBACK_PDF_TEMPLATE` (with the `#ff6b35` Automatos
# orange) is gone. When a template has no blocks and no template_content, the no-template
# PDF path renders through the brand-aware block renderer via `blocks_from_legacy(data)`,
# so branding comes from the workspace brand kit — never hardcoded.


class DocumentGenerationService:
    """Generates documents from templates + data."""

    def __init__(self, db: Session, workspace_id: UUID = None):
        self.db = db
        self.workspace_id = workspace_id
        self.template_service = DocumentTemplateService(db)
        # PRD-156 S4: SandboxedEnvironment blocks SSTI (e.g. accessing __globals__
        # via cycler/class chains) in user-authored template content.
        self._jinja_env = SandboxedEnvironment(autoescape=True)

    # ------------------------------------------------------------------
    # Public dispatch
    # ------------------------------------------------------------------

    async def generate(
        self,
        title: str,
        format: str,
        data: dict,
        workspace_id: UUID = None,
        template_name: str = None,
        template_id: UUID = None,
        user_id: int = None,
    ) -> GeneratedDocument:
        """Single entry point — resolve template then dispatch to format engine.

        ``user_id`` (optional) is the requesting user, used to resolve ``{{user.*}}``
        variable chips in block templates (PRD-167 S3).
        """
        ws = workspace_id or self.workspace_id
        if not ws:
            raise ValueError("workspace_id is required")

        # Resolve template
        template = None
        if template_id:
            template = self.template_service.get_template(template_id, ws)
        elif template_name:
            template = self.template_service.get_template_by_name(ws, template_name)

        # Default to Basic Report for PDF if no template specified
        if not template and format == "pdf":
            template = self.template_service.get_template_by_name(ws, "Basic Report")

        # Inject top-level title into data so templates can reference {{ title }}.
        # The tool schema separates title from data, but templates expect it inside data.
        if "title" not in data:
            data["title"] = title

        # Normalize section keys: LLMs may use heading/body instead of title/content.
        # Templates and _data_to_markdown() expect title/content.
        self._normalize_sections(data)

        logger.info(f"[DocGen] Data keys from LLM: {list(data.keys())}")

        # Generate the format-specific file
        if format == "pdf":
            result = await self.generate_pdf(template, data, ws, title, user_id=user_id)
        elif format == "docx":
            result = await self.generate_docx(template, data, ws, title, user_id=user_id)
        elif format == "xlsx":
            result = await self.generate_xlsx(data, ws, title=title, template=template)
        else:
            raise ValueError(f"Unsupported format: {format}. Use pdf, docx, or xlsx.")

        # P2-09 S3 — the finalisation gate. The render-honesty primitives
        # (RenderedHtml/RenderedDocx.unresolved, ResolvedVariables.unknown) used
        # to be logged and discarded, so a client could still receive a document
        # with visible [[variable]] markers. A Deliverable that did not resolve
        # clean is BLOCKED, loudly — never delivered. (The Studio live preview
        # renders outside generate() and keeps its visible red markers.)
        if result.unresolved or result.unknown:
            logger.warning(
                "[DocGen] BLOCKED finalisation of '%s' (%s): %d unresolved / %d unknown variable(s)",
                title, format, len(result.unresolved), len(result.unknown),
            )
            raise UnresolvedDeliverableError(
                unresolved=result.unresolved, unknown=result.unknown
            )

        # Attach markdown content for live widget display
        result.content = self._data_to_markdown(data, title)
        return result

    # ------------------------------------------------------------------
    # Block rendering helpers (PRD-167 S2/S3/S4)
    # ------------------------------------------------------------------

    def _brand_kit_for(self, workspace_id: UUID) -> dict:
        ws = self.db.query(Workspace).filter(Workspace.id == workspace_id).first()
        return get_brand_kit(getattr(ws, "settings", None))

    def _render_block_html(self, block_doc, data, workspace_id, user_id, title):
        """Resolve a block document's variables and render it to a full HTML page.

        Returns ``(html, unresolved, unknown)`` — the render-honesty lists are
        captured for the finalisation gate in :meth:`generate` (P2-09 S3), not
        discarded behind a log line. The renderer marks BOTH empty-known and
        unknown paths as visible ``[[markers]]``; the authoring errors (unknown)
        are split out so each list stays honest.
        """
        paths = collect_variable_paths(block_doc)
        resolver = VariableResolver(self.db)
        resolved = resolver.resolve(workspace_id, user_id, paths, extra_data=data)
        brand_kit = self._brand_kit_for(workspace_id)
        rendered = render_document_html(block_doc, resolved.values, brand_kit, title=title)
        unknown = list(resolved.unknown)
        unresolved = [p for p in rendered.unresolved if p not in set(unknown)]
        return rendered.html, unresolved, unknown

    def register_as_deliverable(
        self,
        result: GeneratedDocument,
        *,
        title: str,
        source_type: str = "document",
        source_id: str = None,
        agent_id: int = None,
        agent_name: str = None,
        template_id: UUID = None,
    ) -> Optional[dict]:
        """Register a generated document as a workspace deliverable (PRD-167 S6).

        Source attribution travels in ``extra`` (template_id) + ``source_type``. Called
        only for real generations — previews never register. Best-effort: a registration
        failure never fails the generation itself.
        """
        ws = self.workspace_id
        if not ws:
            return None
        try:
            from services.deliverable_service import DeliverableService

            extra = {"template_id": str(template_id)} if template_id else None
            return DeliverableService(self.db, ws).register(
                file_path=f"generated/{result.filename}",
                title=title or result.filename,
                source_type=source_type,
                source_id=source_id,
                agent_id=agent_id,
                agent_name=agent_name,
                artifact_type="document",
                storage_type="generated",
                file_type=result.format,
                file_size_bytes=result.size,
                preview_url=result.download_url,
                extra=extra,
            )
        except Exception:
            logger.exception("[DocGen] deliverable registration failed (non-fatal)")
            return None

    # ------------------------------------------------------------------
    # PDF Generation (Jinja2 + WeasyPrint)
    # ------------------------------------------------------------------

    async def generate_pdf(
        self,
        template: Optional[DocumentTemplate],
        data: dict,
        workspace_id: UUID,
        title: str = "Document",
        user_id: int = None,
    ) -> GeneratedDocument:
        """Render → PDF via WeasyPrint.

        Three render paths, in order of preference (PRD-167 S2):
          1. Block template (``template.blocks``) → block→HTML renderer (brand-aware,
             variable chips resolved). The canonical path.
          2. Legacy ``template_content`` (user-authored Jinja HTML) → sandboxed Jinja2.
          3. No template → brand-aware block render of the legacy data shape
             (``blocks_from_legacy``). Replaces the old hardcoded fallback template.
        """
        try:
            from weasyprint import HTML
        except ImportError:
            raise ImportError(
                "weasyprint is required for PDF generation. "
                "Install with: pip install weasyprint>=62.0"
            )

        block_payload = getattr(template, "blocks", None) if template else None
        unresolved: list = []
        unknown: list = []

        if block_payload:
            # Path 1: canonical block template.
            block_doc = validate_blocks(block_payload)
            rendered_html, unresolved, unknown = self._render_block_html(
                block_doc, data, workspace_id, user_id, title
            )
        elif template and template.template_content:
            # Path 2: legacy user-authored Jinja HTML (kept until per-workspace
            # templates are migrated to blocks — see PRD-167 sunset note).
            if hasattr(template, "data_schema"):
                self._validate_and_backfill(data, template.data_schema)
            # PRD-167 S4: expose the brand kit to legacy templates as {{ brand.* }} so
            # they pick up workspace palette instead of hardcoded Automatos colours.
            render_ctx = {**data, "brand": self._brand_kit_for(workspace_id)}
            try:
                jinja_template = self._jinja_env.from_string(template.template_content)
                rendered_html = jinja_template.render(**render_ctx)
            except jinja2.TemplateError as e:
                raise ValueError(f"Template rendering error: {e}")
            rendered_html = self._embed_charts(rendered_html, data)
        else:
            # Path 3: no template — brand-aware block render of the legacy data shape.
            logger.info("No template found — rendering data via brand-aware block fallback")
            block_doc = blocks_from_legacy(data)
            rendered_html, unresolved, unknown = self._render_block_html(
                block_doc, data, workspace_id, user_id, title
            )

        # Generate PDF
        output_path = self._output_path(workspace_id, title, "pdf")
        try:
            HTML(string=rendered_html, url_fetcher=_safe_url_fetcher).write_pdf(output_path)
        except Exception as e:
            raise RuntimeError(f"PDF generation failed: {e}")

        return self._build_result(
            output_path, "pdf", title, workspace_id,
            unresolved=unresolved, unknown=unknown,
        )

    # ------------------------------------------------------------------
    # DOCX Generation (python-docx-template)
    # ------------------------------------------------------------------

    async def generate_docx(
        self,
        template: Optional[DocumentTemplate],
        data: dict,
        workspace_id: UUID,
        title: str = "Document",
        user_id: int = None,
    ) -> GeneratedDocument:
        """Render → DOCX.

        Block templates (PRD-167 S2, Q71) compile directly to a python-docx Document
        from the same block tree — no uploaded ``.docx`` file required. Legacy templates
        with an uploaded ``.docx`` still render via docxtpl.
        """
        output_path = self._output_path(workspace_id, title, "docx")

        block_payload = getattr(template, "blocks", None) if template else None
        if block_payload:
            # Path 1: canonical block template → compiled DOCX (Q71).
            block_doc = validate_blocks(block_payload)
            paths = collect_variable_paths(block_doc)
            resolved = VariableResolver(self.db).resolve(
                workspace_id, user_id, paths, extra_data=data
            )
            brand_kit = self._brand_kit_for(workspace_id)
            rendered = render_document_docx(block_doc, resolved.values, brand_kit)
            # P2-09 S3: capture the render-honesty lists for the finalisation
            # gate in generate() — same unknown/unresolved split as the HTML path.
            unknown = list(resolved.unknown)
            unresolved = [p for p in rendered.unresolved if p not in set(unknown)]
            rendered.document.save(output_path)
            return self._build_result(
                output_path, "docx", title, workspace_id,
                unresolved=unresolved, unknown=unknown,
            )

        # Path 2: legacy uploaded .docx template via docxtpl.
        try:
            from docxtpl import DocxTemplate, InlineImage
            from docx.shared import Mm
        except ImportError:
            raise ImportError(
                "docxtpl is required for legacy .docx template generation. "
                "Install with: pip install docxtpl>=0.18.0"
            )

        if not template or not template.template_file_path:
            raise ValueError(
                "DOCX generation requires either a block template or an uploaded .docx "
                "file. Create a block template in the editor, or upload one via "
                "/api/documents/templates/upload."
            )

        if not os.path.exists(template.template_file_path):
            raise FileNotFoundError(
                f"Template file not found: {template.template_file_path}"
            )

        doc = DocxTemplate(template.template_file_path)

        # Handle inline images
        for key, value in list(data.items()):
            if isinstance(value, dict) and value.get("_type") == "image":
                data[key] = InlineImage(
                    doc, value["path"], width=Mm(value.get("width", 150))
                )

        doc.render(data)
        doc.save(output_path)

        return self._build_result(output_path, "docx", title, workspace_id)

    # ------------------------------------------------------------------
    # XLSX Generation (XlsxWriter)
    # ------------------------------------------------------------------

    async def generate_xlsx(
        self,
        data: dict,
        workspace_id: UUID,
        title: str = "Export",
        template: Optional[DocumentTemplate] = None,
    ) -> GeneratedDocument:
        """Generate an Excel spreadsheet from tabular data."""
        try:
            import xlsxwriter
        except ImportError:
            raise ImportError(
                "xlsxwriter is required for XLSX generation. "
                "Install with: pip install xlsxwriter>=3.2.0"
            )

        columns = data.get("columns", [])
        rows = data.get("rows", [])
        if not columns:
            raise ValueError("XLSX generation requires 'columns' in data.")

        output_path = self._output_path(workspace_id, title, "xlsx")
        workbook = xlsxwriter.Workbook(output_path)
        worksheet = workbook.add_worksheet(title[:31])  # Excel 31-char limit

        # Header formatting
        header_fmt = workbook.add_format(
            {
                "bold": True,
                "bg_color": "#1a1a2e",
                "font_color": "white",
                "border": 1,
                "text_wrap": True,
            }
        )

        # Write headers
        for col, name in enumerate(columns):
            worksheet.write(0, col, name, header_fmt)

        # Write data with type detection
        for row_idx, row in enumerate(rows, 1):
            for col_idx, value in enumerate(row):
                if col_idx >= len(columns):
                    break
                if isinstance(value, (int, float)):
                    worksheet.write_number(row_idx, col_idx, value)
                elif isinstance(value, datetime):
                    date_fmt = workbook.add_format({"num_format": "yyyy-mm-dd"})
                    worksheet.write_datetime(row_idx, col_idx, value, date_fmt)
                else:
                    worksheet.write_string(row_idx, col_idx, str(value) if value is not None else "")

        # Auto-fit column widths
        for col, name in enumerate(columns):
            col_values = [str(r[col]) if col < len(r) and r[col] is not None else "" for r in rows]
            max_width = max(len(str(name)), max((len(v) for v in col_values), default=0))
            worksheet.set_column(col, col, min(max_width + 2, 50))

        workbook.close()
        return self._build_result(output_path, "xlsx", title, workspace_id)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_sections(data: dict) -> None:
        """Normalize section keys so templates always get title/content.

        LLMs sometimes use heading/body, header/text, name/description, etc.
        This maps common variants to the canonical title/content keys.
        """
        sections = data.get("sections")
        if not isinstance(sections, list):
            return

        for section in sections:
            if not isinstance(section, dict):
                continue

            # Normalize title
            if "title" not in section:
                for alt in ("heading", "header", "name", "section_title", "label"):
                    if alt in section:
                        section["title"] = section.pop(alt)
                        break

            # Normalize content
            if "content" not in section:
                for alt in ("body", "text", "description", "section_content", "detail", "details", "paragraph"):
                    if alt in section:
                        section["content"] = section.pop(alt)
                        break

            # Warn on empty content (helps debug LLM output)
            if not section.get("content"):
                logger.warning(
                    "[DocGen] Section '%s' has empty content. Keys present: %s",
                    section.get("title", "?"), list(section.keys()),
                )

    def _validate_and_backfill(self, data: dict, schema: dict) -> None:
        """Validate data against the template's JSON Schema.

        Non-fatal: missing required fields are backfilled with sensible
        defaults so Jinja2 rendering doesn't crash on {% for %} loops.
        """
        if not schema:
            return

        # Backfill missing required fields with type-appropriate defaults
        # so templates render gracefully even with partial LLM output.
        props = schema.get("properties", {})
        for field in schema.get("required", []):
            if field not in data:
                field_type = props.get(field, {}).get("type", "string")
                default = {
                    "string": "",
                    "array": [],
                    "object": {},
                    "number": 0,
                    "integer": 0,
                    "boolean": False,
                }.get(field_type, "")
                data[field] = default
                logger.warning(
                    f"[DocGen] Backfilled missing required field '{field}' "
                    f"with default {type(default).__name__}"
                )

        try:
            jsonschema.validate(instance=data, schema=schema)
        except jsonschema.ValidationError as e:
            # Log but don't raise — let Jinja2 try rendering.
            logger.warning(f"[DocGen] Schema validation warning: {e.message}")

    def _embed_charts(self, html: str, data: dict) -> str:
        """Replace {{ chart:field_name }} tags with base64 PNG images."""
        charts = data.get("_charts", {})
        for chart_ref in re.findall(r"\{\{\s*chart:(\w+)\s*\}\}", html):
            if chart_ref in charts:
                b64 = charts[chart_ref]
                html = html.replace(
                    f"{{{{ chart:{chart_ref} }}}}",
                    f'<img src="data:image/png;base64,{b64}" style="max-width:100%"/>',
                )
        return html

    @staticmethod
    def _data_to_markdown(data: dict, title: str) -> str:
        """Convert structured template data to markdown for widget display."""
        lines: list[str] = [f"# {title}", ""]

        # Author / date metadata
        meta_parts = []
        if data.get("author"):
            meta_parts.append(f"**Author:** {data['author']}")
        if data.get("date"):
            meta_parts.append(f"**Date:** {data['date']}")
        if meta_parts:
            lines.append(" | ".join(meta_parts))
            lines.append("")

        # Sections (most common structure)
        for section in data.get("sections", []):
            if isinstance(section, dict):
                lines.append(f"## {section.get('title', 'Section')}")
                lines.append("")
                lines.append(str(section.get("content", "")))
                lines.append("")
            elif isinstance(section, str):
                lines.append(section)
                lines.append("")

        # Metrics block
        metrics = data.get("metrics", {})
        if metrics and isinstance(metrics, dict):
            lines.append("## Key Metrics")
            lines.append("")
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            for k, v in metrics.items():
                lines.append(f"| {k} | {v} |")
            lines.append("")

        # Highlights (executive summary style)
        highlights = data.get("highlights", [])
        if highlights:
            lines.append("## Highlights")
            lines.append("")
            for h in highlights:
                lines.append(f"- {h}")
            lines.append("")

        # Recommendations
        recs = data.get("recommendations", [])
        if recs:
            lines.append("## Recommendations")
            lines.append("")
            for r in recs:
                lines.append(f"- {r}")
            lines.append("")

        # Tabular data (xlsx-style)
        columns = data.get("columns", [])
        rows = data.get("rows", [])
        if columns and rows:
            lines.append("| " + " | ".join(str(c) for c in columns) + " |")
            lines.append("| " + " | ".join("---" for _ in columns) + " |")
            for row in rows[:50]:  # Cap at 50 rows for widget display
                lines.append("| " + " | ".join(str(v) for v in row) + " |")
            if len(rows) > 50:
                lines.append(f"\n*... and {len(rows) - 50} more rows*")
            lines.append("")

        # Fallback: render any remaining fields the LLM included
        shown = {"title", "author", "date", "sections", "metrics",
                 "highlights", "recommendations", "columns", "rows", "_charts"}
        for key, val in data.items():
            if key in shown or key.startswith("_") or not val:
                continue
            heading = key.replace("_", " ").title()
            if isinstance(val, str):
                lines.append(f"## {heading}")
                lines.append("")
                lines.append(val)
                lines.append("")
            elif isinstance(val, list):
                lines.append(f"## {heading}")
                lines.append("")
                for item in val:
                    if isinstance(item, dict):
                        # List of objects — render each with sub-heading if it has a title/name
                        sub_title = item.get("title") or item.get("name") or item.get("heading") or ""
                        if sub_title:
                            lines.append(f"### {sub_title}")
                            lines.append("")
                        for k, v in item.items():
                            if k in ("title", "name", "heading"):
                                continue
                            lines.append(str(v))
                            lines.append("")
                    else:
                        lines.append(f"- {item}")
                lines.append("")
            elif isinstance(val, dict):
                lines.append(f"## {heading}")
                lines.append("")
                # Render dict as key-value pairs or a table
                lines.append("| Key | Value |")
                lines.append("|-----|-------|")
                for k, v in val.items():
                    lines.append(f"| {k} | {v} |")
                lines.append("")

        return "\n".join(lines)

    def _output_path(self, workspace_id: UUID, title: str, ext: str) -> str:
        """Build output file path, creating directories as needed."""
        safe_title = re.sub(r"[^\w\s-]", "", title).strip().replace(" ", "_")[:80]
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        directory = os.path.join(GENERATED_DIR, str(workspace_id), "generated")
        os.makedirs(directory, exist_ok=True)
        return os.path.join(directory, f"{timestamp}_{safe_title}.{ext}")

    def _build_result(
        self,
        path: str,
        fmt: str,
        title: str,
        workspace_id: UUID = None,
        *,
        unresolved: Optional[list] = None,
        unknown: Optional[list] = None,
    ) -> GeneratedDocument:
        """Build a GeneratedDocument from a file on disk, uploading to S3 for persistence.

        The persisted ``download_url`` is the STABLE app path
        (``/api/documents/generated/{filename}``), never the raw presigned S3 URL —
        a presign expires after an hour, which rotted every persisted Deliverable
        link (P2-09 S1 / F030). ``serve_generated_file`` re-mints a fresh presign
        on each request, so the app path stays live for the Deliverable's lifetime.

        ``unresolved``/``unknown`` are the render-honesty lists captured off the
        block render (P2-09 S3); non-empty lists make :meth:`generate` block
        finalisation.
        """
        filename = os.path.basename(path)
        size = os.path.getsize(path)

        # Upload a persistence copy to S3 (containers are ephemeral); the link we
        # persist stays the app path — the re-mint endpoint owns presigning.
        download_url = f"/api/documents/generated/{filename}"
        self._upload_to_s3(path, filename, workspace_id)

        return GeneratedDocument(
            path=path,
            format=fmt,
            filename=filename,
            size=size,
            download_url=download_url,
            preview_url=download_url if fmt == "pdf" else None,
            unresolved=list(unresolved or []),
            unknown=list(unknown or []),
        )

    def _upload_to_s3(
        self, local_path: str, filename: str, workspace_id: UUID = None
    ) -> bool:
        """Upload a generated document to S3 for persistence across ephemeral containers.

        Returns True when the upload happened, False when S3 is not configured or
        the upload failed (local serving still works for the container's lifetime).
        No download link is minted here — ``serve_generated_file`` presigns on
        demand, so persisted URLs never expire (P2-09 S1).
        """
        if not boto3:
            logger.debug("[DocGen] boto3 not available, skipping S3 upload")
            return False

        if not config.AWS_ACCESS_KEY_ID or not config.AWS_SECRET_ACCESS_KEY:
            logger.debug("[DocGen] AWS credentials not configured, skipping S3 upload")
            return False

        ws_id = workspace_id or self.workspace_id
        bucket = config.S3_DOCUMENTS_BUCKET or "automatos-ai"
        s3_key = f"workspaces/{ws_id}/generated-documents/{filename}"
        content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"

        try:
            boto_cfg = BotoConfig(
                region_name=config.AWS_REGION or "us-east-1",
                signature_version="v4",
                retries={"max_attempts": 3, "mode": "adaptive"},
            )
            client = boto3.client(
                "s3",
                aws_access_key_id=config.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY,
                config=boto_cfg,
            )

            with open(local_path, "rb") as f:
                client.put_object(
                    Bucket=bucket,
                    Key=s3_key,
                    Body=f,
                    ContentType=content_type,
                )

            logger.info(
                "[DocGen] Uploaded to S3: s3://%s/%s (%d bytes)",
                bucket, s3_key, os.path.getsize(local_path),
            )
            return True

        except Exception:
            logger.exception("[DocGen] S3 upload failed, falling back to local serving")
            return False
