"""
Document Generation Service (PRD-63).

Generates PDF (WeasyPrint), DOCX (python-docx-template), and XLSX (XlsxWriter)
documents from templates + data.
"""

import logging
import os
import re
from datetime import datetime
from typing import Optional
from uuid import UUID

import jinja2
import jsonschema
from sqlalchemy.orm import Session

from core.models.core import DocumentTemplate
from modules.documents.models import GeneratedDocument
from modules.documents.template_service import DocumentTemplateService

logger = logging.getLogger(__name__)

# Base directory for generated documents
GENERATED_DIR = os.environ.get("DOCUMENT_STORAGE_DIR", "documents")

# Inline fallback template used when no DB template is available
_FALLBACK_PDF_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<style>
  @page { size: A4; margin: 2cm; }
  body { font-family: 'Inter', 'Segoe UI', system-ui, sans-serif; color: #1a1a2e; line-height: 1.6; }
  .header { border-bottom: 3px solid #ff6b35; padding-bottom: 1rem; margin-bottom: 2rem; }
  .header h1 { margin: 0 0 0.5rem 0; font-size: 28pt; color: #1a1a2e; }
  .header .meta { color: #666; font-size: 10pt; }
  h2 { color: #1a1a2e; border-bottom: 1px solid #eee; padding-bottom: 0.5rem; margin-top: 2rem; }
  .metrics { display: flex; gap: 1rem; margin: 1.5rem 0; flex-wrap: wrap; }
  .metric-card { flex: 1; min-width: 120px; background: #f8f9fa; border-radius: 8px; padding: 1rem; border-left: 4px solid #ff6b35; text-align: center; }
  .metric-card .label { font-size: 9pt; color: #666; text-transform: uppercase; }
  .metric-card .value { font-size: 20pt; font-weight: 700; color: #1a1a2e; }
  .section-content { margin-bottom: 1.5rem; white-space: pre-wrap; }
</style>
</head>
<body>
  <div class="header">
    <h1>{{ title | default('Report') }}</h1>
    <div class="meta">Generated: {{ date | default('') }} | Author: {{ author | default('Automatos AI') }}</div>
  </div>
  {% if metrics %}
  <div class="metrics">
    {% for key, value in metrics.items() %}
    <div class="metric-card"><div class="label">{{ key }}</div><div class="value">{{ value }}</div></div>
    {% endfor %}
  </div>
  {% endif %}
  {% if sections %}
    {% for section in sections %}
    <h2>{{ section.title }}</h2>
    <div class="section-content">{{ section.content }}</div>
    {% endfor %}
  {% elif content %}
    <div class="section-content">{{ content }}</div>
  {% endif %}
</body>
</html>"""


class DocumentGenerationService:
    """Generates documents from templates + data."""

    def __init__(self, db: Session, workspace_id: UUID = None):
        self.db = db
        self.workspace_id = workspace_id
        self.template_service = DocumentTemplateService(db)
        self._jinja_env = jinja2.Environment(autoescape=True)

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
    ) -> GeneratedDocument:
        """Single entry point — resolve template then dispatch to format engine."""
        ws = workspace_id or self.workspace_id
        if not ws:
            raise ValueError("workspace_id is required")

        # Resolve template
        template = None
        if template_id:
            template = self.template_service.get_template(template_id)
        elif template_name:
            template = self.template_service.get_template_by_name(ws, template_name)

        # Default to Basic Report for PDF if no template specified
        if not template and format == "pdf":
            template = self.template_service.get_template_by_name(ws, "Basic Report")

        # Inject top-level title into data so templates can reference {{ title }}.
        # The tool schema separates title from data, but templates expect it inside data.
        if "title" not in data:
            data["title"] = title

        # Generate the format-specific file
        if format == "pdf":
            result = await self.generate_pdf(template, data, ws, title)
        elif format == "docx":
            result = await self.generate_docx(template, data, ws, title)
        elif format == "xlsx":
            result = await self.generate_xlsx(data, ws, title=title, template=template)
        else:
            raise ValueError(f"Unsupported format: {format}. Use pdf, docx, or xlsx.")

        # Attach markdown content for live widget display
        result.content = self._data_to_markdown(data, title)
        return result

    # ------------------------------------------------------------------
    # PDF Generation (Jinja2 + WeasyPrint)
    # ------------------------------------------------------------------

    async def generate_pdf(
        self,
        template: Optional[DocumentTemplate],
        data: dict,
        workspace_id: UUID,
        title: str = "Document",
    ) -> GeneratedDocument:
        """Render Jinja2 HTML → PDF via WeasyPrint."""
        try:
            from weasyprint import HTML
        except ImportError:
            raise ImportError(
                "weasyprint is required for PDF generation. "
                "Install with: pip install weasyprint>=62.0"
            )

        if not template or not template.template_content:
            logger.info("No template found — using inline fallback for PDF generation")
            template_html = _FALLBACK_PDF_TEMPLATE
        else:
            template_html = template.template_content

        # Validate data against schema (skip if using fallback — no schema)
        if template and hasattr(template, 'data_schema'):
            self._validate_and_backfill(data, template.data_schema)

        # Render Jinja2
        try:
            jinja_template = self._jinja_env.from_string(template_html)
            rendered_html = jinja_template.render(**data)
        except jinja2.TemplateError as e:
            raise ValueError(f"Template rendering error: {e}")

        # Embed charts
        rendered_html = self._embed_charts(rendered_html, data)

        # Generate PDF
        output_path = self._output_path(workspace_id, title, "pdf")
        try:
            HTML(string=rendered_html).write_pdf(output_path)
        except Exception as e:
            raise RuntimeError(f"PDF generation failed: {e}")

        return self._build_result(output_path, "pdf", title)

    # ------------------------------------------------------------------
    # DOCX Generation (python-docx-template)
    # ------------------------------------------------------------------

    async def generate_docx(
        self,
        template: Optional[DocumentTemplate],
        data: dict,
        workspace_id: UUID,
        title: str = "Document",
    ) -> GeneratedDocument:
        """Render .docx template with Jinja2 tags using docxtpl."""
        try:
            from docxtpl import DocxTemplate, InlineImage
            from docx.shared import Mm
        except ImportError:
            raise ImportError(
                "docxtpl is required for DOCX generation. "
                "Install with: pip install docxtpl>=0.18.0"
            )

        if not template or not template.template_file_path:
            raise ValueError(
                "DOCX generation requires a template with a .docx file. "
                "Upload a template via /api/documents/templates/upload."
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

        output_path = self._output_path(workspace_id, title, "docx")
        doc.save(output_path)

        return self._build_result(output_path, "docx", title)

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
        return self._build_result(output_path, "xlsx", title)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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

        # Fallback: dump any remaining top-level string fields
        shown = {"title", "author", "date", "sections", "metrics",
                 "highlights", "recommendations", "columns", "rows", "_charts"}
        for key, val in data.items():
            if key in shown or key.startswith("_"):
                continue
            if isinstance(val, str) and val:
                lines.append(f"## {key.replace('_', ' ').title()}")
                lines.append("")
                lines.append(val)
                lines.append("")

        return "\n".join(lines)

    def _output_path(self, workspace_id: UUID, title: str, ext: str) -> str:
        """Build output file path, creating directories as needed."""
        safe_title = re.sub(r"[^\w\s-]", "", title).strip().replace(" ", "_")[:80]
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        directory = os.path.join(GENERATED_DIR, str(workspace_id), "generated")
        os.makedirs(directory, exist_ok=True)
        return os.path.join(directory, f"{timestamp}_{safe_title}.{ext}")

    def _build_result(self, path: str, fmt: str, title: str) -> GeneratedDocument:
        """Build a GeneratedDocument from a file on disk."""
        filename = os.path.basename(path)
        size = os.path.getsize(path)
        return GeneratedDocument(
            path=path,
            format=fmt,
            filename=filename,
            size=size,
            download_url=f"/api/documents/generated/{filename}",
            preview_url=f"/api/documents/generated/{filename}" if fmt == "pdf" else None,
        )
