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

        if format == "pdf":
            return await self.generate_pdf(template, data, ws, title)
        elif format == "docx":
            return await self.generate_docx(template, data, ws, title)
        elif format == "xlsx":
            return await self.generate_xlsx(data, ws, title=title, template=template)
        else:
            raise ValueError(f"Unsupported format: {format}. Use pdf, docx, or xlsx.")

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
            raise ValueError("PDF generation requires a template with HTML content.")

        # Validate data against schema
        self._validate_data(data, template.data_schema)

        # Render Jinja2
        try:
            jinja_template = self._jinja_env.from_string(template.template_content)
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

    def _validate_data(self, data: dict, schema: dict) -> None:
        """Validate data against the template's JSON Schema."""
        if not schema:
            return
        try:
            jsonschema.validate(instance=data, schema=schema)
        except jsonschema.ValidationError as e:
            raise ValueError(f"Data validation failed: {e.message}")

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
