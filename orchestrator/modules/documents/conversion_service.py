"""
Document Conversion Service (PRD-63).

Converts DOCX/XLSX to PDF via Gotenberg (preferred) or LibreOffice CLI fallback.
"""

import logging
import os
import shutil
import subprocess
from typing import Optional

logger = logging.getLogger(__name__)

GOTENBERG_URL = os.environ.get("GOTENBERG_URL", "http://gotenberg:3000")


class ConversionService:
    """DOCX/XLSX → PDF conversion with graceful degradation."""

    def __init__(self, gotenberg_url: str = None):
        self.gotenberg_url = gotenberg_url or GOTENBERG_URL

    def is_gotenberg_available(self) -> bool:
        """Check if Gotenberg is reachable."""
        try:
            import httpx
            resp = httpx.get(f"{self.gotenberg_url}/health", timeout=3.0)
            return resp.status_code == 200
        except Exception:
            return False

    async def docx_to_pdf(self, docx_path: str) -> str:
        """Convert DOCX to PDF. Returns the PDF file path."""
        return await self._convert(docx_path, "docx")

    async def xlsx_to_pdf(self, xlsx_path: str) -> str:
        """Convert XLSX to PDF. Returns the PDF file path."""
        return await self._convert(xlsx_path, "xlsx")

    async def _convert(self, source_path: str, source_format: str) -> str:
        """Try Gotenberg first, then LibreOffice CLI, then raise."""
        pdf_path = source_path.rsplit(".", 1)[0] + ".pdf"

        # Strategy 1: Gotenberg
        if self.is_gotenberg_available():
            try:
                return await self._convert_via_gotenberg(source_path, pdf_path)
            except Exception as e:
                logger.warning(f"Gotenberg conversion failed, trying LibreOffice: {e}")

        # Strategy 2: LibreOffice CLI
        libreoffice_bin = shutil.which("libreoffice") or shutil.which("soffice")
        if libreoffice_bin:
            try:
                return self._convert_via_libreoffice(source_path, libreoffice_bin)
            except Exception as e:
                logger.warning(f"LibreOffice conversion failed: {e}")

        raise RuntimeError(
            "PDF conversion unavailable — install Gotenberg or LibreOffice"
        )

    async def _convert_via_gotenberg(self, source_path: str, pdf_path: str) -> str:
        """Convert via Gotenberg HTTP API."""
        try:
            from gotenberg_client import GotenbergClient
        except ImportError:
            raise ImportError("gotenberg-client not installed: pip install gotenberg-client>=1.0.0")

        async with GotenbergClient(self.gotenberg_url) as client:
            with open(source_path, "rb") as f:
                response = await client.libre_office.convert(f)
                with open(pdf_path, "wb") as out:
                    out.write(response.content)
        logger.info(f"Converted {source_path} → {pdf_path} via Gotenberg")
        return pdf_path

    def _convert_via_libreoffice(self, source_path: str, libreoffice_bin: str) -> str:
        """Convert via LibreOffice headless CLI."""
        output_dir = os.path.dirname(source_path)
        result = subprocess.run(
            [libreoffice_bin, "--headless", "--convert-to", "pdf", "--outdir", output_dir, source_path],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(f"LibreOffice conversion failed: {result.stderr}")

        pdf_path = source_path.rsplit(".", 1)[0] + ".pdf"
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"Expected PDF not found: {pdf_path}")

        logger.info(f"Converted {source_path} → {pdf_path} via LibreOffice")
        return pdf_path
