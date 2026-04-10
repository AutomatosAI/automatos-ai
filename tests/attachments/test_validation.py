"""
Tests for attachment validation (PRD-127)
"""

import pytest

from modules.attachments.validation import (
    ValidationError,
    detect_mime,
    sanitize_filename,
    validate_upload,
)


class TestSanitizeFilename:
    """Tests for filename sanitization."""

    def test_basic_filename(self):
        assert sanitize_filename("test.txt") == "test.txt"

    def test_strips_directory(self):
        assert sanitize_filename("/etc/passwd") == "passwd"
        assert sanitize_filename("../../../etc/passwd") == "passwd"
        # Windows paths: backslashes are replaced with underscores after basename
        # On Unix, basename("C:\path") returns the whole string, then we replace \ with _
        result = sanitize_filename("C:\\Windows\\System32\\file.exe")
        assert "/" not in result
        assert ".." not in result

    def test_removes_null_bytes(self):
        assert sanitize_filename("test\x00.txt") == "test.txt"

    def test_removes_control_characters(self):
        assert sanitize_filename("test\x01\x02\x03.txt") == "test.txt"

    def test_truncates_long_names(self):
        long_name = "a" * 300 + ".txt"
        result = sanitize_filename(long_name)
        assert len(result) <= 200
        assert result.endswith(".txt")

    def test_rejects_empty(self):
        assert sanitize_filename("") == ""
        assert sanitize_filename(".") == ""
        assert sanitize_filename("..") == ""


class TestDetectMime:
    """Tests for MIME type detection."""

    def test_png_magic_bytes(self):
        png_header = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        assert detect_mime(png_header, "image.png") == "image/png"

    def test_jpeg_magic_bytes(self):
        jpeg_header = b"\xff\xd8\xff" + b"\x00" * 100
        assert detect_mime(jpeg_header, "photo.jpg") == "image/jpeg"

    def test_pdf_magic_bytes(self):
        pdf_header = b"%PDF-1.4" + b"\x00" * 100
        assert detect_mime(pdf_header, "document.pdf") == "application/pdf"

    def test_docx_from_extension(self):
        # DOCX files are ZIP archives
        docx_header = b"PK\x03\x04" + b"\x00" * 100
        assert (
            detect_mime(docx_header, "document.docx")
            == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )

    def test_xlsx_from_extension(self):
        xlsx_header = b"PK\x03\x04" + b"\x00" * 100
        assert (
            detect_mime(xlsx_header, "spreadsheet.xlsx")
            == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    def test_text_from_extension(self):
        text_content = b"Hello, world!"
        assert detect_mime(text_content, "readme.txt") == "text/plain"
        assert detect_mime(text_content, "script.py") == "text/x-python"
        assert detect_mime(text_content, "code.js") == "text/javascript"

    def test_fallback_octet_stream(self):
        binary_content = b"\x00\x01\x02\x03"
        assert detect_mime(binary_content, "unknown.bin") == "application/octet-stream"


class TestValidateUpload:
    """Tests for full upload validation."""

    def test_valid_png(self):
        # Minimal valid PNG
        png_content = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01"
            b"\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        result = validate_upload(png_content, "test.png", "image/png")
        assert result["mime"] == "image/png"
        assert result["safe_filename"] == "test.png"

    def test_valid_text(self):
        content = b"Hello, world!"
        result = validate_upload(content, "hello.txt", "text/plain")
        assert result["mime"] == "text/plain"

    def test_rejects_empty_file(self):
        with pytest.raises(ValidationError, match="Empty file"):
            validate_upload(b"", "empty.txt")

    def test_rejects_oversized_file(self):
        # Create content larger than MAX_ATTACHMENT_SIZE
        huge_content = b"x" * (51 * 1024 * 1024)  # 51 MB
        with pytest.raises(ValidationError, match="too large"):
            validate_upload(huge_content, "huge.bin")

    def test_sanitizes_dangerous_filename(self):
        content = b"test content"
        result = validate_upload(content, "../../../etc/passwd.txt")
        assert result["safe_filename"] == "passwd.txt"
        assert "/" not in result["safe_filename"]
        assert "\\" not in result["safe_filename"]

    def test_rejects_invalid_mime(self):
        # Binary content with unknown type
        content = b"\x00\x01\x02\x03" * 100
        with pytest.raises(ValidationError, match="not allowed"):
            validate_upload(content, "malware.exe")

    def test_allows_code_files(self):
        content = b"def hello(): pass"
        result = validate_upload(content, "script.py")
        assert result["mime"] == "text/x-python"

    def test_declared_mime_mismatch_uses_detected(self):
        # PNG content but declared as text
        png_content = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01"
            b"\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        result = validate_upload(png_content, "image.png", "text/plain")
        # Should use detected MIME, not declared
        assert result["mime"] == "image/png"
