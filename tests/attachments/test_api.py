"""
Tests for Attachments API endpoints (PRD-127)
"""

import pytest
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from fastapi.testclient import TestClient


@pytest.fixture
def mock_context():
    """Create a mock request context."""
    ctx = MagicMock()
    ctx.workspace_id = str(uuid4())
    ctx.user_id = "test_user"
    ctx.agent_id = None
    return ctx


@pytest.fixture
def mock_store():
    """Create a mock attachment store."""
    store = MagicMock()
    return store


class TestUploadEndpoint:
    """Tests for POST /api/attachments."""

    @pytest.mark.asyncio
    async def test_upload_image_success(self, mock_context, mock_store):
        """Test successful image upload."""
        from api.attachments import upload_attachment, AttachmentResponse
        from modules.attachments.store import AttachmentRef, MediaType

        # Mock the store
        ref = AttachmentRef(
            attachment_id=uuid4(),
            workspace_id=uuid4(),
            media_type=MediaType.IMAGE,
            mime="image/png",
            filename="test.png",
            size_bytes=1024,
            s3_key="workspaces/ws/ephemeral-attachments/id/test.png",
        )
        mock_store.put = AsyncMock(return_value=ref)

        with patch("api.attachments.get_attachment_store", return_value=mock_store):
            # Create a mock UploadFile
            file = MagicMock()
            file.filename = "test.png"
            file.content_type = "image/png"
            file.read = AsyncMock(return_value=b"\x89PNG...")

            response = await upload_attachment(file=file, ctx=mock_context)

            assert isinstance(response, AttachmentResponse)
            assert response.filename == "test.png"
            assert response.mime == "image/png"
            assert response.media_type == "image"

    @pytest.mark.asyncio
    async def test_upload_validation_error(self, mock_context, mock_store):
        """Test upload with validation failure."""
        from api.attachments import upload_attachment
        from modules.attachments.validation import ValidationError
        from fastapi import HTTPException

        mock_store.put = AsyncMock(side_effect=ValidationError("File too large"))

        with patch("api.attachments.get_attachment_store", return_value=mock_store):
            file = MagicMock()
            file.filename = "huge.bin"
            file.content_type = "application/octet-stream"
            file.read = AsyncMock(return_value=b"x" * 100)

            with pytest.raises(HTTPException) as exc_info:
                await upload_attachment(file=file, ctx=mock_context)

            assert exc_info.value.status_code == 400
            assert "too large" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_upload_requires_workspace(self):
        """Test that upload requires workspace ID."""
        from api.attachments import upload_attachment
        from fastapi import HTTPException

        ctx = MagicMock()
        ctx.workspace_id = None

        file = MagicMock()
        file.read = AsyncMock(return_value=b"content")

        with pytest.raises(HTTPException) as exc_info:
            await upload_attachment(file=file, ctx=ctx)

        assert exc_info.value.status_code == 400
        assert "Workspace ID required" in exc_info.value.detail


class TestGetEndpoint:
    """Tests for GET /api/attachments/{id}."""

    @pytest.mark.asyncio
    async def test_get_existing_attachment(self, mock_context, mock_store):
        """Test getting an existing attachment."""
        from api.attachments import get_attachment
        from modules.attachments.store import AttachmentRef, MediaType

        attachment_id = uuid4()
        ref = AttachmentRef(
            attachment_id=attachment_id,
            workspace_id=uuid4(),
            media_type=MediaType.DOCUMENT,
            mime="text/plain",
            filename="doc.txt",
            size_bytes=512,
            s3_key="workspaces/ws/ephemeral-attachments/id/doc.txt",
        )
        mock_store.get = AsyncMock(return_value=ref)

        with patch("api.attachments.get_attachment_store", return_value=mock_store):
            response = await get_attachment(
                attachment_id=str(attachment_id), ctx=mock_context
            )

            assert response.attachment_id == str(attachment_id)
            assert response.filename == "doc.txt"

    @pytest.mark.asyncio
    async def test_get_not_found(self, mock_context, mock_store):
        """Test getting a non-existent attachment."""
        from api.attachments import get_attachment
        from modules.attachments.store import AttachmentNotFoundError
        from fastapi import HTTPException

        mock_store.get = AsyncMock(
            side_effect=AttachmentNotFoundError("Not found")
        )

        with patch("api.attachments.get_attachment_store", return_value=mock_store):
            with pytest.raises(HTTPException) as exc_info:
                await get_attachment(
                    attachment_id=str(uuid4()), ctx=mock_context
                )

            assert exc_info.value.status_code == 404


class TestDeleteEndpoint:
    """Tests for DELETE /api/attachments/{id}."""

    @pytest.mark.asyncio
    async def test_delete_success(self, mock_context, mock_store):
        """Test successful deletion."""
        from api.attachments import delete_attachment

        mock_store.delete = AsyncMock(return_value=None)

        attachment_id = str(uuid4())

        with patch("api.attachments.get_attachment_store", return_value=mock_store):
            response = await delete_attachment(
                attachment_id=attachment_id, ctx=mock_context
            )

            assert response["deleted"] is True
            assert response["attachment_id"] == attachment_id

    @pytest.mark.asyncio
    async def test_delete_not_found(self, mock_context, mock_store):
        """Test deleting a non-existent attachment."""
        from api.attachments import delete_attachment
        from modules.attachments.store import AttachmentNotFoundError
        from fastapi import HTTPException

        mock_store.delete = AsyncMock(
            side_effect=AttachmentNotFoundError("Not found")
        )

        with patch("api.attachments.get_attachment_store", return_value=mock_store):
            with pytest.raises(HTTPException) as exc_info:
                await delete_attachment(
                    attachment_id=str(uuid4()), ctx=mock_context
                )

            assert exc_info.value.status_code == 404
