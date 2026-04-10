"""
Tests for AttachmentStore (PRD-127)
"""

import pytest
from uuid import uuid4

from modules.attachments.store import (
    AttachmentNotFoundError,
    AttachmentStore,
    MediaType,
)


@pytest.fixture
def store():
    """Create a store using local filesystem fallback."""
    return AttachmentStore()


@pytest.fixture
def workspace_id():
    return uuid4()


@pytest.mark.asyncio
async def test_put_and_get_image(store, workspace_id):
    """Test uploading and retrieving an image."""
    # Minimal valid PNG (1x1 transparent)
    png_content = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01"
        b"\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    )

    ref = await store.put(
        workspace_id=workspace_id,
        uploaded_by="test_user",
        filename="test.png",
        content=png_content,
        declared_mime="image/png",
    )

    assert ref.media_type == MediaType.IMAGE
    assert ref.mime == "image/png"
    assert ref.filename == "test.png"
    assert ref.size_bytes == len(png_content)

    # Retrieve metadata
    retrieved = await store.get(ref.attachment_id, workspace_id)
    assert retrieved.attachment_id == ref.attachment_id
    assert retrieved.mime == "image/png"


@pytest.mark.asyncio
async def test_put_and_get_document(store, workspace_id):
    """Test uploading and retrieving a text document."""
    content = b"Hello, world!"

    ref = await store.put(
        workspace_id=workspace_id,
        uploaded_by="test_user",
        filename="hello.txt",
        content=content,
        declared_mime="text/plain",
    )

    assert ref.media_type == MediaType.DOCUMENT
    assert ref.mime == "text/plain"
    assert ref.filename == "hello.txt"


@pytest.mark.asyncio
async def test_open_returns_bytes(store, workspace_id):
    """Test downloading attachment content."""
    content = b"Test content for download"

    ref = await store.put(
        workspace_id=workspace_id,
        uploaded_by="test_user",
        filename="download.txt",
        content=content,
    )

    downloaded = await store.open(ref.attachment_id, workspace_id)
    assert downloaded == content


@pytest.mark.asyncio
async def test_sign_url_returns_url(store, workspace_id):
    """Test generating a signed URL."""
    content = b"Content for signing"

    ref = await store.put(
        workspace_id=workspace_id,
        uploaded_by="test_user",
        filename="signed.txt",
        content=content,
    )

    url = await store.sign_url(ref.attachment_id, workspace_id)
    # Local dev returns file:// URL
    assert url.startswith("file://") or url.startswith("https://")


@pytest.mark.asyncio
async def test_delete_removes_attachment(store, workspace_id):
    """Test deleting an attachment."""
    content = b"To be deleted"

    ref = await store.put(
        workspace_id=workspace_id,
        uploaded_by="test_user",
        filename="delete_me.txt",
        content=content,
    )

    # Delete
    await store.delete(ref.attachment_id, workspace_id)

    # Should raise not found
    with pytest.raises(AttachmentNotFoundError):
        await store.get(ref.attachment_id, workspace_id)


@pytest.mark.asyncio
async def test_get_nonexistent_raises(store, workspace_id):
    """Test that getting a nonexistent attachment raises."""
    with pytest.raises(AttachmentNotFoundError):
        await store.get(uuid4(), workspace_id)


@pytest.mark.asyncio
async def test_workspace_isolation(store):
    """Test that attachments are isolated by workspace."""
    ws1 = uuid4()
    ws2 = uuid4()
    content = b"Isolated content"

    ref = await store.put(
        workspace_id=ws1,
        uploaded_by="test_user",
        filename="isolated.txt",
        content=content,
    )

    # Should succeed for ws1
    await store.get(ref.attachment_id, ws1)

    # Should fail for ws2
    with pytest.raises(AttachmentNotFoundError):
        await store.get(ref.attachment_id, ws2)


@pytest.mark.asyncio
async def test_inline_metadata_format(store, workspace_id):
    """Test the inline metadata format for JSONB storage."""
    content = b"Metadata test"

    ref = await store.put(
        workspace_id=workspace_id,
        uploaded_by="test_user",
        filename="meta.txt",
        content=content,
    )

    meta = ref.to_inline_metadata()
    assert "attachment_id" in meta
    assert meta["filename"] == "meta.txt"
    assert meta["mime"] == "text/plain"
    assert meta["media_type"] == "document"
