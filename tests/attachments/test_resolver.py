"""
Tests for AttachmentResolver (PRD-127)
"""

import pytest
from uuid import uuid4

from modules.attachments.resolver import (
    AttachmentResolver,
    VisionNotSupportedError,
    inject_parts_into_last_user_message,
)
from modules.attachments.store import AttachmentStore, MediaType


@pytest.fixture
def store():
    """Create a store using local filesystem fallback."""
    return AttachmentStore()


@pytest.fixture
def resolver(store):
    """Create a resolver with the store."""
    return AttachmentResolver(store=store)


@pytest.fixture
def workspace_id():
    return uuid4()


class TestAttachmentResolver:
    """Tests for the main resolve() method."""

    @pytest.mark.asyncio
    async def test_resolve_empty_list(self, resolver, workspace_id):
        """Test that empty attachment list returns empty parts."""
        parts = await resolver.resolve([], workspace_id, "gpt-4o")
        assert parts == []

    @pytest.mark.asyncio
    async def test_resolve_document_returns_text_part(self, resolver, store, workspace_id):
        """Test that document attachments resolve to text parts."""
        content = b"Document content here"
        ref = await store.put(
            workspace_id=workspace_id,
            uploaded_by="test",
            filename="doc.txt",
            content=content,
        )

        parts = await resolver.resolve(
            [ref.attachment_id], workspace_id, "gpt-4o"
        )

        assert len(parts) == 1
        assert parts[0]["type"] == "text"
        assert "doc.txt" in parts[0]["text"]
        assert "Document content here" in parts[0]["text"]

    @pytest.mark.asyncio
    async def test_resolve_image_returns_image_url_part(self, resolver, store, workspace_id):
        """Test that image attachments resolve to image_url parts."""
        # Minimal PNG
        png_content = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01"
            b"\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        ref = await store.put(
            workspace_id=workspace_id,
            uploaded_by="test",
            filename="image.png",
            content=png_content,
            declared_mime="image/png",
        )

        parts = await resolver.resolve(
            [ref.attachment_id], workspace_id, "gpt-4o"
        )

        assert len(parts) == 1
        assert parts[0]["type"] == "image_url"
        assert "url" in parts[0]["image_url"]
        # Small images use base64 inline
        assert parts[0]["image_url"]["url"].startswith("data:image/png;base64,")

    @pytest.mark.asyncio
    async def test_resolve_image_non_vision_model_raises(self, resolver, store, workspace_id):
        """Test that images on non-vision models raise VisionNotSupportedError."""
        png_content = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01"
            b"\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        ref = await store.put(
            workspace_id=workspace_id,
            uploaded_by="test",
            filename="image.png",
            content=png_content,
            declared_mime="image/png",
        )

        with pytest.raises(VisionNotSupportedError) as exc_info:
            await resolver.resolve(
                [ref.attachment_id], workspace_id, "gpt-3.5-turbo"
            )

        assert "gpt-3.5-turbo" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_resolve_mixed_content(self, resolver, store, workspace_id):
        """Test resolving mixed images and documents."""
        # Upload a document
        doc_ref = await store.put(
            workspace_id=workspace_id,
            uploaded_by="test",
            filename="readme.txt",
            content=b"README content",
        )

        # Upload an image
        png_content = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01"
            b"\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        img_ref = await store.put(
            workspace_id=workspace_id,
            uploaded_by="test",
            filename="photo.png",
            content=png_content,
            declared_mime="image/png",
        )

        parts = await resolver.resolve(
            [doc_ref.attachment_id, img_ref.attachment_id],
            workspace_id,
            "gpt-4o",
        )

        assert len(parts) == 2
        types = {p["type"] for p in parts}
        assert types == {"text", "image_url"}

    @pytest.mark.asyncio
    async def test_resolve_skips_missing_attachments(self, resolver, workspace_id):
        """Test that missing attachments are skipped (not errored)."""
        parts = await resolver.resolve(
            [uuid4()],  # Non-existent ID
            workspace_id,
            "gpt-4o",
        )
        assert parts == []


class TestVisionCapabilityCheck:
    """Tests for vision model detection."""

    @pytest.mark.asyncio
    async def test_known_vision_models(self, resolver):
        """Test that known vision models are detected."""
        vision_models = [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-vision-preview",
            "gpt-4-turbo",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-sonnet-4-20250514",
            "claude-opus-4-20250514",
            "gemini-pro-vision",
            "openai/gpt-4o",
            "anthropic/claude-3-opus",
        ]
        for model in vision_models:
            result = await resolver._check_vision_support(model)
            assert result is True, f"{model} should support vision"

    @pytest.mark.asyncio
    async def test_known_text_only_models(self, resolver):
        """Test that known text-only models are detected."""
        text_models = [
            "gpt-3.5-turbo",
            "claude-2",
            "claude-instant-1.2",
            "deepseek/deepseek-chat",
            "mistral-7b-instruct",
            "mixtral-8x7b",
            "meta-llama/llama-2-70b",
        ]
        for model in text_models:
            result = await resolver._check_vision_support(model)
            assert result is False, f"{model} should not support vision"


class TestInjectParts:
    """Tests for inject_parts_into_last_user_message."""

    def test_inject_into_string_content(self):
        """Test injecting parts into a message with string content."""
        messages = [
            {"role": "user", "content": "Hello"},
        ]
        parts = [{"type": "image_url", "image_url": {"url": "http://example.com/img.png"}}]

        result = inject_parts_into_last_user_message(messages, parts)

        assert result[0]["content"][0] == {"type": "text", "text": "Hello"}
        assert result[0]["content"][1] == parts[0]

    def test_inject_into_list_content(self):
        """Test injecting parts into a message with list content."""
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Existing text"}],
            },
        ]
        parts = [{"type": "text", "text": "### doc.txt\nContent"}]

        result = inject_parts_into_last_user_message(messages, parts)

        assert len(result[0]["content"]) == 2
        assert result[0]["content"][1] == parts[0]

    def test_inject_finds_last_user_message(self):
        """Test that injection targets the last user message."""
        messages = [
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Response"},
            {"role": "user", "content": "Second"},
        ]
        parts = [{"type": "text", "text": "Injected"}]

        result = inject_parts_into_last_user_message(messages, parts)

        # First user message unchanged
        assert result[0]["content"] == "First"
        # Last user message has parts
        assert isinstance(result[2]["content"], list)
        assert result[2]["content"][0]["text"] == "Second"

    def test_inject_empty_parts_returns_unchanged(self):
        """Test that empty parts list returns messages unchanged."""
        messages = [{"role": "user", "content": "Hello"}]
        result = inject_parts_into_last_user_message(messages, [])
        assert result == messages

    def test_inject_no_user_message_appends(self):
        """Test behavior when no user message exists."""
        messages = [{"role": "system", "content": "System"}]
        parts = [{"type": "text", "text": "Content"}]

        result = inject_parts_into_last_user_message(messages, parts)

        # Should append a new user message
        assert len(result) == 2
        assert result[1]["role"] == "user"
        assert result[1]["content"] == parts
