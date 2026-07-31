"""PRD-223 S0.4 — the pre-resolution attachment reference must not state a verdict.

Regression cover for the 2026-07-31 incident: an uploaded document resolved
successfully and was injected into the prompt, but the message flattener had
already written "[Attached file: X — content not available]" above it. The
model read the stale claim and reported it as truth while holding the file.

Pure unit tests: no app boot, no DB, no LLM.
"""

from uuid import uuid4

from core.attachment_refs import render_unresolved_file_part
from consumers.chatbot.prompt_analyzer import PromptAnalyzer
from modules.attachments.extract import is_extraction_failure
from modules.context.sections.conversation import _parts_to_text


ATTACHMENT_ID = "6e7e8e6e-1335-48b8-b52b-38db187c6734"
FILENAME = "PRD-223-MODEL-GOVERNANCE-PROMOTION-GATE.md"
STALE_CLAIM = "content not available"


def _file_part(attachment_id=ATTACHMENT_ID, filename=FILENAME):
    part = {"type": "file", "filename": filename, "mediaType": "text/markdown"}
    if attachment_id:
        part["attachment_id"] = attachment_id
    return part


# ---------------------------------------------------------------------------
# render_unresolved_file_part — the single rule
# ---------------------------------------------------------------------------

class TestRenderUnresolvedFilePart:
    def test_resolver_owned_part_renders_nothing(self):
        """The resolver emits '### <filename>' + text; we must not pre-empt it."""
        assert render_unresolved_file_part(_file_part(), [ATTACHMENT_ID]) is None

    def test_part_not_being_resolved_gets_explicit_marker(self):
        """An attachment from an earlier turn is genuinely not in this prompt."""
        marker = render_unresolved_file_part(_file_part(), [str(uuid4())])
        assert marker is not None
        assert STALE_CLAIM in marker
        assert FILENAME in marker

    def test_part_without_attachment_id_gets_marker(self):
        """Nothing downstream will ever resolve it — silence would invite a guess."""
        marker = render_unresolved_file_part(_file_part(attachment_id=None), [])
        assert marker is not None
        assert STALE_CLAIM in marker

    def test_no_resolved_ids_at_all_gets_marker(self):
        assert render_unresolved_file_part(_file_part(), None) is not None

    def test_uuid_objects_in_resolved_set_are_matched(self):
        """Callers pass UUIDs as often as strings; both must match."""
        att_id = uuid4()
        part = _file_part(attachment_id=str(att_id))
        assert render_unresolved_file_part(part, [att_id]) is None

    def test_missing_filename_falls_back_without_crashing(self):
        marker = render_unresolved_file_part({"type": "file"}, [])
        assert marker is not None


# ---------------------------------------------------------------------------
# prompt_analyzer — ATOM + full chat paths
# ---------------------------------------------------------------------------

class TestPromptAnalyzerFlattening:
    def _messages(self):
        return [{
            "role": "user",
            "parts": [
                _file_part(),
                {"type": "text", "text": "say if you can read this"},
            ],
        }]

    def test_resolved_attachment_leaves_no_stale_claim(self):
        analyzer = PromptAnalyzer()
        out = analyzer.convert_to_llm_messages(
            self._messages(),
            system_prompt="",
            resolved_attachment_ids=[ATTACHMENT_ID],
        )
        user = [m for m in out if m["role"] == "user"][0]
        assert STALE_CLAIM not in user["content"]
        assert "say if you can read this" in user["content"]

    def test_unresolved_attachment_still_marked(self):
        analyzer = PromptAnalyzer()
        out = analyzer.convert_to_llm_messages(
            self._messages(), system_prompt="", resolved_attachment_ids=[],
        )
        user = [m for m in out if m["role"] == "user"][0]
        assert STALE_CLAIM in user["content"]

    def test_default_call_site_is_unchanged(self):
        """Callers that never resolve attachments keep the honest marker."""
        analyzer = PromptAnalyzer()
        out = analyzer.convert_to_llm_messages(self._messages(), system_prompt="")
        user = [m for m in out if m["role"] == "user"][0]
        assert STALE_CLAIM in user["content"]

    def test_incident_reproduction(self):
        """The full shape that misled the model, end to end at message level.

        Flatten first (as the ATOM path does), then append the resolver's
        real document part. The stale claim must not survive anywhere.
        """
        from modules.attachments.resolver import inject_parts_into_last_user_message

        analyzer = PromptAnalyzer()
        llm_messages = analyzer.convert_to_llm_messages(
            self._messages(),
            system_prompt="",
            resolved_attachment_ids=[ATTACHMENT_ID],
        )
        inject_parts_into_last_user_message(
            llm_messages,
            [{"type": "text", "text": f"### {FILENAME}\n\n# PRD-223\nreal body"}],
        )

        blob = str(llm_messages)
        assert STALE_CLAIM not in blob
        assert "real body" in blob


# ---------------------------------------------------------------------------
# conversation section — history rendering
# ---------------------------------------------------------------------------

class TestConversationPartsToText:
    def test_resolved_attachment_renders_nothing(self):
        text = _parts_to_text([_file_part()], [ATTACHMENT_ID])
        assert STALE_CLAIM not in text

    def test_history_attachment_keeps_marker(self):
        text = _parts_to_text([_file_part()], [str(uuid4())])
        assert STALE_CLAIM in text

    def test_text_parts_pass_through(self):
        text = _parts_to_text(
            [{"type": "text", "text": "hello"}, _file_part()], [ATTACHMENT_ID]
        )
        assert text == "hello"

    def test_image_parts_still_skipped(self):
        text = _parts_to_text(
            [{"type": "image_url", "image_url": {"url": "data:..."}}], []
        )
        assert text == ""


# ---------------------------------------------------------------------------
# extract — failure sentinels must be explicit, not "starts with a bracket"
# ---------------------------------------------------------------------------

class TestIsExtractionFailure:
    def test_markdown_opening_with_a_link_is_not_a_failure(self):
        """The bug: any '[' was read as a failure, dropping a real document."""
        assert is_extraction_failure("[See the docs](https://x) then read on") is False

    def test_markdown_opening_with_a_ci_badge_is_not_a_failure(self):
        assert is_extraction_failure("[![build](a.svg)](b)\n\n# Title") is False

    def test_real_sentinels_are_detected(self):
        for sentinel in (
            "[Extraction failed for x.pdf: boom]",
            "[PDF extraction requires pdfplumber]",
            "[PDF contains no extractable text — may be image-only or scanned]",
            "[DOCX extraction requires python-docx]",
            "[XLSX extraction requires openpyxl]",
            "[Binary file: x.bin]",
        ):
            assert is_extraction_failure(sentinel) is True

    def test_empty_is_not_a_sentinel(self):
        assert is_extraction_failure("") is False
