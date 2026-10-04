"""F298 (night 8, #0249, 02:27): a document that fails says so in plain words.

The crash text reached the owner's card: "Step 1 failed: its last tool call,
generate_document, failed: Tool generate_document failed: 'str' object has no
attribute 'get'". Now data the tool cannot use is refused with what to send
instead, and an unexpected failure while the document is made is logged and
answered in words: no Python exception text is ever the tool's answer.
"""
from __future__ import annotations

import pytest

from modules.tools.execution.generate_document_tool import NOT_MADE
from tests.f298_fixtures import call_tool, recorded

PYTHON_TEXT = ("object has no attribute", "Traceback", "TypeError", "AttributeError")


def _plain(error: str) -> bool:
    return not any(text in error for text in PYTHON_TEXT)


def test_a_crash_inside_the_render_is_answered_in_words(monkeypatch):
    recorded(monkeypatch, fails_with=AttributeError("'str' object has no attribute 'get'"))

    answer = call_tool({"title": "Checklist", "format": "pdf", "data": {"content": "- [ ] Count the boxes"}})

    assert answer["success"] is False
    assert answer["error"] == NOT_MADE and _plain(answer["error"])


@pytest.mark.parametrize(
    "parameters, says",
    [
        pytest.param({"title": "Checklist", "format": "pdf", "data": 42},
                     "'data' must be an object", id="a-number"),
        pytest.param({"title": "Checklist", "format": "xlsx", "data": "Harbour Blend 19.50"},
                     "'data' must be an object", id="text-for-a-spreadsheet"),
        pytest.param({"title": "Letter", "format": "pdf", "template_name": "Branded Letter", "data": "Dear Tom"},
                     "a template is filled from named fields", id="text-for-a-template"),
        pytest.param({"title": "Checklist", "format": "pdf", "data": "{\"sections\": ["},
                     "looks like JSON but is not a complete object", id="broken-json"),
    ],
)
def test_data_the_tool_cannot_use_is_refused_before_anything_renders(monkeypatch, parameters, says):
    service = recorded(monkeypatch)

    answer = call_tool(parameters)

    assert answer["success"] is False
    assert says in answer["error"] and "The document was not made" in answer["error"]
    assert _plain(answer["error"])
    assert service.asked == []


def test_a_message_the_platform_wrote_itself_is_passed_on(monkeypatch):
    recorded(monkeypatch, fails_with=ValueError("XLSX generation requires 'columns' in data."))

    answer = call_tool({"title": "Export", "format": "xlsx", "data": {"rows": [["a"]]}})

    assert answer["success"] is False
    assert answer["error"] == "XLSX generation requires 'columns' in data."
