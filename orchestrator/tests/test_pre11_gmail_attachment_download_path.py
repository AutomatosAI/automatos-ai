"""The email extractor marks a Gmail attachment as downloadable (7 Oct decision).

A Gmail attachment carries no URL. When the email's message id and the
attachment's attachmentId are both Gmail ids, the extractor gives it
``downloadPath``: the platform's own route for it, which the chat's email widget
fetches with auth. Outlook keeps its provider link; anything that is not a Gmail
id gets no path.
"""
from __future__ import annotations

import base64
from urllib.parse import parse_qs, urlsplit

from core.composio.gmail_attachments import DOWNLOAD_ROUTE_PREFIX
from modules.tools.formatting.schema_detector import GenericDataExtractor

MESSAGE = "18f2a9c4d1e0b7a3"
ATTACHMENT = "ANGjdJ8x-Wq_3vLk0" + "b" * 200


def _emails(*messages: dict) -> list:
    return GenericDataExtractor.extract_emails({"data": {"messages": list(messages), "nextPageToken": "n"}})


def _gmail(**extra) -> dict:
    return {"messageId": MESSAGE, "threadId": "t1", "subject": "Quote", "sender": "a@b.example",
            "messageText": "See attached.", **extra}


def test_a_gmail_attachment_gets_the_download_path_of_its_message():
    [email] = _emails(_gmail(attachmentList=[
        {"attachmentId": ATTACHMENT, "filename": "Quote 42.pdf", "mimeType": "application/pdf", "size": 2048},
    ]))

    [att] = email["attachments"]
    assert att["id"] == ATTACHMENT
    assert att["filename"] == "Quote 42.pdf"
    url = urlsplit(att["downloadPath"])
    assert url.path == f"{DOWNLOAD_ROUTE_PREFIX}/{MESSAGE}/{ATTACHMENT}"
    assert parse_qs(url.query) == {"filename": ["Quote 42.pdf"]}
    assert not url.scheme and not url.netloc
    assert att["downloadUrl"] is None


def test_a_gmail_payload_part_gets_the_download_path_too():
    [email] = _emails({"id": MESSAGE, "snippet": "hi", "payload": {"parts": [
        {"mimeType": "text/plain", "body": {"data": "aGk"}},
        {"partId": "1", "mimeType": "image/png", "filename": "logo.png",
         "body": {"attachmentId": ATTACHMENT, "size": 99}},
    ]}})

    [att] = email["attachments"]
    assert att["filename"] == "logo.png"
    assert att["downloadPath"].startswith(f"{DOWNLOAD_ROUTE_PREFIX}/{MESSAGE}/{ATTACHMENT}?")


def test_ids_that_are_not_gmail_ids_get_no_download_path():
    [email] = _emails(_gmail(messageId="<CAF+x@mail.gmail.com>", attachmentList=[
        {"attachmentId": ATTACHMENT, "filename": "a.pdf"},
    ]))
    assert email["attachments"][0]["downloadPath"] is None

    [email] = _emails(_gmail(attachmentList=[{"attachmentId": "../../admin", "filename": "a.pdf"}]))
    assert email["attachments"][0]["downloadPath"] is None


def test_an_outlook_attachment_keeps_its_link_and_gets_no_path():
    [email] = _emails({"id": "AAMkAG", "subject": "Hi", "bodyPreview": "x", "attachments": [
        {"id": "att-1", "name": "deck.pptx", "contentType": "application/vnd.ms-powerpoint",
         "size": 10, "contentLocation": "https://outlook.example/att-1"},
    ]})

    [att] = email["attachments"]
    assert att["downloadUrl"] == "https://outlook.example/att-1"
    assert "downloadPath" not in att


def test_an_email_without_attachments_has_none():
    [email] = _emails(_gmail())
    assert "attachments" not in email


def test_a_raw_gmail_payload_still_fills_the_headers_and_the_body():
    body = base64.urlsafe_b64encode(b"Plain body, see the logo.").decode().rstrip("=")
    [email] = _emails({"id": MESSAGE, "snippet": "s", "payload": {
        "headers": [{"name": "Subject", "value": "Hello"}, {"name": "From", "value": "a@b.example"},
                    {"name": "Subject", "value": "second subject is ignored"}],
        "body": {"data": body},
    }})

    assert email["subject"] == "Hello"
    assert email["from"] == "a@b.example"
    assert email["body"] == "Plain body, see the logo."
