"""PRD-256 P256-FIX-RVW-26 (FX-008, Decision D7): a send card names everyone the click mails.

The card read the first present key of the recipient fields alone: GMAIL_SEND_EMAIL with
``recipient_email``, ``extra_recipients``, ``cc`` and ``bcc`` read 'to: supplier@…', while
the grant's params hash covers the whole call, so the click mailed the bcc too. Under D7 the
card is the only review of an agent's send on Auto's ticket, and an agent that read an
inbound email writes these params.

Now every recipient-class field is listed, each address on its own line labelled 'to', 'cc'
or 'bcc' and never cut (a list in full), and each attachment is named. A recipient the card
cannot read (an object, a flag, a list holding one) refuses the call before any grant
(:func:`refused_before_the_send_card`): no card approves a recipient it does not show.
"""
from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any, Dict, Iterator, List, Optional, Tuple

from modules.tools.discovery.card_question_text import shown, value_line

TO, CC, BCC = "to", "cc", "bcc"
# A Composio send's recipient fields, by the names its actions use, in the order the card reads them.
TO_KEYS = ("recipient_email", "to", "to_email", "recipient", "recipients", "extra_recipients", "channel",
           "channel_id", "chat_id", "phone_number", "email")
CC_KEYS = ("cc", "cc_email", "cc_emails")
BCC_KEYS = ("bcc", "bcc_email", "bcc_emails")
RECIPIENT_GROUPS = ((TO, TO_KEYS), (CC, CC_KEYS), (BCC, BCC_KEYS))
ATTACHMENT_KEYS = ("attachment", "attachments")
FILE_NAME_KEYS = ("name", "file_name", "filename", "file_path", "path", "s3key")
ATTACHMENT = "attachment"
RECIPIENT_LINE = "{label}: {address}"
NAMED_JOIN = ", "
OTHER_LABEL = "{label} {address}"
UNREAD_RECIPIENT = ("Nothing was sent and no card was raised: the approval card cannot show the recipient "
                    "in '{field}' (a {kind}). Name each recipient as an address, or a list of addresses, "
                    "and ask again.")


def recipients(params: Any) -> List[Tuple[str, str]]:
    """Every ``(label, address)`` the call names, in the card's order: to, cc, bcc."""
    return [(label, address) for label, _, value in _recipient_fields(params) for address in _addresses(value)]


def recipient_lines(params: Any) -> List[str]:
    """One card line per address, uncut: "- to: ana@harbourline.test", "- bcc: x@other.test"."""
    return [value_line(RECIPIENT_LINE.format(label=label, address=address)) for label, address in recipients(params)]


def recipients_said(params: Any) -> Optional[str]:
    """The recipients in one phrase, as a 'Card raised' line and the click's note name them:
    "orders@kerbside.example, ana@x.test, cc bea@x.test, bcc x@other.test"; None when none."""
    said = [address if label == TO else OTHER_LABEL.format(label=label, address=address)
            for label, address in recipients(params)]
    return NAMED_JOIN.join(said) if said else None


def attachment_lines(params: Any) -> List[str]:
    """One card line per file the send carries: "- attachment: invoice-0412.pdf"."""
    if not isinstance(params, dict):
        return []
    named = [_file_name(item) for key in ATTACHMENT_KEYS for item in _listed(params.get(key))
             if str(item or "").strip()]
    return [value_line(RECIPIENT_LINE.format(label=ATTACHMENT, address=name)) for name in named]


def refused_before_the_send_card(params: Any) -> Optional[Dict[str, Any]]:
    """The refusal for a send whose recipient the card cannot read, before any grant; else None."""
    for _, key, value in _recipient_fields(params):
        unread = next((item for item in _listed(value) if not _readable(item)), None)
        if unread is not None:
            return {"success": False, "error": UNREAD_RECIPIENT.format(field=key, kind=type(unread).__name__)}
    return None


def _recipient_fields(params: Any) -> Iterator[Tuple[str, str, Any]]:
    """``(label, key, value)`` for each recipient field the call fills."""
    if not isinstance(params, dict):
        return
    for label, keys in RECIPIENT_GROUPS:
        for key in keys:
            if params.get(key) not in (None, "", []):
                yield label, key, params[key]


def _addresses(value: Any) -> List[str]:
    """A field's addresses, each on one line, whole."""
    said = (" ".join(str(item).split()) for item in _listed(value) if item is not None)
    return [address for address in said if address]


def _readable(item: Any) -> bool:
    """An address the card can show as it is: text or a number (a chat id), or nothing."""
    if item is None or isinstance(item, str):
        return True
    return isinstance(item, (int, float)) and not isinstance(item, bool)


def _file_name(item: Any) -> str:
    """An attachment's file name: a file object's name (or its path's last part), a path's last part."""
    if isinstance(item, dict):
        named = next((item[key] for key in FILE_NAME_KEYS if str(item.get(key) or "").strip()), None)
        return _file_name(named) if named is not None else shown(item)
    text = " ".join(str(item).split())
    return shown(PurePosixPath(text).name or text) if text else shown(None)


def _listed(value: Any) -> List[Any]:
    if value in (None, "", []):
        return []
    return list(value) if isinstance(value, (list, tuple)) else [value]


__all__ = ["BCC_KEYS", "CC_KEYS", "RECIPIENT_GROUPS", "TO_KEYS", "UNREAD_RECIPIENT", "attachment_lines",
           "recipient_lines", "recipients", "recipients_said", "refused_before_the_send_card"]
