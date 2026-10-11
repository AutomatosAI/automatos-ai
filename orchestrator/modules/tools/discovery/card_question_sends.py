"""PRD-256 P256-FIX-RVW-26 (FX-008, Decision D7): a send card names everyone the click mails.

The card read the first present key of the recipient fields alone: GMAIL_SEND_EMAIL with
``recipient_email``, ``extra_recipients``, ``cc`` and ``bcc`` read 'to: supplier@…', while
the grant's params hash covers the whole call, so the click mailed the bcc too. Under D7 the
card is the only review of an agent's send on Auto's ticket, and an agent that read an
inbound email writes these params.

Now every recipient-class field is listed, each address on its own line labelled 'to', 'cc'
or 'bcc' and never cut (a list in full), and each attachment is named with the file it
attaches (a path whole; an object's name and every source it names). Every other field the
call carries is listed too (:func:`other_lines`): a recipient under a name the card does not
know is still on it, a list of plain values item by item, uncut. A recipient the card cannot
read (an object, a flag, a list holding one) refuses the call before any grant
(:func:`refused_before_the_send_card`): no card approves a recipient it does not show.

P256-FIX-RVW-38: a send by reference (GMAIL_SEND_DRAFT {draft_id}, MAILCHIMP_SEND_CAMPAIGN
{campaign_id}) raised a card reading '(no recipient named)' and the id, while the draft could
be changed after the card (GMAIL_UPDATE_DRAFT) and the click sent whatever it held. Such a
send is refused before any grant (:func:`refused_as_sent_by_reference`), with the line that
tells the agent to send it with the action that names the recipient.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Iterator, List, Optional, Tuple

from modules.tools.discovery.card_question_text import said_line, shown, value_line
from modules.tools.discovery.send_words import BY_REFERENCE, BY_REFERENCE_KEYS, slug_words

TO, CC, BCC = "to", "cc", "bcc"
# A Composio send's recipient fields, by the names its actions use, in the order the card reads them.
TO_KEYS = ("recipient_email", "to", "to_email", "recipient", "recipients", "extra_recipients", "channel",
           "channel_id", "chat_id", "phone_number", "email")
CC_KEYS = ("cc", "cc_email", "cc_emails")
BCC_KEYS = ("bcc", "bcc_email", "bcc_emails")
RECIPIENT_GROUPS = ((TO, TO_KEYS), (CC, CC_KEYS), (BCC, BCC_KEYS))
ATTACHMENT_KEYS = ("attachment", "attachments")
FILE_NAME_KEYS = ("name", "file_name", "filename")
FILE_SOURCE_KEYS = ("s3key", "file_path", "path", "url")
SHOWN_KEYS = frozenset((*TO_KEYS, *CC_KEYS, *BCC_KEYS, *ATTACHMENT_KEYS))
ATTACHMENT = "attachment"
NAMED_FROM = "{name} (from {source})"
RECIPIENT_LINE = "{label}: {address}"
NAMED_JOIN = ", "
OTHER_LABEL = "{label} {address}"
UNREAD_RECIPIENT = ("Nothing was sent and no card was raised: the approval card cannot show the recipient "
                    "in '{field}' (a {kind}). Name each recipient as an address, or a list of addresses, "
                    "and ask again.")
SENT_BY_REFERENCE = ("Nothing was sent and no card was raised: {action} sends a {kind} the app keeps, whose "
                     "recipients and body can change after the card, so the approval card cannot show what the "
                     "click would send. Send it with the action that names each recipient, the subject and the "
                     "body itself, and ask again.")


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


def other_lines(params: Any, used: Any) -> List[str]:
    """A card line for each field the call carries besides those already shown (``used``):
    fail closed, so a recipient under a name the card does not know is still on it. A list
    of plain values is listed item by item, uncut; the field's name is one line, as a value is."""
    if not isinstance(params, dict):
        return []
    return [line for key, value in params.items() if key not in used and value not in (None, "", [], {})
            for line in _field_lines(shown(key), value)]


def _field_lines(field: str, value: Any) -> List[str]:
    if isinstance(value, (list, tuple)) and all(_readable(item) for item in value):
        return [value_line(RECIPIENT_LINE.format(label=field, address=item)) for item in _addresses(value)]
    return [said_line(field, value)]


def refused_before_the_send_card(params: Any) -> Optional[Dict[str, Any]]:
    """The refusal for a send whose recipient the card cannot read, before any grant; else None."""
    for _, key, value in _recipient_fields(params):
        unread = next((item for item in _listed(value) if not _readable(item)), None)
        if unread is not None:
            return {"success": False, "error": UNREAD_RECIPIENT.format(field=key, kind=type(unread).__name__)}
    return None


def refused_as_sent_by_reference(action: str, params: Any) -> Optional[Dict[str, Any]]:
    """The refusal for a send of a draft or a campaign the app keeps, named by the action
    (GMAIL_SEND_DRAFT, MAILCHIMP_SEND_CAMPAIGN) or by a draft's id, before any grant; else None."""
    kept = sorted(slug_words(action) & BY_REFERENCE)
    if not kept and isinstance(params, dict):
        kept = [key.removesuffix("_id").removesuffix("Id") for key in BY_REFERENCE_KEYS
                if params.get(key) not in (None, "")]
    if not kept:
        return None
    return {"success": False, "error": SENT_BY_REFERENCE.format(action=action, kind=kept[0].rstrip("s"))}


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
    """The file an attachment attaches, uncut: a path whole; a file object's name with every source it
    names ("invoice.pdf (from ws/7/payroll.xlsx)"), its sources alone, or the object as it is."""
    if not isinstance(item, dict):
        return _whole(item)
    name, sources = _first_text(item, FILE_NAME_KEYS), NAMED_JOIN.join(_texts(item, FILE_SOURCE_KEYS))
    if name and sources:
        return NAMED_FROM.format(name=name, source=sources)
    return name or sources or _whole(json.dumps(item, ensure_ascii=False, sort_keys=True, default=str))


def _whole(value: Any) -> str:
    """``value`` on one line, never cut: the file a click attaches is shown in full."""
    return " ".join(str(value).split())


def _texts(item: Dict[str, Any], keys: Tuple[str, ...]) -> List[str]:
    return [_whole(item[key]) for key in keys if str(item.get(key) or "").strip()]


def _first_text(item: Dict[str, Any], keys: Tuple[str, ...]) -> str:
    return next(iter(_texts(item, keys)), "")


def _listed(value: Any) -> List[Any]:
    if value in (None, "", []):
        return []
    return list(value) if isinstance(value, (list, tuple)) else [value]


__all__ = ["BCC_KEYS", "CC_KEYS", "RECIPIENT_GROUPS", "SENT_BY_REFERENCE", "SHOWN_KEYS", "TO_KEYS", "UNREAD_RECIPIENT",
           "attachment_lines", "other_lines", "recipient_lines", "recipients", "recipients_said", "refused_as_sent_by_reference",
           "refused_before_the_send_card"]
