"""What an agent reads after generate_document: which link goes where (F298).

F298 (night 8): the PDF link on card #0394.2 answered 403 SignatureDoesNotMatch
while the same file opened from Deliverables. The platform's link was sound: it
was presigned for ``localhost:9000``, the host the browser reaches (the three
links on #0249's card, minted the same way that night, verify). The agent
had copied it into its answer, 600 characters, and changed one of the 64 hex
digits of its signature (``78e47a4d…`` became ``78e47a7d…``). The tool's summary had
told it to use that share link "when emailing or messaging the document", and a
card answer is a message.

The summary now hands the agent a short link for the owner, the Deliverables
page opened on this document (``open_url``: no signature, never expires, opens
in both editions because the page fetches the file as the signed-in owner, or
anonymously in the local edition), and keeps the signed share link for people
outside the workspace only, saying a changed character breaks it. A summary
that has to be cut leaves the share link out rather than cutting it mid-way.

F331 (night 10): the summary also names the data keys the template had no place
for, so an agent never reports a document done over a page that lacks them.
F367 (night 10c): and the amounts that printed with no currency sign, the kit
having none, with the one question to ask the owner.
"""
from __future__ import annotations

import functools
from typing import Any, Callable, Dict, List

TOOL_NAME = "generate_document"
DEFAULT_MAX_CHARS = 20000
TRUNCATION_MARK = "..."
# What generate_document made, by the file it returned: a social template
# renders an MP4 or a PNG (PRD-251 US-117); anything else is a document.
GENERATED_FILE_KINDS = {"mp4": "video", "png": "image"}
# F331: eleven invoices went out a title on an empty page, each card saying "done".
UNUSED_KEYS_LINE = (
    "NOT IN THE {kind}: the template has no place for these data keys, so the page does not show them: "
    "{keys}. Do not call it done: make it again with the fields the template lists "
    "(platform_get_template_schema), or, for a PDF, with no template, which prints every key."
)

# F367: "Total due 269.00": the kit had no currency and nothing said so. Ask once, then the kit holds it.
NO_CURRENCY_LINE = (
    "NO CURRENCY SIGN: the brand kit has no currency, so these amounts printed as bare numbers: {keys}. "
    "Ask the owner once which currency they bill in; save it as the brand kit's currency (platform_update_brand_kit, "
    "currency such as GBP, or the owner sets it on the Brand kit page) and make the document again. Never add a "
    "currency sign the owner did not give."
)


def _document(result: Dict[str, Any]) -> Dict[str, Any]:
    results = result.get("results")
    first = (results or [{}])[0] if isinstance(results, list) else result
    return first if isinstance(first, dict) else {}


def document_lines(doc: Dict[str, Any]) -> List[str]:
    """What was made and the owner's link, the share link left out. Pure."""
    fmt = str(doc.get("format") or "pdf")
    kind = GENERATED_FILE_KINDS.get(fmt.lower(), "document")
    lines = [f"Generated {fmt.upper()} {kind}: {doc.get('filename', 'document')} ({doc.get('size_kb', 0)} KB)"]
    if doc.get("template_name"):
        lines.append(f"Template used: {doc['template_name']}")
    unused = doc.get("unused_data_keys") or []
    if unused:
        lines.append(UNUSED_KEYS_LINE.format(kind=kind.upper(), keys=", ".join(str(key) for key in unused)))
    unpriced = doc.get("amounts_without_currency") or []
    if unpriced:
        lines.append(NO_CURRENCY_LINE.format(keys=", ".join(str(key) for key in unpriced)))
    link = doc.get("open_url") or doc.get("app_url")
    if link:
        lines.append(
            f"Saved to Deliverables. The owner's link, which opens this {kind} in the app; give the owner THIS "
            f"link, on the card or in your answer: {link}"
        )
    lines.append("Copy a link exactly as given; never shorten or retype it, and never invent document:// links.")
    return lines


def share_line(doc: Dict[str, Any]) -> str:
    """The signed share link, for people outside the workspace only. Pure."""
    if not doc.get("share_url"):
        return ("No share link is available (object storage has no copy); the owner's link needs a "
                "signed-in workspace member.")
    return ("Share link ONLY for someone outside the workspace, in an email or a Slack message, never on a "
            "card: it is signed, so one changed character breaks it; paste it exactly as given, valid 7 days: "
            f"{doc['share_url']}")


def summarises_generated_documents(format_for_llm: Callable[..., str]) -> Callable[..., str]:
    """Add generate_document's lines to ToolResultFormatter.format_for_llm's summary."""

    @functools.wraps(format_for_llm)
    def summary(result: Dict[str, Any], tool_name: str, max_chars: int = DEFAULT_MAX_CHARS) -> str:
        text = format_for_llm(result, tool_name, max_chars)
        if tool_name != TOOL_NAME or text.startswith(f"Tool {tool_name} failed"):
            return text
        doc = _document(result)
        full = "\n".join([text, "", *document_lines(doc)])
        share = share_line(doc)
        if len(full) + len(share) + 1 <= max_chars:
            return f"{full}\n{share}"
        return full if len(full) <= max_chars else full[:max_chars] + TRUNCATION_MARK

    return summary
