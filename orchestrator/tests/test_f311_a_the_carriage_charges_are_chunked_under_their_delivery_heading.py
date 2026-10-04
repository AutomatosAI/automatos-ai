"""F311 (night 9) — a Markdown document is chunked by section, each under its heading.

wholesale-terms-2026.md (document 1526) was stored as 8 chunks by the topic-coherence
chunker: "## Delivery" and the van's days in one, "Carriage is charged per drop: under
12 kg £8.50, 12 kg and over £5.00" in the next, without its heading and glued to
"## Payment". Asked what a café pays for delivery on 10 kg, Auto was not handed the
carriage lines and said the terms name no delivery charge (ledger L1, L93). Now every
section keeps its heading and the headings above it, a short one stays whole, and no
line is dropped.
"""
from __future__ import annotations

import pytest

from config import config
from modules.rag.chunking.markdown_sections import section_chunks
from modules.rag.ingestion.coverage import kept_pct
from modules.rag.ingestion.manager import DocumentProcessor, DocumentType

WHOLESALE_TERMS = """# Harbourline wholesale terms (cafés) — 2026

Updated after the January review. Send this to any new café before their first order.

## Prices
- Standard wholesale price: **£22.00 per kg** (Harbour Blend and the core range).
- Cafés on our single origins pay **£24.00 per kg**.
- Prices are per kg of roasted coffee. Coffee is zero-rated, so there's no VAT on it.

## Ordering
- Minimum order: **6 kg** per delivery.
- Order by **noon the day before** your delivery day (email or the order form).
- We roast Tuesdays and Thursdays, so anything ordered after Thursday noon goes out on the following week's run.

## Delivery
Our van does fixed days:
- Bristol: Thursday and Friday
- Bath: Monday and Wednesday
- Taunton and Cheltenham: Monday
- Exeter: Tuesday
- Plymouth: Tuesday and Wednesday
- Bournemouth: Wednesday (every other week)

Carriage is charged per drop:
- under 12 kg: **£8.50**
- 12 kg and over: **£5.00**

## Payment
- **30 days** from the invoice date. A handful of accounts are on 14 days by agreement.
- Bank transfer only, please quote the invoice number.

## Returns
If a bag arrives damaged, photograph it and tell us within 48 hours. We'll replace it on the next drop.
"""

IMPORTERS = """# Green coffee — who we buy from

Three importers. Keep this up to date when contacts change!! (Ellie)

## Tidewater Importers (London)
- Supplies: **Guji Shakiso**, **Kirinyaga AA**, **Yirgacheffe Konga** (all our East Africans)
- Contact: Maya Odum, maya@tidewater-importers.example, 020 7946 0381
- Payment terms: **60 days**
- Lead time: **3 weeks** from order to our door
- Minimum: one 60 kg sack per lot

## Northfield Green Coffee Co. (Liverpool)
- Supplies: **Swiss Water Decaf**, **Huila La Esperanza**, **Nariño Buesaco**
- Contact: Sam Price, sam@northfield-green.example, 0151 496 0722
- Payment terms: **30 days**
- Lead time: **10 days**
- They'll split a sack for us if we ask nicely
"""


@pytest.fixture(autouse=True)
def section_settings(monkeypatch):
    monkeypatch.setattr(config, "RAG_SECTION_CHUNKING_ENABLED", True, raising=False)
    monkeypatch.setattr(config, "RAG_SECTION_MIN_CHARS", 200, raising=False)
    monkeypatch.setattr(config, "RAG_SECTION_MAX_CHARS", 1500, raising=False)


def _chunks(text, file_type=DocumentType.MARKDOWN):
    return DocumentProcessor().chunk_document(text, file_type, {"document_id": 1526})


def _holding(chunks, line):
    found = [chunk.content for chunk in chunks if line in chunk.content]
    assert len(found) == 1, f"{line!r} is in {len(found)} chunks"
    return found[0]


def test_the_carriage_charges_are_stored_with_their_delivery_heading_and_without_payment():
    chunks = _chunks(WHOLESALE_TERMS)
    carriage = _holding(chunks, "Carriage is charged per drop:")
    assert "## Delivery" in carriage
    assert "- under 12 kg: **£8.50**" in carriage and "- 12 kg and over: **£5.00**" in carriage
    assert carriage.startswith("# Harbourline wholesale terms (cafés) — 2026\n## Delivery")
    assert "## Payment" not in carriage and "30 days" not in carriage


def test_a_title_never_stands_alone_and_no_line_is_dropped():
    chunks = _chunks(WHOLESALE_TERMS)
    assert all(chunk.content.strip() != "# Harbourline wholesale terms (cafés) — 2026" for chunk in chunks)
    assert "## Prices" in _holding(chunks, "Updated after the January review.")
    assert kept_pct(WHOLESALE_TERMS, [chunk.content for chunk in chunks]) == 100
    assert all(chunk.metadata["sections"] and chunk.metadata["document_id"] == 1526 for chunk in chunks)


def test_each_importer_is_stored_under_the_documents_heading_with_its_own_terms():
    chunks = _chunks(IMPORTERS)
    northfield = _holding(chunks, "- Payment terms: **30 days**")
    assert northfield.startswith("# Green coffee — who we buy from\n## Northfield Green Coffee Co. (Liverpool)")
    assert "Tidewater" not in northfield
    tidewater = _holding(chunks, "Maya Odum")
    assert "- Lead time: **3 weeks** from order to our door" in tidewater and "Kirinyaga AA" in tidewater


def test_a_long_section_is_split_and_every_piece_carries_its_heading(monkeypatch):
    monkeypatch.setattr(config, "RAG_SECTION_MAX_CHARS", 300, raising=False)
    days = "\n".join(f"- Drop {n}: the café on street {n} takes its order on the van's round" for n in range(30))
    text = f"# Rounds\n\n## Drops\n{days}\n"
    chunks = _chunks(text)
    assert len(chunks) > 2
    assert all(len(chunk.content) <= 300 for chunk in chunks)
    assert all(chunk.content.startswith("# Rounds\n## Drops") for chunk in chunks)
    assert kept_pct(text, [chunk.content for chunk in chunks]) == 100


def test_a_document_without_headings_is_chunked_as_before():
    plain = "Carriage is charged per drop. Under 12 kg it is £8.50 and 12 kg and over it is £5.00. " * 3
    assert section_chunks(plain, 200, 1500) == []
    assert all(not chunk.metadata.get("sections") for chunk in _chunks(plain, DocumentType.TEXT))


def test_a_hash_inside_a_code_block_is_not_a_heading():
    text = "# Setup\n\nRun this:\n```\n# not a heading\npip install x\n```\n\n## After\nDone, and the café list is next."
    pieces = section_chunks(text, 10, 1500)
    assert [piece.heading for piece in pieces] == ["# Setup", "## After"]
    assert "# not a heading" in pieces[0].content
