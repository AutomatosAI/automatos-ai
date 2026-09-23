"""How much of a document the knowledge base actually holds (F086).

Night 3: ingestion kept brand-voice.md's two word lists and lost its sign-off,
kept a Christmas plan without a single price, and marked every one of them
``completed``. The owner found out only by reading the stored text back. After
chunking, ingestion now measures what the stored chunks hold against the text it
extracted, and a document under ``RAG_KEPT_WARN_PCT`` is shown as partial.
"""
from __future__ import annotations

from collections import Counter
from typing import Iterable


def kept_pct(text: str, chunks: Iterable[str]) -> int:
    """The share of the extracted text's words the chunks hold, as a whole
    percentage (rounded down). A word counts as often as it occurs in the
    text, so overlap between chunks never counts it twice; text with no words
    is fully kept."""
    source = Counter((text or "").split())
    total = sum(source.values())
    if not total:
        return 100
    held = Counter(word for chunk in chunks for word in (chunk or "").split())
    kept = sum(min(count, held[word]) for word, count in source.items())
    return (100 * kept) // total
