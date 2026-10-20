"""The documents and splitter settings behind PRD-262 US-026's golden chunks.

Until US-026, RAG ingestion split text with two classes from langchain-text-splitters
1.0.0: ``DocumentProcessor`` (``modules/rag/ingestion/manager.py``) builds a
``RecursiveCharacterTextSplitter`` for text and a ``PythonCodeTextSplitter`` for
Python, both at 1,000 characters with 200 of overlap, and calls
``create_documents([text])`` on one of them when the semantic chunker is unavailable
or fails. US-026 vendored the two classes into ``modules/rag/chunking/text_splitters.py``.

Each case here is one ``create_documents`` call: a splitter, its settings and the
documents in ``text_splitters/`` it splits. The documents cover the paths through
the splitter: a Markdown guide (paragraphs merged with overlap, and a table too long
for one chunk, split at its lines), plain notes (a paragraph split at its words, and a
1,119-character run with no space, split at its characters) and a Python module (split
at its classes, then its functions, then its blank lines). Two short labels, split at
60 characters with 20 of overlap, cover what the 1,000-character cases do not reach:
a chunk exactly ``chunk_size`` long, an overlap dropped so that a long word fits,
several texts in one call and their metadata.

Two readers use them:

- ``scripts/ci/reference/text_splitters.py`` runs every case through langchain and
  records the chunks in ``text_splitter_chunks.json`` here;
- ``tests/test_prd262_text_splitters.py`` checks that the vendored splitters return
  the same chunks, and that ``DocumentProcessor.chunk_document`` serves them.

The documents hold nobody else's content, so they carry no licence. Standard library
only: the reference script loads this file in a venv that has none of the app.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DOCUMENTS = Path(__file__).resolve().parent / "text_splitters"

# The splitters DocumentProcessor builds, with the settings it gives them.
TEXT = "RecursiveCharacterTextSplitter"
CODE = "PythonCodeTextSplitter"
MANAGER_SIZES = (("chunk_size", 1000), ("chunk_overlap", 200))
LABEL_SIZES = (("chunk_size", 60), ("chunk_overlap", 20))


@dataclass(frozen=True)
class Case:
    """One ``create_documents`` call: which splitter, its settings, which documents."""

    splitter: str
    sizes: Tuple[Tuple[str, int], ...]
    documents: Tuple[str, ...]
    metadatas: Optional[Tuple[Dict[str, Any], ...]] = None

    def settings(self) -> Dict[str, Any]:
        """The splitter's keyword arguments."""
        return dict(self.sizes)

    def texts(self) -> List[str]:
        """The documents' text, in order."""
        return [read_document(name) for name in self.documents]

    def metadata_list(self) -> Optional[List[Dict[str, Any]]]:
        """Fresh copies of the metadata, one per document, or None."""
        return None if self.metadatas is None else [dict(entry) for entry in self.metadatas]


def read_document(name: str) -> str:
    """A fixture document as the ingest hands it over: text, with ``\\n`` line ends."""
    return (DOCUMENTS / name).read_text(encoding="utf-8")


def _manager_cases(document: str) -> Dict[str, Case]:
    return {
        f"{document}/text": Case(TEXT, MANAGER_SIZES, (document,)),
        f"{document}/code": Case(CODE, MANAGER_SIZES, (document,)),
    }


CASES: Dict[str, Case] = {
    **_manager_cases("guide.md"),
    **_manager_cases("notes.txt"),
    **_manager_cases("inventory.py.txt"),
    "labels/text": Case(
        TEXT,
        LABEL_SIZES,
        ("label-brazil.txt", "label-colombia.txt"),
        (
            {"lot": "lot0001", "origin": "Brazil", "tags": ["natural", "12 kg"]},
            {"lot": "lot0002", "origin": "Colombia", "tags": ["washed"]},
        ),
    ),
}
