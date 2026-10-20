"""The two text splitters RAG ingestion uses, vendored from langchain-text-splitters 1.0.0.

``DocumentProcessor`` (``modules/rag/ingestion/manager.py``) splits a document with one
of these when the semantic chunker is unavailable or fails: Python with
``PythonCodeTextSplitter``, everything else with ``RecursiveCharacterTextSplitter``.
Until PRD-262 US-026 both came from langchain-text-splitters, which brought
langchain-core, langsmith and zstandard into the API image for two classes.

A splitter cuts the text at the coarsest separator it contains, keeps each separator
at the start of the piece after it, cuts any piece still ``chunk_size`` long or longer
at the next separator down, and merges the pieces back into chunks of at most
``chunk_size`` characters. Each chunk starts with the last pieces of the one before
it, as many as fit in ``chunk_overlap`` (fewer when the next piece would not fit), and
is stripped of surrounding whitespace.

Only what the ingest uses is here. The source is ``base.py``, ``character.py`` and
``python.py`` of langchain-text-splitters 1.0.0 (git tag
``langchain-text-splitters==1.0.0``, commit 3b8cb3d4b6122311097c3a5e9801b911525373d0),
restructured into small functions with the settings the ingest never changes fixed at
their defaults: the separator is kept at the start of the piece (``keep_separator=True``),
separators are literal text (``is_separator_regex=False``), chunks are stripped and no
start index is recorded. Python's separators are copied as a constant;
``get_separators_for_language`` and the other languages are not. ``TextChunk`` stands in
for langchain-core's ``Document`` with the two fields ``create_documents`` fills.
``tests/test_prd262_text_splitters.py`` holds this module to the chunks langchain records
on the committed fixtures, which test.yml's reference-goldens job re-records with the
original packages.
"""
# MIT License
#
# Copyright (c) LangChain, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# Paragraphs, lines, words, then single characters.
DEFAULT_SEPARATORS: Tuple[str, ...] = ("\n\n", "\n", " ", "")
# langchain's separators for Language.PYTHON: classes and functions at the left margin
# (and functions after one tab), then the default ladder.
PYTHON_SEPARATORS: Tuple[str, ...] = ("\nclass ", "\ndef ", "\n\tdef ", "\n\n", "\n", " ", "")
DEFAULT_CHUNK_SIZE = 4000
DEFAULT_CHUNK_OVERLAP = 200


@dataclass(frozen=True)
class TextChunk:
    """One chunk ``create_documents`` returns: its text and its own copy of its text's metadata."""

    page_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def _check_sizes(chunk_size: int, chunk_overlap: int) -> None:
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
    if chunk_overlap < 0:
        raise ValueError(f"chunk_overlap must be >= 0, got {chunk_overlap}")
    if chunk_overlap > chunk_size:
        raise ValueError(
            f"Got a larger chunk overlap ({chunk_overlap}) than chunk size ({chunk_size}), should be smaller."
        )


def _pick_separator(text: str, separators: Sequence[str]) -> Tuple[str, Tuple[str, ...]]:
    """The first separator ``text`` contains and the finer ones after it. The empty
    separator, or the last one when none is found, leaves nothing finer."""
    for index, separator in enumerate(separators):
        if not separator:
            return separator, ()
        if separator in text:
            return separator, tuple(separators[index + 1:])
    return separators[-1], ()


def _pieces(text: str, separator: str) -> List[str]:
    """``text`` cut before every ``separator``, which starts the piece after it; every
    character on its own when ``separator`` is empty. No piece is empty."""
    if not separator:
        return list(text)
    parts = re.split(f"({re.escape(separator)})", text)
    pieces = [parts[0]] + [parts[i] + parts[i + 1] for i in range(1, len(parts), 2)]
    return [piece for piece in pieces if piece]


def _chunk(pieces: Sequence[str]) -> List[str]:
    """The pieces joined and stripped, as one chunk; none when only whitespace is left."""
    text = "".join(pieces).strip()
    return [text] if text else []


class RecursiveCharacterTextSplitter:
    """Splits text at the coarsest separator in ``separators`` that it contains, and
    splits again at the finer ones any piece that is still too long."""

    def __init__(
        self,
        separators: Optional[Sequence[str]] = None,
        *,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        length_function: Callable[[str], int] = len,
    ) -> None:
        _check_sizes(chunk_size, chunk_overlap)
        self._separators = tuple(separators or DEFAULT_SEPARATORS)
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._length_function = length_function

    def create_documents(
        self, texts: Sequence[str], metadatas: Optional[Sequence[Dict[str, Any]]] = None
    ) -> List[TextChunk]:
        """Every text's chunks in order, each carrying a deep copy of its text's metadata."""
        metadatas_ = metadatas or [{}] * len(texts)
        return [
            TextChunk(page_content=chunk, metadata=copy.deepcopy(metadatas_[index]))
            for index, text in enumerate(texts)
            for chunk in self.split_text(text)
        ]

    def split_text(self, text: str) -> List[str]:
        """``text`` as chunks of at most ``chunk_size``, in order."""
        return self._split(text, self._separators)

    def _split(self, text: str, separators: Sequence[str]) -> List[str]:
        separator, finer = _pick_separator(text, separators)
        chunks: List[str] = []
        fitting: List[str] = []
        for piece in _pieces(text, separator):
            if self._length_function(piece) < self._chunk_size:
                fitting.append(piece)
                continue
            chunks += self._merge(fitting)
            fitting = []
            chunks += self._split(piece, finer) if finer else [piece]
        return chunks + self._merge(fitting)

    def _merge(self, pieces: Sequence[str]) -> List[str]:
        """Consecutive pieces joined into chunks of at most ``chunk_size``; a piece is never cut."""
        chunks: List[str] = []
        window: List[str] = []
        total = 0
        for piece in pieces:
            length = self._length_function(piece)
            if window and total + length > self._chunk_size:
                chunks += _chunk(window)
                window, total = self._overlap(window, total, length)
            window.append(piece)
            total += length
        return chunks + _chunk(window)

    def _overlap(self, window: Sequence[str], total: int, incoming: int) -> Tuple[List[str], int]:
        """The window's last pieces, as a new list, that start the next chunk: drop pieces from
        the front while more than ``chunk_overlap`` is left, or while the incoming piece would
        not fit."""
        start = 0
        while total > self._chunk_overlap or (total + incoming > self._chunk_size and total > 0):
            total -= self._length_function(window[start])
            start += 1
        return list(window[start:]), total


class PythonCodeTextSplitter(RecursiveCharacterTextSplitter):
    """Splits Python source at its classes, then its functions, then blank lines, lines and words."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(separators=PYTHON_SEPARATORS, **kwargs)
