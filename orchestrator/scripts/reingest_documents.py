"""
F086: re-ingest, in place, the documents the old chunker cut short
==================================================================

Until F086 the chunker dropped every segment under 100 characters at a topic
shift, so documents were marked completed holding part of their text (night 3:
480 of 575 under 98%). New ingestion keeps every line; this script brings the
existing documents up to it.

For each document it finds the SOURCE — the uploaded file on disk, or the copy
object storage keeps for agent reports and cloud or Auto uploads — extracts its
text exactly as ingestion does, and measures how much of it the stored chunks
hold. Dry run (the default) prints that list. ``--apply`` re-ingests every
document under the threshold under the SAME id: its chunks, vectors and table
and formula rows are cleared and it runs through ingestion again, which records
the new kept share. The knowledge graph is left as it is — it is built from the
source, not the chunks — so no extraction is paid for twice.

``--move-uploads`` moves every uploaded file that sits outside
DOCUMENT_UPLOAD_DIR into it and points ``documents.file_path`` there, so the
sources outlive the next container rebuild (F102). Documents with no source
anywhere are listed for the owner to upload again, with where the source should
be. That includes an object storage no longer holds (F128): it is reported and
skipped, never fatal. ``--apply`` re-ingests the rest, then exits 1 with the
count of documents it could not re-ingest.

Usage::

    cd orchestrator
    python scripts/reingest_documents.py                          # the list
    python scripts/reingest_documents.py --workspace <ws> --apply
    python scripts/reingest_documents.py --ids 720,726 --apply
    python scripts/reingest_documents.py --move-uploads [--apply]
"""

from __future__ import annotations

import argparse
import asyncio
import os
import shutil
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SOURCE_LOCAL, SOURCE_OBJECT_STORAGE, SOURCE_NONE = "local file", "object storage", "none"
# What object storage answers for a key it doesn't hold: HeadObject (download_file) says 404.
MISSING_OBJECT_CODES = frozenset({"404", "NoSuchKey", "NotFound"})


class SourceMissing(Exception):
    """The document's source is gone. It is reported as NO SOURCE and skipped, never fatal (F128)."""


@dataclass(frozen=True)
class Candidate:
    id: int
    workspace_id: str
    filename: str
    file_type: Optional[str]
    file_path: Optional[str]
    kept: Optional[int]          # None when there is no source to measure against
    source: str


def source_of(file_path: Optional[str]) -> str:
    if file_path and file_path.startswith("s3://"):
        return SOURCE_OBJECT_STORAGE
    if file_path and os.path.exists(file_path):
        return SOURCE_LOCAL
    return SOURCE_NONE


def missing_from(file_path: Optional[str]) -> str:
    """Where a document's source should have been, for its NO SOURCE line."""
    if not file_path:
        return "no path recorded"
    if file_path.startswith("s3://"):
        return f"object storage has no {file_path}"
    return f"no file at {file_path}"


def no_source_line(c: Candidate) -> str:
    where = missing_from(c.file_path)
    return f"  #{c.id} [{c.workspace_id[:8]}] {c.filename} — NO SOURCE ({where}): the owner must upload it again"


def plan(rows: Sequence[tuple], *, below: int, measure: Callable[[str, Optional[str], Sequence[str]], Optional[int]]
         ) -> List[Candidate]:
    """Every document under ``below`` percent kept, or with no source at all.
    ``rows``: (id, workspace_id, filename, file_type, file_path, [chunk texts])."""
    out: List[Candidate] = []
    for doc_id, workspace_id, filename, file_type, file_path, chunks in rows:
        source = source_of(file_path)
        kept = None
        if source != SOURCE_NONE:
            try:
                kept = measure(str(workspace_id), file_path, chunks)
            except SourceMissing:
                source = SOURCE_NONE
        if kept is None or kept < below:
            out.append(Candidate(doc_id, str(workspace_id), filename, file_type, file_path, kept, source))
    return out


async def reingest(candidate: Candidate, manager, local_path: str) -> None:
    """Clear what ingestion stored for the document and run it again, same id."""
    from api.documents import processing_type

    manager.clear_chunks(candidate.id)
    s3_key = candidate.file_path.split("/", 3)[3] if candidate.source == SOURCE_OBJECT_STORAGE else None
    await manager._process_document(
        candidate.id, local_path, processing_type(candidate.file_type or ""), s3_key,
        filename=candidate.filename, update_graph=False,
    )


def moved_path(file_path: str, upload_dir: Path) -> Optional[Path]:
    """Where an uploaded file outside ``upload_dir`` goes; None if it stays."""
    path = Path(file_path)
    if not path.is_absolute() or upload_dir in path.parents:
        return None
    return upload_dir / path.name


# ── the live stack ──────────────────────────────────────────────────────────

def _rows(db, workspace: Optional[str], ids: Optional[List[int]]) -> List[tuple]:
    from sqlalchemy import text

    sql = """SELECT d.id, d.workspace_id::text, d.filename, d.file_type, d.file_path,
                    COALESCE(array_agg(c.content ORDER BY c.chunk_index) FILTER (WHERE c.id IS NOT NULL), '{}')
               FROM documents d LEFT JOIN document_chunks c ON c.document_id = d.id
              WHERE (CAST(:ws AS text) IS NULL OR d.workspace_id::text = CAST(:ws AS text))
                AND (CAST(:ids AS int[]) IS NULL OR d.id = ANY(CAST(:ids AS int[])))
              GROUP BY d.id ORDER BY d.workspace_id, d.id"""
    return [tuple(r) for r in db.execute(text(sql), {"ws": workspace, "ids": ids}).all()]


def _fetch(manager, key: str, dest: str, file_path: str) -> None:
    """Download one object. A key object storage doesn't hold raises SourceMissing (F128).
    Any other error (access, network, a missing bucket) still stops the run."""
    from botocore.exceptions import ClientError

    try:
        manager.s3_client.download_file(manager.s3_bucket, key, dest)
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in MISSING_OBJECT_CODES:
            raise SourceMissing(missing_from(file_path)) from exc
        raise


class _Stack:
    """Managers per workspace, sources fetched to disk, text measured as ingestion extracts it."""

    def __init__(self):
        from modules.rag.ingestion.manager import DocumentProcessor

        self._managers: Dict[str, object] = {}
        self._processor = DocumentProcessor()

    def manager(self, workspace_id: str):
        if workspace_id not in self._managers:
            from api.documents import get_document_manager

            self._managers[workspace_id] = get_document_manager(workspace_id)
        return self._managers[workspace_id]

    @contextmanager
    def local_copy(self, workspace_id: str, file_path: str) -> Iterator[str]:
        """The source as a local path. An object storage doesn't hold raises SourceMissing."""
        if not file_path.startswith("s3://"):
            yield file_path
            return
        manager = self.manager(workspace_id)
        key = file_path.split("/", 3)[3]
        handle, tmp = tempfile.mkstemp(suffix=os.path.splitext(key)[1])
        os.close(handle)
        try:
            _fetch(manager, key, tmp, file_path)
            yield tmp
        finally:
            os.unlink(tmp)

    def measure(self, workspace_id: str, file_path: Optional[str], chunks: Sequence[str]) -> Optional[int]:
        from modules.rag.ingestion.coverage import kept_pct

        try:
            with self.local_copy(workspace_id, file_path) as local:
                return kept_pct(self._processor.extract_text_from_file(local), chunks)
        except SourceMissing:
            raise                    # the plan lists it as NO SOURCE
        except Exception as exc:  # noqa: BLE001 — an unreadable source is reported, not fatal
            print(f"  ! {file_path}: {exc}")
            return None


def _apply(todo: Sequence[Candidate], stack: _Stack, db) -> List[Candidate]:
    """Re-ingest each candidate in turn and return the ones whose source was gone. The
    source is fetched before anything of the document is cleared, so a missing one is
    skipped untouched and the run carries on (F128)."""
    missing: List[Candidate] = []
    for c in todo:
        try:
            with stack.local_copy(c.workspace_id, c.file_path) as local:
                asyncio.run(reingest(c, stack.manager(c.workspace_id), local))
        except SourceMissing:
            print(no_source_line(c))
            missing.append(c)
            continue
        after = _rows(db, c.workspace_id, [c.id])
        try:
            kept = stack.measure(c.workspace_id, after[0][4], after[0][5]) if after else None
        except SourceMissing:
            kept = None
        print(f"  #{c.id} {c.filename}: {c.kept}% -> {kept}%")
    return missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workspace")
    parser.add_argument("--ids", help="comma-separated document ids")
    parser.add_argument("--below", type=int, default=None, help="kept %% threshold (default RAG_KEPT_WARN_PCT)")
    parser.add_argument("--move-uploads", action="store_true")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import Session

    from api.documents import UPLOAD_DIR
    from config import config

    below = args.below if args.below is not None else config.RAG_KEPT_WARN_PCT
    ids = [int(i) for i in args.ids.split(",")] if args.ids else None
    stack = _Stack()
    with Session(create_engine(config.DATABASE_URL)) as db:
        rows = _rows(db, args.workspace, ids)
        if args.move_uploads:
            UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
            moves = [(r[0], r[4], moved_path(r[4], UPLOAD_DIR)) for r in rows
                     if r[4] and not r[4].startswith("s3://") and os.path.exists(r[4])]
            moves = [m for m in moves if m[2] is not None]
            print(f"uploads to move into {UPLOAD_DIR}: {len(moves)}")
            for doc_id, old, new in moves:
                print(f"  #{doc_id} {old} -> {new}")
                if args.apply:
                    shutil.copy2(old, new)
                    db.execute(text("UPDATE documents SET file_path = :p WHERE id = :id"), {"p": str(new), "id": doc_id})
            if args.apply:
                db.commit()
                rows = _rows(db, args.workspace, ids)

        found = plan(rows, below=below, measure=stack.measure)
        todo = [c for c in found if c.source != SOURCE_NONE]
        gone = [c for c in found if c.source == SOURCE_NONE]
        print(f"documents: {len(rows)} · under {below}%: {len(todo)} to re-ingest · no source: {len(gone)}")
        for c in todo:
            print(f"  #{c.id} [{c.workspace_id[:8]}] {c.filename} ({c.file_type}) kept {c.kept}% — {c.source}")
        for c in gone:
            print(no_source_line(c))
        if not args.apply:
            print("dry run — nothing changed")
            return 0
        missing = _apply(todo, stack, db)
    skipped = gone + missing
    if skipped:
        print(f"no source: {len(skipped)} not re-ingested ({', '.join(f'#{c.id}' for c in skipped)})")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
