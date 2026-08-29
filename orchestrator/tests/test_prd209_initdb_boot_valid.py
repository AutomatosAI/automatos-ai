"""PRD-209 S2 — init_complete_schema.sql applies cleanly on a fresh pgvector DB.

The fresh-clone boot died because init_complete_schema.sql declared pgvector ANN
indexes (``USING hnsw`` / ``USING ivfflat``) on ``vector(4096)`` columns. pgvector
caps hnsw/ivfflat indexes at 2000 dimensions, so ``CREATE INDEX`` errored and
Postgres initdb aborted the schema before the alembic_version stamp at EOF — leaving
the entrypoint's ``alembic upgrade heads`` to replay the forest and crash.

This guard reproduces the invariant purely (parse the SQL — no DB): NO ANN index may
target a vector column whose declared dimension exceeds pgvector's 2000-dim cap. It
would have caught the boot bug and blocks its re-introduction.
"""
from __future__ import annotations

import pathlib
import re

_ORCH = pathlib.Path(__file__).resolve().parents[1]
_INIT_SQL = _ORCH / "core" / "database" / "init_complete_schema.sql"

# pgvector's hard cap for hnsw / ivfflat indexes over the `vector` type.
PGVECTOR_ANN_DIM_CAP = 2000

_CREATE_TABLE = re.compile(r"CREATE TABLE(?:\s+IF NOT EXISTS)?\s+(\w+)\s*\(", re.I)
_VECTOR_COL = re.compile(r"^\s*(\w+)\s+vector\((\d+)\)", re.I)
_ANN_INDEX = re.compile(
    r"CREATE INDEX(?:\s+IF NOT EXISTS)?\s+\w+\s+ON\s+(\w+)\s+USING\s+(hnsw|ivfflat)\s*\(\s*(\w+)",
    re.I,
)


def _column_dims() -> dict[tuple[str, str], int]:
    """(table, column) -> declared vector dimension, by a line-based scan."""
    dims: dict[tuple[str, str], int] = {}
    table = None
    for line in _INIT_SQL.read_text(encoding="utf-8").splitlines():
        m = _CREATE_TABLE.search(line)
        if m:
            table = m.group(1)
            continue
        if line.strip() == ");":
            table = None
            continue
        vm = _VECTOR_COL.match(line)
        if vm and table:
            dims[(table, vm.group(1))] = int(vm.group(2))
    return dims


def test_no_ann_index_over_pgvector_dim_cap():
    text = _INIT_SQL.read_text(encoding="utf-8")
    dims = _column_dims()
    offenders = []
    for m in _ANN_INDEX.finditer(text):
        table, method, col = m.group(1), m.group(2), m.group(3)
        dim = dims.get((table, col))
        if dim is not None and dim > PGVECTOR_ANN_DIM_CAP:
            offenders.append(f"{method} index on {table}.{col} = vector({dim}) > {PGVECTOR_ANN_DIM_CAP}")
    assert not offenders, (
        "init_complete_schema.sql declares pgvector ANN indexes over the 2000-dim cap "
        f"— initdb will abort here and the fresh clone will not boot: {offenders}"
    )


def test_init_sql_has_the_valid_ann_index_anchor():
    # Non-vacuity: the one legitimate ANN index (kb_images.visual_embedding, a
    # vector(512) column) is still present — proving the check isn't passing simply
    # because all ANN indexes were stripped.
    dims = _column_dims()
    assert dims.get(("kb_images", "visual_embedding")) == 512
    text = _INIT_SQL.read_text(encoding="utf-8")
    assert re.search(r"idx_kb_images_visual_embedding.*USING ivfflat", text)
