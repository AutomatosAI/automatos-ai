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


_REFERENCES = re.compile(r"\bREFERENCES\s+(\w+)\s*\(", re.I)


def test_no_forward_or_missing_fk_targets():
    """Every ``REFERENCES <table>`` target must be CREATE-d earlier in the file.

    initdb replays the SQL top-to-bottom in one pass, so a FK to a table defined
    later — or never (e.g. ``workspaces`` was missing entirely) — errors and aborts
    the schema before the alembic_version stamp at EOF. Comment lines are skipped so
    prose mentioning ``REFERENCES`` doesn't trip the check.
    """
    defined: set[str] = set()
    offenders = []
    current = None
    create_re = re.compile(r"CREATE TABLE(?:\s+IF NOT EXISTS)?\s+(\w+)", re.I)
    for line in _INIT_SQL.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith("--"):
            continue
        cm = create_re.search(line)
        if cm:
            current = cm.group(1)
            defined.add(current)  # self-references are legal
        for rm in _REFERENCES.finditer(line):
            target = rm.group(1)
            if target not in defined:
                offenders.append(f"{current} -> {target}")
    assert not offenders, (
        "init_complete_schema.sql has FK targets not defined earlier in the file "
        f"(forward/missing reference — initdb will abort here): {offenders}"
    )


def test_init_sql_has_the_valid_ann_index_anchor():
    # Non-vacuity: the one legitimate ANN index (kb_images.visual_embedding, a
    # vector(512) column) is still present — proving the check isn't passing simply
    # because all ANN indexes were stripped.
    dims = _column_dims()
    assert dims.get(("kb_images", "visual_embedding")) == 512
    text = _INIT_SQL.read_text(encoding="utf-8")
    assert re.search(r"idx_kb_images_visual_embedding.*USING ivfflat", text)
