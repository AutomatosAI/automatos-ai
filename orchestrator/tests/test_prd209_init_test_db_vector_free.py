"""PRD-209 — init_test_db builds the vector-free shape on stock postgres.

CI's orchestrator-tests job runs on postgres:15 without pgvector; the
knowledge_items extra used to declare `embedding vector(4096)` unconditionally
and killed the whole lane ('type "vector" does not exist').
"""
from scripts.init_test_db import _with_embedding

_DDL = "CREATE TABLE t (\n    id SERIAL,\n    __EMBEDDING__\n    metadata JSONB\n)"


def test_full_shape_with_pgvector():
    out = _with_embedding(_DDL, True)
    assert "embedding vector(4096)," in out and "__EMBEDDING__" not in out


def test_vector_free_shape_without_pgvector():
    out = _with_embedding(_DDL, False)
    assert "vector" not in out and "__EMBEDDING__" not in out
    assert "metadata JSONB" in out
