"""F067 — foreign keys are read from pg_catalog, visible to any role.

information_schema.constraint_column_usage shows a constraint only to the
OWNER of the table, so introspecting as a least-privilege, read-only role —
the secure way to connect — found 0 relationships and every join path was
stripped. Proven on real Postgres (pgvector/pgvector:pg16, 2026-09-22), with
two FKs and one composite FK, run as the owner and as a SELECT-only role:

    old query  owner: 6 rows (2 WRONG — the composite key came out as the
                      cartesian product farm->harvest, harvest->farm)
               reader: 0 rows                                  <- F067
    new query  owner: 4 rows, correct
               reader: the same 4 rows

CI cannot be relied on to create roles, so these tests pin the query's shape
and the mapping of its rows; the live proof is recorded above.
"""
from __future__ import annotations

from modules.nl2sql.schema import introspection
from modules.nl2sql.schema.introspection import FOREIGN_KEYS_FROM_PG_CATALOG, DatabaseIntrospectionService

Q = " ".join(FOREIGN_KEYS_FROM_PG_CATALOG.split()).lower()


def test_foreign_keys_come_from_pg_catalog_not_the_owner_only_view():
    assert "pg_catalog.pg_constraint" in Q
    assert "constraint_column_usage" not in Q, "owner-only: a read-only role sees nothing"
    assert "information_schema" not in Q.split("where")[0], "no information_schema views in the FROM"


def test_a_composite_key_is_paired_by_position_not_crossed():
    assert "unnest(con.conkey, con.confkey)" in Q and "with ordinality" in Q
    assert "k.ord" in Q, "pairs are emitted in key order"


def test_system_schemas_are_excluded_like_the_table_listing():
    assert "not in ('pg_catalog', 'information_schema')" in Q


class _Rows:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return list(self._rows)

    def scalar(self):
        return 0


class _Conn:
    """Answers the three shapes introspect() asks for: tables, columns, FKs."""

    def __init__(self):
        self.fk_sql = None

    def execute(self, clause, params=None):
        sql = str(clause)
        if "pg_constraint" in sql:
            self.fk_sql = sql
            return _Rows([
                ("public", "coffees", "importer_id", "public", "importers", "id"),
                ("public", "cuppings", "farm", "public", "lots", "farm"),
                ("public", "cuppings", "harvest", "public", "lots", "harvest"),
            ])
        if "information_schema.tables" in sql:
            return _Rows([])
        return _Rows([])

    def rollback(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_a):
        return False


def test_introspect_maps_each_column_pair_to_one_relationship(monkeypatch):
    conn = _Conn()
    monkeypatch.setattr(DatabaseIntrospectionService, "_create_engine",
                        lambda self: type("E", (), {"connect": lambda _s: conn})())
    service = DatabaseIntrospectionService(credential={}, dialect="postgres")
    metadata = service.introspect(include_samples=False)
    assert conn.fk_sql is not None, "introspect() did not run the pg_catalog query"
    pairs = [(r["from_table"], r["from_column"], r["to_table"], r["to_column"]) for r in metadata["relationships"]]
    assert pairs == [
        ("coffees", "importer_id", "importers", "id"),
        ("cuppings", "farm", "lots", "farm"),
        ("cuppings", "harvest", "lots", "harvest"),
    ]
