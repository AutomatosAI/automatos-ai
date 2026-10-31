"""PRD-262 row #20: the orphan field sweep runs on core.qdrant's REST client.

scripts/nuke_orphan_field_collections.py built a qdrant-client AsyncQdrantClient, which
left the image with grpcio and portalocker; it now builds core.qdrant's QdrantRestClient,
and its main was split into its two passes (so each piece meets the code-shape limits)
with the behaviour it had:

- pass 1 lists the legacy ``field_<uuid>`` collections on a dry run and drops them
  otherwise, never the shared one; a collection that fails to drop is logged and the
  rest still go; a node that can't list its collections ends the run with exit 2;
- pass 2 deletes, through the field adapter, the fields with points in the shared
  collection that no orchestration run names, lists them on a dry run, and does nothing
  when the shared collection is absent.

A fake client, session and adapter; no Qdrant, no database.
"""
from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

import pytest

from core import qdrant as qdrant_package
from core.database import database
from modules.context.adapters import vector_field
from scripts import nuke_orphan_field_collections as sweep

SHARED = vector_field.SHARED_COLLECTION


class _Client:
    """The REST client's methods the sweep calls, over an in-memory node."""

    made: list = []

    def __init__(self, url, api_key=None):
        self.url, self.api_key, self.dropped = url, api_key, []
        self.collections = ["field_a", "field_b", SHARED, "durable_memory"]
        self.fields = ["f-live", "f-orphan", None]
        self.fail_listing, self.fail_drop = False, set()
        self.setup(self)
        _Client.made.append(self)

    @staticmethod
    def setup(client):
        """What a test changes on the node before the sweep runs."""

    async def get_collections(self):
        if self.fail_listing:
            raise ConnectionError("node down")
        return SimpleNamespace(collections=[SimpleNamespace(name=name) for name in self.collections])

    async def delete_collection(self, name):
        if name in self.fail_drop:
            raise RuntimeError(f"cannot drop {name}")
        self.dropped.append(name)

    async def collection_exists(self, name):
        return name in self.collections

    async def scroll(self, collection_name, limit, with_payload, with_vectors):
        assert (collection_name, with_payload, with_vectors) == (SHARED, ["field_id"], False)
        return [SimpleNamespace(payload={"field_id": field} if field else {}) for field in self.fields], None


class _Session:
    def execute(self, _statement):
        return SimpleNamespace(fetchall=lambda: [("f-live",), (None,)])

    def close(self):
        pass


class _Adapter:
    destroyed: list = []

    async def destroy_context(self, field_id):
        _Adapter.destroyed.append(field_id)


@pytest.fixture
def node(monkeypatch):
    _Client.made, _Adapter.destroyed = [], []
    monkeypatch.setattr(qdrant_package, "QdrantRestClient", _Client)
    monkeypatch.setattr(database, "SessionLocal", _Session)
    monkeypatch.setattr(vector_field, "VectorFieldSharedContext", _Adapter)
    return _Client


def _run(dry_run: bool) -> int:
    return asyncio.run(sweep.main(dry_run=dry_run))


def test_the_sweep_drops_the_legacy_collections_and_the_orphan_fields(node):
    assert _run(dry_run=False) == 0
    (client,) = node.made
    assert client.dropped == ["field_a", "field_b"], "never the shared collection"
    assert _Adapter.destroyed == ["f-orphan"]


def test_a_dry_run_lists_and_deletes_nothing(node, capsys):
    assert _run(dry_run=True) == 0
    assert node.made[0].dropped == [] and _Adapter.destroyed == []
    assert capsys.readouterr().out.splitlines() == ["field_a", "field_b", "orphan_field_id: f-orphan"]


def test_a_failed_drop_is_logged_and_the_rest_still_go(node, caplog, monkeypatch):
    caplog.set_level(logging.INFO, logger=sweep.logger.name)
    monkeypatch.setattr(_Client, "setup", staticmethod(lambda client: client.fail_drop.add("field_a")))
    assert _run(dry_run=False) == 0
    assert node.made[0].dropped == ["field_b"]
    assert "Failed to delete field_a" in caplog.text and "deleted=1 failures=1" in caplog.text


def test_a_node_that_cannot_list_its_collections_ends_the_run(node, monkeypatch):
    monkeypatch.setattr(_Client, "setup", staticmethod(lambda client: setattr(client, "fail_listing", True)))
    assert _run(dry_run=False) == sweep.EXIT_NO_LISTING
    assert _Adapter.destroyed == []


def test_no_shared_collection_means_nothing_to_sweep(node, monkeypatch):
    monkeypatch.setattr(_Client, "setup", staticmethod(lambda client: client.collections.remove(SHARED)))
    assert _run(dry_run=False) == 0
    assert _Adapter.destroyed == [] and node.made[0].dropped == ["field_a", "field_b"]
