"""PRD-273 US-273-02 — write the file: the zip, its categories, its files, its manifest, its passphrase.

On CI's Postgres, inside the test's rolled-back transaction, a workspace with something in
every category (``tests/helpers_prd273_workspace.py``) is exported, and the zip is read back:

* only MOVED tables of the chosen categories have a ``sql/`` part; no NOT_MOVED table, no
  secret column and no planted secret value appears in any part (the BYOK keys and
  integrations in the settings, an API key, an SDK key's hash, a paired machine's hash, a
  Playbook's webhook id, a secret-looking column added to a moved table);
* ``workspace.json`` carries the brand kit and Orchestrator block (no model), ``auto.json`` the
  soul from Auto's agent row, not the settings' stale copy; the bundles validate;
* each file is in ``files/`` once by its SHA-256 and the index names every row that has it; a
  file outside the workspace's storage is listed as skipped;
* the manifest's record of every part (rows, bytes, SHA-256) matches the part;
* with a passphrase every part but the manifest is sealed, opens with the passphrase and not
  without it, and an entry changed after sealing is refused;
* a workspace with 100,000 rows exports with flat memory use;
* a table no rule classifies is skipped and listed; the size cap stops an export clearly.
"""
from __future__ import annotations

import hashlib
import io
import json
import tempfile
import tracemalloc
import uuid
import zipfile
from datetime import datetime

import pytest
from sqlalchemy import text

from config import config
from services.marketplace_bundles.validate import validate_bundle
from services.workspace_move import memory as move_memory
from services.workspace_move.crypto import WrongPassphrase, key_from_manifest, open_entry
from services.workspace_move.export import write_workspace_file
from services.workspace_move.rules import looks_secret
from services.workspace_move.table_map import TABLE_RULES
from services.workspace_move.writer import ExportTooLarge, MoveFileWriter
from tests.helpers_prd273_workspace import (
    AUTO_SOUL, DOCUMENT_BYTES, OTHER_BYTES, FakeObjects, seed_workspace_contents,
)

PASSPHRASE = "correct horse battery staple"
ALL_CATEGORIES = ("auto", "agents", "knowledge", "memory", "chats", "deliverables", "board")


@pytest.fixture
def workspace(db_session, seed_workspace, monkeypatch, tmp_path):
    """A seeded workspace whose documents are in a fake object store and whose generated
    Deliverable is on a temporary disk; Qdrant is off (CI's edition)."""
    monkeypatch.setattr(config, "DOCUMENT_STORAGE_DIR", str(tmp_path))
    monkeypatch.setattr(config, "QDRANT_URL", "")
    seeded = seed_workspace_contents(db_session, seed_workspace(name="Harbour Coffee"), tmp_path)
    store = FakeObjects(seeded.objects)
    monkeypatch.setattr("core.storage.get_s3_client", lambda *args, **kwargs: store)
    db_session.execute(text("ALTER TABLE teams ADD COLUMN prd273_api_key text"))
    team_secret = uuid.uuid4().hex * 2
    db_session.execute(text("UPDATE teams SET prd273_api_key = :s WHERE workspace_id = :ws"),
                       {"s": team_secret, "ws": seeded.workspace_id})
    seeded.secrets.append(team_secret)
    return seeded


def _export(db, seeded, **kwargs):
    out = io.BytesIO()
    result = write_workspace_file(db, seeded.workspace_id, out, exported_by="user:1", **kwargs)
    out.seek(0)
    return zipfile.ZipFile(out), result


def _manifest(zf):
    return json.loads(zf.read("manifest.json"))


def _lines(zf, name):
    return [json.loads(line) for line in zf.read(name).decode("utf-8").splitlines() if line]


def test_only_moved_tables_of_the_chosen_categories_have_parts(db_session, workspace):
    zf, _ = _export(db_session, workspace)
    sql = {name[len("sql/"):-len(".jsonl")] for name in zf.namelist() if name.startswith("sql/")}
    assert {"agents", "skills", "agent_skills", "chats", "messages", "documents", "teams", "workflow_recipes"} <= sql
    for table in sql:
        assert TABLE_RULES[table].moved and TABLE_RULES[table].category != "board", table
    assert not {"user_api_keys", "sdk_api_keys", "cli_hosts", "credentials", "audit_logs"} & sql
    assert "board_tasks" not in sql  # the Board is off by default (§8-Q3)
    zf_all, _ = _export(db_session, workspace, categories=ALL_CATEGORIES)
    assert "sql/board_tasks.jsonl" in zf_all.namelist()


def test_no_secret_column_or_planted_secret_in_any_part(db_session, workspace):
    zf, _ = _export(db_session, workspace, categories=ALL_CATEGORIES)
    for name in zf.namelist():
        data = zf.read(name)
        for secret in workspace.secrets:
            assert secret.encode() not in data, f"a planted secret is in {name}"
        if name.startswith("sql/"):
            table = name[len("sql/"):-len(".jsonl")]
            for row in _lines(zf, name):
                assert not [key for key in row if key in TABLE_RULES[table].secret_columns or looks_secret(key)]
    teams = _manifest(zf)["tables"]["teams"]
    assert teams["dropped"] == {"prd273_api_key": "secret"}
    (playbook,) = _lines(zf, "sql/workflow_recipes.jsonl")
    assert playbook["schedule_config"] == {"type": "webhook"}


def test_workspace_and_auto_carry_the_settings_and_the_agent_rows_soul(db_session, workspace):
    zf, _ = _export(db_session, workspace)
    settings = json.loads(zf.read("workspace.json"))
    assert settings["name"] == "Harbour Coffee"
    assert settings["settings"]["brand_kit"]["primary_color"] == "#1F6FEB"
    assert set(settings["settings"]) == {"brand_kit", "orchestrator", "voice_live", "push"}
    assert "llm" not in settings["settings"]["orchestrator"] and "model" not in settings["settings"]["orchestrator"]
    auto = json.loads(zf.read("auto.json"))["agent"]
    assert auto["custom_persona_prompt"] == AUTO_SOUL and auto["use_custom_persona"] is True
    assert auto["source_agent_id"] == workspace.auto_id and auto["configuration"]["proactive_level"] == "notify"
    assert {row["id"] for row in _lines(zf, "sql/agents.jsonl")} == {workspace.auto_id, workspace.writer_id}


def test_the_bundles_validate_and_name_what_the_agents_reference(db_session, workspace):
    zf, result = _export(db_session, workspace)
    agents = {line["source_id"]: line for line in _lines(zf, "bundles/agents.jsonl")}
    assert set(agents) == {workspace.auto_id, workspace.writer_id}
    for line in agents.values():
        validate_bundle(line["bundle"])
    writer = agents[workspace.writer_id]["bundle"]["agent"]
    assert writer["skills"] == [{"name": workspace.global_skill, "package": None}]
    assert writer["model"]["modelId"] == "anthropic/claude-haiku-5-5"
    (playbook,) = _lines(zf, "bundles/playbooks.jsonl")
    validate_bundle(playbook["bundle"])
    assert playbook["bundle"]["playbook"]["agents"] == {"a1": {"item": agents[workspace.writer_id]["bundle"]["externalId"]}}
    skills = {row["name"] for row in _lines(zf, "refs/skills.jsonl")}
    assert workspace.global_skill in skills
    assert result.summary["unbundled"] == 0


def test_files_are_carried_once_by_hash_and_the_index_names_every_row(db_session, workspace):
    zf, result = _export(db_session, workspace)
    plan_sha, invoice_sha = hashlib.sha256(DOCUMENT_BYTES).hexdigest(), hashlib.sha256(OTHER_BYTES).hexdigest()
    assert zf.read(f"files/{plan_sha}") == DOCUMENT_BYTES and zf.read(f"files/{invoice_sha}") == OTHER_BYTES
    assert len([name for name in zf.namelist() if name.startswith("files/") and name != "files/index.jsonl"]) == 2
    index = _lines(zf, "files/index.jsonl")
    assert sorted((line["table"], line["sha256"]) for line in index) == sorted(
        [("documents", plan_sha), ("documents", plan_sha), ("deliverables", invoice_sha)])
    skipped = _manifest(zf)["skipped_files"]
    assert [entry["reason"] for entry in skipped] == ["not in this workspace's storage"]
    assert result.summary["files"] == 3 and result.summary["skipped_files"] == 1


def test_the_manifest_matches_every_part(db_session, workspace):
    zf, _ = _export(db_session, workspace, categories=ALL_CATEGORIES)
    manifest = _manifest(zf)
    assert manifest["format"] == "automatos.workspace/v1"
    assert manifest["source"]["workspace_id"] == workspace.workspace_id and manifest["exported_by"] == "user:1"
    assert manifest["categories"] == list(ALL_CATEGORIES) and manifest["encryption"] is None
    assert set(manifest["parts"]) == set(zf.namelist()) - {"manifest.json"}
    for name, record in manifest["parts"].items():
        data = zf.read(name)
        assert record["bytes"] == len(data) and record["sha256"] == hashlib.sha256(data).hexdigest(), name
        if name.endswith(".jsonl"):
            assert record["rows"] == len(data.splitlines()), name
    assert manifest["memory"] == {"durable": False, "field": False}
    assert manifest["tables"]["agents"]["category"] == "agents"
    assert "semantic_embedding" not in manifest["tables"]["agents"]["columns"]


def test_a_passphrase_seals_every_part_but_the_manifest(db_session, workspace):
    zf, _ = _export(db_session, workspace, passphrase=PASSPHRASE)
    manifest = _manifest(zf)
    assert manifest["encryption"]["cipher"] == "AES-256-GCM" and manifest["encryption"]["kdf"] == "scrypt"
    with pytest.raises(WrongPassphrase):
        key_from_manifest("not the passphrase at all", manifest["encryption"])
    key = key_from_manifest(PASSPHRASE, manifest["encryption"])
    for name, record in manifest["parts"].items():
        sealed = zf.read(name)
        assert zf.getinfo(name).compress_type == zipfile.ZIP_STORED
        plain = open_entry(sealed, key, name)
        assert hashlib.sha256(plain).hexdigest() == record["sha256"], name
        if len(plain) > 16:
            assert plain not in sealed, f"{name} is stored in the clear"
    auto = json.loads(open_entry(zf.read("auto.json"), key, "auto.json"))
    assert auto["agent"]["custom_persona_prompt"] == AUTO_SOUL
    tampered = bytearray(zf.read("auto.json"))
    tampered[20] ^= 1
    with pytest.raises(WrongPassphrase):
        open_entry(bytes(tampered), key, "auto.json")
    with pytest.raises(WrongPassphrase):
        open_entry(zf.read("auto.json"), key, "workspace.json")  # an entry moved to another name


def test_memory_parts_carry_both_stores_when_qdrant_is_on(db_session, workspace, monkeypatch):
    async def durable(workspace_id):
        yield [move_memory.durable_line("p1", {"content": "Prefers mornings", "namespace": f"ws:{workspace_id}",
                                               "subject_id": "user:1", "metadata": {"kind": "preference"}})]

    async def field(workspace_id):
        yield [move_memory.field_line({"key": "tone", "value": "direct", "strength": 0.8})]

    monkeypatch.setattr(move_memory, "memory_available", lambda: True)
    monkeypatch.setattr(move_memory, "durable_pages", durable)
    monkeypatch.setattr(move_memory, "field_pages", field)
    zf, _ = _export(db_session, workspace)
    (memory,) = _lines(zf, "memory/durable.jsonl")
    assert (memory["text"], memory["subject_id"], memory["metadata"]) == ("Prefers mornings", "user:1",
                                                                           {"kind": "preference"})
    assert _lines(zf, "memory/field.jsonl")[0]["value"] == "direct"
    assert _manifest(zf)["memory"] == {"durable": True, "field": True}


def test_a_table_no_rule_classifies_is_skipped_and_listed(db_session, workspace):
    probe = f"prd273_probe_{uuid.uuid4().hex[:10]}"
    db_session.execute(text(f"CREATE TABLE {probe} (id serial PRIMARY KEY, workspace_id uuid, note text)"))
    db_session.execute(text(f"INSERT INTO {probe} (workspace_id, note) VALUES (:ws, 'unknown data')"),
                       {"ws": workspace.workspace_id})
    zf, result = _export(db_session, workspace)
    assert f"sql/{probe}.jsonl" not in zf.namelist()
    assert {"table": probe, "reason": "no rule classifies it"} in _manifest(zf)["skipped_tables"]
    assert result.summary["skipped_tables"] >= 1


def test_the_size_cap_stops_an_export_clearly():
    writer = MoveFileWriter(io.BytesIO(), key=None, stamp=datetime(2026, 11, 6), max_bytes=10)
    with pytest.raises(ExportTooLarge):
        writer.write_json("workspace.json", {"name": "more than ten bytes"})


ROWS_SMALL, ROWS_LARGE = 10_000, 100_000
# Holding 100,000 rows in memory takes well over 100 MiB; streaming them stays near one batch.
FLAT_PEAK_BYTES = 48 * 1024 * 1024


def _fill_memory(db, workspace_id: str, rows: int) -> None:
    db.execute(text(
        "INSERT INTO memory_short_term (id, workspace_id, content, content_type, importance, decay_score, "
        "access_count, metadata, promoted_to_l3) SELECT gen_random_uuid(), :ws, 'remembered line ' || g, "
        "'exchange', 0.5, 1.0, 0, '{}'::jsonb, false FROM generate_series(1, :rows) g"),
        {"ws": workspace_id, "rows": rows})


def _peak(db, workspace_id: str):
    """(the peak of Python memory while exporting, the manifest). The file goes to disk, as the
    job writes it, so the zip's own bytes are not counted."""
    with tempfile.TemporaryFile() as out:
        tracemalloc.start()
        try:
            result = write_workspace_file(db, workspace_id, out, categories=("memory",))
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
    return peak, result.manifest


def test_a_workspace_with_100k_rows_exports_with_flat_memory(db_session, seed_workspace, monkeypatch):
    monkeypatch.setattr(config, "QDRANT_URL", "")
    small, large = seed_workspace(), seed_workspace()
    _fill_memory(db_session, small, ROWS_SMALL)
    _fill_memory(db_session, large, ROWS_LARGE)
    with tempfile.TemporaryFile() as warm:  # the first export compiles and caches the statements
        write_workspace_file(db_session, small, warm, categories=("memory",))
    (small_peak, _), (large_peak, manifest) = _peak(db_session, small), _peak(db_session, large)
    assert manifest["parts"]["sql/memory_short_term.jsonl"]["rows"] == ROWS_LARGE
    assert large_peak < FLAT_PEAK_BYTES, f"peak {large_peak} bytes for {ROWS_LARGE} rows"
    assert large_peak < 2.5 * small_peak, f"peak grew from {small_peak} to {large_peak} bytes with 10x the rows"
