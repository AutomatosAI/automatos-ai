"""PRD-233 S4 — one storage-client factory (absorbs PRD-151).

Pure tests, boto3 mocked at the ``boto3.client`` boundary; no AWS, no MinIO.

  Guard   ``boto3.client(`` appears nowhere in ``orchestrator/`` outside
          ``core/storage/s3.py`` and the allow-list (Bedrock runtime, the
          SaaS-only S3 Vectors backend and its two scripts) — PRD-151 G4.
  Factory SaaS shape (``S3_ENDPOINT_URL`` unset) reproduces the legacy
          per-site kwargs: region from ``AWS_REGION``, explicit creds only
          when both halves exist, SigV4, adaptive retries, NO ``endpoint_url``,
          NO path-style — PRD-151 G5. Local shape adds the endpoint and
          path-style addressing. ``FAST_FAIL`` keeps DocumentManager's PRD-164
          timeouts. Memoized per profile; ``reset_s3_client`` drops the memo.
  G3      nothing configured ⇒ one ``StorageNotConfigured`` with the
          actionable message, and boto3 is never touched.
  Buckets ``ensure_bucket`` creates only against a configured endpoint
          (MinIO); on AWS it is a no-op that never even builds a client.
  Sites   the three bespoke local fallbacks are gone; construction is lazy;
          presigns that leave the backend go through the public client.
"""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

import boto3
import pytest
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError

import core.storage.s3 as s3mod
from core.storage import (
    FAST_FAIL,
    STANDARD,
    StorageNotConfigured,
    ensure_bucket,
    get_public_s3_client,
    get_s3_client,
    is_storage_configured,
    reset_s3_client,
)

ORCHESTRATOR_ROOT = Path(__file__).resolve().parent.parent

# The only files allowed to construct a boto3 client directly.
BOTO_CLIENT_ALLOWLIST = frozenset({
    "core/storage/s3.py",
    # Bedrock runtime client — an LLM provider, not object storage.
    "core/llm/clients/bedrock_client.py",
    # AWS S3 Vectors — SaaS-only opt-in (S3_VECTORS_ENABLED); MinIO cannot serve it.
    "modules/search/vector_store/backends/s3_vectors_backend.py",
    "scripts/migrate_to_s3_vectors.py",
    "scripts/recreate_s3_index.py",
})
SKIP_DIRS = frozenset({"tests", "__pycache__", "node_modules", ".venv", "venv"})
BOTO_CLIENT_CALL = re.compile(r"\bboto3\.client\(")

MINIO = "http://minio:9000"
PUBLIC = "http://localhost:9000"


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def _fresh_factory():
    reset_s3_client()
    yield
    reset_s3_client()


@pytest.fixture
def boto_spy(monkeypatch):
    """Record every boto3.client() call and hand back a MagicMock client."""
    calls = []

    def _client(service, **kwargs):
        client = MagicMock(name=f"{service}-client-{len(calls)}")
        calls.append(SimpleNamespace(service=service, kwargs=kwargs, client=client))
        return client

    monkeypatch.setattr(boto3, "client", _client)
    return calls


def _configure(monkeypatch, *, endpoint="", public="", path_style=None,
               region="us-east-1", key="AKIATEST", secret="test-secret"):
    """Set the storage knobs the factory reads (defaults = the SaaS shape)."""
    cfg = s3mod.config
    monkeypatch.setattr(cfg, "S3_ENDPOINT_URL", endpoint, raising=False)
    monkeypatch.setattr(cfg, "S3_PUBLIC_ENDPOINT_URL", public, raising=False)
    monkeypatch.setattr(
        cfg, "S3_USE_PATH_STYLE", bool(endpoint) if path_style is None else path_style, raising=False
    )
    monkeypatch.setattr(cfg, "AWS_REGION", region, raising=False)
    monkeypatch.setattr(cfg, "AWS_ACCESS_KEY_ID", key, raising=False)
    monkeypatch.setattr(cfg, "AWS_SECRET_ACCESS_KEY", secret, raising=False)


def _client_error(code: str, op: str = "HeadBucket") -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": code}}, op)


# --------------------------------------------------------------------------- #
# source guard (PRD-151 G4)
# --------------------------------------------------------------------------- #


def _python_sources():
    for path in ORCHESTRATOR_ROOT.rglob("*.py"):
        rel = path.relative_to(ORCHESTRATOR_ROOT)
        if SKIP_DIRS.intersection(rel.parts):
            continue
        yield rel.as_posix(), path


def test_no_boto3_client_outside_the_factory():
    offenders = sorted(
        rel for rel, path in _python_sources()
        if rel not in BOTO_CLIENT_ALLOWLIST
        and BOTO_CLIENT_CALL.search(path.read_text(encoding="utf-8", errors="ignore"))
    )
    assert offenders == [], (
        "boto3.client( must only be called from core/storage/s3.py "
        f"(+ allow-list); found in: {offenders}"
    )


def test_allowlist_entries_still_exist():
    """A stale allow-list silently widens the guard; keep it honest."""
    missing = [rel for rel in BOTO_CLIENT_ALLOWLIST if not (ORCHESTRATOR_ROOT / rel).exists()]
    assert missing == []


def test_bespoke_local_fallbacks_are_deleted():
    config_src = (ORCHESTRATOR_ROOT / "config.py").read_text(encoding="utf-8")
    assert 'os.getenv("MARKETPLACE_LOCAL_DIR")' not in config_src
    assert 'os.getenv("IMAGE_STORE_LOCAL_DIR")' not in config_src

    marketplace = (ORCHESTRATOR_ROOT / "core/services/marketplace_s3.py").read_text(encoding="utf-8")
    images = (ORCHESTRATOR_ROOT / "core/services/image_store.py").read_text(encoding="utf-8")
    attachments = (ORCHESTRATOR_ROOT / "modules/attachments/store.py").read_text(encoding="utf-8")
    assert "class LocalStorageService" not in marketplace
    assert "class LocalImageStore" not in images
    assert "_local_dir" not in attachments and "file://" not in attachments


def test_browser_facing_presigns_use_the_public_client():
    """Links that leave the backend must be minted against S3_PUBLIC_ENDPOINT_URL."""
    for rel in ("api/documents.py", "api/document_generation.py", "modules/attachments/store.py"):
        src = (ORCHESTRATOR_ROOT / rel).read_text(encoding="utf-8")
        assert "get_public_s3_client().generate_presigned_url(" in src, rel
        assert "get_s3_client().generate_presigned_url(" not in src, rel


# --------------------------------------------------------------------------- #
# factory — SaaS shape reproduces the legacy kwargs table (PRD-151 G5)
# --------------------------------------------------------------------------- #


def test_saas_kwargs_match_the_legacy_constructors(monkeypatch, boto_spy):
    _configure(monkeypatch, region="eu-west-2")

    client = get_s3_client()

    assert len(boto_spy) == 1 and client is boto_spy[0].client
    call = boto_spy[0]
    assert call.service == "s3"
    # Exactly the legacy kwarg set: no endpoint_url, no session token, nothing else.
    assert set(call.kwargs) == {"region_name", "config", "aws_access_key_id", "aws_secret_access_key"}
    assert call.kwargs["region_name"] == "eu-west-2"
    assert call.kwargs["aws_access_key_id"] == "AKIATEST"
    assert call.kwargs["aws_secret_access_key"] == "test-secret"

    cfg = call.kwargs["config"]
    assert isinstance(cfg, BotoConfig)
    assert cfg.signature_version == "v4"
    assert cfg.retries == {"max_attempts": 3, "mode": "adaptive"}
    assert cfg.s3 is None, "AWS keeps boto's virtual-host addressing"
    assert cfg.region_name is None, "region rides the client kwarg, not the Config"
    assert cfg.connect_timeout == BotoConfig().connect_timeout
    assert cfg.read_timeout == BotoConfig().read_timeout


def test_saas_blank_region_falls_back_to_us_east_1(monkeypatch, boto_spy):
    _configure(monkeypatch, region="")
    get_s3_client()
    assert boto_spy[0].kwargs["region_name"] == "us-east-1"


def test_fast_fail_profile_keeps_document_manager_timeouts(monkeypatch, boto_spy):
    _configure(monkeypatch)

    get_s3_client(FAST_FAIL)

    cfg = boto_spy[0].kwargs["config"]
    assert cfg.connect_timeout == 3
    assert cfg.read_timeout == 5
    assert cfg.retries == {"max_attempts": 1}
    assert cfg.signature_version == "v4"


def test_local_endpoint_adds_endpoint_and_path_style(monkeypatch, boto_spy):
    _configure(monkeypatch, endpoint=MINIO, public=PUBLIC)

    get_s3_client()

    call = boto_spy[0]
    assert call.kwargs["endpoint_url"] == MINIO
    assert call.kwargs["config"].s3 == {"addressing_style": "path"}
    assert call.kwargs["aws_access_key_id"] == "AKIATEST"


def test_endpoint_without_explicit_creds_leaves_the_default_chain(monkeypatch, boto_spy):
    _configure(monkeypatch, endpoint=MINIO, key=None, secret=None)

    get_s3_client()

    assert "aws_access_key_id" not in boto_spy[0].kwargs
    assert "aws_secret_access_key" not in boto_spy[0].kwargs


def test_half_a_key_pair_is_not_a_credential(monkeypatch, boto_spy):
    _configure(monkeypatch, endpoint=MINIO, secret=None)
    get_s3_client()
    assert "aws_access_key_id" not in boto_spy[0].kwargs


def test_path_style_can_be_forced_off_with_an_endpoint(monkeypatch, boto_spy):
    _configure(monkeypatch, endpoint=MINIO, path_style=False)
    get_s3_client()
    assert boto_spy[0].kwargs["config"].s3 is None


# --------------------------------------------------------------------------- #
# memoization / public client / G3
# --------------------------------------------------------------------------- #


def test_memoized_per_profile_and_reset(monkeypatch, boto_spy):
    _configure(monkeypatch)

    a = get_s3_client()
    b = get_s3_client(STANDARD)
    fast = get_s3_client(FAST_FAIL)

    assert a is b
    assert fast is not a
    assert len(boto_spy) == 2

    reset_s3_client()
    assert get_s3_client() is not a
    assert len(boto_spy) == 3


def test_public_client_is_the_same_object_in_saas(monkeypatch, boto_spy):
    _configure(monkeypatch)
    assert get_public_s3_client() is get_s3_client()
    assert len(boto_spy) == 1


def test_public_client_uses_the_public_endpoint_locally(monkeypatch, boto_spy):
    _configure(monkeypatch, endpoint=MINIO, public=PUBLIC)

    internal = get_s3_client()
    public = get_public_s3_client()

    assert public is not internal
    endpoints = {c.kwargs["endpoint_url"] for c in boto_spy}
    assert endpoints == {MINIO, PUBLIC}
    assert get_public_s3_client() is public


def test_nothing_configured_raises_one_actionable_error(monkeypatch, boto_spy):
    _configure(monkeypatch, key=None, secret=None)

    assert is_storage_configured() is False
    with pytest.raises(StorageNotConfigured) as excinfo:
        get_s3_client()
    with pytest.raises(StorageNotConfigured):
        get_public_s3_client()

    message = str(excinfo.value)
    assert "S3_ENDPOINT_URL" in message and "AWS_ACCESS_KEY_ID" in message
    assert isinstance(excinfo.value, RuntimeError)
    assert boto_spy == [], "an unconfigured store must never touch boto3"


def test_is_storage_configured_shapes(monkeypatch):
    _configure(monkeypatch)
    assert is_storage_configured() is True
    _configure(monkeypatch, endpoint=MINIO, key=None, secret=None)
    assert is_storage_configured() is True
    _configure(monkeypatch, key="only-half", secret=None)
    assert is_storage_configured() is False


def test_importing_the_factory_builds_nothing(boto_spy):
    import importlib

    importlib.import_module("core.storage")
    assert boto_spy == []


# --------------------------------------------------------------------------- #
# ensure_bucket
# --------------------------------------------------------------------------- #


def test_ensure_bucket_never_creates_on_aws(monkeypatch, boto_spy):
    _configure(monkeypatch)

    assert ensure_bucket("automatos-ai") is False
    assert boto_spy == [], "no endpoint ⇒ no bucket probe, not even a client"


def test_ensure_bucket_creates_a_missing_bucket_on_minio_once(monkeypatch, boto_spy):
    _configure(monkeypatch, endpoint=MINIO)
    created = ensure_bucket("automatos-marketplace")  # builds the client first
    client = boto_spy[0].client
    # First call saw a MagicMock head_bucket that "succeeded" → no create.
    assert created is False
    client.create_bucket.assert_not_called()

    reset_s3_client()
    client = get_s3_client()
    client.head_bucket.side_effect = _client_error("404")

    assert ensure_bucket("automatos-marketplace") is True
    client.create_bucket.assert_called_once_with(Bucket="automatos-marketplace")

    # Memoized per process: a second call makes no further requests.
    client.head_bucket.reset_mock()
    assert ensure_bucket("automatos-marketplace") is False
    client.head_bucket.assert_not_called()


def test_ensure_bucket_passes_a_location_constraint_outside_us_east_1(monkeypatch, boto_spy):
    _configure(monkeypatch, endpoint=MINIO, region="eu-west-1")
    client = get_s3_client()
    client.head_bucket.side_effect = _client_error("NoSuchBucket")

    assert ensure_bucket("automatos-ai") is True
    client.create_bucket.assert_called_once_with(
        Bucket="automatos-ai", CreateBucketConfiguration={"LocationConstraint": "eu-west-1"}
    )


def test_ensure_bucket_is_idempotent_under_a_create_race(monkeypatch, boto_spy):
    _configure(monkeypatch, endpoint=MINIO)
    client = get_s3_client()
    client.head_bucket.side_effect = _client_error("404")
    client.create_bucket.side_effect = _client_error("BucketAlreadyOwnedByYou", "CreateBucket")

    assert ensure_bucket("automatos-ai") is False


def test_ensure_bucket_reraises_real_errors(monkeypatch, boto_spy):
    _configure(monkeypatch, endpoint=MINIO)
    client = get_s3_client()
    client.head_bucket.side_effect = _client_error("403")

    with pytest.raises(ClientError):
        ensure_bucket("automatos-ai")
    client.create_bucket.assert_not_called()


def test_ensure_bucket_ignores_blank_names(monkeypatch, boto_spy):
    _configure(monkeypatch, endpoint=MINIO)
    assert ensure_bucket("") is False
    assert boto_spy == []


def test_ensure_bucket_probes_with_the_client_it_is_handed(monkeypatch, boto_spy):
    """A site writes with the client it probed with (fast-fail manager, injected test client)."""
    _configure(monkeypatch, endpoint=MINIO)
    handed = MagicMock(name="handed-client")
    handed.head_bucket.side_effect = _client_error("404")

    assert ensure_bucket("automatos-ai", handed) is True

    handed.create_bucket.assert_called_once_with(Bucket="automatos-ai")
    assert boto_spy == [], "the factory client must not be built when one is handed in"


# --------------------------------------------------------------------------- #
# migrated sites — lazy construction, no fallbacks, factory-only clients
# --------------------------------------------------------------------------- #


def test_marketplace_service_is_s3_only_and_lazy(monkeypatch, boto_spy):
    from core.services.marketplace_s3 import MarketplaceS3Service

    _configure(monkeypatch, key=None, secret=None)
    svc = MarketplaceS3Service()  # constructing never needs storage
    assert boto_spy == []
    with pytest.raises(StorageNotConfigured):
        svc.client

    _configure(monkeypatch)
    assert svc.client is get_s3_client()


def test_image_store_singleton_is_the_s3_store(monkeypatch, boto_spy):
    import core.services.image_store as image_store

    monkeypatch.setattr(image_store, "_image_store", None)
    _configure(monkeypatch, key=None, secret=None)

    store = image_store.get_image_store()

    assert isinstance(store, image_store.S3ImageStore)
    assert boto_spy == []


@pytest.mark.asyncio
async def test_attachment_store_has_no_local_fallback(monkeypatch, boto_spy):
    import modules.attachments.store as store_mod

    monkeypatch.setattr(
        store_mod, "validate_upload",
        lambda content, filename, declared_mime: {"safe_filename": filename, "mime": "text/plain"},
    )
    _configure(monkeypatch, key=None, secret=None)
    store = store_mod.AttachmentStore()
    assert boto_spy == []

    with pytest.raises(StorageNotConfigured):
        await store.put(workspace_id=uuid4(), uploaded_by="u", filename="a.txt", content=b"hi")


@pytest.mark.asyncio
async def test_attachment_put_bootstraps_the_bucket_on_minio(monkeypatch, boto_spy):
    import modules.attachments.store as store_mod

    monkeypatch.setattr(
        store_mod, "validate_upload",
        lambda content, filename, declared_mime: {"safe_filename": filename, "mime": "text/plain"},
    )
    _configure(monkeypatch, endpoint=MINIO)
    monkeypatch.setattr(s3mod.config, "S3_DOCUMENTS_BUCKET", "automatos-ai", raising=False)
    store = store_mod.AttachmentStore(bucket="automatos-ai")
    ws = uuid4()

    ref = await store.put(workspace_id=ws, uploaded_by="u", filename="a.txt", content=b"hi")

    client = boto_spy[0].client
    client.head_bucket.assert_called_once_with(Bucket="automatos-ai")
    client.put_object.assert_called_once()
    assert client.put_object.call_args.kwargs["Key"] == ref.s3_key
    assert ref.s3_key.startswith(f"workspaces/{ws}/ephemeral-attachments/")


def test_workspace_purge_skips_cleanly_when_unconfigured(monkeypatch, boto_spy):
    from services import workspace_purge

    _configure(monkeypatch, key=None, secret=None)
    assert workspace_purge._build_s3_client() is None
    assert boto_spy == []

    _configure(monkeypatch)
    assert workspace_purge._build_s3_client() is get_s3_client()


def test_document_manager_builds_its_client_on_first_use(monkeypatch, boto_spy):
    pytest.importorskip("docx")
    from modules.rag.ingestion import manager as mgr_mod

    monkeypatch.setattr("core.llm.create_embedding_manager", lambda *a, **kw: _StubEmbeddings())
    _configure(monkeypatch, key=None, secret=None)

    dm = mgr_mod.DocumentManager(db_config={}, workspace_id="ws-test", s3_bucket="automatos-ai")
    assert boto_spy == [], "construction must perform no storage I/O (PRD-151 FR-4)"
    with pytest.raises(StorageNotConfigured):
        dm.s3_client
    dm._ensure_s3_bucket_exists()  # unconfigured ⇒ logged, never raised
    assert boto_spy == []

    _configure(monkeypatch)
    assert dm.s3_client is get_s3_client(FAST_FAIL)
    assert boto_spy[0].kwargs["config"].connect_timeout == 3

    # The bucket probe rides the same fast-fail client the manager writes with.
    _configure(monkeypatch, endpoint=MINIO)
    reset_s3_client()
    dm._ensure_s3_bucket_exists()
    assert len(boto_spy) == 2 and boto_spy[1].kwargs["config"].connect_timeout == 3
    boto_spy[1].client.head_bucket.assert_called_once_with(Bucket="automatos-ai")


class _StubEmbeddings:
    def get_provider_info(self):
        return {"provider": "stub"}
