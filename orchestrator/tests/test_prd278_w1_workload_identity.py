"""PRD-278 Wave 1 — storage and the search index from the pod's own role.

US-101  ``AWS_USE_WORKLOAD_IDENTITY`` (off by default). On with no key pair:
        object storage counts as configured and no client gets key arguments.
        On with a key pair: start-up refuses, naming both. Off: today's shapes.
US-102  On: start-up makes one HeadBucket on the documents bucket, in a worker
        thread, inside a time limit, and a refusal names the bucket and the kind
        of failure, never a credential. Off: no call.
US-104  ``S3_VECTORS_CREATE_IF_MISSING`` (on by default). Off: no create call,
        ever; a missing index is a start-up error naming the bucket and index;
        a role refused both creates starts against an index that exists; the
        size check still applies.

Pure tests: boto3 is replaced at ``boto3.client`` and every client is a fake. No AWS.
"""
from __future__ import annotations

import ast
import asyncio
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import boto3
import pytest
from botocore.exceptions import ClientError, NoCredentialsError

import core.storage.workload_identity as wi
import modules.search.vector_store.backends.s3_vectors_backend as s3b_mod
from config import config
from core.storage import FAST_FAIL, get_s3_client, is_storage_configured, reset_s3_client
from modules.search.vector_store.backends.s3_vectors_backend import (
    IndexDimensionMismatchError,
    S3VectorsBackend,
    VectorIndexCheckError,
    VectorIndexMissingError,
    verify_vector_index,
)

ORCHESTRATOR = Path(__file__).resolve().parent.parent
MINIO = "http://minio:9000"
DOCS_BUCKET = "example-documents"
VECTOR_BUCKET = "example-vectors"
INDEX = "documents-index"
# Stands in for whatever AWS writes in an error message: it must never reach ours.
AWS_MESSAGE = "aws-message-sentinel-not-for-the-operator"


@pytest.fixture(autouse=True)
def _fresh_factory():
    reset_s3_client()
    yield
    reset_s3_client()


@pytest.fixture
def boto_spy(monkeypatch):
    """Every boto3.client() call, each answered with a MagicMock client."""
    calls = []

    def _client(service, **kwargs):
        client = MagicMock(name=f"{service}-client-{len(calls)}")
        calls.append(SimpleNamespace(service=service, kwargs=kwargs, client=client, thread=threading.get_ident()))
        return client

    monkeypatch.setattr(boto3, "client", _client)
    return calls


def _storage(monkeypatch, *, identity, key=None, secret=None, endpoint=""):
    """The knobs the factory and the start-up check read. Raises if a setting is missing."""
    monkeypatch.setattr(config, "AWS_USE_WORKLOAD_IDENTITY", identity)
    monkeypatch.setattr(config, "AWS_ACCESS_KEY_ID", key)
    monkeypatch.setattr(config, "AWS_SECRET_ACCESS_KEY", secret)
    monkeypatch.setattr(config, "S3_ENDPOINT_URL", endpoint)
    monkeypatch.setattr(config, "S3_PUBLIC_ENDPOINT_URL", "")
    monkeypatch.setattr(config, "S3_USE_PATH_STYLE", bool(endpoint))
    monkeypatch.setattr(config, "AWS_REGION", "eu-west-2")
    monkeypatch.setattr(config, "S3_DOCUMENTS_BUCKET", DOCS_BUCKET)


def _error(code: str, op: str) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": AWS_MESSAGE}}, op)


def _raised_without_its_cause(error: BaseException) -> None:
    """The AWS exception is neither chained nor shown as context, so no traceback carries it."""
    assert error.__cause__ is None
    assert error.__suppress_context__ is True


# --------------------------------------------------------------------------- #
# the settings
# --------------------------------------------------------------------------- #


def test_both_settings_exist_with_their_defaults_and_are_in_the_config_surface():
    source = (ORCHESTRATOR / "config.py").read_text(encoding="utf-8")
    assert 'os.getenv("AWS_USE_WORKLOAD_IDENTITY", "false")' in source
    assert 'os.getenv("S3_VECTORS_CREATE_IF_MISSING", "true")' in source
    assert isinstance(config.AWS_USE_WORKLOAD_IDENTITY, bool)
    assert isinstance(config.S3_VECTORS_CREATE_IF_MISSING, bool)
    surface = json.loads((ORCHESTRATOR / "reports" / "config-surface.json").read_text())["settings"]
    assert {"AWS_USE_WORKLOAD_IDENTITY", "S3_VECTORS_CREATE_IF_MISSING"} <= set(surface)
    assert surface == sorted(surface)


# --------------------------------------------------------------------------- #
# US-101 — the setting and the gate
# --------------------------------------------------------------------------- #


def test_on_with_no_key_storage_is_configured_and_the_client_gets_no_key(monkeypatch, boto_spy):
    _storage(monkeypatch, identity=True)

    assert is_storage_configured() is True
    get_s3_client()

    [call] = boto_spy
    assert set(call.kwargs) == {"region_name", "config"}
    assert call.kwargs["region_name"] == "eu-west-2"


def test_on_the_client_never_gets_a_key_even_if_one_is_set(monkeypatch, boto_spy):
    _storage(monkeypatch, identity=True, key="AKIAEXAMPLEEXAMPLE00", secret="example-secret")
    get_s3_client()
    assert "aws_access_key_id" not in boto_spy[0].kwargs
    assert "aws_secret_access_key" not in boto_spy[0].kwargs


@pytest.mark.parametrize("key,secret", [("AKIAEXAMPLEEXAMPLE00", "example-secret"), ("AKIAEXAMPLEEXAMPLE00", None),
                                        (None, "example-secret")])
def test_on_with_a_key_set_start_up_fails_naming_both(monkeypatch, boto_spy, key, secret):
    _storage(monkeypatch, identity=True, key=key, secret=secret)

    with pytest.raises(wi.WorkloadIdentityError) as refused:
        asyncio.run(wi.verify_workload_identity())

    message = str(refused.value)
    assert "AWS_USE_WORKLOAD_IDENTITY" in message
    assert "AWS_ACCESS_KEY_ID" in message and "AWS_SECRET_ACCESS_KEY" in message
    assert "example-secret" not in message and "AKIAEXAMPLE" not in message
    assert boto_spy == [], "a conflict is refused before any call"


def test_off_every_existing_shape_behaves_as_today(monkeypatch):
    _storage(monkeypatch, identity=False, key="AKIATEST", secret="test-secret")
    assert is_storage_configured() is True
    _storage(monkeypatch, identity=False, endpoint=MINIO)
    assert is_storage_configured() is True
    _storage(monkeypatch, identity=False, key="only-half")
    assert is_storage_configured() is False
    _storage(monkeypatch, identity=False)
    assert is_storage_configured() is False, "the ambient chain alone still does not count"


def test_off_the_client_keeps_the_key_pair(monkeypatch, boto_spy):
    _storage(monkeypatch, identity=False, key="AKIATEST", secret="test-secret")
    get_s3_client()
    assert boto_spy[0].kwargs["aws_access_key_id"] == "AKIATEST"
    assert boto_spy[0].kwargs["aws_secret_access_key"] == "test-secret"


def _vectors(monkeypatch, *, create=True, enabled=True, bucket=VECTOR_BUCKET):
    monkeypatch.setattr(config, "S3_VECTORS_ENABLED", enabled)
    monkeypatch.setattr(config, "S3_VECTORS_CREATE_IF_MISSING", create)
    monkeypatch.setattr(config, "S3_VECTORS_BUCKET", bucket)
    monkeypatch.setattr(config, "S3_VECTORS_INDEX_NAME", INDEX)
    monkeypatch.setattr(config, "S3_VECTORS_DIMENSION", 2048)
    monkeypatch.setattr(config, "S3_VECTORS_METRIC", "cosine")


def test_on_the_search_index_client_gets_no_key(monkeypatch, boto_spy):
    _storage(monkeypatch, identity=True)
    _vectors(monkeypatch)
    S3VectorsBackend(workspace_id="ws-a")
    assert boto_spy[0].service == "s3vectors"
    assert set(boto_spy[0].kwargs) == {"region_name"}


def test_off_the_search_index_client_is_built_as_today(monkeypatch, boto_spy):
    _storage(monkeypatch, identity=False, key="AKIATEST", secret=None)
    _vectors(monkeypatch)
    S3VectorsBackend(workspace_id="ws-a")
    assert boto_spy[0].kwargs == {"region_name": "eu-west-2", "aws_access_key_id": "AKIATEST",
                                  "aws_secret_access_key": None}


# --------------------------------------------------------------------------- #
# US-102 — proving the role at start-up
# --------------------------------------------------------------------------- #


def test_on_start_up_makes_one_head_bucket_on_the_documents_bucket_off_the_loop(monkeypatch, boto_spy):
    _storage(monkeypatch, identity=True)

    asyncio.run(wi.verify_workload_identity())

    [call] = boto_spy
    call.client.head_bucket.assert_called_once_with(Bucket=DOCS_BUCKET)
    assert call.kwargs["config"].connect_timeout == FAST_FAIL.connect_timeout
    assert call.kwargs["config"].read_timeout == FAST_FAIL.read_timeout
    assert call.kwargs["config"].retries == {"max_attempts": 1}
    assert call.thread != threading.get_ident(), "the probe runs in a worker thread, not on the event loop"


@pytest.mark.parametrize("code,kind", [("403", "access denied"), ("AccessDenied", "access denied"),
                                       ("404", "does not exist"), ("InvalidIdentityToken", "InvalidIdentityToken")])
def test_a_refusal_fails_start_up_naming_the_bucket_and_the_kind(monkeypatch, boto_spy, code, kind):
    _storage(monkeypatch, identity=True)
    client = get_s3_client(FAST_FAIL)
    client.head_bucket.side_effect = _error(code, "HeadBucket")

    with pytest.raises(wi.WorkloadIdentityError) as refused:
        asyncio.run(wi.verify_workload_identity())

    message = str(refused.value)
    assert DOCS_BUCKET in message and kind in message
    assert AWS_MESSAGE not in message, "AWS's own words (which may carry ARNs) are not quoted"
    _raised_without_its_cause(refused.value)


def test_a_refusal_is_logged_once_at_error_level_with_code_and_class(monkeypatch, boto_spy, caplog):
    _storage(monkeypatch, identity=True)
    get_s3_client(FAST_FAIL).head_bucket.side_effect = _error("AccessDenied", "HeadBucket")

    with caplog.at_level("ERROR", logger=wi.__name__), pytest.raises(wi.WorkloadIdentityError):
        asyncio.run(wi.verify_workload_identity())

    errors = [r for r in caplog.records if r.name == wi.__name__ and r.levelname == "ERROR"]
    assert len(errors) == 1
    logged = errors[0].getMessage()
    assert "AccessDenied" in logged and "ClientError" in logged
    assert AWS_MESSAGE not in logged


def test_no_credentials_from_the_chain_is_its_own_kind(monkeypatch, boto_spy):
    _storage(monkeypatch, identity=True)
    get_s3_client(FAST_FAIL).head_bucket.side_effect = NoCredentialsError()

    with pytest.raises(wi.WorkloadIdentityError, match=f"{DOCS_BUCKET}.*no credentials"):
        asyncio.run(wi.verify_workload_identity())


def test_the_probe_has_a_short_time_limit(monkeypatch, boto_spy):
    assert 0 < wi.PROBE_TIMEOUT_SECONDS <= 30, "short: a start-up check, not a request"
    _storage(monkeypatch, identity=True)
    monkeypatch.setattr(wi, "PROBE_TIMEOUT_SECONDS", 0.05)
    get_s3_client(FAST_FAIL).head_bucket.side_effect = lambda **_kw: time.sleep(0.5)

    with pytest.raises(wi.WorkloadIdentityError, match=f"{DOCS_BUCKET}.*no answer within"):
        asyncio.run(wi.verify_workload_identity())


def test_off_no_call_is_made_and_a_key_pair_is_fine(monkeypatch, boto_spy):
    _storage(monkeypatch, identity=False, key="AKIATEST", secret="test-secret")
    asyncio.run(wi.verify_workload_identity())
    assert boto_spy == []


def _lifespan_awaits() -> list:
    tree = ast.parse((ORCHESTRATOR / "main.py").read_text(encoding="utf-8"))
    lifespan = next(node for node in tree.body
                    if isinstance(node, ast.AsyncFunctionDef) and node.name == "lifespan")
    calls = sorted((node for node in ast.walk(lifespan) if isinstance(node, ast.Call)),
                   key=lambda node: (node.lineno, node.col_offset))
    return [getattr(node.func, "attr", None) or getattr(node.func, "id", None) for node in calls]


def test_lifespan_runs_both_checks_in_the_hard_fail_phase_before_the_boot():
    """Outside any swallowing run_stage: after validate_security, before _boot_core."""
    calls = _lifespan_awaits()
    order = [calls.index(name) for name in
             ("validate_security", "verify_workload_identity", "verify_vector_index", "_boot_core")]
    assert order == sorted(order), calls


# --------------------------------------------------------------------------- #
# US-104 — an index core does not create
# --------------------------------------------------------------------------- #


def _backend(monkeypatch, boto_spy, *, create, bucket=VECTOR_BUCKET):
    _storage(monkeypatch, identity=True)
    _vectors(monkeypatch, create=create, bucket=bucket)
    backend = S3VectorsBackend(workspace_id="ws-a")
    return backend, boto_spy[-1].client


def _no_create(client) -> None:
    client.create_vector_bucket.assert_not_called()
    client.create_index.assert_not_called()


def test_off_and_no_index_fails_naming_the_bucket_and_the_index(monkeypatch, boto_spy):
    backend, client = _backend(monkeypatch, boto_spy, create=False)
    client.get_index.side_effect = _error("NotFoundException", "GetIndex")

    with pytest.raises(VectorIndexMissingError) as missing:
        backend._ensure_setup()

    assert VECTOR_BUCKET in str(missing.value) and INDEX in str(missing.value)
    assert "S3_VECTORS_CREATE_IF_MISSING" in str(missing.value)
    assert AWS_MESSAGE not in str(missing.value)
    _raised_without_its_cause(missing.value)
    _no_create(client)


def test_off_and_no_index_fails_start_up(monkeypatch, boto_spy):
    _storage(monkeypatch, identity=True)
    _vectors(monkeypatch, create=False)
    fake = MagicMock(name="s3vectors")
    fake.get_index.side_effect = _error("NotFoundException", "GetIndex")
    monkeypatch.setattr(s3b_mod.boto3, "client", lambda *_a, **_kw: fake)

    with pytest.raises(VectorIndexMissingError, match=f"{INDEX}.*{VECTOR_BUCKET}"):
        asyncio.run(verify_vector_index())
    _no_create(fake)


def test_off_a_role_refused_both_creates_starts_against_an_index_that_exists(monkeypatch):
    _storage(monkeypatch, identity=True)
    _vectors(monkeypatch, create=False)
    fake = MagicMock(name="s3vectors")
    fake.create_vector_bucket.side_effect = _error("AccessDeniedException", "CreateVectorBucket")
    fake.create_index.side_effect = _error("AccessDeniedException", "CreateIndex")
    fake.query_vectors.return_value = {"vectors": []}
    threads = []

    def _get_index(**_kwargs):
        threads.append(threading.get_ident())
        return {"index": {"dimension": 2048}}

    fake.get_index.side_effect = _get_index
    monkeypatch.setattr(s3b_mod.boto3, "client", lambda *_a, **_kw: fake)

    asyncio.run(verify_vector_index())  # start-up passes
    assert threads and threading.get_ident() not in threads, "the start-up check runs off the event loop"

    assert S3VectorsBackend(workspace_id="ws-a").search(query_embedding=[0.1] * 4) == []
    _no_create(fake)


def test_off_the_index_is_checked_once_per_backend(monkeypatch, boto_spy):
    backend, client = _backend(monkeypatch, boto_spy, create=False)
    client.get_index.return_value = {"dimension": 2048}
    client.query_vectors.return_value = {"vectors": []}

    backend.search(query_embedding=[0.1] * 4)
    checked = client.get_index.call_count
    backend.search(query_embedding=[0.1] * 4)
    backend.add_documents([{"external_file_id": "f1", "chunk_index": 0, "chunk_text": "t"}], [[0.1] * 4])

    assert checked >= 1 and client.get_index.call_count == checked
    _no_create(client)


def test_off_the_size_check_still_applies(monkeypatch, boto_spy):
    backend, client = _backend(monkeypatch, boto_spy, create=False)
    client.get_index.return_value = {"dimension": 1024}

    with pytest.raises(IndexDimensionMismatchError, match="1024"):
        backend._ensure_setup()
    _no_create(client)


def test_off_any_other_refusal_names_the_bucket_the_index_and_the_code_only(monkeypatch, boto_spy):
    backend, client = _backend(monkeypatch, boto_spy, create=False)
    client.get_index.side_effect = _error("AccessDeniedException", "GetIndex")

    with pytest.raises(VectorIndexCheckError) as refused:
        backend._ensure_setup()

    message = str(refused.value)
    assert VECTOR_BUCKET in message and INDEX in message and "AccessDeniedException" in message
    assert AWS_MESSAGE not in message, "AWS's own words are not quoted"
    assert not isinstance(refused.value, ClientError), "the raw AWS error is not re-raised"
    _raised_without_its_cause(refused.value)
    _no_create(client)


def test_on_setup_creates_as_today(monkeypatch, boto_spy):
    backend, client = _backend(monkeypatch, boto_spy, create=True)
    client.create_index.side_effect = _error("ConflictException", "CreateIndex")
    client.get_index.return_value = {"dimension": 2048}

    backend._ensure_setup()

    client.create_vector_bucket.assert_called_once_with(vectorBucketName=VECTOR_BUCKET)
    client.create_index.assert_called_once_with(vectorBucketName=VECTOR_BUCKET, indexName=INDEX, dimension=2048,
                                                dataType="float32", distanceMetric="cosine")
    client.get_index.assert_called_once()


@pytest.mark.parametrize("create,enabled,bucket", [(True, True, VECTOR_BUCKET), (False, False, VECTOR_BUCKET),
                                                   (False, True, "vectors-{workspace_id}")])
def test_the_start_up_check_runs_only_for_one_shared_bucket_with_creation_off(monkeypatch, boto_spy,
                                                                              create, enabled, bucket):
    _storage(monkeypatch, identity=True)
    _vectors(monkeypatch, create=create, enabled=enabled, bucket=bucket)
    asyncio.run(verify_vector_index())
    assert boto_spy == [], "no client, no call"


def test_the_backend_never_creates_when_off_in_its_source():
    """Every create call sits on the creating branch, never on the checking one."""
    source = (ORCHESTRATOR / "modules/search/vector_store/backends/s3_vectors_backend.py").read_text()
    tree = ast.parse(source)
    backend = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "S3VectorsBackend")
    creators = {
        method.name for method in backend.body if isinstance(method, ast.FunctionDef)
        for call in ast.walk(method)
        if isinstance(call, ast.Call) and getattr(call.func, "attr", "") in ("create_vector_bucket", "create_index")
    }
    assert creators == {"_create_if_missing"}
