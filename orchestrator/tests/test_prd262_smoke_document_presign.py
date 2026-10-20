"""PRD-262 US-021: the smoke stores a document in object storage and fetches it back by its presigned URL.

scripts/ci/smoke_document_presign.py runs against the stack the fresh-clone smoke and
smoke-images boot, from the host. The step itself needs that stack and runs only in CI;
these tests hold its parts:

- its multipart body is one FastAPI parses as the upload route's ``file`` field;
- against a stand-in backend and store on 127.0.0.1, the round trip passes, and each way
  it can go wrong fails with its own reason: the upload refused, a size that is not the
  one uploaded, an object the backend cannot find, a link on another origin or without a
  SigV4 signature, other bytes coming back, and nothing listening;
- the link is minted by ``docker exec`` of a snippet in the backend's container, which
  imports what the attachments store defines; a failed, silent or missing exec fails with
  its reason;
- its record keeps the link without the signature, and it writes the step summary;
- the routes it calls are the backend's, and they store the file and nothing else: the
  attachments store creates the bucket and puts the object, imports nothing of the RAG
  ingestion or the LLM clients (so no embedding provider is needed, which CI has not),
  and its sign_url presigns against the public endpoint;
- smoke-fresh-clone.sh runs it in every mode, after the image checks and before the warm
  restarts, against the origin compose publishes MinIO on (S3_PUBLIC_ENDPOINT_URL), with
  the backend's container and the local workspace, and both smoke workflows watch it.
"""
from __future__ import annotations

import ast
import http.server
import importlib.util
import json
import re
import socket
import subprocess
import sys
import threading
from pathlib import Path

import pytest
import yaml
from fastapi import FastAPI, File, UploadFile  # module level: the route's annotations resolve against these
from fastapi.testclient import TestClient

_REPO = Path(__file__).resolve().parents[2]
_TOOL = _REPO / "scripts" / "ci" / "smoke_document_presign.py"
_SMOKE = _REPO / "scripts" / "ci" / "smoke-fresh-clone.sh"
_STORE = _REPO / "orchestrator" / "modules" / "attachments" / "store.py"
_WORKFLOWS = _REPO / ".github" / "workflows"
_spec = importlib.util.spec_from_file_location("smoke_document_presign", _TOOL)
presign = importlib.util.module_from_spec(_spec)
sys.modules["smoke_document_presign"] = presign
_spec.loader.exec_module(presign)

ATTACHMENT_ID = "0f6c2a4e-8d3b-4c1a-9e7f-2b5d6a8c1e30"
WORKSPACE = "00000000-0000-0000-0000-0000000000c1"
CONTAINER = "automatos_backend"
BUCKET = "automatos-ai"
SIGV4_QUERY = ("X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=minioadmin%2F20261019%2Fus-east-1%2Fs3%2Faws4_request"
               "&X-Amz-Date=20261019T120000Z&X-Amz-Expires=900&X-Amz-SignedHeaders=host&X-Amz-Signature=5e1f")


def _file_bytes(body: bytes) -> bytes:
    """The one part's bytes: after its headers, before the closing delimiter (FastAPI's parse is tested apart)."""
    return body.split(b"\r\n\r\n", 1)[1].rsplit(b"\r\n--", 1)[0]


# --------------------------------------------------------------------------- a stand-in stack


class _Stack(http.server.ThreadingHTTPServer):
    """The backend's two attachment routes and the store's GET, on one port. ``mode`` picks how it goes wrong."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.mode, self.stored, self.store_headers = "ok", b"", {}

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"

    def link(self, attachment_id: str) -> str:
        """What AttachmentStore.sign_url would mint for the stand-in store."""
        origin = "http://minio:9000" if self.mode == "other-origin" else self.base
        query = "AWSAccessKeyId=minioadmin&Signature=x&Expires=1" if self.mode == "sigv2" else SIGV4_QUERY
        key = f"workspaces/{WORKSPACE}/ephemeral-attachments/{attachment_id}/{presign.FILENAME}"
        return f"{origin}/{BUCKET}/{key}?{query}"

    def metadata(self) -> bytes:
        size = len(self.stored) + (1 if self.mode == "wrong-size" else 0)
        return json.dumps({"attachment_id": ATTACHMENT_ID, "filename": presign.FILENAME, "mime": presign.CONTENT_TYPE,
                           "media_type": "document", "size_bytes": size}).encode()


class _Handler(http.server.BaseHTTPRequestHandler):
    server: _Stack

    def log_message(self, *args) -> None:  # quiet test output
        return None

    def _send(self, status: int, body: bytes, content_type: str = presign.JSON_TYPE) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        body = self.rfile.read(int(self.headers["Content-Length"]))
        if self.path != presign.UPLOAD_PATH or self.server.mode == "upload-refused":
            self._send(500, b'{"detail": "Upload failed"}')
            return
        self.server.stored = _file_bytes(body)
        self._send(200, self.server.metadata())

    def do_GET(self) -> None:
        if self.path == presign.METADATA_PATH.format(attachment_id=ATTACHMENT_ID):
            if self.server.mode == "not-stored":
                self._send(404, b'{"detail": "Attachment not found or expired"}')
            else:
                self._send(200, self.server.metadata())
        elif self.path.startswith(f"/{BUCKET}/"):
            self.server.store_headers = dict(self.headers)
            tampered = self.server.mode == "other-bytes"
            self._send(200, self.server.stored + (b"!" if tampered else b""), presign.CONTENT_TYPE)
        else:
            self._send(404, b'{"detail": "Not Found"}')


@pytest.fixture
def stack(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "*")  # urllib must reach 127.0.0.1 directly
    monkeypatch.setenv("no_proxy", "*")
    server = _Stack(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()


def _exec_runner(stack: _Stack, calls: list):
    """A docker that answers the sign exec as the backend would: a log line, then the marked link."""
    def runner(command, **kwargs):
        calls.append((command, kwargs))
        out = f"INFO config loaded\n{presign.LINK_MARKER}{stack.link(command[-2])}\n".encode()
        return subprocess.CompletedProcess(command, 0, out, b"")
    return runner


# --------------------------------------------------------------------------- the round trip


def test_the_upload_body_is_one_fastapi_reads_as_the_file_field():
    app = FastAPI()

    @app.post(presign.UPLOAD_PATH)
    async def upload(file: UploadFile = File(...)) -> dict:
        return {"filename": file.filename, "type": file.content_type, "body": (await file.read()).decode()}

    content = presign.document("n0nce")
    body, content_type = presign.multipart(content, "prd262-us021-n0nce")
    answer = TestClient(app).post(presign.UPLOAD_PATH, content=body, headers={"Content-Type": content_type})

    assert answer.status_code == 200, answer.text
    assert answer.json() == {"filename": presign.FILENAME, "type": presign.CONTENT_TYPE, "body": content.decode()}


def test_each_run_uploads_a_document_of_its_own():
    assert presign.document("a") != presign.document("b")
    assert presign.FILENAME.endswith(".md")


def test_the_document_goes_in_and_comes_back_by_its_presigned_url(stack):
    report = presign.run(stack.base, stack.base, "n0nce", stack.link)

    assert report["failures"] == [], report
    assert stack.stored == presign.document("n0nce")
    assert report["attachment_id"] == ATTACHMENT_ID and report["fetched_bytes"] == len(stack.stored)
    key = f"workspaces/{WORKSPACE}/ephemeral-attachments/{ATTACHMENT_ID}/{presign.FILENAME}"
    assert report["link"] == f"{stack.base}/{BUCKET}/{key}"
    assert "Authorization" not in stack.store_headers, "the host fetches the link with no credentials"


@pytest.mark.parametrize("mode,reason", [
    ("upload-refused", f"POST {presign.UPLOAD_PATH} answered 500"),
    ("wrong-size", "answered size_bytes"),
    ("not-stored", f"GET {presign.METADATA_PATH.format(attachment_id=ATTACHMENT_ID)} answered 404"),
    ("other-origin", "the link points at http://minio:9000, not http://127.0.0.1:"),
    ("sigv2", "is not a SigV4 presigned URL"),
    ("other-bytes", "bytes that are not the"),
])
def test_each_way_the_round_trip_breaks_fails_with_its_reason(stack, mode, reason):
    stack.mode = mode
    report = presign.run(stack.base, stack.base, "n0nce", stack.link)
    assert len(report["failures"]) == 1 and reason in report["failures"][0], report["failures"]


def test_nothing_listening_fails_with_no_answer():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]  # closed again on leaving: nothing listens there
    report = presign.run(f"http://127.0.0.1:{port}", f"http://127.0.0.1:{port}", "n0nce", lambda _id: "")
    assert len(report["failures"]) == 1 and "got no answer" in report["failures"][0]


def test_a_link_must_be_sigv4_on_the_published_origin():
    good = f"http://localhost:9000/{BUCKET}/k?{SIGV4_QUERY}"
    assert presign.url_problems(good, "http://localhost:9000") == []
    assert presign.url_problems(good, "http://localhost:9100") == [
        "the link points at http://localhost:9000, not http://localhost:9100, where object storage is "
        "published on this host (S3_PUBLIC_ENDPOINT_URL)"]
    unsigned = presign.url_problems(f"http://localhost:9000/{BUCKET}/k", "http://localhost:9000")
    assert unsigned == ["the link has no X-Amz-Algorithm, X-Amz-Credential, X-Amz-Signature: it is not a SigV4 "
                        "presigned URL"]
    other = good.replace("AWS4-HMAC-SHA256", "AWS4-ECDSA-P256-SHA256")
    assert presign.url_problems(other, "http://localhost:9000") == [
        "the link is signed with AWS4-ECDSA-P256-SHA256, not AWS4-HMAC-SHA256"]


def test_the_record_keeps_no_signature_and_the_summary_says_how_it_went(stack, tmp_path):
    out, summary = tmp_path / "document-presign.json", tmp_path / "summary.md"
    argv = ["--api", stack.base, "--storage", stack.base, "--container", CONTAINER, "--workspace", WORKSPACE,
            "--out", str(out), "--summary", str(summary)]
    calls: list = []

    assert presign.main(argv, runner=_exec_runner(stack, calls)) == 0
    record = json.loads(out.read_text(encoding="utf-8"))
    assert record["failures"] == [] and "?" not in record["link"]
    assert "Signature" not in out.read_text(encoding="utf-8")
    assert "(PRD-262 US-021): passed" in summary.read_text(encoding="utf-8")
    (command, kwargs), = calls
    assert command == presign.sign_command(CONTAINER, ATTACHMENT_ID, WORKSPACE)
    assert kwargs["timeout"] == presign.SIGN_TIMEOUT_SECONDS and "shell" not in kwargs

    stack.mode = "not-stored"
    assert presign.main(argv, runner=_exec_runner(stack, [])) == 1
    assert "(PRD-262 US-021): FAILED" in summary.read_text(encoding="utf-8")


def test_without_query_drops_the_query_and_the_fragment():
    assert presign.without_query("http://h:9/b/k?X-Amz-Signature=s#f") == "http://h:9/b/k"


# --------------------------------------------------------------------------- the link, minted in the backend


def test_the_link_is_minted_by_the_backends_own_python_in_its_container():
    assert presign.sign_command(CONTAINER, ATTACHMENT_ID, WORKSPACE) == [
        "docker", "exec", CONTAINER, "python", "-c", presign.SIGN_SNIPPET, ATTACHMENT_ID, WORKSPACE]


def test_the_snippet_calls_what_the_attachments_store_defines():
    tree = ast.parse(presign.SIGN_SNIPPET)
    (imported,) = [node for node in tree.body if isinstance(node, ast.ImportFrom)]
    assert imported.module == "modules.attachments.store"
    assert [alias.name for alias in imported.names] == ["get_attachment_store"]
    called = {node.func.attr for node in ast.walk(tree)
              if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
    assert "sign_url" in called
    store = ast.parse(_STORE.read_text(encoding="utf-8"))
    functions = {node.name for node in ast.walk(store) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    assert {"get_attachment_store", "sign_url"} <= functions
    assert presign.LINK_MARKER in presign.SIGN_SNIPPET


@pytest.mark.parametrize("answer,reason", [
    (subprocess.CompletedProcess([], 1, b"", b"AttachmentNotFoundError: not found or expired"),
     "exited 1: AttachmentNotFoundError: not found or expired"),
    (subprocess.CompletedProcess([], 0, b"http://localhost:9000/b/k\n", b""), "printed no link"),
    (FileNotFoundError("docker"), "could not run"),
    (subprocess.TimeoutExpired(["docker"], 60), "could not run"),
])
def test_a_sign_that_fails_says_why(answer, reason):
    def runner(command, **kwargs):
        if isinstance(answer, BaseException):
            raise answer
        return answer

    with pytest.raises(presign.StepFailed, match=reason):
        presign.presigned_url(CONTAINER, ATTACHMENT_ID, WORKSPACE, runner)


# --------------------------------------------------------------------------- the backend's routes


def _route_manifest() -> set:
    manifest = json.loads((_REPO / "orchestrator" / "reports" / "route-manifest.json").read_text(encoding="utf-8"))
    return {(route["method"], route["path"]) for route in manifest["routes"]}


def _function(path: Path, name: str) -> ast.AST:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return next(node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == name)


def test_the_tool_calls_routes_the_backend_has():
    assert ("POST", presign.UPLOAD_PATH) in _route_manifest()
    assert ("GET", presign.METADATA_PATH) in _route_manifest()


def test_the_attachment_upload_stores_the_file_and_nothing_else():
    """No RAG ingestion on this path, so it needs no embedding provider (CI has none)."""
    route = ast.unparse(_function(_REPO / "orchestrator" / "api" / "attachments.py", "upload_attachment"))
    assert "await store.put(" in route
    put = ast.unparse(_function(_STORE, "put"))
    assert "ensure_bucket(self._bucket)" in put and "self._client.put_object(" in put
    for module in ("api/attachments.py", "modules/attachments/store.py", "modules/attachments/validation.py"):
        tree = ast.parse((_REPO / "orchestrator" / module).read_text(encoding="utf-8"))
        imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
        imported |= {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
        assert not {name for name in imported if name.startswith(("modules.rag", "core.llm"))}, module


def test_sign_url_presigns_a_get_against_the_public_endpoint():
    sign = ast.unparse(_function(_STORE, "sign_url"))
    assert "get_public_s3_client().generate_presigned_url('get_object'" in sign


# --------------------------------------------------------------------------- the wiring


def _shell_function(text: str, name: str) -> str:
    match = re.search(rf"^{name}\(\) \{{\n(.*?)^\}}", text, re.M | re.S)
    assert match, f"smoke-fresh-clone.sh has no {name}() function"
    return match.group(1)


def test_the_smoke_runs_it_in_every_mode_after_the_image_checks_and_before_the_warm_restarts():
    text = _SMOKE.read_text(encoding="utf-8")
    assert f'DOCUMENT_TOOL="{_TOOL.relative_to(_REPO).as_posix()}"' in text
    step = _shell_function(text, "check_document_storage")
    assert '"$DOCUMENT_TOOL" --api "$API_BASE" --storage "$STORAGE_URL"' in step
    assert '--container "$BACKEND_CONTAINER" --workspace "$DEFAULT_WORKSPACE_ID"' in step
    assert '--out "$DOCUMENT_FILE"' in step and "|| fail " in step
    assert "SMOKE_IMAGES" not in step, "the source build and both image modes all run it"
    main = _shell_function(text, "main")
    order = [main.index(call) for call in ("check_images", "check_document_storage", "warm_restarts")]
    assert order == sorted(order)


def test_the_container_and_workspace_are_the_ones_compose_runs():
    text = _SMOKE.read_text(encoding="utf-8")
    assert re.search(rf'^BACKEND_CONTAINER="{CONTAINER}"$', text, re.M)
    assert re.search(rf'^export DEFAULT_WORKSPACE_ID="\$\{{DEFAULT_WORKSPACE_ID:-{WORKSPACE}\}}"$', text, re.M)
    compose = yaml.safe_load((_REPO / "docker-compose.yml").read_text(encoding="utf-8"))
    assert compose["services"]["backend"]["container_name"] == CONTAINER
    defaults = (_REPO / "envs" / "api.defaults").read_text(encoding="utf-8")
    assert re.search(rf"^DEFAULT_WORKSPACE_ID={WORKSPACE}$", defaults, re.M)


def test_the_link_must_point_where_compose_publishes_minio():
    text = _SMOKE.read_text(encoding="utf-8")
    assert re.search(r'^STORAGE_URL="http://localhost:\$\{MINIO_PORT:-9000\}"$', text, re.M)
    defaults = (_REPO / "envs" / "api.defaults").read_text(encoding="utf-8")
    assert re.search(r"^S3_PUBLIC_ENDPOINT_URL=http://localhost:9000$", defaults, re.M)
    compose = (_REPO / "docker-compose.yml").read_text(encoding="utf-8")
    assert '"${BIND_ADDRESS:-127.0.0.1}:${MINIO_PORT:-9000}:9000"' in compose


def _paths(name: str, event: str) -> list:
    workflow = yaml.safe_load((_WORKFLOWS / name).read_text(encoding="utf-8"))
    return workflow.get("on", workflow.get(True))[event]["paths"]  # PyYAML reads `on` as True


def test_both_smoke_workflows_watch_the_tool():
    relative = _TOOL.relative_to(_REPO).as_posix()
    assert relative in _paths("smoke-fresh-clone.yml", "push")
    assert relative in _paths("smoke-fresh-clone.yml", "pull_request")
    assert relative in _paths("smoke-images.yml", "pull_request")
