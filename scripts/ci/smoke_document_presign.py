#!/usr/bin/env python3
"""Store a document in object storage through the backend and fetch it back by its presigned URL (PRD-262 US-021).

pybuild keeps only botocore's s3, s3vectors, bedrock-runtime, sts, sso and sso-oidc
models. The fresh-clone smoke builds the production stage from the checkout (PRD-262
US-035), and smoke-images boots the images CI builds; both start MinIO with the backend.
This step proves, from the host, that the pruned botocore still serves the app's object
storage. It goes through the chat's attachments, which store a file and do nothing else
with it:

1. it POSTs a short Markdown document to ``/api/attachments``. ``AttachmentStore.put``
   creates the bucket on first use and puts the bytes in object storage, with no
   extraction, chunks or embeddings. (``/api/missions/upload`` also embeds the document
   for retrieval and answers 500 when no embedding provider is configured, as on a CI
   runner, which has none and calls no paid API. ``/api/documents/upload`` keeps its
   source on the backend's volume, so it never touches the store.)
2. it GETs ``/api/attachments/<id>``: the backend lists and HEADs the object in the store
   and answers its size, which must be the size uploaded;
3. inside the backend's container (``docker exec``), ``AttachmentStore.sign_url`` mints
   the presigned GET the app hands out for an attachment, against the browser-reachable
   endpoint (S3_PUBLIC_ENDPOINT_URL). No route answers that link to a client, so the
   step runs the app's own method where the app runs;
4. the link must point at ``--storage`` (where MinIO is published on this host) and
   carry a SigV4 signature, and a GET from the host, with no credentials, must answer
   200 with the bytes uploaded.

Each run uploads a document of its own (``--nonce``). The link goes into the record
without its query string, so no signature is kept.

The standard library only, like the other smoke tools. Exit 0 = the document came back,
1 = it did not.

    python3 scripts/ci/smoke_document_presign.py --api http://localhost:8000 \\
        --storage http://localhost:9000 --container automatos_backend \\
        --workspace "$DEFAULT_WORKSPACE_ID" --out "$OUT_DIR/document-presign.json"
"""
from __future__ import annotations

import argparse
import http.client
import json
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

UPLOAD_PATH = "/api/attachments"
METADATA_PATH = "/api/attachments/{attachment_id}"
FIELD = "file"
FILENAME = "prd262-us021-smoke.md"
CONTENT_TYPE = "text/markdown"
JSON_TYPE = "application/json"
REQUEST_TIMEOUT_SECONDS = 30
SIGN_TIMEOUT_SECONDS = 60
SIGV4_ALGORITHM = "AWS4-HMAC-SHA256"
SIGV4_PARAMS = ("X-Amz-Algorithm", "X-Amz-Credential", "X-Amz-Signature")
EXCERPT_CHARS = 300
_HTTP_ERRORS = (urllib.error.URLError, http.client.HTTPException, OSError)
# Run by the backend's own python, in its working directory (/app): the app's presign
# for an attachment, printed after LINK_MARKER so no log line is read as the link.
# argv: the attachment's id, its workspace.
LINK_MARKER = "presigned-link: "
SIGN_SNIPPET = f"""import asyncio, sys, uuid
from modules.attachments.store import get_attachment_store
attachment_id, workspace_id = (uuid.UUID(value) for value in sys.argv[1:3])
print({LINK_MARKER!r} + asyncio.run(get_attachment_store().sign_url(attachment_id, workspace_id)))"""

Runner = Callable[..., subprocess.CompletedProcess]
Signer = Callable[[str], str]


class StepFailed(Exception):
    """One step of the round trip failed; the message says which and why."""


def document(nonce: str) -> bytes:
    """A short Markdown document, unique to ``nonce``. Pure."""
    return (
        f"# PRD-262 US-021 storage smoke {nonce}\n\n"
        "This document travels from the host to the backend, into object storage through the "
        "pruned botocore, and back to the host through a presigned link.\n\n"
        "When its bytes come back unchanged, the s3 client still builds, uploads and signs.\n"
    ).encode("utf-8")


def multipart(content: bytes, boundary: str) -> Tuple[bytes, str]:
    """``content`` as the one file field of a multipart/form-data body, and that body's Content-Type. Pure."""
    head = (f'--{boundary}\r\nContent-Disposition: form-data; name="{FIELD}"; filename="{FILENAME}"\r\n'
            f"Content-Type: {CONTENT_TYPE}\r\n\r\n").encode("utf-8")
    return head + content + f"\r\n--{boundary}--\r\n".encode("utf-8"), f"multipart/form-data; boundary={boundary}"


def _excerpt(body: bytes) -> str:
    return " ".join(body.decode("utf-8", "replace").split())[:EXCERPT_CHARS]


def request(url: str, *, data: Optional[bytes] = None, headers: Optional[Dict[str, str]] = None,
            timeout: float = REQUEST_TIMEOUT_SECONDS) -> Tuple[int, str, bytes]:
    """(status, Content-Type, body). A POST when ``data`` is given. An HTTP error status is
    returned, not raised; no answer at all raises StepFailed."""
    method = "GET" if data is None else "POST"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, data=data, headers=headers or {}, method=method),
                                    timeout=timeout) as response:
            return response.status, response.headers.get("Content-Type", ""), response.read()
    except urllib.error.HTTPError as error:
        return error.code, (error.headers or {}).get("Content-Type", ""), error.read()
    except _HTTP_ERRORS as error:
        raise StepFailed(f"{method} {url} got no answer: {error}") from error


def _json_field(answer: bytes, field: str, call: str):
    """``field`` of a JSON object answer, or StepFailed naming ``call``. Pure."""
    try:
        value = json.loads(answer).get(field)
    except (ValueError, AttributeError) as error:
        raise StepFailed(f"{call} answered no JSON object: {_excerpt(answer)}") from error
    if value is None:
        raise StepFailed(f"{call} answered no {field}: {_excerpt(answer)}")
    return value


def _check_size(answer: bytes, expected: int, call: str) -> None:
    """The answer's ``size_bytes`` is ``expected``, or StepFailed naming ``call``. Pure."""
    size = _json_field(answer, "size_bytes", call)
    if size != expected:
        raise StepFailed(f"{call} answered size_bytes {size!r}, not the {expected} uploaded")


def upload(api: str, content: bytes, boundary: str) -> str:
    """POST the document; the attachment id the backend stored it under."""
    body, content_type = multipart(content, boundary)
    status, _, answer = request(f"{api.rstrip('/')}{UPLOAD_PATH}", data=body,
                                headers={"Content-Type": content_type})
    call = f"POST {UPLOAD_PATH}"
    if status != 200:
        raise StepFailed(f"{call} answered {status}: {_excerpt(answer)}")
    attachment_id = _json_field(answer, "attachment_id", call)
    try:
        uuid.UUID(str(attachment_id))
    except ValueError as error:
        raise StepFailed(f"{call} answered attachment_id {attachment_id!r}, not a UUID") from error
    _check_size(answer, len(content), call)
    return str(attachment_id)


def stored(api: str, attachment_id: str, size: int) -> None:
    """The backend finds the object in the store (it lists and HEADs it) and answers its size."""
    path = METADATA_PATH.format(attachment_id=attachment_id)
    status, _, answer = request(f"{api.rstrip('/')}{path}")
    call = f"GET {path}"
    if status != 200:
        raise StepFailed(f"{call} answered {status}: {_excerpt(answer)}")
    _check_size(answer, size, call)


def sign_command(container: str, attachment_id: str, workspace: str) -> List[str]:
    """``docker exec`` of SIGN_SNIPPET in the backend's container, as its own user. Pure."""
    return ["docker", "exec", container, "python", "-c", SIGN_SNIPPET, attachment_id, workspace]


def presigned_url(container: str, attachment_id: str, workspace: str, runner: Runner = subprocess.run) -> str:
    """The link AttachmentStore.sign_url mints inside the backend's container."""
    command = sign_command(container, attachment_id, workspace)
    call = f"AttachmentStore.sign_url in {container}"
    try:
        proc = runner(command, capture_output=True, check=False, timeout=SIGN_TIMEOUT_SECONDS)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise StepFailed(f"{call} could not run: {error}") from error
    if proc.returncode != 0:
        raise StepFailed(f"{call} exited {proc.returncode}: {_excerpt(proc.stderr or b'')}")
    links = [line[len(LINK_MARKER):].strip() for line in (proc.stdout or b"").decode("utf-8", "replace").splitlines()
             if line.startswith(LINK_MARKER)]
    if len(links) != 1 or not links[0].startswith(("http://", "https://")):
        raise StepFailed(f"{call} printed no link: {_excerpt(proc.stdout or b'')}")
    return links[0]


def url_problems(url: str, storage: str) -> List[str]:
    """Why ``url`` is not a SigV4 presigned link on ``storage``'s origin. Pure."""
    link, origin = urllib.parse.urlsplit(url), urllib.parse.urlsplit(storage)
    problems = []
    if (link.scheme, link.netloc) != (origin.scheme, origin.netloc):
        problems.append(f"the link points at {link.scheme}://{link.netloc}, not {origin.scheme}://{origin.netloc}, "
                        "where object storage is published on this host (S3_PUBLIC_ENDPOINT_URL)")
    query = urllib.parse.parse_qs(link.query)
    missing = [name for name in SIGV4_PARAMS if not query.get(name)]
    if missing:
        problems.append(f"the link has no {', '.join(missing)}: it is not a SigV4 presigned URL")
    elif query["X-Amz-Algorithm"] != [SIGV4_ALGORITHM]:
        problems.append(f"the link is signed with {query['X-Amz-Algorithm'][0]}, not {SIGV4_ALGORITHM}")
    return problems


def fetch(url: str, expected: bytes) -> int:
    """GET the link from the host, as a browser would; the bytes must be the ones uploaded."""
    status, _, answer = request(url)
    if status != 200:
        raise StepFailed(f"the presigned link answered {status}: {_excerpt(answer)}")
    if answer != expected:
        raise StepFailed(f"the presigned link answered {len(answer)} bytes that are not the {len(expected)} uploaded")
    return len(answer)


def without_query(url: str) -> str:
    """``url`` without its query string, so the record keeps no signature. Pure."""
    return urllib.parse.urlunsplit(urllib.parse.urlsplit(url)._replace(query="", fragment=""))


def _round_trip(api: str, storage: str, content: bytes, nonce: str, sign: Signer) -> Dict[str, object]:
    """Upload, find, presign, check and fetch; what each step found. Raises StepFailed at the first failure."""
    started = time.monotonic()
    attachment_id = upload(api, content, f"prd262-us021-{nonce}")
    uploaded = round(time.monotonic() - started, 2)
    stored(api, attachment_id, len(content))
    url = sign(attachment_id)
    found = {"attachment_id": attachment_id, "upload_seconds": uploaded, "link": without_query(url)}
    problems = url_problems(url, storage)
    if problems:
        return {**found, "failures": problems}
    return {**found, "fetched_bytes": fetch(url, content), "failures": []}


def run(api: str, storage: str, nonce: str, sign: Signer) -> Dict[str, object]:
    """The whole round trip; ``failures`` lists what went wrong, empty when the document came back.
    ``sign`` turns an attachment id into its presigned link (presigned_url, in CI)."""
    content = document(nonce)
    report = {"api": api, "storage": storage, "filename": FILENAME, "nonce": nonce, "bytes": len(content)}
    try:
        return {**report, **_round_trip(api, storage, content, nonce, sign)}
    except StepFailed as error:
        return {**report, "failures": [str(error)]}


def summary_markdown(report: Dict[str, object]) -> str:
    """The step summary's section for this check. Pure."""
    verdict = "passed" if not report["failures"] else "FAILED"
    rows = [
        ("uploaded", f"{report.get('filename')}, {report.get('bytes')} bytes, to {UPLOAD_PATH}"),
        ("attachment id", report.get("attachment_id")),
        ("upload answered after (s)", report.get("upload_seconds")),
        ("presigned link (query left out)", report.get("link")),
        ("bytes the host fetched back", report.get("fetched_bytes")),
    ]
    lines = [f"### A document goes into object storage and comes back by its presigned URL (PRD-262 US-021): "
             f"{verdict}", "", "| | |", "|---|---|"]
    lines += [f"| {name} | {value if value is not None else '—'} |" for name, value in rows]
    lines += [f"- {failure}" for failure in report["failures"]]
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None, runner: Runner = subprocess.run) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--api", required=True, help="the backend's base URL on this host")
    parser.add_argument("--storage", required=True, help="object storage's base URL on this host")
    parser.add_argument("--container", required=True, help="the backend's container, where the link is minted")
    parser.add_argument("--workspace", required=True, help="the workspace the backend stores the upload under")
    parser.add_argument("--nonce", default=uuid.uuid4().hex[:12], help="makes this run's document unique")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args(argv)

    def sign(attachment_id: str) -> str:
        return presigned_url(args.container, attachment_id, args.workspace, runner)

    report = run(args.api, args.storage, args.nonce, sign)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    text = summary_markdown(report)
    print(text)
    if args.summary:
        with args.summary.open("a", encoding="utf-8") as handle:
            handle.write(text)
    return 1 if report["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
