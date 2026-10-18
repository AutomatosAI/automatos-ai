#!/usr/bin/env python3
"""Upload a document to the running backend and fetch it back by its presigned URL (PRD-262 US-021).

pybuild keeps only botocore's s3, s3vectors, bedrock-runtime, sts, sso and sso-oidc
models. The fresh-clone smoke builds the development stage, which shares that base, and
smoke-images boots the production stage; both start MinIO with the backend. This step
proves, from the host, that the pruned botocore still serves the app's object storage:

1. it POSTs a short Markdown document to ``/api/missions/upload``, which runs
   ``DocumentManager.upload_document``: it creates the bucket on first use and puts the
   source in object storage. (``/api/documents/upload`` keeps its source on the
   backend's volume, and the backend serves that download itself, so it never touches
   the store.)
2. it GETs ``/api/documents/<id>/download?mode=url``, the call the Documents page makes:
   the backend HEADs the object and answers JSON with a presigned GET, minted against
   the browser-reachable endpoint (S3_PUBLIC_ENDPOINT_URL);
3. the link must point at ``--storage`` (where MinIO is published on this host) and
   carry a SigV4 signature, and a GET from the host, with no credentials, must answer
   200 with the bytes uploaded.

Each run uploads a document of its own (``--nonce``), so no earlier copy stands in for
it. The link goes into the record without its query string, so no signature is kept.

The standard library only, like the other smoke tools. Exit 0 = the document came back,
1 = it did not.

    python3 scripts/ci/smoke_document_presign.py --api http://localhost:8000 \\
        --storage http://localhost:9000 --out "$OUT_DIR/document-presign.json"
"""
from __future__ import annotations

import argparse
import http.client
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

UPLOAD_PATH = "/api/missions/upload"
DOWNLOAD_PATH = "/api/documents/{document_id}/download?mode=url"
FIELD = "file"
FILENAME = "prd262-us021-smoke.md"
CONTENT_TYPE = "text/markdown"
JSON_TYPE = "application/json"
UPLOAD_TIMEOUT_SECONDS = 180  # the backend processes the document before it answers
REQUEST_TIMEOUT_SECONDS = 30
SIGV4_ALGORITHM = "AWS4-HMAC-SHA256"
SIGV4_PARAMS = ("X-Amz-Algorithm", "X-Amz-Credential", "X-Amz-Signature")
EXCERPT_CHARS = 300
_HTTP_ERRORS = (urllib.error.URLError, http.client.HTTPException, OSError)


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


def upload(api: str, content: bytes, boundary: str) -> int:
    """POST the document; the id the backend stored it under."""
    body, content_type = multipart(content, boundary)
    status, _, answer = request(f"{api.rstrip('/')}{UPLOAD_PATH}", data=body,
                                headers={"Content-Type": content_type}, timeout=UPLOAD_TIMEOUT_SECONDS)
    call = f"POST {UPLOAD_PATH}"
    if status != 200:
        raise StepFailed(f"{call} answered {status}: {_excerpt(answer)}")
    document_id = _json_field(answer, "document_id", call)
    if not isinstance(document_id, int) or isinstance(document_id, bool):
        raise StepFailed(f"{call} answered document_id {document_id!r}, not an integer")
    return document_id


def presigned_url(api: str, document_id: int) -> str:
    """The link the Documents page's download call answers with."""
    path = DOWNLOAD_PATH.format(document_id=document_id)
    status, content_type, answer = request(f"{api.rstrip('/')}{path}")
    call = f"GET {path}"
    if status != 200:
        raise StepFailed(f"{call} answered {status}: {_excerpt(answer)}")
    if not content_type.startswith(JSON_TYPE):
        raise StepFailed(f"{call} answered {content_type or 'no Content-Type'}, not a JSON link: the backend served "
                         "the file itself, so the document is not in object storage")
    url = _json_field(answer, "url", call)
    if not isinstance(url, str):
        raise StepFailed(f"{call} answered url {url!r}, not a string")
    return url


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


def _round_trip(api: str, storage: str, content: bytes, nonce: str) -> Dict[str, object]:
    """Upload, presign, check and fetch; what each step found. Raises StepFailed at the first failure."""
    started = time.monotonic()
    document_id = upload(api, content, f"prd262-us021-{nonce}")
    uploaded = round(time.monotonic() - started, 2)
    url = presigned_url(api, document_id)
    found = {"document_id": document_id, "upload_seconds": uploaded, "link": without_query(url)}
    problems = url_problems(url, storage)
    if problems:
        return {**found, "failures": problems}
    return {**found, "fetched_bytes": fetch(url, content), "failures": []}


def run(api: str, storage: str, nonce: str) -> Dict[str, object]:
    """The whole round trip; ``failures`` lists what went wrong, empty when the document came back."""
    content = document(nonce)
    report = {"api": api, "storage": storage, "filename": FILENAME, "nonce": nonce, "bytes": len(content)}
    try:
        return {**report, **_round_trip(api, storage, content, nonce)}
    except StepFailed as error:
        return {**report, "failures": [str(error)]}


def summary_markdown(report: Dict[str, object]) -> str:
    """The step summary's section for this check. Pure."""
    verdict = "passed" if not report["failures"] else "FAILED"
    rows = [
        ("uploaded", f"{report.get('filename')}, {report.get('bytes')} bytes, to {UPLOAD_PATH}"),
        ("document id", report.get("document_id")),
        ("upload answered after (s)", report.get("upload_seconds")),
        ("presigned link (query left out)", report.get("link")),
        ("bytes the host fetched back", report.get("fetched_bytes")),
    ]
    lines = [f"### A document goes into object storage and comes back by its presigned URL (PRD-262 US-021): "
             f"{verdict}", "", "| | |", "|---|---|"]
    lines += [f"| {name} | {value if value is not None else '—'} |" for name, value in rows]
    lines += [f"- {failure}" for failure in report["failures"]]
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--api", required=True, help="the backend's base URL on this host")
    parser.add_argument("--storage", required=True, help="object storage's base URL on this host")
    parser.add_argument("--nonce", default=uuid.uuid4().hex[:12], help="makes this run's document unique")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args(argv)
    report = run(args.api, args.storage, args.nonce)
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
