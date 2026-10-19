"""Read-only OCI registry access for the image reports (PRD-262 US-001, scripts/ci).

``images.yml`` pushes every build by digest: to GHCR when it publishes, and to the
job's own ``registry:2`` service on ``localhost:5000`` otherwise. The reports read
those digests back, and the current ``edge`` from GHCR, through the registry's HTTP
API with the standard library only, because the arm64 runner image has no
``crane`` or ``skopeo`` and the reports also need layer blobs, which
``docker buildx imagetools inspect`` cannot fetch.

* ``connect(ref)`` returns a :class:`Session` for the repository ``ref`` names. A
  registry that answers ``/v2/`` with a Bearer challenge gets a pull token,
  anonymous unless ``REGISTRY_USER``/``REGISTRY_TOKEN`` are set (the GHCR packages
  are public, so only a brand-new private package needs them). ``localhost``
  speaks plain HTTP and needs no token.
* ``platform_digest(index, arch)`` picks one architecture's manifest out of the
  index BuildKit pushes. With ``provenance`` and ``sbom`` on, a per-arch push is an
  index of two manifests: the image, and an ``unknown/unknown`` attestation.
* ``resolve(ref)`` is the digest a tag points at: for a published tag, the
  multi-arch index (smoke-images, PRD-262 US-003, checks that a pulled image's
  ``RepoDigests`` holds it, and promote-edge that ``edge`` lands on it).
* Reads retry three times on a network error, a 5xx or a 429.
* ``READ_ERRORS`` is what a read here can raise, for the callers that report a
  failed read instead of crashing on it.
"""
from __future__ import annotations

import base64
import hashlib
import http.client
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import IO, NamedTuple, Optional, Tuple

MANIFEST_TYPES = ",".join((
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
    "application/vnd.docker.distribution.manifest.v2+json",
))
ATTEMPTS = 3
TIMEOUT_SECONDS = 120
LOCAL_HOSTS = ("localhost", "127.0.0.1")
CHALLENGE_PARAM = re.compile(r'(\w+)="([^"]*)"')
# A failed request (URLError, HTTPError included), a body cut short mid-read
# (http.client.IncompleteRead is an HTTPException, neither a URLError nor an OSError),
# a platform the index lacks (LookupError), or a reference or JSON that does not parse.
READ_ERRORS = (urllib.error.URLError, http.client.HTTPException, LookupError, ValueError, OSError)


class Ref(NamedTuple):
    """``host/repo`` plus a tag or ``sha256:`` digest (``reference``)."""

    host: str
    repo: str
    reference: str


class Session(NamedTuple):
    """One repository on one registry, with the pull token it needs (or none)."""

    base: str
    repo: str
    token: Optional[str]


def split_ref(ref: str) -> Ref:
    """``ghcr.io/o/automatos-api:edge`` or ``localhost:5000/automatos-api@sha256:…``."""
    host, _, rest = ref.partition("/")
    if not rest or ("." not in host and ":" not in host and host not in LOCAL_HOSTS):
        raise ValueError(f"not a registry reference with a host: {ref!r}")
    if "@" in rest:
        repo, _, reference = rest.partition("@")
    elif ":" in rest.rsplit("/", 1)[-1]:
        repo, _, reference = rest.rpartition(":")
    else:
        repo, reference = rest, "latest"
    return Ref(host, repo, reference)


def parse_challenge(header: str) -> dict:
    """The parameters of a ``WWW-Authenticate: Bearer realm=…,service=…`` header."""
    scheme, _, params = header.partition(" ")
    if scheme.lower() != "bearer":
        raise ValueError(f"unsupported registry auth scheme: {scheme!r}")
    return dict(CHALLENGE_PARAM.findall(params))


def platform_digest(index: dict, arch: str) -> Optional[str]:
    """The digest of ``arch``'s image manifest in ``index``; None for a plain manifest.

    The attestation manifest BuildKit adds is ``unknown/unknown`` and never matches.
    """
    if "manifests" not in index:
        return None
    for entry in index["manifests"]:
        platform = entry.get("platform", {})
        if platform.get("os") == "linux" and platform.get("architecture") == arch:
            return entry["digest"]
    raise LookupError(f"the index has no linux/{arch} manifest")


def _credentials() -> Optional[Tuple[str, str]]:
    user, token = os.environ.get("REGISTRY_USER", ""), os.environ.get("REGISTRY_TOKEN", "")
    return (user, token) if user and token else None


def _open(request: urllib.request.Request) -> IO[bytes]:
    """``urlopen`` with retries on network errors, 5xx and 429; other HTTP errors raise."""
    for attempt in range(1, ATTEMPTS + 1):
        try:
            return urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS)
        except urllib.error.HTTPError as error:
            if (error.code < 500 and error.code != 429) or attempt == ATTEMPTS:
                raise
        except urllib.error.URLError:
            if attempt == ATTEMPTS:
                raise
        time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def _request(url: str, accept: str = "", token: Optional[str] = None) -> urllib.request.Request:
    request = urllib.request.Request(url, headers={"Accept": accept} if accept else {})
    if token:
        # Unredirected: a blob GET redirects to signed storage, which rejects a bearer.
        request.add_unredirected_header("Authorization", f"Bearer {token}")
    return request


def _token(challenge: dict, repo: str) -> str:
    query = {"scope": f"repository:{repo}:pull"}
    if challenge.get("service"):
        query["service"] = challenge["service"]
    request = _request(f"{challenge['realm']}?{urllib.parse.urlencode(query)}")
    credentials = _credentials()
    if credentials:
        basic = base64.b64encode(":".join(credentials).encode()).decode()
        request.add_unredirected_header("Authorization", f"Basic {basic}")
    with _open(request) as response:
        body = json.load(response)
    return body.get("token") or body["access_token"]


def connect(ref: str) -> Tuple[Session, str]:
    """A session for ``ref``'s repository, and the tag or digest ``ref`` names."""
    host, repo, reference = split_ref(ref)
    scheme = "http" if host.split(":")[0] in LOCAL_HOSTS else "https"
    base = f"{scheme}://{host}"
    try:
        _open(_request(f"{base}/v2/")).close()
        return Session(base, repo, None), reference
    except urllib.error.HTTPError as error:
        if error.code != 401:
            raise
        challenge = parse_challenge(error.headers.get("WWW-Authenticate", ""))
    return Session(base, repo, _token(challenge, repo)), reference


def manifest(session: Session, reference: str) -> dict:
    """The index or manifest ``reference`` (tag or digest) names, as JSON."""
    url = f"{session.base}/v2/{session.repo}/manifests/{reference}"
    with _open(_request(url, MANIFEST_TYPES, session.token)) as response:
        return json.load(response)


def resolve(ref: str) -> str:
    """The ``sha256:`` digest ``ref``'s tag points at, as the registry serves it.

    A HEAD request answers with ``Docker-Content-Digest``; a registry that leaves the
    header out is read with a GET and the digest is the hash of the bytes it served.
    """
    session, reference = connect(ref)
    if reference.startswith("sha256:"):
        return reference
    url = f"{session.base}/v2/{session.repo}/manifests/{reference}"
    head = _request(url, MANIFEST_TYPES, session.token)
    head.method = "HEAD"
    with _open(head) as response:
        digest = response.headers.get("Docker-Content-Digest", "")
    if digest:
        return digest
    with _open(_request(url, MANIFEST_TYPES, session.token)) as response:
        return "sha256:" + hashlib.sha256(response.read()).hexdigest()


def blob(session: Session, digest: str) -> IO[bytes]:
    """An open stream of one blob (a config or a compressed layer); close it after use."""
    return _open(_request(f"{session.base}/v2/{session.repo}/blobs/{digest}", token=session.token))


def image(ref: str, arch: str) -> Tuple[dict, dict, Session]:
    """``arch``'s manifest and image config for ``ref``, and the session that read them."""
    session, reference = connect(ref)
    top = manifest(session, reference)
    digest = platform_digest(top, arch)
    image_manifest = manifest(session, digest) if digest else top
    with blob(session, image_manifest["config"]["digest"]) as response:
        config = json.load(response)
    return {**image_manifest, "digest": digest or reference}, config, session
