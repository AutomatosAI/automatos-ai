"""PRD-270 W2 US-270-15a — what the plugin import tests share: a plugin's bundle and package, Automatos Cloud
as an ``httpx.MockTransport``, the object store as a fake S3 client, and ``run_async`` as ``asyncio.run``.

``tests/test_prd270w2_plugin_import_checks.py`` (the zip rules, the local scan and the pass's seams, no
database) and ``tests/test_prd270w2_plugin_import.py`` (the import on Postgres) both use them. Fake data only.
"""
from __future__ import annotations

import asyncio
import hashlib
import io
import json
import random
import warnings
import zipfile
from pathlib import Path

import httpx

from config import config
from core.services import marketplace_s3, plugin_security_scanner
from services.marketplace_bundles import items_feed, plugin_import
from services.marketplace_bundles.canonical import canonical_bytes
from services.marketplace_bundles.catalog_items import entry_of
from services.marketplace_bundles.catalog_paths import plugin_package_path
from services.marketplace_bundles.exported import ExportedItem, ItemEntry

ORCH = Path(__file__).resolve().parents[1]
FIXTURES = ORCH / "tests" / "fixtures" / "marketplace_bundles"
MODULE = ORCH / "services" / "marketplace_bundles" / "plugin_import.py"
UPLOAD_SERVICE = ORCH / "core" / "services" / "plugin_upload_service.py"
S3_SERVICE = ORCH / "core" / "services" / "marketplace_s3.py"
CLOUD = "https://cloud.example.test"
BUNDLES = CLOUD + "/api/marketplace/catalog/bundles/"
MARK = "SECRET-MARK-7f3a"
BINARY = bytes(range(256))
APPEND_LINE = "    open(notes_path, 'a')"  # a medium static finding (file_write), never critical
SENT_HEADERS = {"host", "accept", "accept-encoding", "user-agent", "connection"}


# --------------------------------------------------------------------------- the plugin, its bundle and package


def external_id(slug):
    return hashlib.sha256(slug.encode()).hexdigest()[:32]


def bundle_of(slug, version="1.0.0", name="Test Research Kit", **changes):
    base = json.loads((FIXTURES / "plugin.json").read_text(encoding="utf-8"))
    plugin = {**base["plugin"], "slug": slug}
    return {**base, "externalId": external_id(slug), "version": version, "name": name, "plugin": plugin, **changes}


def skill_md(skill, body="Summarise the sources."):
    return f"---\nname: {skill}\ndescription: A test skill.\n---\n\n# Summarise\n\n{body}\n".encode()


def files_of(slug, version, skill=None, manifest=None, extra=None, body="Summarise the sources."):
    """A plugin's files: its manifest (``manifest`` overriding its keys), a SKILL.md (its skill named after the
    slug unless ``skill`` names it), a command with a medium static finding and a binary file."""
    named = {"slug": slug, "name": "Test Research Kit", "version": version, **(manifest or {})}
    skill = skill or f"{slug}-skill"
    return {
        "manifest.json": json.dumps(named).encode(),
        f"skills/{skill}/SKILL.md": skill_md(skill, body),
        "commands/save-notes.md": f"Save the notes:\n\n{APPEND_LINE}\n".encode(),
        "assets/logo.png": BINARY,
        **(extra or {}),
    }


def zip_of(files, compression=zipfile.ZIP_DEFLATED, folders=()):
    buffer = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # zipfile warns on a name written twice
        with zipfile.ZipFile(buffer, "w", compression=compression) as archive:
            for folder in folders:
                archive.writestr(folder, b"")
            for path, data in (files.items() if isinstance(files, dict) else files):
                archive.writestr(path, data)
    return buffer.getvalue()


def incompressible(size, seed=0):
    return random.Random(seed).randbytes(size)


def package_url(slug, version):
    return CLOUD + plugin_package_path(slug, version)


class Cloud:
    """Automatos Cloud as a mock transport: a responder per URL (404 for any other), each request recorded."""

    def __init__(self):
        self.requests = []
        self.answers = {}
        self.on_request = None  # called on every request (the open-transaction check)

    def answer(self, url, body):
        self.answers[url] = body

    def serve(self, bundle, package):
        """``bundle`` at its derived URL and ``package`` at its plugin's; the parsed catalogue entry."""
        data = canonical_bytes(bundle)
        revision = hashlib.sha256(data).hexdigest()
        self.answer(BUNDLES + revision, data)
        self.answer(package_url(bundle["plugin"]["slug"], bundle["version"]), package)
        item = ExportedItem(ItemEntry("plugin", bundle["externalId"], bundle["version"], frozenset()), bundle, revision)
        return items_feed.parse_item_entry(entry_of(item))

    def handler(self, request):
        self.requests.append(request)
        if self.on_request is not None:
            self.on_request()
        body = self.answers.get(str(request.url))
        return httpx.Response(404) if body is None else httpx.Response(200, content=body)

    def factory(self):
        return httpx.Client(transport=httpx.MockTransport(self.handler), follow_redirects=True)

    def urls(self):
        return [str(request.url) for request in self.requests]


class _Body:
    def __init__(self, data):
        self.data = data

    def read(self, amt=None):
        return self.data if amt is None else self.data[:amt]


class Bucket:
    """The four S3 calls MarketplaceS3Service makes, on a dict of (bucket, key) -> bytes."""

    def __init__(self):
        self.objects = {}
        self.puts = 0
        self.fail_put_at = None  # the put (1-based) that raises
        self.on_write = None  # called on every put and delete (the open-transaction check)

    def put_object(self, Bucket, Key, Body, **_):  # noqa: N803 — boto3's argument names
        if self.on_write is not None:
            self.on_write()
        self.puts += 1
        if self.fail_put_at is not None and self.puts >= self.fail_put_at:
            raise OSError("storage went away")
        self.objects[(Bucket, Key)] = bytes(Body)

    def list_objects_v2(self, Bucket, Prefix, **_):  # noqa: N803
        keys = sorted(key for bucket, key in self.objects if bucket == Bucket and key.startswith(Prefix))
        return {"Contents": [{"Key": key} for key in keys], "IsTruncated": False}

    def get_object(self, Bucket, Key):  # noqa: N803
        return {"Body": _Body(self.objects[(Bucket, Key)])}

    def delete_objects(self, Bucket, Delete):  # noqa: N803
        if self.on_write is not None:
            self.on_write()
        for item in Delete["Objects"]:
            self.objects.pop((Bucket, item["Key"]), None)
        return {"Deleted": Delete["Objects"]}

    def keys(self, prefix=""):
        return sorted(key for _, key in self.objects if key.startswith(prefix))


class Runner:
    """``run_async``: ``asyncio.run``, recording the name of every coroutine it runs."""

    def __init__(self):
        self.ran = []

    def __call__(self, coroutine):
        self.ran.append(coroutine.__name__)
        return asyncio.run(coroutine)

    def kinds(self):
        """The coroutines it ran, each name once, in the order first run."""
        return list(dict.fromkeys(self.ran))



def _no_model(*_args, **_kwargs):
    raise AssertionError("a Sync never calls a model")


def refuse_models(monkeypatch):
    """The LLM scan and ``PluginScanService.scan_plugin`` fail any test that reaches them."""
    monkeypatch.setattr(plugin_security_scanner, "llm_security_scan", _no_model)
    monkeypatch.setattr(plugin_security_scanner.PluginScanService, "scan_plugin", _no_model)


def cloud_settings(monkeypatch):
    """The local edition syncing from ``CLOUD``, with the wave's default caps and no plugin refusal remembered
    from an earlier test; Automatos Cloud, mocked."""
    plugin_import.forget_refused()
    monkeypatch.setattr(config, "AUTH_EDITION", "local")
    monkeypatch.setattr(config, "AUTOMATOS_CLOUD_URL", CLOUD)
    monkeypatch.setattr(config, "AUTOMATOS_CLOUD_BUNDLE_MAX_KB", 256)
    monkeypatch.setattr(config, "AUTOMATOS_CLOUD_TIMEOUT_SECONDS", 15)
    monkeypatch.setattr(config, "AUTOMATOS_CLOUD_PLUGINS_PER_SYNC", 5)
    monkeypatch.setattr(config, "PLUGIN_MAX_UPLOAD_SIZE_MB", 10)
    return Cloud()


def fake_bucket(monkeypatch):
    """A fake S3 client for ``MarketplaceS3Service`` (and so the materialiser too)."""
    fake = Bucket()
    monkeypatch.setattr(marketplace_s3, "get_s3_client", lambda *_args, **_kwargs: fake)
    monkeypatch.setattr(marketplace_s3, "ensure_bucket", lambda *_args, **_kwargs: True)
    return fake
