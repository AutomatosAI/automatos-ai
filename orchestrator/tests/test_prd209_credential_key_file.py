"""PRD-209 — the auto-generated credential-encryption key lives where config says.

Live-test finding (2026-08-29): with CREDENTIAL_ENCRYPTION_KEY unset the backend
auto-generates a Fernet key and wrote it to core/.credential_key INSIDE the image's
writable layer — gone on the next container recreate, taking every stored API key
with it (undecryptable). config.CREDENTIAL_KEY_FILE now points the local edition at
a named volume (envs/api.defaults + compose `backend_data`). Pure test: no server.
"""
from __future__ import annotations

import importlib
import pathlib

import pytest


def _fresh_encryption_module():
    import core.credentials.encryption as enc
    return importlib.reload(enc)


def test_key_file_path_is_taken_from_config(tmp_path, monkeypatch):
    from config import config as app_config

    target = tmp_path / "data" / ".credential_key"
    monkeypatch.setattr(app_config, "CREDENTIAL_ENCRYPTION_KEY", None, raising=False)
    monkeypatch.setattr(app_config, "CREDENTIAL_KEY_FILE", str(target), raising=False)
    enc = _fresh_encryption_module()
    cls = next(v for v in vars(enc).values() if isinstance(v, type) and "Encrypt" in v.__name__ and hasattr(v, "_generate_and_save_key"))
    cls()  # constructing resolves/creates the key
    assert target.exists(), "key must be generated at config.CREDENTIAL_KEY_FILE"
    assert target.parent.is_dir(), "parent dir must be created (a named volume mount point)"


def test_local_defaults_and_compose_persist_the_key():
    repo = pathlib.Path(__file__).resolve().parents[2]
    defaults = (repo / "envs" / "api.defaults").read_text(encoding="utf-8")
    compose = (repo / "docker-compose.yml").read_text(encoding="utf-8")
    assert "CREDENTIAL_KEY_FILE=/app/data/.credential_key" in defaults
    assert "backend_data:/app/data" in compose, "compose must mount the backend_data volume at /app/data"
    assert "\n  backend_data:" in compose, "compose must declare the backend_data named volume"
