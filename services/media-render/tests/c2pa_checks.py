"""Shared by the Content Credentials suites (PRD-264 EUAI-06b): read a file's C2PA manifest
back with c2pa-python (the signer's own SDK), and hold it to what the service puts in.

A manifest this service signs carries one ``c2pa.created`` action with the file's IPTC
digital source type, the claim generator "Automatos AI media-render/<version>", the XMP's
own random instance id, and nothing that names a user, workspace, prompt or path: its keys
are held to an allowlist, every assertion is a ``c2pa.*`` one, and no canary string a test
names appears anywhere in the manifest store.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from media_render import content_credentials
from media_render.versions import SERVICE_VERSION

ACTIONS_LABELS = ("c2pa.actions", "c2pa.actions.v2")
# What c2pa reports about a manifest (c2pa-rs ``Manifest``), less what this service never
# writes: vendor, metadata and credentials, which could carry personal data.
MANIFEST_KEYS = frozenset({
    "claim_generator", "claim_generator_info", "title", "format", "instance_id", "thumbnail",
    "ingredients", "assertions", "redactions", "signature_info", "label", "claim_version",
})
# A self-signed chain reads "Valid" (signed, the signer unknown), a trust-list one "Trusted".
GOOD_STATES = ("Valid", "Trusted")


def store_of(path: Path) -> Dict[str, Any]:
    """The file's manifest store as c2pa reads it; fails when it carries none."""
    mime = content_credentials.mime_type(path)
    assert mime is not None, f"{path.name} is not a format this service signs"
    store = content_credentials.read_store(path, mime)
    assert store is not None, f"{path.name} carries no C2PA manifest"
    return store


def active_manifest(store: Dict[str, Any]) -> Dict[str, Any]:
    return store["manifests"][store["active_manifest"]]


def created_actions(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Every ``c2pa.created`` action in the manifest's actions assertions."""
    actions = [
        action
        for assertion in manifest.get("assertions", [])
        if assertion.get("label") in ACTIONS_LABELS
        for action in assertion.get("data", {}).get("actions", [])
    ]
    return [action for action in actions if action.get("action") == content_credentials.CREATED]


def assert_signed(
    path: Path, expected: str, *, instance_id: Optional[str] = None, absent: Iterable[str] = ()
) -> Dict[str, Any]:
    """``path`` carries a valid manifest this service signed, created as ``expected``; the store."""
    store = store_of(path)
    assert store.get("validation_state") in GOOD_STATES, json.dumps(store.get("validation_results"), indent=2)[:4000]
    manifest = active_manifest(store)
    extra = set(manifest) - MANIFEST_KEYS
    assert not extra, f"manifest fields outside the allowlist: {sorted(extra)}"
    (generator, *_) = manifest["claim_generator_info"]
    assert (generator["name"], generator["version"]) == (content_credentials.CLAIM_GENERATOR_NAME, SERVICE_VERSION)
    assert f"{generator['name']}/{generator['version']}" == content_credentials.claim_generator()
    created = created_actions(manifest)
    assert [action.get("digitalSourceType") for action in created] == [expected], created
    labels = [assertion.get("label", "") for assertion in manifest.get("assertions", [])]
    assert all(label.startswith("c2pa.") for label in labels), labels
    if instance_id is not None:
        assert manifest["instance_id"] == instance_id, "the manifest's instance id is not the XMP's"
    text = json.dumps(store)
    for secret in absent:
        assert secret not in text, f"the manifest store carries {secret!r}"
    return store


def ingredient_labels(manifest: Dict[str, Any]) -> List[Optional[str]]:
    """The manifest each ingredient carried (its label in the store), in order."""
    return [ingredient.get("active_manifest") for ingredient in manifest.get("ingredients", [])]
