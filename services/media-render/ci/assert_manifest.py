#!/usr/bin/env python3
"""Assert a rendered file's signed C2PA manifest (PRD-264 EUAI-06b; the media-render CI job).

Stdlib only: it runs on the runner, not in the image. The image's own c2pa reads the file
(``media_render.content_credentials.read_store``) and prints its manifest store as JSON;
this checks it against the render job as GET /render/{id} returned it, and prints one
PASS/FAIL line per expectation so the job log is the evidence:

- the store validates (``Valid``: signed, the per-install or self-signed signer unknown;
  ``Trusted`` once a trust-list certificate signs);
- its active manifest has one ``c2pa.created`` action with the expected digital source type
  and the claim generator "Automatos AI media-render/<version>";
- the job's output names the file's SHA-256 and that manifest's id.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import List

ACTIONS_LABELS = ("c2pa.actions", "c2pa.actions.v2")
GENERATOR = "Automatos AI media-render"
GOOD_STATES = ("Valid", "Trusted")


def _created(manifest: dict) -> List[dict]:
    return [
        action
        for assertion in manifest.get("assertions", [])
        if assertion.get("label") in ACTIONS_LABELS
        for action in assertion.get("data", {}).get("actions", [])
        if action.get("action") == "c2pa.created"
    ]


def check(store: dict, job: dict, media: Path, name: str, expected: str) -> List[str]:
    """Each failed expectation, in words; printing a PASS line for each that holds."""
    manifest = store.get("manifests", {}).get(store.get("active_manifest"), {})
    generator = (manifest.get("claim_generator_info") or [{}])[0]
    (output,) = [entry for entry in job.get("outputs", []) if entry.get("name") == name] or [{}]
    expectations = [
        (f"the store validates ({store.get('validation_state')})", store.get("validation_state") in GOOD_STATES),
        (f"one c2pa.created action, {expected}", [a.get("digitalSourceType") for a in _created(manifest)] == [expected]),
        (f"the claim generator is {GENERATOR}/{generator.get('version')}", generator.get("name") == GENERATOR and bool(generator.get("version"))),
        ("the job names the file's SHA-256", output.get("sha256") == hashlib.sha256(media.read_bytes()).hexdigest()),
        ("the job names the active manifest's id", bool(output.get("manifest_id")) and output.get("manifest_id") == store.get("active_manifest")),
        (f"the job names the digital source type {expected}", output.get("digital_source_type") == expected),
    ]
    failures = []
    for label, held in expectations:
        print(f"{'PASS' if held else 'FAIL'}: {label}")
        if not held:
            failures.append(label)
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("manifest", type=Path, help="the manifest store c2pa printed, as JSON")
    parser.add_argument("--job", type=Path, required=True, help="the render job, as GET /render/{id} returned it")
    parser.add_argument("--file", type=Path, required=True, help="the rendered file")
    parser.add_argument("--name", required=True, help="the output's name in the job")
    parser.add_argument("--expected", required=True, help="the IPTC digital source type")
    args = parser.parse_args()
    store = json.loads(args.manifest.read_text() or "null")
    if not isinstance(store, dict):
        print(f"FAIL: {args.file.name} carries no C2PA manifest")
        return 1
    failures = check(store, json.loads(args.job.read_text()), args.file, args.name, args.expected)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
