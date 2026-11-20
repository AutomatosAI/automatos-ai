"""Every pod a Helm release creates, as a bare Pod, one per line (PRD-280 US-100).

chart.yml's install lane runs it on a release's manifest and hooks, converted to JSON by
``kubectl create --dry-run=client -o json``, and creates each Pod it prints with
``kubectl create --dry-run=server`` in a namespace that enforces the "restricted" pod
security level. Pod Security admission only warns on a Deployment or a Job; on a Pod it
refuses, and names it. The real install's pods are made by controllers, and the migration
Job's pod is gone once its hook succeeds, so their specs are checked this way.

Input on stdin: JSON documents one after another, each an object or a ``List`` of them
(``kubectl create`` prints each object on its own). Output on stdout: one line per pod
template, ``<owner name><TAB><Pod as compact JSON>``. The Pod keeps its template's spec,
without the service account: the probe namespace has only its ``default`` account, and
the account plays no part in the pod security level.

Pure: no cluster, no network. Python's standard library only, so it runs on the runner.
"""
from __future__ import annotations

import copy
import json
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Where each kind keeps its pod template's spec.
TEMPLATE_PATHS: Dict[str, Tuple[str, ...]] = {
    "Pod": ("spec",),
    "Deployment": ("spec", "template", "spec"),
    "StatefulSet": ("spec", "template", "spec"),
    "DaemonSet": ("spec", "template", "spec"),
    "ReplicaSet": ("spec", "template", "spec"),
    "Job": ("spec", "template", "spec"),
    "CronJob": ("spec", "jobTemplate", "spec", "template", "spec"),
}
ACCOUNT_FIELDS = ("serviceAccountName", "serviceAccount")


def documents(text: str) -> List[Dict[str, Any]]:
    """The JSON documents kubectl printed, one after another (``create`` prints each object
    on its own; ``apply`` and ``get`` print a List)."""
    decoder = json.JSONDecoder()
    found: List[Dict[str, Any]] = []
    position = 0
    while True:
        while position < len(text) and text[position].isspace():
            position += 1
        if position >= len(text):
            return found
        document, position = decoder.raw_decode(text, position)
        found.append(document)


def objects(document: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The objects in one printed document: a List's items, or the one object."""
    if document.get("kind") == "List":
        return list(document.get("items") or [])
    return [document]


def pod_spec(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The pod template's spec of an object that makes pods, or None for any other kind."""
    path = TEMPLATE_PATHS.get(obj.get("kind", ""))
    if path is None:
        return None
    node: Any = obj
    for key in path:
        node = (node or {}).get(key)
    return node if isinstance(node, dict) else None


def probe_pod(name: str, spec: Dict[str, Any]) -> Dict[str, Any]:
    """A bare Pod named after its owner, with the template's spec and no service account."""
    kept = {key: value for key, value in copy.deepcopy(spec).items() if key not in ACCOUNT_FIELDS}
    return {"apiVersion": "v1", "kind": "Pod", "metadata": {"name": name}, "spec": kept}


def probe_pods(document: Dict[str, Any]) -> Iterable[Tuple[str, Dict[str, Any]]]:
    """(owner name, Pod) for every object in the document that makes pods."""
    for obj in objects(document):
        spec = pod_spec(obj)
        if spec is not None:
            name = obj["metadata"]["name"]
            yield name, probe_pod(name, spec)


def main() -> int:
    """Read kubectl's JSON on stdin and print one ``name<TAB>pod`` line per pod template."""
    count = 0
    for document in documents(sys.stdin.read()):
        for name, pod in probe_pods(document):
            sys.stdout.write(f"{name}\t{json.dumps(pod, separators=(',', ':'), sort_keys=True)}\n")
            count += 1
    if count == 0:
        sys.stderr.write("chart_pod_specs: the release has no object that makes a pod\n")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
