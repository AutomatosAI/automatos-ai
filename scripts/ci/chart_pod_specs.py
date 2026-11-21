"""Every pod a Helm release creates, for the strict install lane's checks (PRD-280 US-100).

chart.yml's install lane runs it on a release's manifest and hooks, converted to JSON by
``kubectl create --dry-run=client -o json``. Three modes, each reading stdin:

- no argument: one line per pod template, ``<owner name><TAB><Pod as compact JSON>``.
  ``deploy/kind/strict.sh`` creates each Pod with ``kubectl create --dry-run=server`` in a
  namespace that enforces the "restricted" pod security level. Pod Security admission only
  warns on a Deployment or a Job; on a Pod it refuses, and names it. The real install's
  pods are made by controllers, and the migration Job's pod is gone once its hook
  succeeds, so their specs are checked this way. The Pod keeps its template's spec,
  without the service account: the probe namespace has only its ``default`` account, and
  the account plays no part in the pod security level.
- ``read-only-roots``: one line, ``<owner name>/<container>``, for each container or init
  container whose root file system is not read-only; exit 1 if there is one.
- ``violations``: stdin is the API server's refusal of one Pod; one line per violation of
  the level it names, ``<reason> (<detail>)``, as Pod Security writes them.

Input for the first two: JSON documents one after another, each an object or a ``List``
of them (``kubectl create`` prints each object on its own).

Pure: no cluster, no network. Python's standard library only, so it runs on the runner.
"""
from __future__ import annotations

import copy
import json
import re
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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
CONTAINER_FIELDS = ("initContainers", "containers")
# Pod Security's refusal: `violates PodSecurity "<level>:<version>": <reason> (<detail>), ...`.
# A detail holds no parenthesis; a reason starts the message or follows ", ".
VIOLATIONS_START = re.compile(r'violates PodSecurity "[^"]*": ')
VIOLATION = re.compile(r"(?:^|, )([^(),][^()]*?) \(([^()]*)\)")


class InputError(ValueError):
    """kubectl's output is not what the lane expects; the message says where."""


def _mapping(value: Any, where: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise InputError(f"{where} is a JSON {type(value).__name__}, not an object")
    return value


def documents(text: str) -> List[Dict[str, Any]]:
    """The JSON documents kubectl printed, one after another (``create`` prints each object
    on its own; ``apply`` and ``get`` print a List). A document that is not an object is
    refused."""
    decoder = json.JSONDecoder()
    found: List[Dict[str, Any]] = []
    position = 0
    while True:
        while position < len(text) and text[position].isspace():
            position += 1
        if position >= len(text):
            return found
        document, position = decoder.raw_decode(text, position)
        found.append(_mapping(document, f"document {len(found) + 1}"))


def objects(document: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The objects in one printed document: a List's items, or the one object."""
    if document.get("kind") == "List":
        items = document.get("items") or []
        return [_mapping(item, f"List item {index + 1}") for index, item in enumerate(items)]
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


def writable_roots(spec: Dict[str, Any]) -> List[str]:
    """The containers and init containers of a pod spec without a read-only root."""
    return [
        container.get("name", "?")
        for field in CONTAINER_FIELDS
        for container in spec.get(field) or []
        if (container.get("securityContext") or {}).get("readOnlyRootFilesystem") is not True
    ]


def violations(message: str) -> List[str]:
    """Each violation a Pod Security refusal names, ``<reason> (<detail>)``; none when the
    message is not one."""
    start = VIOLATIONS_START.search(message)
    if start is None:
        return []
    listed = message[start.end():].strip()
    return [f"{reason} ({detail})" for reason, detail in VIOLATION.findall(listed)]


def _all_pods(text: str) -> List[Tuple[str, Dict[str, Any]]]:
    pods = [pair for document in documents(text) for pair in probe_pods(document)]
    if not pods:
        raise InputError("the release has no object that makes a pod")
    return pods


def _print_pods(pods: List[Tuple[str, Dict[str, Any]]]) -> int:
    for name, pod in pods:
        sys.stdout.write(f"{name}\t{json.dumps(pod, separators=(',', ':'), sort_keys=True)}\n")
    return 0


def _print_writable_roots(pods: List[Tuple[str, Dict[str, Any]]]) -> int:
    found = [f"{name}/{container}" for name, pod in pods for container in writable_roots(pod["spec"])]
    for line in found:
        sys.stdout.write(f"{line}\n")
    return 1 if found else 0


def main(argv: Sequence[str] = ()) -> int:
    """Run one mode (module docstring) on stdin; 2 for an unknown mode, 1 for bad input."""
    mode = argv[0] if argv else "pods"
    text = sys.stdin.read()
    if mode == "violations":
        for line in violations(text):
            sys.stdout.write(f"{line}\n")
        return 0
    if mode not in ("pods", "read-only-roots"):
        sys.stderr.write(f"chart_pod_specs: unknown mode {mode!r}: pods, read-only-roots or violations\n")
        return 2
    try:
        pods = _all_pods(text)
    except (InputError, json.JSONDecodeError) as error:
        sys.stderr.write(f"chart_pod_specs: {error}\n")
        return 1
    return _print_pods(pods) if mode == "pods" else _print_writable_roots(pods)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
