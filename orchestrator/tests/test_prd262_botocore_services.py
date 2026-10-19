"""PRD-262 US-021: botocore ships only the services we call.

pybuild deletes every service model under ``botocore/data`` except s3, s3vectors,
bedrock-runtime, sts, sso and sso-oidc, and keeps the top-level files. A client for any
other service then raises UnknownServiceError in the image, which the backend shards
would never see: they install the full botocore on the runner. This suite checks:

- the Dockerfile: the prune RUN sits in pybuild, after US-020's strip and before the
  base stage copies /install, and keeps exactly the six. botocore is in the second half
  of US-022's two-layer install, /install/rest, which the prune names. Run here with
  ``sh`` on a fake ``botocore/data``, it keeps those six and the top-level files, deletes every other
  service (names that share a prefix with a kept one included), and fails before it
  deletes anything when a kept service is missing;
- the guard for future code and botocore bumps: every boto3 client or resource built
  anywhere in orchestrator/ names one of the six, as a string literal;
- scripts/ci/image_botocore.py, which CI runs inside the image: its list is the
  Dockerfile's, the botocore the runner installed has its top-level files and kept
  services, it builds each kept client offline (the s3 one presigning with SigV4, as the
  app's does), and it reports a leftover, a missing service and a client that fails;
- both workflows run it inside the image they built.

File reads, ASTs, a fake tree in tmp_path and offline boto3 clients; no Docker, no network.
"""
from __future__ import annotations

import ast
import fnmatch
import functools
import importlib.util
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, FrozenSet, List, NamedTuple, Optional

import boto3
import botocore
import pytest
import yaml
from botocore.config import Config
from botocore.exceptions import UnknownServiceError

ROOT = Path(__file__).resolve().parents[2]
ORCHESTRATOR = ROOT / "orchestrator"
WORKFLOWS = ROOT / ".github" / "workflows"
DOCKERFILE = (ORCHESTRATOR / "Dockerfile").read_text(encoding="utf-8")
SCRIPT = ROOT / "scripts" / "ci" / "image_botocore.py"
SCRIPT_RUN = "docker run --rm -i --entrypoint python parity-be - < scripts/ci/image_botocore.py"
PRD_SERVICES = frozenset({"s3", "s3vectors", "bedrock-runtime", "sts", "sso", "sso-oidc"})
# PRD-262 US-022: botocore is outside the lock's stable set, so it installs into the rest half.
DATA = "/install/rest/lib/python3.11/site-packages/botocore/data"
_CONTINUATION = re.compile(r"\\[ \t]*\n")
_KEEP = re.compile(r'\bkeep="([^"]+)"')


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


image_botocore = _load("image_botocore")
_split_spec = importlib.util.spec_from_file_location("split_lock", ORCHESTRATOR / "scripts" / "split_lock.py")
split_lock = importlib.util.module_from_spec(_split_spec)
_split_spec.loader.exec_module(split_lock)


# --------------------------------------------------------------------------- the Dockerfile


def _instructions(text: str) -> List[str]:
    """Instructions with continuation lines joined, comment lines dropped, spaces collapsed."""
    code = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    return [" ".join(line.split()) for line in _CONTINUATION.sub(" ", code).splitlines() if line.strip()]


def _pybuild() -> List[str]:
    """pybuild's instructions, from its FROM up to the next FROM."""
    instructions = _instructions(DOCKERFILE)
    start = next(i for i, line in enumerate(instructions) if re.match(r"FROM\s.*\sas\s+pybuild$", line, re.I))
    end = next(i for i, line in enumerate(instructions) if i > start and line.upper().startswith("FROM "))
    return instructions[start:end]


def _prune_run() -> str:
    runs = [line for line in _instructions(DOCKERFILE) if "botocore/data" in line]
    assert len(runs) == 1, f"one RUN prunes botocore/data, found {len(runs)}"
    return runs[0]


def _kept() -> FrozenSet[str]:
    match = _KEEP.search(_prune_run())
    assert match, "the prune names its services in keep=\"...\""
    return frozenset(match.group(1).split())


def test_the_dockerfile_keeps_exactly_the_services_the_prd_names():
    assert _kept() == PRD_SERVICES
    assert set(image_botocore.KEPT_SERVICES) == PRD_SERVICES, "the in-image check checks the Dockerfile's list"
    assert len(image_botocore.KEPT_SERVICES) == len(PRD_SERVICES)


def test_the_prune_runs_in_pybuild_after_the_strip_and_before_the_base_copies_it():
    stage = _pybuild()
    install = next(i for i, line in enumerate(stage) if "pip install" in line)
    strip = next(i for i, line in enumerate(stage) if "strip --strip-debug" in line)
    prune = next(i for i, line in enumerate(stage) if "botocore/data" in line)
    assert install < strip < prune
    instructions = _instructions(DOCKERFILE)
    assert instructions.index(_prune_run()) < instructions.index("COPY --from=pybuild /install/rest /usr/local")
    assert "botocore" not in split_lock.STABLE, "the prune names the half botocore installs into"
    run = _prune_run()
    assert run.startswith("RUN ") and not run.startswith("RUN --mount"), "Railway builds this file: no cache mounts"
    assert "|| true" not in run and f"data={DATA}" in run


def _fake_data(root: Path, services, extra=("ec2", "dynamodb", "s3control", "sso-admin", "stsx",
                                              "bedrock-agent-runtime", "s3vectors-beta")) -> Path:
    """A botocore/data under ``root`` with one model per service and botocore's top-level files."""
    data = root / DATA.lstrip("/")
    for name in (*services, *extra):
        (data / name / "2020-01-01").mkdir(parents=True)
        (data / name / "2020-01-01" / "service-2.json.gz").write_bytes(name.encode())
    for name in image_botocore.TOP_LEVEL_FILES:
        (data / name).write_text(f'{{"name": "{name}"}}', encoding="utf-8")
    return data


def _run_prune(root: Path) -> subprocess.CompletedProcess:
    run = _prune_run()
    assert run.count("/install/") == 1, "the prune names /install once, so the test can point it at a fake tree"
    script = run.removeprefix("RUN ").replace("/install/", f"{root}/install/")
    return subprocess.run(["sh", "-c", script], capture_output=True, text=True, timeout=60, check=False)


@pytest.mark.skipif(shutil.which("sh") is None, reason="needs sh, as the Docker build has")
def test_the_prune_keeps_the_kept_services_and_the_top_level_files_and_nothing_else(tmp_path):
    data = _fake_data(tmp_path, sorted(PRD_SERVICES))

    done = _run_prune(tmp_path)

    assert done.returncode == 0, done.stderr
    assert sorted(path.name for path in data.iterdir() if path.is_dir()) == sorted(PRD_SERVICES)
    assert sorted(path.name for path in data.iterdir() if path.is_file()) == sorted(image_botocore.TOP_LEVEL_FILES)
    for name in PRD_SERVICES:
        assert (data / name / "2020-01-01" / "service-2.json.gz").read_bytes() == name.encode()
    assert image_botocore.data_problems(data) == []


@pytest.mark.skipif(shutil.which("sh") is None, reason="needs sh, as the Docker build has")
def test_a_missing_kept_service_fails_the_build_before_anything_goes(tmp_path):
    data = _fake_data(tmp_path, sorted(PRD_SERVICES - {"sso-oidc"}))

    done = _run_prune(tmp_path)

    assert done.returncode != 0
    assert "botocore/data has no sso-oidc" in done.stderr
    assert (data / "ec2").is_dir() and (data / "dynamodb").is_dir(), "nothing is deleted when the check fails"


@pytest.mark.skipif(shutil.which("sh") is None, reason="needs sh, as the Docker build has")
def test_no_botocore_at_all_fails_the_build(tmp_path):
    assert _run_prune(tmp_path).returncode != 0


# --------------------------------------------------------------------------- the guard on the code


class BotoCall(NamedTuple):
    line: int
    factory: str  # client or resource
    service: Optional[str]  # None when the service is not a string literal


FACTORIES = frozenset({"client", "resource"})
MODULE_FACTORIES = frozenset({"boto3.client", "boto3.resource"})
SESSIONS = frozenset({"boto3.Session", "boto3.session.Session"})
SKIP_PARTS = frozenset({"__pycache__", "node_modules", ".venv", "venv"})


def _bindings(node: ast.AST) -> List[tuple]:
    """(local name, dotted path) for what one import statement binds from boto3."""
    if isinstance(node, ast.Import):
        return [(alias.asname, alias.name) if alias.asname else ("boto3", "boto3")
                for alias in node.names if alias.name.split(".")[0] == "boto3"]
    if isinstance(node, ast.ImportFrom) and not node.level and (node.module or "").split(".")[0] == "boto3":
        return [(alias.asname or alias.name, f"{node.module}.{alias.name}") for alias in node.names]
    return []


def _aliases(tree: ast.AST) -> Dict[str, str]:
    """Each name a module binds to boto3, or to something imported from it, as a dotted path."""
    return dict(binding for node in ast.walk(tree) for binding in _bindings(node))


def _dotted(node: ast.AST, names: Dict[str, str]) -> Optional[str]:
    """``node`` as a dotted path through boto3, aliases resolved; None when it is not one."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name) or node.id not in names:
        return None
    return ".".join([names[node.id], *reversed(parts)])


def _session_targets(tree: ast.AST, names: Dict[str, str]) -> FrozenSet[str]:
    """What a module assigns a boto3 Session to (``session``, ``self._session``), as source text."""
    found = set()
    for node in ast.walk(tree):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        if targets and isinstance(node.value, ast.Call) and _dotted(node.value.func, names) in SESSIONS:
            found.update(ast.unparse(target) for target in targets)
    return frozenset(found)


def _factory(call: ast.Call, names: Dict[str, str], sessions: FrozenSet[str]) -> Optional[str]:
    """``client`` or ``resource`` when ``call`` builds a boto3 one, else None."""
    dotted = _dotted(call.func, names)
    if dotted in MODULE_FACTORIES:
        return dotted.rsplit(".", 1)[1]
    if not isinstance(call.func, ast.Attribute) or call.func.attr not in FACTORIES:
        return None
    receiver = call.func.value
    built_here = isinstance(receiver, ast.Call) and _dotted(receiver.func, names) in SESSIONS
    return call.func.attr if built_here or ast.unparse(receiver) in sessions else None


def _service(call: ast.Call) -> Optional[str]:
    """The service a client call names: its first argument or ``service_name=``, when a literal."""
    keyword = next((kw.value for kw in call.keywords if kw.arg == "service_name"), None)
    node = call.args[0] if call.args else keyword
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def boto_calls(source: str) -> List[BotoCall]:
    """Every boto3 client or resource ``source`` builds, with the service it names."""
    tree = ast.parse(source)
    names = _aliases(tree)
    if not names:
        return []
    sessions = _session_targets(tree, names)
    calls = (BotoCall(node.lineno, factory, _service(node)) for node in ast.walk(tree)
             if isinstance(node, ast.Call) and (factory := _factory(node, names, sessions)))
    return sorted(calls)


@functools.lru_cache(maxsize=1)
def _orchestrator_calls() -> Dict[str, List[BotoCall]]:
    """The boto3 clients and resources every Python file under orchestrator/ builds, by file."""
    found = {}
    for path in sorted(ORCHESTRATOR.rglob("*.py")):
        relative = path.relative_to(ORCHESTRATOR)
        text = "" if SKIP_PARTS.intersection(relative.parts) else path.read_text(encoding="utf-8", errors="replace")
        calls = boto_calls(text) if "boto3" in text else []
        if calls:
            found[relative.as_posix()] = calls
    return found


def test_every_boto3_client_in_orchestrator_names_a_kept_service():
    """A client outside the list raises UnknownServiceError in the image (the shards can't see it)."""
    offenders = [f"orchestrator/{path}:{call.line} {call.factory}({call.service or '<not a string literal>'})"
                 for path, calls in _orchestrator_calls().items() for call in calls if call.service not in _kept()]
    assert offenders == [], (
        "the API image's botocore has the models of these services only: "
        f"{sorted(_kept())}. Name the service as a literal, or add it to the keep list of the "
        f"botocore prune in orchestrator/Dockerfile and to scripts/ci/image_botocore.py: {offenders}")


def test_the_walk_finds_the_clients_the_app_builds():
    found = _orchestrator_calls()
    services = {call.service for calls in found.values() for call in calls}
    assert {"s3", "s3vectors", "bedrock-runtime"} <= services, services
    assert [call.service for call in found["core/storage/s3.py"]] == ["s3"], "the storage factory's client"
    assert {call.service for call in found["core/llm/clients/bedrock_client.py"]} == {"bedrock-runtime"}


@pytest.mark.parametrize("source,expected", [
    ('import boto3\nboto3.client("ec2", region_name="x")', [("client", "ec2")]),
    ('import boto3\nboto3.client(service_name="dynamodb")', [("client", "dynamodb")]),
    ('import boto3\nboto3.resource("sqs")', [("resource", "sqs")]),
    ('import boto3 as b3\nb3.client("lambda")', [("client", "lambda")]),
    ('from boto3 import client\nclient("sns")', [("client", "sns")]),
    ('from boto3 import resource as make\nmake("s3")', [("resource", "s3")]),
    ('import boto3\nboto3.Session(profile_name="p").client("kms")', [("client", "kms")]),
    ('import boto3.session\ns = boto3.session.Session()\ns.client("ses")', [("client", "ses")]),
    ('from boto3.session import Session\nclass A:\n    def f(self):\n        self._s = Session()\n'
     '        return self._s.resource("iam")', [("resource", "iam")]),
    ('def f(name):\n    import boto3\n    return boto3.client(name)', [("client", None)]),
    ('import boto3\nboto3.client(service_name=SERVICE)', [("client", None)]),
])
def test_the_guard_reads_every_way_code_builds_a_client(source, expected):
    assert [(call.factory, call.service) for call in boto_calls(source)] == expected


@pytest.mark.parametrize("source", [
    'import httpx\nhttpx.client("ec2")',
    'import boto3\nclass A:\n    def f(self):\n        return self.client("ec2")',
    'import boto3\ndef spy(monkeypatch, fake):\n    monkeypatch.setattr(boto3, "client", fake)',
    'import openai\nsession = openai.Session()\nsession.client("x")',
    'boto3 = None  # a name, not the module\nboto3.client("ec2")',
])
def test_the_guard_ignores_what_is_not_a_boto3_client(source):
    assert boto_calls(source) == []


# --------------------------------------------------------------------------- the in-image check


def test_the_check_wants_exactly_the_kept_directories_and_the_top_level_files(tmp_path):
    data = _fake_data(tmp_path, sorted(PRD_SERVICES), extra=("ec2",))
    shutil.rmtree(data / "sso")
    (data / "endpoints.json").unlink()

    assert image_botocore.data_problems(data) == [
        "botocore/data still has ec2/",
        "botocore/data lost sso/, a service we call",
        "botocore/data lost endpoints.json",
    ]
    assert image_botocore.data_problems(tmp_path / "missing") == [f"{tmp_path / 'missing'} is not a directory"]


def test_the_installed_botocore_has_the_top_level_files_and_every_kept_service():
    """The runner installs the same requirements: a bump that renames or drops one fails here first."""
    data = Path(botocore.__file__).parent / "data"
    assert all((data / name).is_file() for name in image_botocore.TOP_LEVEL_FILES)
    assert all((data / name).is_dir() for name in image_botocore.KEPT_SERVICES)
    assert (data / image_botocore.PRUNED_PROBE).is_dir(), "a probe that is no service would prove nothing"


def test_boto3_builds_each_kept_client_offline_and_the_s3_one_presigns_with_sigv4():
    session = image_botocore._session(boto3)
    s3_config = Config(signature_version=image_botocore.APP_SIGNATURE_VERSION)
    assert image_botocore.client_problems(session, image_botocore.KEPT_SERVICES, s3_config) == []
    # The full botocore on the runner still has the probe's model, which the image must not.
    assert image_botocore.pruned_problems(session, UnknownServiceError) == [
        f"boto3 still builds a {image_botocore.PRUNED_PROBE} client: the prune left its model"]


def test_the_check_signs_like_the_app():
    from core.storage import s3 as storage

    assert image_botocore.APP_SIGNATURE_VERSION == storage.SIGNATURE_VERSION


class _FakeSession:
    """Builds a stand-in client, or raises ``errors[name]``; presigns ``url``."""

    def __init__(self, errors=None, url="https://probe/?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Signature=1"):
        self.errors, self.url = errors or {}, url

    def client(self, name, config=None):
        if name in self.errors:
            raise self.errors[name]
        meta = SimpleNamespace(service_model=SimpleNamespace(api_version="2020-01-01"), endpoint_url="https://e")
        return SimpleNamespace(meta=meta, generate_presigned_url=lambda *args, **kwargs: self.url)


def test_the_check_names_the_client_that_fails_and_a_presign_without_sigv4():
    session = _FakeSession({"sso": RuntimeError("no model\nfor sso")})
    assert image_botocore.client_problems(session, ("s3", "sso"), None) == [
        "boto3 can't build a sso client: RuntimeError: no model for sso"]

    sigv2 = _FakeSession(url="https://probe/?AWSAccessKeyId=probe&Signature=x&Expires=1")
    assert image_botocore.client_problems(sigv2, ("s3",), None) == [
        "the s3 client presigned https://probe/?AWSAccessKeyId=probe&Signature=x&Expires=1 with no signature"]


def test_the_check_wants_the_pruned_service_unknown():
    probe = image_botocore.PRUNED_PROBE
    unknown = UnknownServiceError(service_name=probe, known_service_names="s3")
    assert image_botocore.pruned_problems(_FakeSession({probe: unknown}), UnknownServiceError) == []
    assert image_botocore.pruned_problems(_FakeSession({probe: ValueError("bad")}), UnknownServiceError) == [
        f"a {probe} client failed with ValueError: bad, not UnknownServiceError"]


def test_the_script_loads_on_a_bare_python_and_imports_boto3_only_in_main():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    top = {alias.name.split(".")[0] for node in tree.body if isinstance(node, ast.Import) for alias in node.names}
    top |= {node.module.split(".")[0] for node in tree.body if isinstance(node, ast.ImportFrom) and node.module}
    assert top - {"__future__"} <= set(sys.stdlib_module_names), top
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    inside = {alias.name for node in ast.walk(main) if isinstance(node, ast.Import) for alias in node.names}
    assert {"boto3", "botocore"} <= inside


# --------------------------------------------------------------------------- CI runs it inside the image


def _workflow(name: str) -> dict:
    return yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))


def test_prod_image_parity_runs_it_inside_the_railway_build():
    steps = _workflow("test.yml")["jobs"]["prod-image-parity"]["steps"]
    built = next(i for i, step in enumerate(steps) if "-t parity-be orchestrator/" in step.get("run", ""))
    check = next(i for i, step in enumerate(steps) if "image_botocore.py" in step.get("run", ""))
    assert built < check
    assert steps[check]["run"].strip() == SCRIPT_RUN
    assert "if" not in steps[check] and "continue-on-error" not in steps[check]


def test_images_yml_runs_it_inside_every_api_build_on_both_arches():
    workflow = _workflow("images.yml")
    steps = workflow["jobs"]["build"]["steps"]
    built = next(i for i, step in enumerate(steps) if step.get("id") == "build")
    check = next(i for i, step in enumerate(steps) if "image_botocore.py" in step.get("run", ""))
    step = steps[check]
    assert built < check
    assert "matrix.name == 'api'" in step["if"] and "steps.build.outcome == 'success'" in step["if"]
    assert "!cancelled()" in step["if"], "a failed check before it does not hide this one"
    assert step["env"] == {"IMAGE": "${{ matrix.image }}", "DIGEST": "${{ steps.build.outputs.digest }}"}
    assert 'docker pull --quiet "${IMAGE}@${DIGEST}"' in step["run"]
    assert 'docker run --rm -i --entrypoint python "${IMAGE}@${DIGEST}" - < scripts/ci/image_botocore.py' in step["run"]
    assert "continue-on-error" not in step
    pick = next(step for step in workflow["jobs"]["changes"]["steps"] if step.get("id") == "pick")
    api = next(image for image in json.loads(pick["env"]["IMAGES"]) if image["name"] == "api")
    assert api["arches"] == ["amd64", "arm64"]


def test_a_change_to_the_script_reruns_both_checks():
    relative = SCRIPT.relative_to(ROOT).as_posix()
    changes = _workflow("test.yml")["jobs"]["changes"]["steps"]
    match = next(step for step in changes if "PROD_IMAGE_PATHS" in (step.get("env") or {}))
    assert re.search(match["env"]["PROD_IMAGE_PATHS"], relative)
    images = _workflow("images.yml")
    paths = images.get("on", images.get(True))["pull_request"]["paths"]  # PyYAML reads `on` as True
    assert any(fnmatch.fnmatch(relative, path) for path in paths)
