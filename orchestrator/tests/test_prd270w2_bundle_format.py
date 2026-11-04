"""PRD-270 W2 US-270-11b (K8) — the item bundle format, schema 1: the envelope, agents and Playbooks.

* Every valid fixture in ``tests/fixtures/marketplace_bundles/`` validates and is canonical: its
  bytes are ``canonical_bytes`` of what it parses to, newline aside.
* ``cases.json`` is the corpus (its ``_comment`` says how a case is built): every case of the agent
  and Playbook sections gets its code, and every code and every cap of those specs has a case.
* Strictness is the allowlist: a key outside the spec at the top, inside agent, inside a step and
  inside a step's config is 'unknown-key'; neither spec has a configuration, schedule_config, id,
  email or workspace field; a step's keys are K8's STEP_KEYS plus agent.
* A step agent naming no local id and a reference of the wrong kind are 'ref'; a document over the
  cap is 'too-large'; the revision does not move with key order, and the canonical bytes of a
  value JSON cannot carry raise a BundleError, never another error.
* Messages echo no input text, and the validator imports only the standard library and its own
  package: the hosted export and the local import share it.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple

import pytest

ORCH_ROOT = Path(__file__).resolve().parent.parent
if str(ORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(ORCH_ROOT))

from services.marketplace_bundles import schema  # noqa: E402
from services.marketplace_bundles.canonical import canonical_bytes, capped_bytes, revision_of  # noqa: E402
from services.marketplace_bundles.errors import BUNDLE_ERROR_CODES  # noqa: E402
from services.marketplace_bundles.validate import BundleError, validate_bundle  # noqa: E402

FIXTURES = ORCH_ROOT / "tests" / "fixtures" / "marketplace_bundles"
CASES = json.loads((FIXTURES / "cases.json").read_text(encoding="utf-8"))
SECTIONS = ("agent", "playbook")
PACKAGE = ORCH_ROOT / "services" / "marketplace_bundles"
VALIDATOR_FILES = ("errors.py", "schema.py", "refs.py", "validate.py")
OWN_PACKAGE = "services.marketplace_bundles"
# The caps of the envelope, agent and Playbook specs, and of the references a Playbook's agents take.
CAPS = (
    "EXTERNAL_ID_LENGTH", "VERSION_MAX", "NAME_MAX", "DESCRIPTION_MAX", "CATEGORY_MAX", "ICON_MAX", "TAGS_MAX",
    "TAG_MAX", "PUBLISHER_NAME_MAX", "AGENT_TYPE_MAX", "PROMPT_TEXT_MAX", "PERSONA_SLUG_MAX", "MODEL_ID_MAX",
    "PROVIDER_MAX", "SKILLS_MAX", "SKILL_NAME_MAX", "APPS_MAX", "APP_MAX", "PLUGINS_MAX", "PLUGIN_SLUG_MAX",
    "AGENT_SLUG_MAX", "REF_NAME_MAX", "AGENTS_MAX", "STEPS_MAX", "STEP_TOKEN_MAX", "STEP_ORDER_MAX",
    "PROMPT_TEMPLATE_MAX", "STEP_TEXT_MAX", "STEP_UNTIL_MAX", "STEP_CONFIG_VALUE_MAX", "IO_MAX", "IO_TYPE_MAX",
    "IO_DESCRIPTION_MAX", "SCALAR_TEXT_MAX", "SCALAR_NUMBER_MAX", "SCALAR_LIST_MAX", "EXECUTION_NUMBER_MAX",
)
# Fields no agent or Playbook spec may have (AC: neither has a configuration, schedule_config, id,
# email or workspace field). Exact names, then substrings, case-insensitive.
FORBIDDEN_NAMES = ("configuration", "schedule_config", "id", "agent_id")
FORBIDDEN_PARTS = ("email", "workspace")
MARKER = "zq-marker-7f3a"


def _fixture(name: str) -> Dict[str, Any]:
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


def _expand(value: Any) -> Any:
    """A case value with its fillers ($text, $list, $map) written out."""
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if not isinstance(value, dict):
        return value
    if "$text" in value:
        return "x" * value["$text"]
    if "$list" in value:
        return [_expand(value["of"]) for _ in range(value["$list"])]
    if "$map" in value:
        return {f"{value['prefix']}{index}": _expand(value["of"]) for index in range(value["$map"])}
    return {key: _expand(item) for key, item in value.items()}


def _parent(document: Any, path: str) -> Tuple[Any, Any]:
    """The container a dotted path's last part lives in, and that part (a list position as an int)."""
    parts = path.split(".")
    target = document
    for part in parts[:-1]:
        target = target[int(part)] if isinstance(target, list) else target[part]
    last = parts[-1]
    return target, int(last) if isinstance(target, list) else last


def _document(section: str, case: Mapping[str, Any]) -> Dict[str, Any]:
    """The case's input: its section's base with ``set`` applied and ``unset`` removed."""
    document = json.loads((FIXTURES / CASES[section]["base"]).read_text(encoding="utf-8"))
    for path, value in case.get("set", {}).items():
        target, key = _parent(document, path)
        target[key] = _expand(value)
    for path in case.get("unset", []):
        target, key = _parent(document, path)
        del target[key]
    return document


def _error(document: Any, max_bytes: Optional[int] = None) -> Optional[BundleError]:
    """The BundleError the validator (then the size check, given a cap) raises, or None."""
    try:
        validate_bundle(document)
        if max_bytes is not None:
            capped_bytes(document, max_bytes)
    except BundleError as error:
        return error
    return None


def _code(document: Any, max_bytes: Optional[int] = None) -> Optional[str]:
    error = _error(document, max_bytes)
    return None if error is None else error.code


def _cases() -> List[Any]:
    return [pytest.param(section, case, id=f"{section}: {case['name']}")
            for section in SECTIONS for case in CASES[section]["cases"]]


def _field_names(spec: Mapping[str, Any]) -> Iterator[str]:
    """Every key name an object spec allows, at any depth."""
    kind = spec["kind"]
    if kind == "object":
        for name, field in spec["fields"].items():
            yield name
            yield from _field_names(field)
    elif kind == "list":
        yield from _field_names(spec["items"])
    elif kind == "map":
        yield from _field_names(spec["values"])


@pytest.mark.parametrize("name", SECTIONS)
def test_every_valid_fixture_validates_and_is_canonical(name):
    raw = (FIXTURES / f"{name}.json").read_bytes()
    bundle = json.loads(raw)
    assert validate_bundle(bundle) is bundle
    assert canonical_bytes(bundle) == raw.rstrip(b"\n")
    assert bundle["publisher"] == {"name": "Test Publisher"}


@pytest.mark.parametrize("section, case", _cases())
def test_every_case_gets_its_code(section, case):
    assert _code(_document(section, case), case.get("maxBytes")) == case["code"]


def test_every_code_and_every_cap_has_a_case():
    cases = [case for section in SECTIONS for case in CASES[section]["cases"]]
    assert {case["code"] for case in cases} >= set(BUNDLE_ERROR_CODES) | {None}
    caps = {case["cap"] for case in cases if "cap" in case}
    assert caps >= set(CAPS), sorted(set(CAPS) - caps)
    assert all(isinstance(getattr(schema, cap, None), int) for cap in caps)
    for section in SECTIONS:
        names = [case["name"] for case in CASES[section]["cases"]]
        assert len(names) == len(set(names)), section


@pytest.mark.parametrize("place", [(), ("agent",), ("playbook", "steps", 0), ("playbook", "steps", 1, "config")])
def test_an_extra_key_at_any_depth_is_unknown_key(place):
    document = _fixture("playbook" if place and place[0] == "playbook" else "agent")
    target = document
    for part in place:
        target = target[part]
    target[MARKER] = "x"
    assert _code(document) == "unknown-key"


def test_the_specs_have_only_k8_fields():
    names = set(_field_names(schema.TYPE_SPECS["agent"])) | set(_field_names(schema.TYPE_SPECS["playbook"]))
    for name in names:
        assert name.lower() not in FORBIDDEN_NAMES, name
        assert not any(part in name.lower() for part in FORBIDDEN_PARTS), name
    playbook = schema.TYPE_SPECS["playbook"]["fields"]["playbook"]["fields"]
    step = playbook["steps"]["items"]["fields"]
    assert set(step) == set(schema.STEP_KEYS) | {"agent"}
    assert set(step["config"]["fields"]) == {"title", "format", "template_name", "data"}
    assert set(playbook["execution"]["fields"]) == {
        "mode", "max_retries", "timeout_per_step", "total_timeout", "parallel_limit", "wait_for_me"}
    agent = schema.TYPE_SPECS["agent"]["fields"]["agent"]["fields"]
    assert set(agent) == {"agentType", "prompt", "persona", "model", "skills", "apps", "plugins"}
    assert schema.BUNDLE_SCHEMA == 1 and set(schema.BUNDLE_TYPES) >= {"agent", "playbook"}


@pytest.mark.parametrize("agent", ["a9", "a1 ", "A1", ""])
def test_a_step_agent_naming_no_local_id_is_ref(agent):
    document = _fixture("playbook")
    document["playbook"]["steps"][0]["agent"] = agent
    assert _code(document) == "ref"


@pytest.mark.parametrize("reference", [
    {"templateId": "test-weekly-digest"}, {"modelId": "test-vendor/test-model-1"}, {"app": "SLACK"},
])
def test_a_reference_of_the_wrong_kind_is_ref(reference):
    document = _fixture("playbook")
    document["playbook"]["agents"]["a1"] = reference
    assert _code(document) == "ref"


def test_a_document_over_the_cap_is_too_large():
    bundle = _fixture("agent")
    size = len(canonical_bytes(bundle))
    assert capped_bytes(bundle, size) == canonical_bytes(bundle)
    with pytest.raises(BundleError) as caught:
        capped_bytes(bundle, size - 1)
    assert caught.value.code == "too-large"


def _reordered(value: Any) -> Any:
    """``value`` with every object's keys in reverse order."""
    if isinstance(value, dict):
        return {key: _reordered(value[key]) for key in reversed(list(value))}
    if isinstance(value, list):
        return [_reordered(item) for item in value]
    return value


@pytest.mark.parametrize("name", SECTIONS)
def test_the_revision_is_stable_under_key_order(name):
    bundle = _fixture(name)
    shuffled = _reordered(bundle)
    assert list(shuffled) != list(bundle)
    assert revision_of(shuffled) == revision_of(bundle)
    raw = (FIXTURES / f"{name}.json").read_bytes().rstrip(b"\n")
    assert revision_of(bundle) == hashlib.sha256(raw).hexdigest()
    assert len(revision_of(bundle)) == 64 and revision_of(bundle) == revision_of(bundle).lower()
    changed = copy.deepcopy(bundle)
    changed["description"] += "."
    assert revision_of(changed) != revision_of(bundle)


def _marked_documents() -> Iterator[Any]:
    """Bundles that carry MARKER where the validator refuses them, in every kind of place."""
    edits = (
        ("agent", lambda d: d.__setitem__(MARKER, 1)),
        ("agent", lambda d: d["agent"].__setitem__(MARKER, 1)),
        ("agent", lambda d: d.__setitem__("type", MARKER)),
        ("agent", lambda d: d.__setitem__("version", f"{MARKER} 1")),
        ("agent", lambda d: d.__setitem__("name", MARKER * 30)),
        ("agent", lambda d: d["agent"].__setitem__("apps", [MARKER])),
        ("agent", lambda d: d["agent"].__setitem__("persona", MARKER.upper())),
        ("playbook", lambda d: d["playbook"]["steps"][0].__setitem__("agent", MARKER)),
        ("playbook", lambda d: d["playbook"]["steps"][0].__setitem__("error_handling", MARKER)),
        ("playbook", lambda d: d["playbook"]["agents"].__setitem__("a1", {MARKER: "x"})),
        ("playbook", lambda d: d["playbook"]["agents"].__setitem__("a1", {"slug": MARKER.upper()})),
        ("playbook", lambda d: d["playbook"]["inputs"].__setitem__(MARKER, {"type": "string"})),
        ("playbook", lambda d: d["playbook"]["steps"][1]["config"].__setitem__(MARKER, "x")),
    )
    for name, edit in edits:
        document = _fixture(name)
        edit(document)
        yield document


def test_messages_echo_no_input_text():
    for document in _marked_documents():
        error = _error(document)
        assert error is not None and error.code in BUNDLE_ERROR_CODES
        for text in (error.message, str(error)):
            assert MARKER not in text.lower(), text
            assert "\n" not in text and text.strip()


def test_a_number_that_is_not_finite_is_refused():
    document = _fixture("playbook")
    document["playbook"]["execution"]["total_timeout"] = float("nan")
    assert _code(document) == "type"
    document["playbook"]["execution"]["total_timeout"] = float("inf")
    assert _code(document) == "type"


@pytest.mark.parametrize("value", [float("nan"), "lone \ud800 surrogate"], ids=["nan", "lone surrogate"])
def test_canonical_bytes_refuse_what_json_cannot_carry(value):
    bundle = _fixture("agent")
    bundle["description"] = value
    with pytest.raises(BundleError) as caught:
        canonical_bytes(bundle)
    assert caught.value.code == "type"
    with pytest.raises(BundleError):
        revision_of(bundle)


def test_anything_but_an_object_is_refused():
    for document in (None, [], "bundle", 1, b"{}"):
        assert _code(document) == "type"


def _imported_modules(path: Path) -> Iterator[str]:
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, path.name
            yield node.module


def test_the_validator_imports_only_the_standard_library_and_its_package():
    for name in VALIDATOR_FILES:
        for module in _imported_modules(PACKAGE / name):
            own = module == OWN_PACKAGE or module.startswith(f"{OWN_PACKAGE}.")
            assert own or module.split(".")[0] in sys.stdlib_module_names, (name, module)
    assert (PACKAGE / "__init__.py").read_text(encoding="utf-8") == ""
