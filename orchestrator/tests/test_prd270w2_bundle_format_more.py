"""PRD-270 W2 US-270-11c (K8, K10) — the item bundle format for packages and plugins; the local version.

* ``package.json`` and ``plugin.json`` validate and are canonical, and every case of the package
  and plugin sections of ``cases.json`` gets its code (the corpus helpers are US-270-11b's).
* A package member's type says which references it takes (refs.MEMBER_REFS, the K8 table): every
  kind is tried on every type. The member types are core's (marketplace_packages.MEMBER_TYPES).
* A package's setup objects take only their list's keys, and every seeded package's setup manifest
  and matching fit them.
* A plugin's scan report: the verdict is 'safe' only, the rest says what the scanner says of a plugin
  it calls safe, and a finding carries no file content (no matched_text, pattern or evidence).
* ``local_version`` is '<version>+<the revision's first 12>', at most 45 characters.
* No spec, at any depth, has a field whose name holds id, email, workspace, owner, token, secret,
  password, credential, webhook, s3 or path, save the K8 fields named below, each at its one place,
  with the reason it carries none of them. A map's names (a Playbook's input names, its agents'
  local ids) are data on a pattern, not fields, so the walk does not see them.
"""
from __future__ import annotations

import ast
import copy
import json
import re
from typing import Any, Dict, Iterator, List, Mapping, Tuple

import pytest

# US-270-11b's corpus helpers: the case builder, the code a document gets, the fixtures.
from tests.test_prd270w2_bundle_format import CASES, FIXTURES, ORCH_ROOT, _code, _document, _error, _fixture

from services.marketplace_bundles import refs, schema
from services.marketplace_bundles.canonical import canonical_bytes, local_version, revision_of
from services.marketplace_bundles.errors import BUNDLE_ERROR_CODES
from services.marketplace_bundles.validate import BundleError, validate_bundle

SECTIONS = ("package", "plugin")
PACKAGE_MODEL = ORCH_ROOT / "core" / "models" / "marketplace_packages.py"
# The caps the package and plugin specs (and the references their members take) sit on.
CAPS = (
    "PACKAGE_SLUG_MAX", "VERTICAL_TAGS_MAX", "TAG_MAX", "MATCHING_ITEMS_MAX", "MATCHING_TEXT_MAX", "MEMBERS_MAX",
    "NAME_MAX", "DESCRIPTION_MAX", "SETUP_LIST_MAX", "SCALAR_TEXT_MAX", "SCALAR_LIST_MAX",
    "SCALAR_NUMBER_MAX", "AGENT_SLUG_MAX", "TEMPLATE_ID_MAX", "EXTERNAL_ID_LENGTH", "REF_NAME_MAX",
    "PLUGIN_SLUG_MAX", "MODEL_ID_MAX", "APP_MAX", "LONG_DESCRIPTION_MAX", "AUTHOR_MAX", "LICENSE_MAX",
    "CONTENT_COUNT_MAX", "RECOMMENDED_MODELS_MAX", "TOKEN_ESTIMATE_MAX", "FINDINGS_MAX", "FINDING_TYPE_MAX",
    "FINDING_FILE_MAX", "FINDING_LINE_MAX", "FINDING_DESCRIPTION_MAX", "SAFE_RISK_SCORE_MAX", "SCAN_SUMMARY_MAX",
    "SCANNED_AT_MAX",
)
# A field name holding any of these (case-insensitive) could carry what never leaves Automatos Cloud.
FORBIDDEN_PARTS = ("id", "email", "workspace", "owner", "token", "secret", "password", "credential",
                   "webhook", "s3", "path")
# K8's own fields that hold one of them, by their one place (the type, then the field path), and why
# each carries none of what the list guards against.
K8_FIELDS_HOLDING_A_PART = {
    **{f"{kind}.externalId": "K10's opaque hash of an item's lineage, never an id a route takes"
       for kind in schema.BUNDLE_TYPES},
    "agent.agent.model.modelId": "a model's public id, such as test-vendor/test-model-1",
    "agent.agent.model.provider": "a model's provider; 'id' by spelling only",
    "playbook.playbook.steps[].step_id": "a step's own name inside its Playbook (core.py PLAYBOOK_STEP_FIELDS)",
    "package.package.setup.guide_steps": "a package's setup guide; 'id' by spelling only",
    "package.package.setup.questions[].id": "a setup question's own name, as the seeds write it",
    "plugin.plugin.tokenEstimate": "a count of model tokens, not a credential",
    "ref.templateId": "a reference to a seeded Playbook by its public template id",
    "ref.modelId": "a reference to a model by its public id",
}
# One target of each reference kind, in a form every field taking that kind accepts.
TARGETS = {
    "slug": "test-research-kit", "templateId": "test-seeded-playbook", "item": "0123456789abcdef0123456789abcdef",
    "name": "Test Name", "modelId": "test-vendor/test-model-1", "app": "SLACK",
}
MARKER = "zq-marker-7f3a"


def _cases() -> List[Any]:
    return [pytest.param(section, case, id=f"{section}: {case['name']}")
            for section in SECTIONS for case in CASES[section]["cases"]]


def _field_paths(spec: Mapping[str, Any], path: str) -> Iterator[Tuple[str, str]]:
    """(path, name) of every key a spec allows, at any depth, in every variant of a tagged object."""
    kind = spec["kind"]
    if kind == "object":
        for name, field in spec["fields"].items():
            yield f"{path}.{name}", name
            yield from _field_paths(field, f"{path}.{name}")
    elif kind == "tagged":
        for tag, variant in spec["variants"].items():
            yield from _field_paths(variant, f"{path}<{tag}>")
    elif kind == "list":
        yield from _field_paths(spec["items"], f"{path}[]")
    elif kind == "map":
        yield from _field_paths(spec["values"], f"{path}.*")


def _all_fields() -> Dict[str, str]:
    """path -> name of every type's fields, and of every key a reference can carry."""
    fields = {}
    for type_name, spec in schema.TYPE_SPECS.items():
        fields.update(_field_paths(spec, type_name))
    fields.update({f"ref.{kind}": kind for kind in refs.REF_KINDS})
    for field, qualifiers in refs.REF_QUALIFIERS.items():
        fields.update({f"ref({field}).{key}": key for key in qualifiers})
    return fields


def _fields(spec: Mapping[str, Any]) -> Mapping[str, Any]:
    return spec["fields"]


@pytest.mark.parametrize("name", SECTIONS)
def test_the_two_fixtures_validate_and_are_canonical(name):
    raw = (FIXTURES / f"{name}.json").read_bytes()
    bundle = json.loads(raw)
    assert validate_bundle(bundle) is bundle
    assert canonical_bytes(bundle) == raw.rstrip(b"\n")
    assert bundle["type"] == name and bundle["publisher"] == {"name": "Test Publisher"}


@pytest.mark.parametrize("section, case", _cases())
def test_every_new_case_gets_its_code(section, case):
    assert _code(_document(section, case), case.get("maxBytes")) == case["code"]


def test_every_code_and_every_cap_of_the_new_specs_has_a_case():
    cases = [case for section in SECTIONS for case in CASES[section]["cases"]]
    assert {case["code"] for case in cases} >= set(BUNDLE_ERROR_CODES) | {None}
    caps = {case["cap"] for case in cases if "cap" in case}
    assert caps >= set(CAPS), sorted(set(CAPS) - caps)
    assert all(isinstance(getattr(schema, cap, None), int) for cap in caps)
    for section in SECTIONS:
        names = [case["name"] for case in CASES[section]["cases"]]
        assert len(names) == len(set(names)), section
        assert CASES[section]["base"] == f"{section}.json"


def test_the_package_and_plugin_specs_are_k8s():
    package = _fields(_fields(schema.TYPE_SPECS["package"])["package"])
    assert set(package) == {"slug", "verticalTags", "matching", "members", "setup"}
    assert set(_fields(package["matching"])) == {"platforms", "url_patterns", "vocabulary"}
    assert set(_fields(package["setup"])) == {"questions", "required_connects", "guide_steps", "report_templates"}
    members = package["members"]["items"]
    assert members["kind"] == "tagged" and members["key"] == "type"
    for member_type, variant in members["variants"].items():
        assert set(_fields(variant)) == {"type", "ref", "name", "description"}
        assert _fields(variant)["ref"]["field"] == schema.member_ref_field(member_type)
    plugin = _fields(_fields(schema.TYPE_SPECS["plugin"])["plugin"])
    assert set(plugin) == {"slug", "longDescription", "author", "license", "contents", "recommendedModels",
                           "tokenEstimate", "scan"}
    assert set(_fields(plugin["contents"])) == {"skills", "commands", "agents", "hooks"}
    assert schema.BUNDLE_TYPES == ("agent", "playbook", "package", "plugin")


def _core_member_types() -> tuple:
    """marketplace_packages.MEMBER_TYPES, read from the model's source (no SQLAlchemy import)."""
    for node in ast.walk(ast.parse(PACKAGE_MODEL.read_text(encoding="utf-8"))):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == "MEMBER_TYPES" for target in targets):
                return ast.literal_eval(node.value)
    raise AssertionError("MEMBER_TYPES not found in the package model")


def test_the_member_types_are_cores_and_each_has_its_kinds():
    assert schema.MEMBER_TYPES == _core_member_types()
    assert set(refs.MEMBER_REFS) == set(schema.MEMBER_TYPES)
    members = _fields(_fields(schema.TYPE_SPECS["package"])["package"])["members"]["items"]
    assert set(members["variants"]) == set(schema.MEMBER_TYPES)
    assert dict(refs.MEMBER_REFS) == {
        "agent": ("slug", "item"), "playbook": ("templateId", "item"), "skill": ("name",),
        "plugin": ("slug",), "llm": ("modelId",), "tool": ("app",),
    }


@pytest.mark.parametrize("member_type", schema.MEMBER_TYPES)
@pytest.mark.parametrize("kind", sorted(refs.REF_KINDS))
def test_a_member_takes_only_the_kinds_of_its_type(member_type, kind):
    document = _fixture("package")
    member = {"type": member_type, "ref": {kind: TARGETS[kind]}, "name": "Test Member", "description": ""}
    document["package"]["members"] = [member]
    expected = None if kind in refs.MEMBER_REFS[member_type] else "ref"
    assert _code(document) == expected


@pytest.mark.parametrize("member_type", ["theme", "persona", "AGENT", ""])
def test_a_member_of_an_unknown_type_is_refused(member_type):
    document = _fixture("package")
    document["package"]["members"][0]["type"] = member_type
    assert _code(document) == "pattern"


def test_only_a_skill_member_names_its_plugin():
    for index, member in enumerate(_fixture("package")["package"]["members"]):
        document = _fixture("package")
        document["package"]["members"][index]["ref"]["package"] = "test-research-kit"
        assert _code(document) == (None if member["type"] == "skill" else "ref"), member["type"]


@pytest.mark.parametrize("verdict", ["review_required", "blocked", "pending", "SAFE"])
def test_a_scan_verdict_other_than_safe_is_refused(verdict):
    document = _fixture("plugin")
    document["plugin"]["scan"]["verdict"] = verdict
    assert _code(document) == "pattern"


@pytest.mark.parametrize("scan, code", [
    ({"llmStatus": "failed"}, "pattern"), ({"llmStatus": "flagged"}, "pattern"), ({"llmStatus": None}, "type"),
    ({"riskScore": 20}, "too-long"), ({"riskScore": 100}, "too-long"), ({"riskScore": None}, "type"),
    ({"staticFindings": [{"type": "dangerous_call", "severity": "critical", "file": "run.py", "line": 3,
                          "description": "Uses eval()"}]}, "pattern"),
], ids=["llm failed", "llm flagged", "no llm status", "risk 20", "risk 100", "no risk score", "static critical"])
def test_a_safe_report_says_what_the_scanner_says_of_a_safe_plugin(scan, code):
    document = _fixture("plugin")
    document["plugin"]["scan"].update(scan)
    assert _code(document) == code


def test_a_setup_object_takes_only_its_lists_keys():
    keys = {name: set(listed) for name, listed in schema.SETUP_OBJECT_KEYS.items()}
    assert set(keys) == {"questions", "required_connects", "guide_steps", "report_templates"}
    for list_name, allowed in keys.items():
        others = set().union(*keys.values()) - allowed
        for key in sorted(others) + ["api_key", "webhook_url", "workspace_id"]:
            document = _fixture("package")
            document["package"]["setup"][list_name] = [{**_one_setup_object(list_name), key: "x"}]
            assert _code(document) == "unknown-key", (list_name, key)


def _one_setup_object(list_name: str) -> Dict[str, Any]:
    """The smallest object a setup list takes: its required keys only."""
    return {key: "x" for key in schema.SETUP_REQUIRED_KEYS.get(list_name, ())}


def test_every_seeded_package_fits_the_setup_and_matching_specs():
    from core.seeds.seed_packages import PACKAGES  # the seeded definitions (imports the models)

    assert PACKAGES
    for package in PACKAGES:
        document = _fixture("package")
        document["package"]["setup"] = copy.deepcopy(package["setup_manifest"])
        document["package"]["matching"] = copy.deepcopy(package["matching"])
        document["package"]["verticalTags"] = list(package["vertical_tags"])
        assert _code(document) is None, package["slug"]


@pytest.mark.parametrize("key", ["matched_text", "pattern", "evidence"])
@pytest.mark.parametrize("findings", ["staticFindings", "findings"])
def test_a_finding_carries_no_file_content(findings, key):
    scan = _fields(_fields(_fields(schema.TYPE_SPECS["plugin"])["plugin"])["scan"])
    assert set(_fields(scan["staticFindings"]["items"])) == {"type", "severity", "file", "line", "description"}
    assert set(_fields(scan["findings"]["items"])) == {"severity", "description"}
    document = _fixture("plugin")
    document["plugin"]["scan"][findings][0][key] = "open('notes', 'a')"
    assert _code(document) == "unknown-key"


def test_the_local_version_has_its_length_and_shape():
    assert schema.LOCAL_VERSION_MAX == 45 and schema.LOCAL_VERSION_MAX <= 50  # Agent.version is String(50)
    for name in ("agent", "playbook", "package", "plugin"):
        bundle = _fixture(name)
        revision = revision_of(bundle)
        version = local_version(bundle["version"], revision)
        assert version == f"{bundle['version']}+{revision[:12]}"
        assert re.fullmatch(r"[0-9A-Za-z][0-9A-Za-z.-]{0,31}\+[0-9a-f]{12}", version)
    longest = local_version("9" * schema.VERSION_MAX, "f" * 64)
    assert len(longest) == schema.LOCAL_VERSION_MAX


@pytest.mark.parametrize("version, revision", [
    ("", "a" * 64), ("1.0.0+abc", "a" * 64), ("9" * 33, "a" * 64), (None, "a" * 64), ("1.0.0\n", "a" * 64),
    ("1.0.0", "a" * 63), ("1.0.0", "A" * 64), ("1.0.0", "g" * 64), ("1.0.0", None), ("1.0.0", "a" * 65),
])
def test_the_local_version_refuses_what_is_not_a_version_and_a_revision(version, revision):
    with pytest.raises(BundleError) as caught:
        local_version(version, revision)
    assert caught.value.code == "pattern"


def test_no_spec_has_a_field_that_could_carry_what_never_leaves():
    fields = _all_fields()
    names = set(fields.values())
    assert {"verdict", "members", "app_name", "package"} <= names and "matched_text" not in names
    assert set(K8_FIELDS_HOLDING_A_PART) <= set(fields), sorted(set(K8_FIELDS_HOLDING_A_PART) - set(fields))
    for path, name in fields.items():
        if path not in K8_FIELDS_HOLDING_A_PART:
            assert not any(part in name.lower() for part in FORBIDDEN_PARTS), path


def _marked_documents() -> Iterator[Dict[str, Any]]:
    """Package and plugin bundles carrying MARKER where the validator refuses them."""
    edits = (
        ("package", lambda d: d["package"]["members"][0].__setitem__("type", MARKER)),
        ("package", lambda d: d["package"]["members"][0].__setitem__("ref", {MARKER: "x"})),
        ("package", lambda d: d["package"]["members"][4]["ref"].__setitem__("package", MARKER.upper())),
        ("package", lambda d: d["package"]["setup"]["questions"][0].__setitem__(MARKER, "x")),
        ("package", lambda d: d["package"]["members"][5]["ref"].__setitem__("name", f"{MARKER}\n")),
        ("package", lambda d: d["package"].__setitem__("slug", MARKER.upper())),
        ("plugin", lambda d: d["plugin"]["scan"].__setitem__("verdict", MARKER)),
        ("plugin", lambda d: d["plugin"]["scan"]["staticFindings"][0].__setitem__(MARKER, "x")),
        ("plugin", lambda d: d["plugin"]["scan"].__setitem__("scannedAt", MARKER)),
        ("plugin", lambda d: d["plugin"]["scan"].__setitem__("scannedAt", "2026-02-30T09:15:00")),
        ("plugin", lambda d: d["plugin"]["scan"]["staticFindings"][0].__setitem__("file", f"/{MARKER}")),
        ("plugin", lambda d: d["plugin"].__setitem__("author", f"{MARKER}@example.com")),
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
