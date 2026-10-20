"""PRD-262 US-042: the self-host docs tell the truth about what starts.

``docs/getting-started/self-hosting.md`` §3 lists what runs in three tables: the
default stack, the core override (``docker-compose.core.yml``, US-040) and the
optional profiles. This suite holds them to the compose files:

* ``docker compose config --services``, with no profile, with each documented
  profile, and with the core override alone, with the profile that brings the worker
  back and with each documented profile (where the docker CLI is installed; CI's
  runners always have it, so there a missing CLI fails instead of skipping);
* from the files: every service and profile documented, each container name, image or
  build context, and host port with its variable;
* the rest of US-042's docs: every clone of this repository the self-host docs give is
  ``git clone --depth 1``; §10 says what W5 changes for an existing install; the
  figures for the first download are §8's; nothing describes ``minio-init`` or the old
  3.7 GB of images; ``envs/frontend.defaults`` says the default stack runs the
  production target, whose NODE_ENV compose sets.

No Docker daemon, no network.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest
import yaml

_REPO = Path(__file__).resolve().parents[2]
_SECRETS = {"POSTGRES_PASSWORD": "test", "REDIS_PASSWORD": "test", "API_KEY": "test"}
BASE_FILE = "docker-compose.yml"
CORE_FILE = "docker-compose.core.yml"
GUIDE_PATH = _REPO / "docs" / "getting-started" / "self-hosting.md"
GUIDE = GUIDE_PATH.read_text(encoding="utf-8")
# The self-host docs: what a reader installs and updates from.
SELF_HOST_DOCS = ("README.md", "QUICKSTART.md", "SETUP.md", *sorted(
    path.relative_to(_REPO).as_posix() for path in (_REPO / "docs" / "getting-started").glob("*.md")))

DEFAULT_HEADER = ("Service", "Container", "Image / build", "Host port (variable)", "What it does")
CORE_HEADER = ("Override", "Command", "Leaves out", "Profile that brings it back")
PROFILE_HEADER = ("Profile", *DEFAULT_HEADER)
_DIVIDER = re.compile(r"^\|(?:\s*:?-+:?\s*\|)+\s*$")
_FIRST_CODE = re.compile(r"`([^`]+)`")
_PORT = re.compile(r"(\d+)[^`(,]*\(`([A-Z_]+)`\)")
_CLONE = re.compile(r"git clone(?P<flags>[^\n`]*?)\s+https://github\.com/AutomatosAI/automatos-ai(?:\.git)?\b")


def _compose(name: str) -> dict:
    return yaml.safe_load((_REPO / name).read_text(encoding="utf-8"))["services"]


BASE = _compose(BASE_FILE)
CORE = _compose(CORE_FILE)


def _section(text: str, heading: str) -> str:
    start = text.index(f"\n## {heading}\n")
    end = text.find("\n## ", start + 1)
    return text[start:] if end == -1 else text[start:end]


def _cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _tables(text: str) -> dict[tuple[str, ...], list[dict[str, str]]]:
    """Each markdown table in ``text``, keyed by its header, as rows of {column: cell}."""
    lines = text.splitlines()
    tables: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for index, line in enumerate(lines[:-1]):
        if not (line.startswith("|") and _DIVIDER.match(lines[index + 1])):
            continue
        header = tuple(_cells(line))
        rows = []
        for row in lines[index + 2:]:
            if not row.startswith("|"):
                break
            cells = _cells(row)
            assert len(cells) == len(header), f"a row this test cannot split: {row}"
            rows.append(dict(zip(header, cells)))
        tables[header] = rows
    return tables


def _code(cell: str) -> str:
    """The first `code` span in a cell: the name the row is about."""
    match = _FIRST_CODE.search(cell)
    assert match, f"no `name` in the cell: {cell}"
    return match.group(1)


TABLES = _tables(_section(GUIDE, "3. What runs where"))
DEFAULT_ROWS = TABLES.get(DEFAULT_HEADER, [])
CORE_ROWS = TABLES.get(CORE_HEADER, [])
PROFILE_ROWS = TABLES.get(PROFILE_HEADER, [])
DEFAULT = {_code(row["Service"]) for row in DEFAULT_ROWS}
PROFILED = {(_code(row["Profile"]), _code(row["Service"])) for row in PROFILE_ROWS}
PROFILES = sorted({profile for profile, _ in PROFILED})


def _with_profile(profile: str) -> set[str]:
    return {service for named, service in PROFILED if named == profile}


# --------------------------------------------------------------------------- §3 against the files


def test_section_3_has_the_three_tables():
    assert DEFAULT_ROWS and CORE_ROWS and PROFILE_ROWS, (
        f"§3's tables were not found under these headers: {sorted(TABLES)}")


def test_the_default_table_is_every_service_without_a_profile():
    assert DEFAULT == {name for name, body in BASE.items() if not body.get("profiles")}
    assert {"minio-volume-owner", "workspace-worker"} <= DEFAULT


def test_the_profile_table_is_every_service_with_a_profile():
    assert PROFILED == {(profile, name) for name, body in BASE.items() for profile in body.get("profiles") or []}


def test_the_core_table_names_the_override_what_it_leaves_out_and_the_profile_back():
    (row,) = CORE_ROWS
    assert _code(row["Override"]) == CORE_FILE
    assert f"docker compose -f {BASE_FILE} -f {CORE_FILE} up -d" in row["Command"]
    assert "make up CORE=1" in row["Command"]
    profiled = {name: body["profiles"] for name, body in CORE.items() if body.get("profiles")}
    assert profiled == {_code(row["Leaves out"]): [_code(row["Profile that brings it back"])]}


@pytest.mark.parametrize("row", DEFAULT_ROWS + PROFILE_ROWS, ids=lambda row: _code(row["Service"]))
def test_each_row_names_composes_container_image_and_ports(row):
    body = BASE[_code(row["Service"])]
    assert _code(row["Container"]) == body["container_name"]
    named = _code(row["Image / build"]).removeprefix("docker.io/")
    image = str(body.get("image", "")).removeprefix("docker.io/")
    if image:
        assert image == named or image.startswith((f"{named}:", f"{named}@")), (named, image)
    else:
        assert named == body["build"]["context"], (named, body["build"])
    ports = " ".join(body.get("ports") or [])
    if row["Host port (variable)"].startswith("none"):
        assert not ports
    else:
        pairs = _PORT.findall(row["Host port (variable)"])
        assert pairs, row["Host port (variable)"]
        for port, variable in pairs:
            assert f"${{{variable}:-{port}}}" in ports, (variable, port, ports)


# --------------------------------------------------------------------------- docker compose config --services


def _services(*files: str, profiles: tuple[str, ...] = ()) -> set[str]:
    """``docker compose config --services`` for ``files`` and ``profiles``, with only the three secrets set."""
    docker = shutil.which("docker")
    if docker is None:
        if os.environ.get("CI") == "true":
            pytest.fail("the docker CLI is missing on this CI runner: these checks need `docker compose config`")
        pytest.skip("needs the docker CLI, which CI's runners have")
    environ = {key: value for key, value in os.environ.items()
               if not key.startswith(("AUTOMATOS_", "NEXT_PUBLIC_", "COMPOSE_", "POSTGRES_", "REDIS_", "API_"))}
    with tempfile.NamedTemporaryFile("w", suffix=".env") as empty:  # never a checkout's own .env
        command = [docker, "compose", "--project-directory", str(_REPO), "--env-file", empty.name]
        for name in files:
            command += ["-f", str(_REPO / name)]
        for profile in profiles:
            command += ["--profile", profile]
        proc = subprocess.run([*command, "config", "--services"], cwd=str(_REPO), capture_output=True,
                              text=True, timeout=120, env={**environ, **_SECRETS})
    assert proc.returncode == 0, proc.stderr
    return set(proc.stdout.split())


def _core_leaves_out() -> set[str]:
    return {_code(row["Leaves out"]) for row in CORE_ROWS}


def test_the_default_stack_is_the_default_table():
    assert _services(BASE_FILE) == DEFAULT


@pytest.mark.parametrize("profile", PROFILES)
def test_each_profile_adds_its_documented_services(profile):
    assert _services(BASE_FILE, profiles=(profile,)) == DEFAULT | _with_profile(profile)


def test_the_core_override_leaves_out_what_its_table_says():
    assert _services(BASE_FILE, CORE_FILE) == DEFAULT - _core_leaves_out()


def test_the_documented_profile_brings_the_worker_back_to_the_core_stack():
    (row,) = CORE_ROWS
    assert _services(BASE_FILE, CORE_FILE, profiles=(_code(row["Profile that brings it back"]),)) == DEFAULT


@pytest.mark.parametrize("profile", PROFILES)
def test_each_profile_adds_its_services_to_the_core_stack(profile):
    expected = (DEFAULT - _core_leaves_out()) | _with_profile(profile)
    assert _services(BASE_FILE, CORE_FILE, profiles=(profile,)) == expected


# --------------------------------------------------------------------------- the rest of the docs


def _read(name: str) -> str:
    return (_REPO / name).read_text(encoding="utf-8")


def test_every_clone_the_self_host_docs_give_is_shallow():
    clones = [(name, match.group("flags")) for name in SELF_HOST_DOCS for match in _CLONE.finditer(_read(name))]
    assert len(clones) >= 6, f"the clone search found too few: {clones}"
    assert [(name, flags) for name, flags in clones if "--depth 1" not in flags] == []


def test_section_10_says_what_w5_changes_for_an_existing_install():
    updating = _section(GUIDE, "10. Updating")
    for needle in ("`make up` pulls instead of building", "The source mount moves to `make dev`",
                   "AUTOMATOS_IMAGE_TAG", "minio-volume-owner", "65532"):
        assert needle in updating, needle


def test_the_first_download_is_the_measured_one():
    """§8, 16 Oct: the default first pull is 1,330 MB on amd64 and 1,284 MB on arm64."""
    disk = next(line for line in _section(GUIDE, "1. Prerequisites").splitlines() if line.startswith("| Disk |"))
    assert "1.33 GB" in disk and "1.28 GB" in disk and "make up CORE=1" in disk
    assert "1.33 GB" in _read("SETUP.md") and "1.28 GB" in _read("QUICKSTART.md")


@pytest.mark.parametrize("name", SELF_HOST_DOCS)
def test_no_self_host_doc_describes_what_no_longer_starts(name):
    text = _read(name)
    assert "minio-init" not in text, "MinIO's bucket comes from the backend; minio-init is long gone"
    assert "3.7 GB" not in text, "the images measured 1.33 GB to download on 16 Oct (§8)"


def test_the_compose_files_header_lists_what_starts():
    header = _read(BASE_FILE).split("\nservices:", 1)[0]
    assert "minio-init" not in header
    assert "minio-volume-owner" in header and CORE_FILE in header
    for profile in sorted({profile for body in BASE.values() for profile in body.get("profiles") or []}):
        assert f"--profile {profile}" in header, profile


def test_the_frontend_defaults_say_the_default_stack_runs_the_production_target():
    defaults = _read("envs/frontend.defaults")
    assert "The dev target (compose default)" not in defaults
    assert "production target, which the default stack runs" in defaults
    assert not re.search(r"^NODE_ENV=", defaults, re.M), "compose sets NODE_ENV; environment: wins over this file"
    assert BASE["frontend"]["environment"]["NODE_ENV"] == "production"
    assert _compose("docker-compose.dev.yml")["frontend"]["environment"]["NODE_ENV"] == "development"
