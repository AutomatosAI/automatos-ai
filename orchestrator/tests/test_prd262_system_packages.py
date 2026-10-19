"""PRD-262 US-019: the API image's system packages shrink to what the code calls.

Debian ghostscript left the base stage's apt list (nothing calls gs), and with it the
fonts-urw-base35 it brought, which the list now names: its faces are what a brand kit
naming Helvetica, Times, Courier, Palatino or Bookman prints in. libgdk-pixbuf went
(WeasyPrint 69 never opens it), pangoft2 and harfbuzz-subset are named (WeasyPrint opens
them, and only tesseract brought them), and the same RUN deletes tesseract's orientation
model and git's server-side binaries, linking git-core's copy of git to /usr/bin/git.

The same RUN deletes ICU's libraries and tesseract's 15 training tools, which alone
link ICU (the 19 Oct breakdown's row #13), and fonts-urw-base35's Type1 copies, which
Pango can't use (row #35). This suite checks:

- the Dockerfile's apt RUN: what it installs, deletes and links, in the one layer, and
  that scripts/ci/image_system_tools.py looks for exactly what it changed; after the
  ICU deletion the build itself fails if ldd finds a library tesseract needs missing or
  tesseract can't list its languages, and fontconfig's cache is rebuilt after the fonts
  go;
- every CI list that mirrors the image's (test.yml, nl2sql-eval-scheduled.yml,
  .oss-scanner/Dockerfile) installs fonts-urw-base35 and not ghostscript;
- test.yml's prod-image-parity checks the built image: no gs, the system tools, an HTTPS
  clone, the Helvetica render against `edge`, and the clone tests in the throwaway test
  image (scripts/ci/image_test_image.py), which is never pushed; images.yml runs the
  system tools on every API build, on both arches;
- scripts/ci/image_pdf_fixture.py: its comparison, and one real render here.

File reads and pure functions, plus one WeasyPrint render; no Docker, no network.
"""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"
DOCKERFILE = (ROOT / "orchestrator" / "Dockerfile").read_text(encoding="utf-8")
OSS_SCANNER = (ROOT / ".oss-scanner" / "Dockerfile").read_text(encoding="utf-8")
KEPT = ("curl", "git", "postgresql-client", "libmagic1", "tesseract-ocr", "libpango-1.0-0", "libcairo2",
        "fonts-dejavu-core")
NAMED = ("fonts-urw-base35", "libpangoft2-1.0-0", "libharfbuzz-subset0")
DROPPED = ("ghostscript", "libgdk-pixbuf-2.0-0")
OSD_DELETE = "rm -f /usr/share/tesseract-ocr/5/tessdata/osd.traineddata"
GIT_LINK = "ln -sf /usr/bin/git /usr/lib/git-core/git"
EDGE = "ghcr.io/automatosai/automatos-api:edge"
CLONE_TESTS = "tests/test_prd262_codegraph_clone.py"
_CONTINUATION = re.compile(r"\\[ \t]*\n")
_APT_INSTALL = re.compile(r"apt-get install\s+(?:-\S+\s+)*([^&|;\n]*)")


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "ci" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


system_tools = _load("image_system_tools")
pdf_fixture = _load("image_pdf_fixture")
test_image = _load("image_test_image")


def _instructions(text: str) -> list[str]:
    """Instructions with continuation lines joined; comment lines dropped."""
    code = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    return [" ".join(line.split()) for line in _CONTINUATION.sub(" ", code).splitlines() if line.strip()]


def _base_apt_run() -> str:
    """The base stage's apt RUN: the one instruction after `FROM … as base` that runs apt-get install."""
    instructions = _instructions(DOCKERFILE)
    start = next(i for i, line in enumerate(instructions) if re.match(r"FROM\s.*\sas\s+base$", line, re.I))
    return next(line for line in instructions[start + 1:] if "apt-get install" in line)


def _packages(command: str) -> list[str]:
    """The packages one apt-get install names, up to the first && | or ;."""
    match = _APT_INSTALL.search(command)
    assert match, command
    return [word for word in match.group(1).split() if word != "\\"]


def _workflow(name: str) -> dict:
    return yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- the image


def test_the_apt_layer_names_what_the_code_calls_and_drops_the_rest():
    packages = _packages(_base_apt_run())

    assert set(KEPT) <= set(packages), "every package the code calls stays"
    assert set(NAMED) <= set(packages), "named, now that ghostscript and tesseract no longer bring them"
    assert not set(DROPPED) & set(packages)
    assert set(system_tools.NAMED_PACKAGES) == set(NAMED)
    assert set(DROPPED) <= set(system_tools.ABSENT_PACKAGES)


def test_the_same_layer_deletes_the_osd_model_and_trims_git():
    run = _base_apt_run()
    git_rm = next(part for part in run.split("&&") if "/usr/lib/git-core/" in part and "rm -f" in part)

    assert OSD_DELETE in run
    assert set(git_rm.split()[2:]) == set(system_tools.GIT_REMOVED), git_rm
    assert GIT_LINK in run
    assert run.index(GIT_LINK) > run.index(git_rm), "the link replaces git-core's copy after the deletions"
    assert run.rstrip().endswith("apt-get clean")


def _parts(run: str) -> list[str]:
    return [" ".join(part.split()) for part in run.split("&&")]


def test_the_same_layer_deletes_icu_and_the_training_tools_and_proves_tesseract_still_runs():
    """PRD-262 row #13: only the training tools link ICU; tesseract and libtesseract don't."""
    parts = _parts(_base_apt_run())
    icu = parts.index("icu=$(dpkg -L libicu76 | grep -E '/libicu[a-z0-9]*\\.so\\.')")
    assert parts[icu + 1] == "rm -f $icu", "an empty match fails the RUN instead of deleting nothing"
    tools = next(part for part in parts if part.startswith("rm -f /usr/bin/ambiguous_words"))
    assert tools.split()[2:] == [f"/usr/bin/{tool}" for tool in system_tools.TRAINING_TOOLS]
    assert "/usr/bin/tesseract" not in tools.split()
    ldd, langs = parts.index("! ldd /usr/bin/tesseract | grep 'not found'"), parts.index("tesseract --list-langs")
    assert icu < parts.index(tools) < parts.index("ldconfig") < ldd < langs
    assert "dpkg --purge" not in _base_apt_run() and "--force-depends" not in _base_apt_run(), (
        "deleted like the other files, so dpkg records no broken dependency")


def test_the_same_layer_deletes_the_type1_fonts_and_rebuilds_fontconfigs_cache():
    """PRD-262 row #35: Pango can't use Type1; the OpenType faces of fonts-urw-base35 stay."""
    parts = _parts(_base_apt_run())
    fonts = parts.index("rm -rf /usr/share/fonts/type1/urw-base35 /usr/share/fonts/X11/Type1")
    assert parts.index("fc-cache -f") == fonts + 1
    assert "opentype" not in _base_apt_run()
    assert system_tools.URW_OPENTYPE.as_posix() == "/usr/share/fonts/opentype/urw-base35"
    assert set(system_tools.URW_ALIASED) == {"Helvetica", "Times", "Courier", "Palatino", "Bookman"}
    assert {".t1", ".pfb", ".afm"} <= set(system_tools.TYPE1_SUFFIXES)


def test_the_icu_scan_reads_only_elf_files_that_name_icu(tmp_path):
    script = tmp_path / "script.sh"
    script.write_bytes(b"#!/bin/sh\n# libicuuc.so.76 named in a comment\n")
    unrelated = tmp_path / "libunrelated.so"
    unrelated.write_bytes(system_tools.ELF_MAGIC + b"\0" * 64)
    assert not system_tools._links_missing_icu(script)
    assert not system_tools._links_missing_icu(unrelated), "no ICU name, so ldd never runs"


def test_the_dockerfile_no_longer_installs_ghostscript_or_gdk_pixbuf():
    code = "\n".join(_instructions(DOCKERFILE))
    assert "ghostscript" not in code and "gdk-pixbuf" not in code
    assert "gdk-pixbuf" not in DOCKERFILE, "the comment no longer says WeasyPrint needs it"
    assert "tesseract/ghostscript" not in DOCKERFILE


def test_the_https_helper_and_every_builtin_stay():
    """Only server-side and mail binaries go: a clone needs git-remote-http(s) and the builtins."""
    removed = {Path(path).name for path in system_tools.GIT_REMOVED}
    assert removed == {"git-daemon", "git-http-backend", "git-http-push", "git-imap-send", "git-shell", "scalar"}
    assert not removed & {system_tools.GIT_HTTPS_HELPER, "git-remote-http", *system_tools.GIT_BUILTINS}


# --------------------------------------------------------------------------- CI's lists


def _ci_package_lists() -> list[tuple[str, list[str]]]:
    """(where, packages) for every apt list in CI that installs the API's libraries (libmagic1)."""
    found = []
    for path in sorted(WORKFLOWS.glob("*.yml")):
        for job_name, job in (_workflow(path.name).get("jobs") or {}).items():
            listed = (job.get("env") or {}).get("SYSTEM_PACKAGES")
            if listed:
                found.append((f"{path.name} {job_name} env", listed.split()))
            for step in job.get("steps") or []:
                run = _CONTINUATION.sub(" ", step.get("run", ""))
                if "apt-get install" in run and "libmagic1" in run:
                    found.append((f"{path.name} {job_name}", _packages(run)))
    oss = next(line for line in _instructions(OSS_SCANNER) if "apt-get install" in line)
    found.append((".oss-scanner/Dockerfile", _packages(oss)))
    return found


def test_every_ci_list_that_mirrors_the_image_has_the_fonts_and_no_ghostscript():
    lists = _ci_package_lists()
    where = [name.split()[0] for name, _ in lists]

    assert where.count("test.yml") == 4 and "nl2sql-eval-scheduled.yml" in where, where
    assert ".oss-scanner/Dockerfile" in where
    for name, packages in lists:
        assert "fonts-urw-base35" in packages, name
        assert "ghostscript" not in packages, name


def test_the_shards_install_the_list_they_name():
    shard = _workflow("test.yml")["jobs"]["orchestrator-tests-shard"]
    install = next(step for step in shard["steps"] if step.get("name") == "Install system libraries")
    assert "$SYSTEM_PACKAGES" in install["run"]


# --------------------------------------------------------------------------- CI's checks in the image


def _parity_steps() -> list[dict]:
    return _workflow("test.yml")["jobs"]["prod-image-parity"]["steps"]


def _after_the_api_build(predicate) -> dict:
    steps = _parity_steps()
    built = next(i for i, step in enumerate(steps) if "-t parity-be orchestrator/" in step.get("run", ""))
    found = [i for i, step in enumerate(steps) if predicate(step)]
    assert len(found) == 1, f"{len(found)} steps match"
    assert built < found[0], "the check runs after the API image is built"
    return steps[found[0]]


def test_ci_finds_no_gs_in_the_image():
    step = _after_the_api_build(lambda step: "command -v gs" in step.get("run", ""))
    assert "docker run --rm --entrypoint sh parity-be -c '! command -v gs'" in step["run"]


def test_ci_runs_the_system_tools_script_in_the_image():
    step = _after_the_api_build(lambda step: "image_system_tools.py" in step.get("run", ""))
    assert step["run"].strip() == "docker run --rm -i --entrypoint python parity-be - < scripts/ci/image_system_tools.py"


def test_images_yml_runs_the_system_tools_script_in_every_api_build():
    """PRD-262 rows #13 and #35 change the apt layer on both arches, so both are checked."""
    steps = _workflow("images.yml")["jobs"]["build"]["steps"]
    built = next(i for i, step in enumerate(steps) if step.get("id") == "build")
    at, step = next((i, step) for i, step in enumerate(steps) if "image_system_tools.py" in step.get("run", ""))
    assert built < at
    assert "matrix.name == 'api'" in step["if"] and "!cancelled()" in step["if"]
    assert ('docker run --rm -i --entrypoint python "${IMAGE}@${DIGEST}" - < scripts/ci/image_system_tools.py'
            in step["run"])
    assert "continue-on-error" not in step


def test_ci_clones_a_public_repository_over_https_in_the_image():
    step = _after_the_api_build(lambda step: "git clone" in step.get("run", ""))
    assert "--entrypoint sh parity-be" in step["run"]
    assert re.search(r"git clone --depth 1 https://github\.com/\S+", step["run"])


def test_ci_lays_out_the_helvetica_fixture_in_edge_and_in_the_new_image():
    step = _after_the_api_build(lambda step: "image_pdf_fixture.py" in step.get("run", ""))
    run = step["run"]
    assert step["env"]["EDGE"] == EDGE
    assert step["if"] == "github.event_name == 'pull_request'", "on main, edge is the image the push replaces"
    assert 'docker pull --quiet "$EDGE"' in run
    assert ("docker run --rm -i --network none -w /app --entrypoint python parity-be - html "
            '< scripts/ci/image_pdf_fixture.py > "$out/page.txt"') in run, "the new image writes the page once"
    assert ('docker run --rm -i --network none -v "$out/page.txt:/tmp/page.txt:ro" --entrypoint python "$1" '
            "- layout /tmp/page.txt < scripts/ci/image_pdf_fixture.py") in run
    assert 'layout "$EDGE" > "$out/edge.txt"' in run and 'layout parity-be > "$out/new.txt"' in run
    assert 'image_pdf_fixture.py compare "$out/edge.txt" "$out/new.txt"' in run


def test_ci_runs_the_clone_tests_in_the_test_image():
    # US-028 adds its parser tests to the test image; this finds the clone tests' step.
    step = _after_the_api_build(lambda step: CLONE_TESTS in step.get("run", ""))
    run = " ".join(_CONTINUATION.sub(" ", step["run"]).split())
    assert "image_test_image.py build parity-be parity-be-test" in run
    assert f"image_test_image.py pytest parity-be-test -q --timeout=120 {CLONE_TESTS}" in run
    # On the network: the clone tests' module imports the skill loader, which loads
    # modules/rag/service.py, and that fetches tiktoken's encoding when imported (until
    # PRD-262 US-033). The clones themselves are of a repository made on the spot.
    assert "--network" not in run
    assert (ROOT / "orchestrator" / CLONE_TESTS).is_file()


def test_the_parity_job_pushes_nothing():
    """The test image is built in the job and never published."""
    assert not [step for step in _parity_steps() if "push" in step.get("run", "")]


@pytest.mark.parametrize("path", [
    "scripts/ci/image_system_tools.py",
    "scripts/ci/image_pdf_fixture.py",
    "scripts/ci/image_test_image.py",
    f"orchestrator/{CLONE_TESTS}",
])
def test_a_change_to_a_check_reruns_prod_image_parity(path):
    changes = next(step for step in _workflow("test.yml")["jobs"]["changes"]["steps"]
                   if "PROD_IMAGE_PATHS" in (step.get("env") or {}))
    assert re.search(changes["env"]["PROD_IMAGE_PATHS"], path)


# --------------------------------------------------------------------------- the test image


def test_the_test_image_is_the_image_under_test_plus_the_dev_requirements():
    """PRD-262 US-022: the dev lock, hashes required, compiled against the image's own lock."""
    dockerfile = test_image.dockerfile("parity-be")
    assert dockerfile.splitlines() == [
        "FROM parity-be",
        "COPY requirements-dev.lock /tmp/requirements-dev.lock",
        "RUN python -m pip install --user --no-cache-dir --no-warn-script-location --require-hashes "
        "-r /tmp/requirements-dev.lock",
    ]
    assert test_image.DEV_LOCK == ROOT / "orchestrator" / "requirements-dev.lock"
    assert test_image.DEV_LOCK.is_file()
    assert test_image.build_command("t", "/ctx") == ["docker", "build", "--tag", "t", "--file", "-", "/ctx"]


def test_the_test_image_mounts_the_tests_read_only():
    command = test_image.pytest_command("t", ["-q", CLONE_TESTS], "none")
    assert command[:2] == ["docker", "run"]
    assert f"{ROOT / 'orchestrator' / 'tests'}:/app/tests:ro" in command
    assert command[command.index("--network") + 1] == "none"
    assert command[command.index("t"):] == ["t", "-m", "pytest", "-p", "no:cacheprovider", "-q", CLONE_TESTS]
    assert "--network" not in test_image.pytest_command("t", [])


def test_the_test_image_runs_with_the_apps_throwaway_database_settings():
    """The image sets no database settings, and core/database/database.py raises when imported
    without them; the clone tests' module reaches it through the skill loader. The run gets
    the ones image_site_packages.py imports the app with (PRD-262 US-020): nothing listens."""
    command = test_image.pytest_command("t", ["-q", CLONE_TESTS])
    settings = test_image.site_packages.APP_IMPORT_ENV
    assert {"POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_HOST", "POSTGRES_PORT", "POSTGRES_DB"} <= set(settings)
    passed = [command[i + 1] for i, word in enumerate(command[:command.index("t")]) if word == "--env"]
    assert [f"{key}={value}" for key, value in settings.items()] == passed[1:], passed
    assert passed[0] == "PYTHONDONTWRITEBYTECODE=1"


@pytest.mark.parametrize("rest,expected", [
    (["--network", "none", "-q", "x.py"], ("none", ["-q", "x.py"])),
    (["-q", "x.py"], (None, ["-q", "x.py"])),
    (["--network"], (None, ["--network"])),
])
def test_the_network_flag_is_read_before_pytests_arguments(rest, expected):
    assert test_image.pytest_split(rest) == expected


def test_the_api_dockerfile_gains_no_test_stage():
    """No stage is added for it, so `production` stays the API's last stage (US-010)."""
    stages = re.findall(r"^FROM\s.*\sas\s+(\S+)\s*$", DOCKERFILE, re.M | re.I)
    assert stages[-1] == "production" and "test" not in stages


# --------------------------------------------------------------------------- the Helvetica render


def _facts(pages, fonts=("NimbusSansRegular", "NimbusSansBold")):
    return {"pages": [list(page) for page in pages], "fonts": sorted(fonts)}


EDGE_FACTS = _facts([["Harbourline wholesale terms", "1. Orders placed before noon"], ["2. Every invoice"]])


def test_the_same_pages_in_nimbus_sans_pass():
    assert pdf_fixture.problems(EDGE_FACTS, _facts(EDGE_FACTS["pages"])) == []


def test_a_moved_line_break_fails_with_the_lines_that_moved():
    moved = _facts([["Harbourline wholesale terms", "1. Orders placed", "before noon"], ["2. Every invoice"]])

    found = pdf_fixture.problems(EDGE_FACTS, moved)

    assert len(found) == 1 and found[0].startswith("the line breaks differ from edge's")
    assert "-page 1: 1. Orders placed before noon" in found[0] and "+page 1: before noon" in found[0]


def test_a_page_more_fails():
    longer = _facts([*EDGE_FACTS["pages"], ["3. Returns"]])
    assert any("has 3 pages; edge printed 2" in problem for problem in pdf_fixture.problems(EDGE_FACTS, longer))


def test_dejavu_fails_even_when_edge_printed_it_too():
    """The proof is Nimbus Sans: two images that both fell back to DejaVu do not pass."""
    dejavu = _facts(EDGE_FACTS["pages"], fonts=("DejaVuSans", "DejaVuSansBold"))
    found = pdf_fixture.problems(dejavu, dejavu)
    assert found == [f"Helvetica did not print in NimbusSans (fonts-urw-base35): {dejavu['fonts']}"]


def test_other_fonts_than_edges_fail():
    other = _facts(EDGE_FACTS["pages"], fonts=("NimbusSansRegular", "DejaVuSans"))
    assert any(problem.startswith("the text is set in") for problem in pdf_fixture.problems(EDGE_FACTS, other))


def test_the_last_marked_line_is_the_one_read():
    output = ('a warning on stdout\nFIXTURE_PDF {"pages": [], "fonts": []}\n'
              'FIXTURE_HTML "<p>x</p>"\nFIXTURE_PDF {"pages": [["x"]], "fonts": ["f"]}\n')
    assert pdf_fixture.marked(output, pdf_fixture.PDF_MARKER) == {"pages": [["x"]], "fonts": ["f"]}
    assert pdf_fixture.marked(output, pdf_fixture.HTML_MARKER) == "<p>x</p>"
    with pytest.raises(ValueError, match="no FIXTURE_PDF line"):
        pdf_fixture.marked("Traceback (most recent call last):\n", pdf_fixture.PDF_MARKER)


def test_pdf_font_names_lose_their_subset_prefix():
    assert pdf_fixture.flat("ABCDEF+NimbusSans-Regular") == "NimbusSansRegular"
    assert pdf_fixture.flat("DejaVu Sans Mono") == "DejaVuSansMono"


def test_the_fixture_names_helvetica_and_fills_two_pages_here():
    """The render the image runs, run here: the kit's font reaches the page's CSS, and the text
    is long enough that the page count is a check too."""
    html = pdf_fixture.fixture_html()
    facts = pdf_fixture.read_pdf(pdf_fixture.render_pdf(html))
    text = " ".join(line for page in facts["pages"] for line in page)

    assert re.search(r"font-family:\s*Helvetica\b", html), "the kit's font is in the page"
    assert len(facts["pages"]) >= 2 and all(facts["pages"]), facts["pages"]
    assert pdf_fixture.TITLE in text and "Returns are accepted within fourteen days" in text
    assert facts["fonts"]


def test_the_ocr_fixture_is_read_line_by_line():
    read = "Automatos reads this page\n\nInvoice 4827 paid in full\n\x0c"
    assert system_tools.missing_lines(read) == []
    assert system_tools.missing_lines("Automatos reads this page") == ["Invoice 4827 paid in full"]
