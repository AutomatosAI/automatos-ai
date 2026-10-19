"""PRD-262 US-020: site-packages ships without debug info and test suites.

After its install, pybuild strips the debug sections and the static symbol table from
every compiled library outside the auditwheel ``*.libs`` directories (``strip
--strip-unneeded``, which leaves ``.dynsym`` alone; the 19 Oct breakdown's row #25 moved
it on from ``--strip-debug``) and deletes every directory named exactly ``tests`` under
site-packages. Its apt list names binutils for the strip. CI proves the result inside
the built image with scripts/ci/image_site_packages.py: test.yml's prod-image-parity
runs it on the Railway build, and images.yml on the GHCR build, on both arches.

This suite checks:

- the Dockerfile: the prune RUN in pybuild, after the install and before the base stage
  copies /install's two halves (PRD-262 US-022), with the glob and the ``*.libs``
  exclusion the script also uses; and the RUN after it that deletes the C, C++ and
  Cython sources and headers (row #37), with the patterns the script looks for;
- the script's ELF reader, on ELF files built here (32- and 64-bit, both byte orders)
  and, on Linux, on this interpreter's own extension modules: it finds the debug
  sections and the ``.symtab`` the strip removes;
- how the script names, finds, loads and judges files, including an extension its
  package loads under another name (opencv's cv2), the tree check, and the throwaway
  settings it imports the app with;
- the script runs on the image's bare Python: it imports the standard library only;
- both workflows run it inside the image they built.

File reads, ELF bytes built in memory and copies of this interpreter's own extension
modules; no Docker, no network.
"""
from __future__ import annotations

import ast
import fnmatch
import importlib
import importlib.util
import json
import re
import shutil
import struct
import sys
from pathlib import Path, PurePosixPath

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"
DOCKERFILE = (ROOT / "orchestrator" / "Dockerfile").read_text(encoding="utf-8")
SCRIPT = ROOT / "scripts" / "ci" / "image_site_packages.py"
SCRIPT_RUN = "docker run --rm -i --entrypoint python parity-be - < scripts/ci/image_site_packages.py"
STRIP = "find /install -type f -name '*.so*' -not -path '*.libs/*' -exec strip --strip-unneeded {} +"
PRUNE_TESTS = "find /install -type d -name tests -path '*/site-packages/*' -prune -exec rm -rf {} +"
_CONTINUATION = re.compile(r"\\[ \t]*\n")
LINUX_ONLY = pytest.mark.skipif(not sys.platform.startswith("linux"),
                                reason="ELF and /proc/self/maps are Linux's; CI runs on Linux")

STB_LOCAL, STB_GLOBAL, STB_WEAK = 0, 1, 2
STV_DEFAULT, STV_HIDDEN, STV_PROTECTED = 0, 2, 3
STT_FUNC = 2
SHT_PROGBITS, SHT_SYMTAB, SHT_STRTAB, SHT_DYNSYM = 1, 2, 3, 11
DEFINED, UNDEFINED = 7, 0  # a section index, and SHN_UNDEF
# Standard-library modules CPython builds as shared extensions on Linux, small ones first.
STDLIB_EXTENSIONS = ("_json", "_csv", "_struct", "array", "math", "select", "_ctypes")


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


site_packages = _load("image_site_packages")


# --------------------------------------------------------------------------- ELF files built here


def _symbol(fmt, order: str, name_at: int, bind: int, visibility: int, index: int) -> bytes:
    values = [0] * 6
    name_field, info_field, other_field, index_field = fmt.symbol_fields
    values[name_field], values[info_field] = name_at, (bind << 4) | STT_FUNC
    values[other_field], values[index_field] = visibility, index
    return struct.pack(order + fmt.symbol, *values)


def _strings(names: list[str]) -> tuple[bytes, list[int]]:
    """A string table holding ``names``, and where each one starts."""
    table, offsets = b"\0", []
    for name in names:
        offsets.append(len(table))
        table += name.encode() + b"\0"
    return table, offsets


def _elf(cls: int = 2, order: str = "<", symbols=(), debug: bool = True, symtab: bool = False) -> bytes:
    """A small ELF shared object: .dynstr, .dynsym holding ``symbols`` ((name, binding,
    visibility, section index) each), an optional .debug_info, an optional (empty) .symtab,
    and .shstrtab."""
    fmt = site_packages.FORMATS[cls]
    dynstr, name_offsets = _strings([name for name, *_ in symbols])
    dynsym = bytes(struct.calcsize(order + fmt.symbol)) + b"".join(
        _symbol(fmt, order, at, bind, visibility, index)
        for at, (_, bind, visibility, index) in zip(name_offsets, symbols))
    contents = [(".dynstr", SHT_STRTAB, dynstr, 0, 0), (".dynsym", SHT_DYNSYM, dynsym, 1,
                                                       struct.calcsize(order + fmt.symbol))]
    contents += [(".debug_info", SHT_PROGBITS, b"\x01" * 32, 0, 0)] if debug else []
    contents += [(".symtab", SHT_SYMTAB, b"", 0, 0)] if symtab else []
    shstrtab, section_names = _strings([name for name, *_ in contents] + [".shstrtab"])
    contents.append((".shstrtab", SHT_STRTAB, shstrtab, 0, 0))
    ident_size = site_packages.EI_NIDENT
    header_size = ident_size + struct.calcsize(order + fmt.header)
    body, offsets = b"", []
    for _, _, data, _, _ in contents:
        offsets.append(header_size + len(body))
        body += data
    section_size = struct.calcsize(order + fmt.section)
    headers = bytes(section_size) + b"".join(
        struct.pack(order + fmt.section, name_at, kind, 0, 0, offset, len(data), link, 0, 1, entsize)
        for name_at, offset, (_, kind, data, link, entsize) in zip(section_names, offsets, contents))
    ident = site_packages.ELF_MAGIC + bytes([cls, 1 if order == "<" else 2, 1]) + bytes(9)
    header = struct.pack(order + fmt.header, 3, 62, 1, 0, 0, header_size + len(body), 0, header_size,
                         0, 0, section_size, len(contents) + 1, len(contents))
    return ident + header + body + headers


LAYOUTS = [(2, "<"), (2, ">"), (1, "<"), (1, ">")]


@pytest.mark.parametrize("cls,order", LAYOUTS)
def test_the_reader_finds_an_exported_init_and_the_debug_sections(cls, order):
    data = _elf(cls, order, [("PyInit_demo", STB_GLOBAL, STV_DEFAULT, DEFINED)])
    elf = site_packages.parse(data)

    assert site_packages.exports(data, elf, "PyInit_demo")
    assert not site_packages.exports(data, elf, "PyInit_other")
    assert site_packages.debug_sections(elf) == (".debug_info",)
    assert site_packages.debug_sections(site_packages.parse(_elf(cls, order, debug=False))) == ()


@pytest.mark.parametrize("cls,order", LAYOUTS)
def test_the_reader_finds_the_static_symbol_table_strip_unneeded_removes(cls, order):
    """PRD-262 row #25: --strip-unneeded removes .symtab with the debug sections; .dynsym stays."""
    kept = _elf(cls, order, [("PyInit_demo", STB_GLOBAL, STV_DEFAULT, DEFINED)], debug=True, symtab=True)
    elf = site_packages.parse(kept)

    assert site_packages.unneeded_sections(elf) == (".debug_info", ".symtab")
    assert site_packages.exports(kept, elf, "PyInit_demo"), "the loader's table is a section of its own"
    stripped = site_packages.parse(_elf(cls, order, [("PyInit_demo", STB_GLOBAL, STV_DEFAULT, DEFINED)], debug=False))
    assert site_packages.unneeded_sections(stripped) == ()
    assert site_packages.SYMBOL_TABLE == ".symtab"


@pytest.mark.parametrize("binding,visibility,index,exported", [
    (STB_GLOBAL, STV_DEFAULT, DEFINED, True),
    (STB_WEAK, STV_DEFAULT, DEFINED, True),
    (STB_GLOBAL, STV_PROTECTED, DEFINED, True),
    (STB_GLOBAL, STV_DEFAULT, UNDEFINED, False),  # imported from elsewhere, not defined here
    (STB_LOCAL, STV_DEFAULT, DEFINED, False),
    (STB_GLOBAL, STV_HIDDEN, DEFINED, False),
])
def test_only_a_defined_visible_global_or_weak_symbol_is_exported(binding, visibility, index, exported):
    data = _elf(symbols=[("PyInit_demo", binding, visibility, index)])
    assert site_packages.exports(data, site_packages.parse(data), "PyInit_demo") is exported


def test_a_longer_name_that_ends_in_the_symbol_is_not_the_symbol():
    data = _elf(symbols=[("xPyInit_demo", STB_GLOBAL, STV_DEFAULT, DEFINED)])
    elf = site_packages.parse(data)
    assert not site_packages.exports(data, elf, "PyInit_demo")
    assert site_packages.exports(data, elf, "xPyInit_demo")


def test_a_file_that_is_not_a_whole_elf_file_is_refused():
    whole = _elf(symbols=[("PyInit_demo", STB_GLOBAL, STV_DEFAULT, DEFINED)])
    no_sections = whole[:40] + bytes(8) + whole[48:]  # ELF64 e_shoff is at offset 40
    for data in (b"", b"#!/bin/sh\n", whole[:100], no_sections):
        with pytest.raises(ValueError):
            site_packages.parse(data)


@LINUX_ONLY
def test_the_reader_agrees_with_this_interpreters_own_extension_modules():
    module = _stdlib_extension()
    data = Path(module.__file__).read_bytes()
    elf = site_packages.parse(data)
    assert site_packages.exports(data, elf, f"PyInit_{module.__name__}")
    assert not site_packages.exports(data, elf, "PyInit_no_such_module")


def _stdlib_extension():
    """The first of STDLIB_EXTENSIONS this interpreter loads from a .so file."""
    modules = (importlib.import_module(name) for name in STDLIB_EXTENSIONS)
    found = next((module for module in modules if str(getattr(module, "__file__", "")).endswith(".so")), None)
    assert found is not None, f"none of {STDLIB_EXTENSIONS} is a shared extension here"
    return found


# --------------------------------------------------------------------------- naming, finding, judging


@pytest.mark.parametrize("relative,name", [
    ("PIL/_imaging.cpython-311-x86_64-linux-gnu.so", "PIL._imaging"),
    ("_cffi_backend.cpython-311-aarch64-linux-gnu.so", "_cffi_backend"),
    ("cv2/cv2.abi3.so", "cv2.cv2"),
    ("chardet/models/__init__.cpython-311-x86_64-linux-gnu.so", "chardet.models"),
    ("650fdafa2468af73c11f__mypyc.cpython-311-x86_64-linux-gnu.so", "650fdafa2468af73c11f__mypyc"),
    ("pypdfium2_raw/libpdfium.so", "pypdfium2_raw.libpdfium"),
])
def test_a_file_is_named_as_the_module_python_loads_it_as(relative, name):
    assert site_packages.module_name(PurePosixPath(relative)) == name


def test_the_walk_finds_the_files_the_strip_touches(tmp_path):
    files = ["a/x.cpython-311-x86_64-linux-gnu.so", "a/libz.so.1", "top.abi3.so",
             "pillow.libs/libjpeg-1a2b.so.62", "a/.libs/libq.so", "a/readme.txt", "a/b/deep.so"]
    for relative in files:
        (tmp_path / relative).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / relative).write_bytes(b"x")
    (tmp_path / "a" / "link.so").symlink_to(tmp_path / "a" / "libz.so.1")

    found = [path.relative_to(tmp_path).as_posix() for path in site_packages.native_files(tmp_path)]

    assert found == ["a/b/deep.so", "a/libz.so.1", "a/x.cpython-311-x86_64-linux-gnu.so", "top.abi3.so"]
    assert all(fnmatch.fnmatch(Path(path).name, site_packages.LIBRARY_GLOB) for path in found)


def _result(key, error=None, debug=(), kind="extension"):
    return site_packages.Result(key, kind, error, tuple(debug))


def test_the_failures_must_be_exactly_the_expected_ones_this_image_has():
    expected = {"PIL._imagingtk": "no Tk", "matplotlib.backends._tkagg": "no Tk"}
    loads = [_result("numpy._core._multiarray_umath"), _result("PIL._imagingtk", "ImportError: tkinter")]
    assert site_packages.problems_with(loads, expected) == [], "a listed file the image lacks is not required"

    unexpected = loads + [_result("cv2.cv2", "ImportError: libavcodec.so: cannot open shared object file")]
    assert site_packages.problems_with(unexpected, expected) == [
        "cv2.cv2 does not load: ImportError: libavcodec.so: cannot open shared object file"]

    healed = [_result("PIL._imagingtk")]
    assert site_packages.problems_with(healed, expected) == [
        "PIL._imagingtk loads now: remove it from EXPECTED_FAILURES (no Tk)"]


def test_a_debug_section_left_behind_fails_and_so_does_an_empty_walk():
    debug = [_result("zstandard.backend_c", debug=(".debug_info", ".debug_line"))]
    assert site_packages.problems_with(debug, {}) == ["zstandard.backend_c still carries .debug_info, .debug_line"]
    assert site_packages.problems_with([], {}) == ["no *.so file under site-packages: the walk found nothing to check"]


def test_the_tree_check_wants_no_tests_directory_and_every_kept_testing_package(tmp_path):
    for package in site_packages.KEPT_TESTING:
        (tmp_path / package).mkdir(parents=True)
        (tmp_path / package / "__init__.py").write_text("")
    (tmp_path / "jinja2").mkdir()
    (tmp_path / "jinja2" / "tests.py").write_text("")  # a module named tests is not a test suite
    assert site_packages.tree_problems(tmp_path) == []

    (tmp_path / "pandas" / "tests" / "frame").mkdir(parents=True)
    (tmp_path / "alembic" / "testing" / "__init__.py").unlink()
    assert site_packages.tree_problems(tmp_path) == [
        "site-packages still has pandas/tests/",
        "site-packages lost alembic/testing/: only directories named exactly tests go",
    ]


def test_the_tree_check_leaves_the_base_images_own_packages_alone(tmp_path):
    """python:3.11-slim installs setuptools 79.0.1 after it sweeps test directories, so its
    tests sit in the base layer, where pybuild's prune never reaches; deleting them later
    would add whiteouts and save nothing. A package pybuild installs is still judged."""
    for package in site_packages.KEPT_TESTING:
        (tmp_path / package).mkdir(parents=True)
        (tmp_path / package / "__init__.py").write_text("")
    for path in ("setuptools/tests", "setuptools/_distutils/tests", "setuptools/_distutils/compilers/C/tests",
                 "pkg_resources/tests"):
        (tmp_path / path).mkdir(parents=True)
    assert site_packages.tree_problems(tmp_path) == []
    assert {"setuptools", "pkg_resources"} <= set(site_packages.BASE_IMAGE_PACKAGES)

    (tmp_path / "setuptools_scm" / "tests").mkdir(parents=True)  # a name that only starts like one
    assert site_packages.tree_problems(tmp_path) == ["site-packages still has setuptools_scm/tests/"]


def test_the_source_check_finds_what_pybuild_deletes(tmp_path):
    """PRD-262 row #37: a C, C++ or Cython source or header left in site-packages fails; the
    modules, stubs and libraries beside it, and the base image's own packages, do not."""
    for relative in ("numpy/__init__.py", "numpy/__init__.pyi", "lxml/etree.cpython-311-x86_64-linux-gnu.so",
                     "fontTools/cu2qu/cu2qu.py", "setuptools/_vendor/thing.h"):
        (tmp_path / relative).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / relative).write_text("")
    assert site_packages.compiler_source_problems(tmp_path) == []

    left = ["numpy/_core/include/numpy/ndarraytypes.h", "fontTools/cu2qu/cu2qu.c", "scipy/linalg/cython_blas.pxd",
            "lxml/includes/etree_defs.h", "lxml/public-api.pxi", "sqlalchemy/cyextension/util.pyx",
            "greenlet/TGreenlet.cpp", "pkg/vendor.hpp"]
    for relative in left:
        (tmp_path / relative).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / relative).write_text("")
    (problem,) = site_packages.compiler_source_problems(tmp_path)
    assert problem.startswith(f"site-packages still has {len(left)} C, C++ or Cython sources or headers: "), problem
    assert "fontTools/cu2qu/cu2qu.c" in problem and "setuptools" not in problem


def test_every_expected_failure_says_why():
    assert "PIL._imagingtk" in site_packages.EXPECTED_FAILURES
    for key, reason in site_packages.EXPECTED_FAILURES.items():
        assert re.fullmatch(r"[A-Za-z_][\w.]*|[\w./-]+\.so[\w.]*", key), key
        assert len(reason.split()) >= 8, f"{key}: say why it fails and why that is fine"


def test_the_app_is_imported_with_the_backend_test_jobs_kind_of_settings():
    """Throwaway database settings with nothing listening, in the local edition, as the shards set them."""
    shard = _workflow("test.yml")["jobs"]["orchestrator-tests-shard"]["env"]
    settings = site_packages.APP_IMPORT_ENV
    assert set(settings) <= set(shard)
    assert {key: settings[key] for key in ("AUTH_EDITION", "DEFAULT_WORKSPACE_ID", "POSTGRES_HOST")} == {
        key: shard[key] for key in ("AUTH_EDITION", "DEFAULT_WORKSPACE_ID", "POSTGRES_HOST")}
    assert {"POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_PORT", "POSTGRES_DB"} <= set(settings), (
        "core/database/database.py builds its URL from these when imported")
    assert site_packages.APP_DIR == "/app" and site_packages.APP_IMPORT == "import main"


def test_a_failed_import_is_reported_on_one_line(tmp_path, monkeypatch):
    package = tmp_path / "us020broken"
    package.mkdir()
    (package / "__init__.py").write_text('raise ImportError("first line\\nsecond line")\n')
    monkeypatch.syspath_prepend(str(tmp_path))

    error = site_packages.load_extension("us020broken._speedups", package / "_speedups.so")

    assert error == "ImportError: first line second line"


# --------------------------------------------------------------------------- loading real extensions


@pytest.fixture
def isolated_imports(monkeypatch):
    """Drops whatever the test imported under its own package names afterwards."""
    before = set(sys.modules)
    yield monkeypatch
    for name in set(sys.modules) - before:
        if name.startswith("us020"):
            del sys.modules[name]


@LINUX_ONLY
def test_probe_imports_an_extension_and_opens_a_plain_library_with_ctypes(tmp_path, isolated_imports):
    source = Path(_stdlib_extension().__file__)
    package = tmp_path / "us020pkg"
    package.mkdir()
    extension = package / source.name
    shutil.copyfile(source, extension)
    library = package / "libplain.so"
    shutil.copyfile(source, library)  # no PyInit_libplain: a library, not a module
    (package / "notes.so.txt").write_text("not a library")
    isolated_imports.syspath_prepend(str(tmp_path))
    stem = source.name.split(".")[0]

    results = [site_packages.probe(tmp_path, path) for path in site_packages.native_files(tmp_path)]

    by_kind = {result.kind: result for result in results}
    assert by_kind["extension"].key == f"us020pkg.{stem}" and by_kind["extension"].error is None
    assert f"us020pkg.{stem}" in sys.modules
    assert by_kind["library"].key == "us020pkg/libplain.so" and by_kind["library"].error is None
    assert by_kind["unreadable"].key == "us020pkg/notes.so.txt"
    assert by_kind["unreadable"].error == "ValueError: not an ELF file"


@LINUX_ONLY
def test_an_extension_its_package_loads_under_another_name_counts_as_loaded(tmp_path, isolated_imports):
    """opencv's cv2 loads cv2/cv2.abi3.so as the top-level cv2 and keeps it as cv2._native."""
    source = Path(_stdlib_extension().__file__)
    stem = source.name.split(".")[0]
    package = tmp_path / "us020cv"
    package.mkdir()
    shutil.copyfile(source, package / source.name)
    (package / "__init__.py").write_text(
        "import importlib.util, os\n"
        f"_spec = importlib.util.spec_from_file_location({stem!r}, os.path.join(os.path.dirname(__file__), "
        f"{source.name!r}))\n"
        "_native = importlib.util.module_from_spec(_spec)\n"
        "_spec.loader.exec_module(_native)\n"
    )
    isolated_imports.syspath_prepend(str(tmp_path))

    assert site_packages.load_extension(f"us020cv.{stem}", package / source.name) is None
    assert f"us020cv.{stem}" not in sys.modules, "its init ran once, under the package's name for it"


# --------------------------------------------------------------------------- the script runs on a bare image


def test_the_script_imports_the_standard_library_only():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    names = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import)
             for alias in node.names}
    names |= {node.module.split(".")[0] for node in ast.walk(tree)
              if isinstance(node, ast.ImportFrom) and node.module and not node.level}
    assert names - {"__future__"} <= set(sys.stdlib_module_names), names - set(sys.stdlib_module_names)


# --------------------------------------------------------------------------- the Dockerfile


def _instructions(text: str) -> list[str]:
    """Instructions with continuation lines joined, comment lines dropped, spaces collapsed."""
    code = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    return [" ".join(line.split()) for line in _CONTINUATION.sub(" ", code).splitlines() if line.strip()]


def _pybuild() -> list[str]:
    """pybuild's instructions, from its FROM up to the next FROM."""
    instructions = _instructions(DOCKERFILE)
    start = next(i for i, line in enumerate(instructions) if re.match(r"FROM\s.*\sas\s+pybuild$", line, re.I))
    end = next(i for i, line in enumerate(instructions) if i > start and line.upper().startswith("FROM "))
    return instructions[start:end]


def test_pybuild_names_binutils_for_the_strip():
    apt = next(line for line in _pybuild() if "apt-get install" in line)
    assert "binutils" in apt.split()


def test_pybuild_strips_debug_info_and_deletes_tests_after_the_install():
    stage = _pybuild()
    install = next(i for i, line in enumerate(stage) if "pip install" in line)
    prune = next(i for i, line in enumerate(stage) if "strip" in line)
    run = stage[prune]

    assert install < prune, "the prune works on what the install put in /install"
    assert run == f"RUN {STRIP} && {PRUNE_TESTS}", run
    # --strip-unneeded (PRD-262 row #25): the debug sections and .symtab; --strip-all would
    # also drop what a relocatable object needs, and --strip-debug leaves .symtab.
    assert "|| true" not in run and "--strip-all" not in run and "--strip-debug" not in run
    assert "-name testing" not in DOCKERFILE, "numpy.testing, sqlalchemy.testing and alembic.testing stay"
    assert sum("strip" in line for line in _instructions(DOCKERFILE)) == 1, "only pybuild strips"


def _source_find() -> str:
    runs = [line for line in _pybuild() if line.startswith("RUN ") and "-name '*.pyx'" in line]
    assert len(runs) == 1, runs
    return runs[0]


def test_pybuild_deletes_the_compiler_sources_after_the_strip_and_before_the_compile():
    """PRD-262 row #37: the find names exactly the patterns the check looks for, under
    site-packages only, and deletes them; site-packages' bytecode is compiled after it."""
    stage = _pybuild()
    run = _source_find()
    at = stage.index(run)
    names = re.findall(r"-name '([^']+)'", run)

    assert sorted(names) == sorted(site_packages.COMPILER_SOURCES), names
    assert run.startswith("RUN find /install -type f -path '*/site-packages/*' \\( ") and run.endswith(" \\) -delete")
    assert next(i for i, line in enumerate(stage) if "strip" in line) < at
    assert at < next(i for i, line in enumerate(stage) if "compileall" in line)
    assert "|| true" not in run


def test_the_strip_names_the_files_the_check_loads():
    assert f"-name '{site_packages.LIBRARY_GLOB}'" in STRIP
    assert f"-not -path '*{site_packages.AUDITWHEEL_SUFFIX}/*'" in STRIP
    assert f"-name {site_packages.TESTS_DIR} " in PRUNE_TESTS


def test_the_base_stage_copies_the_pruned_tree():
    """Both halves of PRD-262 US-022's two-layer install sit under /install, which the strip walks,
    and so does the development stage's part (row #47), copied by that stage alone."""
    instructions = _instructions(DOCKERFILE)
    prune = next(i for i, line in enumerate(instructions) if STRIP in line)
    copies = [i for i, line in enumerate(instructions) if line.startswith("COPY --from=pybuild ")]
    assert [instructions[i] for i in copies] == ["COPY --from=pybuild /install/stable /usr/local",
                                                 "COPY --from=pybuild /install/rest /usr/local",
                                                 "COPY --from=pybuild /install/development /usr/local"]
    development = next(i for i, line in enumerate(instructions) if re.match(r"FROM\s.*\sas\s+development$", line, re.I))
    assert copies[1] < development < copies[2], "the base stage copies the halves; the development stage its part"
    assert prune < copies[0]
    assert not any("RUN --mount" in line for line in instructions), "Railway builds this file: no cache mounts"


# --------------------------------------------------------------------------- CI runs it inside the image


def _workflow(name: str) -> dict:
    return yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))


def test_prod_image_parity_runs_it_inside_the_railway_build():
    steps = _workflow("test.yml")["jobs"]["prod-image-parity"]["steps"]
    built = next(i for i, step in enumerate(steps) if "-t parity-be orchestrator/" in step.get("run", ""))
    check = next(i for i, step in enumerate(steps) if "image_site_packages.py" in step.get("run", ""))
    assert built < check
    assert steps[check]["run"].strip() == SCRIPT_RUN
    assert "if" not in steps[check] and "continue-on-error" not in steps[check]


def test_images_yml_runs_it_inside_every_api_build_on_both_arches():
    workflow = _workflow("images.yml")
    steps = workflow["jobs"]["build"]["steps"]
    built = next(i for i, step in enumerate(steps) if step.get("id") == "build")
    check = next(i for i, step in enumerate(steps) if "image_site_packages.py" in step.get("run", ""))
    step = steps[check]

    assert built < check
    assert "matrix.name == 'api'" in step["if"] and "steps.build.outcome == 'success'" in step["if"]
    assert "!cancelled()" in step["if"], "a failed check before it does not hide this one"
    assert step["env"] == {"IMAGE": "${{ matrix.image }}", "DIGEST": "${{ steps.build.outputs.digest }}"}
    assert 'docker pull --quiet "${IMAGE}@${DIGEST}"' in step["run"]
    assert ('docker run --rm -i --entrypoint python "${IMAGE}@${DIGEST}" - < scripts/ci/image_site_packages.py'
            in step["run"])
    assert "continue-on-error" not in step
    pick = next(step for step in workflow["jobs"]["changes"]["steps"] if step.get("id") == "pick")
    api = next(image for image in json.loads(pick["env"]["IMAGES"]) if image["name"] == "api")
    assert api["arches"] == ["amd64", "arm64"] and api["build_args"] == [], "one API image (US-027)"


def test_a_change_to_the_script_reruns_both_checks():
    relative = SCRIPT.relative_to(ROOT).as_posix()
    changes = _workflow("test.yml")["jobs"]["changes"]["steps"]
    match = next(step for step in changes if "PROD_IMAGE_PATHS" in (step.get("env") or {}))
    assert re.search(match["env"]["PROD_IMAGE_PATHS"], relative)
    images = _workflow("images.yml")
    paths = images.get("on", images.get(True))["pull_request"]["paths"]  # PyYAML reads `on` as True
    assert any(fnmatch.fnmatch(relative, path) for path in paths)
    assert "orchestrator/Dockerfile" in paths
