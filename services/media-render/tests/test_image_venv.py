"""PRD-262 #18: the venv carries what the service runs, stripped. Runs inside the image.

onnxruntime's transformers, quantization and tools are gone, and protobuf and
flatbuffers (which only those import) are not installed; numpy's f2py is gone;
espeakng_loader's extra copies of libespeak-ng.so are links to the one it loads;
no shared library carries a symbol table, and every compiled extension still
imports.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import shutil
import sysconfig
from pathlib import Path

import pytest

from elf import section_names, shared_libraries

SITE = Path(sysconfig.get_paths()["purelib"])
ONNXRUNTIME_UNUSED = ("transformers", "quantization", "tools")


def _package(name: str) -> Path:
    return Path(importlib.util.find_spec(name).origin).parent


def test_onnxruntimes_unused_subpackages_and_numpys_f2py_are_gone():
    assert [name for name in ONNXRUNTIME_UNUSED if (_package("onnxruntime") / name).exists()] == []
    assert not (_package("numpy") / "f2py").exists()
    assert shutil.which("f2py") is None


@pytest.mark.parametrize("distribution", ["protobuf", "flatbuffers"])
def test_onnxruntimes_unused_dependencies_are_not_installed(distribution):
    with pytest.raises(importlib.metadata.PackageNotFoundError):
        importlib.metadata.distribution(distribution)


def test_espeak_ng_ships_one_library_and_links_the_other_names_to_it():
    package = _package("espeakng_loader")
    libraries = sorted(package.glob("libespeak-ng.so*"))
    assert [path.name for path in libraries if not path.is_symlink()] == ["libespeak-ng.so"]
    assert len(libraries) > 1
    assert {path.resolve() for path in libraries} == {(package / "libespeak-ng.so").resolve()}


def test_no_shared_library_in_the_venv_carries_a_symbol_table():
    libraries = shared_libraries(SITE)
    assert libraries, f"no shared library under {SITE}"
    assert [str(path) for path in libraries if ".symtab" in section_names(path)] == []


def test_every_compiled_extension_in_the_venv_still_imports():
    suffixes = (sysconfig.get_config_var("EXT_SUFFIX"), ".abi3.so")
    modules = []
    for path in SITE.rglob("*.so"):
        suffix = next((s for s in suffixes if path.name.endswith(s)), None)
        if suffix:
            modules.append(".".join(path.relative_to(SITE).with_name(path.name[: -len(suffix)]).parts))
    assert modules, "no compiled extension found"
    failed = {}
    for module in sorted(modules):
        try:
            importlib.import_module(module)
        except ImportError as exc:
            failed[module] = str(exc)
    assert failed == {}
