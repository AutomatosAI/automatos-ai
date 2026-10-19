"""PRD-262 §3-D5, decided by Gerard on 19 Oct 2026: the worker keeps its toolchain, trimmed.

gcc, g++, make, node, npm, pnpm, uv, git and Perl stay. Two parts of gcc go, in the apt
layer that installs build-essential, so no layer keeps them:

- row 12 of the 19 Oct size breakdown (42.2 MB on arm64): the sanitizer runtimes (asan,
  hwasan, lsan, tsan, ubsan) and gcc's archives, preinit objects and links for them, so
  ``-fsanitize`` no longer links;
- row 31: libc.a and libstdc++.a (12.2 MB), so ``-static`` and ``-static-libstdc++`` no
  longer link. libstdc++exp.a, which the breakdown also listed (0.9 MB), stays: GCC 14
  ships it as a static archive only, so an ordinary dynamic link of std::stacktrace
  (-lstdc++exp) needs it.

Both lists come from ``dpkg -L`` of the installed packages, never a hard-coded triplet,
so the RUN holds on amd64 (which Railway builds) and arm64. A package one arch lacks is
skipped, but a list that comes back empty fails the RUN. Files are deleted, packages
are not purged, so dpkg records no broken dependency for the browser step's apt-get.

The RUN's own commands run here under sh, with stand-in dpkg-query, dpkg, rm and
ldconfig fed the file lists of Debian trixie's packages for both arches.
scripts/ci/image_worker_toolchain.py proves in the built image that the files are gone,
that gcc still finds what a dynamic link needs, and that make, gcc and g++ build hello
world that links libc.so.6 and libstdc++.so.6 dynamically and runs.

No Docker, no network.
"""
from __future__ import annotations

import importlib.util
import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = (ROOT / "services" / "workspace-worker" / "Dockerfile").read_text()
TOOLCHAIN_SCRIPT = "scripts/ci/image_worker_toolchain.py"


def _load(relative: str):
    """A CI script, loaded by path as the other image tests load theirs."""
    name = Path(relative).stem
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


toolchain = _load(TOOLCHAIN_SCRIPT)


def _instructions(text: str) -> list[str]:
    """Instructions with continuation lines joined; comment lines dropped."""
    code = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    return [line.strip() for line in re.sub(r"\\[ \t]*\n", " ", code).splitlines() if line.strip()]


(APT_RUN,) = [line for line in _instructions(DOCKERFILE) if line.startswith("RUN") and "build-essential" in line]
TRIM_START = "sanitizers=$("
TRIM_END = "&& ldconfig"


def _trim_fragment() -> str:
    """The RUN's commands from the sanitizer list to ldconfig, as the build's sh runs them."""
    return APT_RUN[APT_RUN.index(TRIM_START):APT_RUN.index(TRIM_END) + len(TRIM_END)]


# --------------------------------------------------------------------------- the packages

# dpkg -L of Debian trixie's packages (gcc-14 14.2.0-19, glibc 2.41-12+deb13u4): files
# only, and of the headers only the sanitizer ones. {t} is the arch's triplet; the two
# arches lay the packages out alike, except amd64's x87 crtprec*.o and libquadmath.
TRIPLETS = {"amd64": "x86_64-linux-gnu", "arm64": "aarch64-linux-gnu"}
GCC_DIR, LIB_DIR = "/usr/lib/gcc/{t}/14", "/usr/lib/{t}"
SANITIZER_PACKAGES = {
    "libasan8": "libasan.so.8", "libhwasan0": "libhwasan.so.0", "liblsan0": "liblsan.so.0",
    "libtsan2": "libtsan.so.2", "libubsan1": "libubsan.so.1",
}


def _in(directory: str, names: str) -> list[str]:
    return [f"{directory}/{name}" for name in names.split()]


def _runtime(package: str, soname: str) -> list[str]:
    return [f"{LIB_DIR}/{soname}", f"{LIB_DIR}/{soname}.0.0",
            f"/usr/share/doc/{package}", f"/usr/share/lintian/overrides/{package}"]


PACKAGE_FILES = {
    **{package: _runtime(package, soname) for package, soname in SANITIZER_PACKAGES.items()},
    "libgcc-14-dev": _in(GCC_DIR, (
        "crtbegin.o crtbeginS.o crtbeginT.o crtend.o crtendS.o crtfastmath.o crtoffloadbegin.o"
        " crtoffloadend.o crtoffloadtable.o crtoffloadtableS.o libasan.a libasan.so libasan_preinit.o"
        " libatomic.a libatomic.so libbacktrace.a libgcc.a libgcc_eh.a libgcc_s.so libgcov.a libgomp.a"
        " libgomp.so libhwasan.a libhwasan.so libitm.a libitm.so liblsan.a liblsan.so liblsan_preinit.o"
        " libssp_nonshared.a libtsan.a libtsan.so libtsan_preinit.o libubsan.a libubsan.so"
    )) + _in(f"{GCC_DIR}/include/sanitizer", (
        "asan_interface.h common_interface_defs.h hwasan_interface.h lsan_interface.h tsan_interface.h"
    )),
    "libstdc++-14-dev": _in(GCC_DIR, "libstdc++.a libstdc++.so libstdc++exp.a libstdc++fs.a libsupc++.a"),
    "libc6-dev": _in(LIB_DIR, (
        "Mcrt1.o Scrt1.o crt1.o crti.o crtn.o gcrt1.o grcrt1.o rcrt1.o audit/sotruss-lib.so"
        " libBrokenLocale.a libBrokenLocale.so libanl.a libanl.so libc.a libc.so libc_malloc_debug.so"
        " libc_nonshared.a libdl.a libg.a libm-2.41.a libm.a libm.so libmcheck.a libmvec.a libmvec.so"
        " libnss_compat.so libnss_hesiod.so libpthread.a libpthread_nonshared.a libresolv.a libresolv.so"
        " librt.a libthread_db.so libutil.a"
    )),
    # Installed, and named like the packages above, but not theirs: the RUN leaves them be.
    "libgcc-s1": _in(LIB_DIR, "libgcc_s.so.1"),
    "libstdc++6": _in(LIB_DIR, "libstdc++.so.6 libstdc++.so.6.0.33"),
    "libc6": _in(LIB_DIR, "libc.so.6"),
}
AMD64_ONLY = {"libgcc-14-dev": _in(GCC_DIR, "crtprec32.o crtprec64.o crtprec80.o libquadmath.a libquadmath.so")}

# What row 12 deletes: the runtimes, and libgcc-14-dev's archives, preinit objects and links.
SANITIZERS_REMOVED = [path for soname in SANITIZER_PACKAGES.values()
                      for path in (f"{LIB_DIR}/{soname}", f"{LIB_DIR}/{soname}.0.0")] + _in(GCC_DIR, (
    "libasan.a libasan.so libasan_preinit.o libhwasan.a libhwasan.so liblsan.a liblsan.so"
    " liblsan_preinit.o libtsan.a libtsan.so libtsan_preinit.o libubsan.a libubsan.so"
))
ARCHIVES_REMOVED = [f"{LIB_DIR}/libc.a", f"{GCC_DIR}/libstdc++.a"]


def _installed(arch: str, without: tuple[str, ...] = ()) -> dict[str, list[str]]:
    """Each installed package's file list on ``arch``, less the packages ``without`` names."""
    extras = AMD64_ONLY if arch == "amd64" else {}
    return {package: [path.format(t=TRIPLETS[arch]) for path in files + extras.get(package, [])]
            for package, files in PACKAGE_FILES.items() if package not in without}


# --------------------------------------------------------------------------- the stand-ins

_STANDINS = {
    # dpkg-query -W -f FORMAT PATTERN...: "ii  <package>:<arch>" for each installed package
    # a pattern matches; a pattern that matches none is an error, as in dpkg-query.
    "dpkg-query": (
        '#!/bin/sh\nshift 3\nstatus=0\nfor pattern in "$@"; do\n  found=\n'
        '  for package in $(cat "$FAKE_DPKG/installed"); do\n'
        '    case "$package" in $pattern) echo "ii  $package:$FAKE_ARCH"; found=1;; esac\n  done\n'
        '  [ -n "$found" ] || { echo "dpkg-query: no packages found matching $pattern" >&2; status=1; }\n'
        'done\nexit $status\n'
    ),
    # dpkg -L PACKAGE...: dpkg's "/." line, then the package's list.
    "dpkg": (
        '#!/bin/sh\n[ "$1" = -L ] || exit 2\nshift\n[ $# -gt 0 ] || exit 2\nstatus=0\n'
        'for package in "$@"; do\n  list="$FAKE_DPKG/${package%%:*}.list"\n'
        '  if [ -f "$list" ]; then echo /.; cat "$list"; '
        'else echo "dpkg-query: package \'$package\' is not installed" >&2; status=1; fi\n'
        'done\nexit $status\n'
    ),
    "rm": '#!/bin/sh\necho "rm $*" >> "$CALL_LOG"\n',
    "ldconfig": '#!/bin/sh\necho "ldconfig $*" >> "$CALL_LOG"\n',
}


def _standin_env(tmp_path: Path, arch: str, packages: dict[str, list[str]]) -> dict[str, str]:
    """A PATH whose dpkg-query, dpkg, rm and ldconfig are the stand-ins, over ``packages``."""
    bin_dir, dpkg_dir = tmp_path / "bin", tmp_path / "dpkg"
    bin_dir.mkdir()
    dpkg_dir.mkdir()
    for name, body in _STANDINS.items():
        (bin_dir / name).write_text(body, encoding="utf-8")
        (bin_dir / name).chmod((bin_dir / name).stat().st_mode | stat.S_IXUSR)
    for package, files in packages.items():
        (dpkg_dir / f"{package}.list").write_text("".join(f"{path}\n" for path in files), encoding="utf-8")
    (dpkg_dir / "installed").write_text("".join(f"{package}\n" for package in packages), encoding="utf-8")
    path = os.pathsep.join((str(bin_dir), "/usr/bin", "/bin"))
    for name in _STANDINS:
        # The RUN deletes real paths: only ever under the stand-in rm, never the host's.
        assert shutil.which(name, path=path) == str(bin_dir / name), name
    return {"PATH": path, "FAKE_DPKG": str(dpkg_dir), "FAKE_ARCH": arch, "CALL_LOG": str(tmp_path / "calls.log")}


def _run_trims(tmp_path: Path, arch: str, without: tuple[str, ...] = ()) -> tuple[int, list[list[str]]]:
    """(exit code, the stand-in calls as words) of the RUN's trims on ``arch``."""
    env = _standin_env(tmp_path, arch, _installed(arch, without))
    proc = subprocess.run(["sh", "-c", _trim_fragment()], env=env, stdin=subprocess.DEVNULL,
                          capture_output=True, text=True, timeout=30)
    log = Path(env["CALL_LOG"])
    calls = log.read_text(encoding="utf-8").splitlines() if log.exists() else []
    return proc.returncode, [call.split() for call in calls]


def _on(arch: str, paths: list[str]) -> set[str]:
    return {path.format(t=TRIPLETS[arch]) for path in paths}


needs_sh = pytest.mark.skipif(shutil.which("sh") is None, reason="needs sh")


# --------------------------------------------------------------------------- the Dockerfile


def test_the_trims_run_in_the_layer_that_installs_gcc_after_the_install():
    order = ["apt-get install", TRIM_START, "rm -f $sanitizers", "libc_a=$(", "libstdcxx_a=$(",
             "rm -f $libc_a $libstdcxx_a", TRIM_END, "rm -rf /var/lib/apt/lists/*"]
    found = [APT_RUN.index(needle) for needle in order]
    assert found == sorted(found), "installed, trimmed, ldconfig, then apt's lists go"
    assert APT_RUN.endswith("rm -rf /var/lib/apt/lists/*")
    assert [line for line in _instructions(DOCKERFILE) if TRIM_START in line or "libc_a=" in line] == [APT_RUN]


def test_the_trims_delete_files_and_leave_every_package_registered():
    """A purge would leave an unmet dependency, and the browser step's apt-get install after it would fail."""
    for verb in ("dpkg --purge", "dpkg -r", "apt-get purge", "apt-get remove", "apt-get autoremove", "--force-depends"):
        assert verb not in APT_RUN, verb


def test_each_list_comes_from_dpkg_and_an_empty_one_fails_the_run():
    """``name=$(… | grep …) &&``: the assignment's status is grep's, so no match stops the RUN."""
    for name, source, pattern in (
        ("sanitizers", "xargs -r dpkg -L", "grep -E '^/usr/lib/.+/lib(asan|hwasan|lsan|tsan|ubsan)[._][^/]*$')"),
        ("libc_a", "dpkg -L libc6-dev", "grep '/libc\\.a$')"),
        ("libstdcxx_a", "xargs -r dpkg -L", "grep '/libstdc++\\.a$')"),
    ):
        command = re.search(rf"&& {name}=\$\((.*?)\) +&&", APT_RUN).group(1) + ")"
        assert source in command and command.endswith(pattern), name
    for packages in ("'libgcc-[0-9]*-dev'", "'libstdc++-[0-9]*-dev'"):
        assert packages in APT_RUN, packages
    for name in toolchain.SANITIZER_NAMES:
        assert f"'lib{name}[0-9]*'" in APT_RUN, name


def test_the_run_never_names_what_a_dynamic_link_needs():
    for kept in ("libgcc.a", "libgcc_eh", "libgcc_s", "libc_nonshared", "libssp_nonshared", "crt1", "crti",
                 "crtn", "crtbegin", "crtend", "libsupc++", "libstdc++exp", "libstdc++fs", "libc.so", "libstdc++.so"):
        assert kept not in APT_RUN, kept


# --------------------------------------------------------------------------- the RUN, run


@needs_sh
@pytest.mark.parametrize("arch", sorted(TRIPLETS))
def test_on_each_arch_the_run_deletes_the_sanitizers_and_the_two_archives_only(tmp_path, arch):
    code, calls = _run_trims(tmp_path, arch)
    assert code == 0
    assert [call[:2] for call in calls] == [["rm", "-f"], ["rm", "-f"], ["ldconfig"]]
    assert set(calls[0][2:]) == _on(arch, SANITIZERS_REMOVED) and len(calls[0]) == 2 + len(SANITIZERS_REMOVED)
    assert calls[1][2:] == [path.format(t=TRIPLETS[arch]) for path in ARCHIVES_REMOVED]


@needs_sh
def test_an_arch_without_one_sanitizer_package_still_builds(tmp_path):
    code, calls = _run_trims(tmp_path, "amd64", without=("libhwasan0",))
    assert code == 0
    runtime = {f"/usr/lib/x86_64-linux-gnu/libhwasan.so.0{suffix}" for suffix in ("", ".0.0")}
    assert set(calls[0][2:]) == _on("amd64", SANITIZERS_REMOVED) - runtime
    assert calls[-1] == ["ldconfig"]


@needs_sh
def test_no_sanitizer_file_at_all_fails_the_run(tmp_path):
    code, calls = _run_trims(tmp_path, "arm64", without=(*SANITIZER_PACKAGES, "libgcc-14-dev"))
    assert code != 0
    assert calls == [], "nothing deleted, and the RUN stops before ldconfig"


@needs_sh
@pytest.mark.parametrize("package", ["libc6-dev", "libstdc++-14-dev"])
def test_a_missing_static_archive_fails_the_run(tmp_path, package):
    code, calls = _run_trims(tmp_path, "arm64", without=(package,))
    assert code != 0
    assert [call[:2] for call in calls] == [["rm", "-f"]], "the sanitizers went, then the RUN stopped"


# --------------------------------------------------------------------------- the in-image check


def test_the_in_image_check_finds_what_the_run_deletes(tmp_path):
    """Its matcher, on the arm64 tree, finds the files the RUN's grep picks and nothing else."""
    for path in [path for files in _installed("arm64").values() for path in files]:
        (tmp_path / path.lstrip("/")).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path.lstrip("/")).write_bytes(b"")
    found = {path[len(str(tmp_path)):] for path in toolchain.sanitizer_files(str(tmp_path / "usr" / "lib"))}
    assert found == _on("arm64", SANITIZERS_REMOVED)
    assert toolchain.LIBRARY_ROOT == "/usr/lib"


def test_the_in_image_check_looks_for_the_two_archives_and_the_link_files(monkeypatch):
    assert {name for _, name in toolchain.STATIC_ARCHIVES_REMOVED} == {"libc.a", "libstdc++.a"}
    assert {"libgcc.a", "libgcc_eh.a", "libgcc_s.so", "libc_nonshared.a", "libssp_nonshared.a", "crt1.o",
            "Scrt1.o", "crti.o", "crtn.o", "crtbeginS.o", "crtendS.o", "libc.so", "libstdc++.so"} <= set(
        toolchain.LINK_FILES_KEPT)
    # gcc -print-file-name: a full path when found, the bare name when not.
    answers = {"libc.a": "/usr/lib/aarch64-linux-gnu/libc.a", "libstdc++.a": "libstdc++.a"}
    monkeypatch.setattr(toolchain, "must", lambda argv, cwd: answers.get(argv[1].split("=", 1)[1], "/nonexistent"))
    failures = toolchain.check_static_archives(Path("/"))
    assert failures[0] == "gcc still finds /usr/lib/aarch64-linux-gnu/libc.a"
    assert len(failures) == 1 + len(toolchain.LINK_FILES_KEPT), "and every link file reported missing"


def test_the_in_image_hello_world_is_built_by_make_with_gcc_and_g_plus_plus():
    makefile = toolchain.HELLO_SOURCES["Makefile"]
    assert "\tgcc -O2 -o $@ $<\n" in makefile and "\tg++ -O2 -o $@ $<\n" in makefile
    assert not [flag for flag in ("-static", "-fsanitize") if flag in makefile]
    assert toolchain.HELLO_PROGRAMS["hello-cpp"][1] == {"libstdc++.so.6", "libc.so.6"}
    line = " 0x0000000000000001 (NEEDED)             Shared library: [libstdc++.so.6]"
    assert toolchain.NEEDED_LIBRARY.findall(line) == ["libstdc++.so.6"]


def test_the_in_image_checks_for_rows_12_and_31_follow_the_lto_check():
    checks = [check for _, check in toolchain.CHECKS]
    at = checks.index(toolchain.check_lto)
    assert checks[at + 1:at + 4] == [
        toolchain.check_sanitizers_gone, toolchain.check_static_archives, toolchain.check_hello_dynamic]
