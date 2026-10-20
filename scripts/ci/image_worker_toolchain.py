"""PRD-262: the workspace worker's agent toolchain still works after the image's trims.

Agents compile, clone and run Node in the worker (executor.py ALLOWED_COMMANDS), so a
cut that saves bytes must leave each of those working. images.yml runs this inside the
worker image it built, on each architecture, with the image's own Python, as root
(the paths it checks are root's), before anything else in the container runs npm:

    docker run --rm -i --entrypoint python IMAGE - < scripts/ci/image_worker_toolchain.py

Each check names the row of the 19 Oct size breakdown it proves, or the finding:

- row 47, what the npm install left: npm's docs and man pages, Node's compile cache in
  /tmp and npm's cache and logs in /root/.npm are not in the image (checked first,
  before any check below runs npm and writes them again);
- row 16, gcc's lto-dump: nothing named lto-dump is left in /usr/bin, and two units
  compiled with ``-flto`` still link (lto-wrapper, lto1 and the linker plugin) into a
  program that runs;
- rows 12 and 31, PRD-262 §3-D5 (decided 19 Oct 2026: the toolchain stays, trimmed): no
  sanitizer runtime, archive or gcc link to one is left in /usr/lib or in ld.so's
  cache; gcc finds neither libc.a nor libstdc++.a, but still finds every file a dynamic
  link needs (libgcc*.a, libc_nonshared.a, libssp_nonshared.a, crt*.o, the libc.so,
  libgcc_s.so and libstdc++.so links); and make builds a C and a C++ hello world with
  gcc and g++ that link libc.so.6 and libstdc++.so.6 dynamically and run;
- row 19, the API's git trim: git's server-side and mail binaries are gone, git-core's
  ``git`` is a link to /usr/bin/git, and a repository can still be made, committed to,
  branched, checked out, cloned and pulled; an HTTPS clone works (git-remote-https), and
  git's credential protocol answers;
- row 21, the node binary: no .symtab, .strtab or debug sections, .dynsym kept; it
  runs JavaScript and crypto, a native N-API addon compiled here resolves Node's
  symbols and loads, and npm, npx, corepack and pnpm run on it;
- row 47, gosu: it is gone, and setpriv (util-linux, in the base image) drops root to
  the worker the way entrypoint.sh does, with the worker's uid, gid, groups and HOME,
  and with no_new_privs set (F452);
- F452: no file on the image's filesystem keeps a setuid or setgid bit, so nothing the
  worker or an agent runs can gain privileges through one.

Standard library only. Exit 0 when every check passes, 1 otherwise.
"""
from __future__ import annotations

import glob
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable, List, Sequence, Tuple

COMMAND_TIMEOUT_S = 300
STDERR_TAIL_CHARS = 800

# Row 16: the names gcc's lto-dump goes by (aarch64-linux-gnu-lto-dump-14 on arm64).
LTO_DUMP_GLOB = "/usr/bin/*lto-dump*"
LTO_UNITS = {
    "main.c": '#include <stdio.h>\nint answer(void);\nint main(void) { printf("lto %d\\n", answer()); return 0; }\n',
    "answer.c": "int answer(void) { return 42; }\n",
}
LTO_OUTPUT = "lto 42"

# Rows 12 and 31: what the apt layer deletes, found the way the Dockerfile's grep finds it.
SANITIZER_NAMES = ("asan", "hwasan", "lsan", "tsan", "ubsan")
SANITIZER_FILE = re.compile(rf"^lib(?:{'|'.join(SANITIZER_NAMES)})[._]")
LIBRARY_ROOT = "/usr/lib"
# gcc -print-file-name prints the bare name when no directory the linker searches holds it.
STATIC_ARCHIVES_REMOVED = (("gcc", "libc.a"), ("g++", "libstdc++.a"))
LINK_FILES_KEPT = (
    "libgcc.a", "libgcc_eh.a", "libgcc_s.so", "libc_nonshared.a", "libssp_nonshared.a",
    "libc.so", "libstdc++.so", "Scrt1.o", "crt1.o", "crti.o", "crtn.o", "crtbeginS.o", "crtendS.o",
)
HELLO_SOURCES = {
    "hello.c": '#include <stdio.h>\nint main(void) { puts("hello c"); return 0; }\n',
    "hello.cpp": (
        "#include <iostream>\n#include <string>\n"
        'int main() { std::cout << std::string("hello c++") << std::endl; return 0; }\n'
    ),
    "Makefile": "all: hello-c hello-cpp\nhello-c: hello.c\n\tgcc -O2 -o $@ $<\nhello-cpp: hello.cpp\n\tg++ -O2 -o $@ $<\n",
}
# Each program, what it prints, and the shared libraries it must name as NEEDED.
HELLO_PROGRAMS = {
    "hello-c": ("hello c", frozenset({"libc.so.6"})),
    "hello-cpp": ("hello c++", frozenset({"libstdc++.so.6", "libc.so.6"})),
}
NEEDED_LIBRARY = re.compile(r"\(NEEDED\)\s+Shared library: \[([^\]]+)\]")

# Row 19: what the Dockerfile's git trim deletes, as orchestrator/Dockerfile's US-019 does.
GIT_REMOVED = (
    "/usr/bin/git-shell",
    "/usr/bin/scalar",
    "/usr/lib/git-core/git-daemon",
    "/usr/lib/git-core/git-http-backend",
    "/usr/lib/git-core/git-http-push",
    "/usr/lib/git-core/git-imap-send",
    "/usr/lib/git-core/git-shell",
    "/usr/lib/git-core/scalar",
)
GIT_CORE_GIT = "/usr/lib/git-core/git"
GIT_BINARY = "/usr/bin/git"
GIT_REMOTE_HTTPS = "/usr/lib/git-core/git-remote-https"
GIT_IDENTITY = ("-c", "user.name=probe", "-c", "user.email=probe@example.invalid")
# A public repository of a few KB that GitHub keeps for exactly this kind of check.
HTTPS_REPO = "https://github.com/octocat/Hello-World.git"
HTTPS_REPO_FILE = "README"
CREDENTIAL_INPUT = "protocol=https\nhost=example.invalid\nusername=probe\npassword=probe-secret\n\n"

# Row 21: the node binary the node-stripped stage strips.
NODE = "/usr/local/bin/node"
NODE_STRIPPED_SECTIONS = (".symtab", ".strtab")
DEBUG_PREFIXES = (".debug_", ".zdebug_")
SECTION_NAME = re.compile(r"^\s*\[\s*\d+\]\s+(\S+)", re.M)
CRYPTO_INPUT = "probe"
NODE_PROBE_JS = (
    "process.stdout.write(process.version + ' ' + "
    f"require('crypto').createHash('sha256').update('{CRYPTO_INPUT}').digest('hex'))"
)
NODE_MAJOR = "v24."
# A Node-API addon written against the ABI alone (no headers: the image has none), loaded
# by the well-known napi_register_module_v1 symbol. Its napi_* calls resolve against the
# node binary's .dynsym, which a strip that went too far would take with it.
NODE_ADDON_SOURCE = """#include <stddef.h>
typedef struct napi_env__ *napi_env;
typedef struct napi_value__ *napi_value;
typedef int napi_status;
napi_status napi_create_string_utf8(napi_env env, const char *str, size_t length, napi_value *result);
napi_status napi_set_named_property(napi_env env, napi_value object, const char *name, napi_value value);
napi_value napi_register_module_v1(napi_env env, napi_value exports) {
    napi_value text;
    if (napi_create_string_utf8(env, "n-api ok", (size_t)-1, &text) != 0) return NULL;
    napi_set_named_property(env, exports, "probe", text);
    return exports;
}
"""
NODE_ADDON_TEXT = "n-api ok"
NODE_TOOLS = ("npm", "npx", "corepack", "pnpm")

# Row 47: npm's docs and what the pnpm install left behind.
NODE_LEFTOVERS = (
    "/usr/local/lib/node_modules/npm/docs",
    "/usr/local/lib/node_modules/npm/man",
    "/tmp/node-compile-cache",
    "/root/.npm",
)

# Row 47: entrypoint.sh's drop to the worker, minus the exec of the worker itself.
# --no-new-privs (F452, #1309): no setuid binary takes the worker back to root.
SETPRIV_DROP = ("setpriv", "--reuid=worker", "--regid=worker", "--init-groups", "--no-new-privs",
                "env", "HOME=/home/worker")
WORKER_IDENTITY = "1000 1000 /home/worker"
# F452: the NoNewPrivs field of /proc/<pid>/status in the dropped process.
NO_NEW_PRIVS = "1"
DROPPED_PROBE = 'echo "$(id -u) $(id -g) $HOME $(sed -n "s/^NoNewPrivs:[[:space:]]*//p" /proc/$$/status)"'
# F452: every file with a setuid or setgid bit; -xdev keeps find off /proc, /sys and /dev.
SETID_FILES = ("find", "/", "-xdev", "-perm", "/6000", "-type", "f")

Check = Callable[[Path], List[str]]


def run(argv: Sequence[str], cwd: Path, stdin: str = "") -> Tuple[int, str]:
    """(exit code, stdout + stderr) of ``argv``, never raising for a failed command."""
    done = subprocess.run(list(argv), cwd=str(cwd), input=stdin, capture_output=True, text=True,
                          timeout=COMMAND_TIMEOUT_S, check=False)
    return done.returncode, (done.stdout + done.stderr).strip()


def must(argv: Sequence[str], cwd: Path, stdin: str = "") -> str:
    """stdout + stderr of ``argv``; a failed command raises with its output's tail."""
    code, output = run(argv, cwd, stdin)
    if code != 0:
        raise RuntimeError(f"{' '.join(argv)} exited {code}: {output[-STDERR_TAIL_CHARS:]}")
    return output


# --------------------------------------------------------------------------- row 47, npm


def check_node_leftovers(work: Path) -> List[str]:
    """npm's docs and the install's caches and logs never reached the image."""
    return [f"{path} is in the image" for path in NODE_LEFTOVERS if os.path.lexists(path)]


# --------------------------------------------------------------------------- row 16


def check_lto(work: Path) -> List[str]:
    """No lto-dump, and an -flto build still links and runs."""
    failures = [f"{path} is still in the image" for path in sorted(glob.glob(LTO_DUMP_GLOB))]
    for name, source in LTO_UNITS.items():
        (work / name).write_text(source, encoding="utf-8")
    objects = []
    for name in LTO_UNITS:
        objects.append(name.replace(".c", ".o"))
        must(["gcc", "-flto", "-O2", "-c", name, "-o", objects[-1]], work)
    must(["gcc", "-flto", "-O2", *objects, "-o", "lto-probe"], work)
    output = must(["./lto-probe"], work)
    if output != LTO_OUTPUT:
        failures.append(f"the -flto program printed {output!r}, not {LTO_OUTPUT!r}")
    return failures


# --------------------------------------------------------------------------- rows 12 and 31


def sanitizer_files(root: str) -> List[str]:
    """Every file, link or directory under ``root`` named like a sanitizer runtime or archive."""
    found = []
    for directory, subdirectories, files in os.walk(root):
        found += [os.path.join(directory, name) for name in subdirectories + files if SANITIZER_FILE.match(name)]
    return sorted(found)


def check_sanitizers_gone(work: Path) -> List[str]:
    """No sanitizer runtime, archive or gcc link is left, and ld.so's cache lists none."""
    failures = [f"{path} is still in the image" for path in sanitizer_files(LIBRARY_ROOT)]
    # The first line counts the libraries; each one after names one: "\tlibfoo.so.1 (libc6,...) => ...".
    cached = [line.split()[0] for line in must(["ldconfig", "-p"], work).splitlines()[1:] if line.strip()]
    failures += [f"ld.so's cache still lists {name}" for name in cached if SANITIZER_FILE.match(name)]
    return failures


def check_static_archives(work: Path) -> List[str]:
    """gcc finds neither libc.a nor libstdc++.a, and still finds what a dynamic link needs."""
    failures = []
    for compiler, name in STATIC_ARCHIVES_REMOVED:
        found = must([compiler, f"-print-file-name={name}"], work)
        if found != name:
            failures.append(f"{compiler} still finds {found}")
    for name in LINK_FILES_KEPT:
        found = must(["gcc", f"-print-file-name={name}"], work)
        if not os.path.isfile(found):
            failures.append(f"gcc no longer finds {name}, which a dynamic link needs")
    return failures


def check_hello_dynamic(work: Path) -> List[str]:
    """make builds hello world with gcc and g++; each runs, linked against shared libraries."""
    for name, text in HELLO_SOURCES.items():
        (work / name).write_text(text, encoding="utf-8")
    must(["make", "-s"], work)
    failures = []
    for program, (greeting, libraries) in HELLO_PROGRAMS.items():
        output = must([f"./{program}"], work)
        if output != greeting:
            failures.append(f"{program} printed {output!r}, not {greeting!r}")
        needed = set(NEEDED_LIBRARY.findall(must(["readelf", "-d", program], work)))
        if not libraries <= needed:
            failures.append(f"{program} does not link {sorted(libraries - needed)} dynamically; NEEDED: {sorted(needed)}")
    return failures


# --------------------------------------------------------------------------- row 19


def _git(work: Path, *args: str) -> str:
    return must(["git", *GIT_IDENTITY, *args], work)


def _commit(repo: Path, name: str, text: str) -> None:
    (repo / name).write_text(text, encoding="utf-8")
    _git(repo, "add", name)
    _git(repo, "commit", "-q", "-m", f"add {name}")


def check_git_trim(work: Path) -> List[str]:
    """The server-side binaries are gone, and git-core's git is a link to /usr/bin/git."""
    failures = [f"{path} is still in the image" for path in GIT_REMOVED if os.path.lexists(path)]
    if not os.path.islink(GIT_CORE_GIT) or os.path.realpath(GIT_CORE_GIT) != GIT_BINARY:
        failures.append(f"{GIT_CORE_GIT} is not a link to {GIT_BINARY}")
    if not os.access(GIT_REMOTE_HTTPS, os.X_OK):
        failures.append(f"{GIT_REMOTE_HTTPS} is missing: HTTPS clones need it")
    return failures


def check_git_local(work: Path) -> List[str]:
    """init, commit, clone, branch, checkout and pull on repositories in ``work``."""
    origin, clone = work / "origin", work / "clone"
    must(["git", "init", "-q", "-b", "main", str(origin)], work)
    _commit(origin, "one.txt", "one\n")
    must(["git", "clone", "-q", f"file://{origin}", str(clone)], work)
    _git(clone, "checkout", "-q", "-b", "feature")
    _commit(clone, "two.txt", "two\n")
    _git(clone, "checkout", "-q", "main")
    _commit(origin, "three.txt", "three\n")
    _git(clone, "pull", "-q", "--no-rebase", "origin", "main")
    expected = {"one.txt", "three.txt"}
    found = {path.name for path in clone.iterdir() if path.suffix == ".txt"}
    return [] if found == expected else [f"the clone's main holds {sorted(found)}, not {sorted(expected)}"]


def check_git_https(work: Path) -> List[str]:
    """An HTTPS clone, through git-remote-https, and git's credential protocol."""
    target = work / "https-clone"
    must(["git", "clone", "-q", "--depth", "1", HTTPS_REPO, str(target)], work)
    failures = [] if (target / HTTPS_REPO_FILE).is_file() else [f"{HTTPS_REPO} cloned without its {HTTPS_REPO_FILE}"]
    answer = must(["git", "credential", "fill"], work, stdin=CREDENTIAL_INPUT)
    if "password=probe-secret" not in answer.splitlines():
        failures.append("git credential fill did not hand back the credential it was given")
    return failures


# --------------------------------------------------------------------------- row 21


def elf_sections(path: str) -> List[str]:
    """The section names ``readelf -S`` lists (binutils comes with build-essential)."""
    return SECTION_NAME.findall(must(["readelf", "-S", "-W", path], Path("/")))


def check_node_binary(work: Path) -> List[str]:
    """Stripped, with .dynsym kept, and still running JavaScript and crypto."""
    sections = elf_sections(NODE)
    failures = [f"{NODE} still has {name}" for name in sections
                if name in NODE_STRIPPED_SECTIONS or name.startswith(DEBUG_PREFIXES)]
    if ".dynsym" not in sections:
        failures.append(f"{NODE} lost .dynsym: native addons cannot resolve Node's symbols")
    expected = f"{hashlib.sha256(CRYPTO_INPUT.encode()).hexdigest()}"
    version, _, digest = must([NODE, "-e", NODE_PROBE_JS], work).partition(" ")
    if not version.startswith(NODE_MAJOR) or digest != expected:
        failures.append(f"node printed {version} {digest}, not {NODE_MAJOR}x and sha256({CRYPTO_INPUT!r})")
    return failures


def check_node_addon(work: Path) -> List[str]:
    """A native addon built here loads into the stripped node and runs."""
    (work / "probe.c").write_text(NODE_ADDON_SOURCE, encoding="utf-8")
    must(["gcc", "-shared", "-fPIC", "-O2", "probe.c", "-o", "probe.node"], work)
    loaded = must([NODE, "-e", "process.stdout.write(require('./probe.node').probe)"], work)
    return [] if loaded == NODE_ADDON_TEXT else [f"the addon answered {loaded!r}, not {NODE_ADDON_TEXT!r}"]


def check_node_tools(work: Path) -> List[str]:
    """npm, npx, corepack and pnpm each run on the stripped node."""
    failures = []
    for tool in NODE_TOOLS:
        code, output = run([tool, "--version"], work)
        if code != 0:
            failures.append(f"{tool} --version exited {code}: {output[-STDERR_TAIL_CHARS:]}")
    return failures


# --------------------------------------------------------------------------- row 47


def check_setpriv(work: Path) -> List[str]:
    """No gosu, and setpriv drops root to the worker with the worker's HOME and no_new_privs."""
    failures = [] if shutil.which("gosu") is None else ["gosu is still in the image"]
    # From /: the worker may not enter root's temporary directory.
    identity = must([*SETPRIV_DROP, "sh", "-c", DROPPED_PROBE], Path("/"))
    expected = f"{WORKER_IDENTITY} {NO_NEW_PRIVS}"
    if identity != expected:
        failures.append(f"setpriv ran the command as {identity!r} (uid gid HOME NoNewPrivs), not {expected!r}")
    return failures


# --------------------------------------------------------------------------- F452


def check_no_setid_files(work: Path) -> List[str]:
    """No file keeps a setuid or setgid bit: the Dockerfile's last install step strips them."""
    found = must(SETID_FILES, Path("/")).splitlines()
    return [f"{path} keeps its setuid or setgid bit" for path in found if path.strip()]


# --------------------------------------------------------------------------- the run


CHECKS: Tuple[Tuple[str, Check], ...] = (
    ("row 47: no npm docs, compile cache or npm logs", check_node_leftovers),
    ("row 16: no lto-dump, and gcc -flto links", check_lto),
    ("row 12: no sanitizer runtime, archive or link is left", check_sanitizers_gone),
    ("row 31: no libc.a or libstdc++.a, and what a dynamic link needs is kept", check_static_archives),
    ("rows 12 and 31: make, gcc and g++ build hello world, linked dynamically", check_hello_dynamic),
    ("row 19: git's server-side binaries are gone", check_git_trim),
    ("row 19: git init, commit, clone, branch, checkout and pull", check_git_local),
    ("row 19: an HTTPS clone and git's credential protocol", check_git_https),
    ("row 21: node is stripped, keeps .dynsym and runs", check_node_binary),
    ("row 21: a native N-API addon loads into the stripped node", check_node_addon),
    ("row 21: npm, npx, corepack and pnpm run on the stripped node", check_node_tools),
    ("row 47: no gosu, and setpriv drops to the worker with no_new_privs", check_setpriv),
    ("F452: no file keeps a setuid or setgid bit", check_no_setid_files),
)


def run_check(title: str, check: Check) -> List[str]:
    """The check's failures; one that raises is a failure with its message."""
    with tempfile.TemporaryDirectory(prefix="worker-toolchain-") as work:
        try:
            return check(Path(work))
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            return [f"{type(exc).__name__}: {exc}"]


def main() -> int:
    failed = 0
    for title, check in CHECKS:
        failures = run_check(title, check)
        print(f"{'FAIL' if failures else 'PASS'}  {title}")
        for failure in failures:
            print(f"      {failure}")
        failed += bool(failures)
    print(f"{len(CHECKS) - failed} of {len(CHECKS)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
