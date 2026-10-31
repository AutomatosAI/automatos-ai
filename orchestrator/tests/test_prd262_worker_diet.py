"""PRD-262: the worker image's code cuts from the 19 Oct size breakdown, pinned.

Each cut keeps behaviour: agents keep gcc, g++, make, node, npm, pnpm, uv, git and
Perl. The Dockerfile edits are checked here against the in-image proofs that images.yml
runs on the image it built, on both arches:

- row 16: gcc's lto-dump is deleted in the apt layer that installs build-essential;
- row 19: the API's git trim (US-019) is applied to the worker: the same server-side
  and mail binaries go, and git-core's git becomes a link to /usr/bin/git;
- row 21: Node's binary comes from a `node-stripped` stage that media-render carries
  byte for byte, copied with --link as the production stage's first instruction, so
  the two images' layers are one blob on the same base;
- row 27: psycopg2 replaces psycopg2-binary at the same version; a `psycopg2` stage
  builds it from the lock's hash-checked sdist against libpq-dev, without build
  isolation, on setuptools from the hashed requirements-build.lock, and the production
  stage installs Debian's libpq5 and copies the build in;
- row 36: aiohttp and its six helpers leave the lock; the worker's server runs on the
  Starlette and uvicorn mcp already brings, now listed as direct dependencies
  (test_prd262_worker_starlette.py and the O2b suites test the server itself);
- row 40: the stable layer's RUN strips the debug sections from every *.so* it
  installed outside the auditwheel *.libs directories, as the API's US-020 does;
- row 47, tests: the same RUN deletes every directory named exactly `tests`;
- row 47, npm: an `npm` stage drops npm's docs and man pages before the copy, and the
  pnpm install's RUN deletes Node's compile cache and npm's logs it leaves behind;
- row 47, hiredis and python-dotenv: out of requirements.in and the lock; redis-py
  parses replies in Python, and no worker module imports dotenv;
- row 47, gosu: gone from the apt list; entrypoint.sh drops to the worker with setpriv
  (test_prd262_worker_boot.py runs the entrypoint, and test.yml's parity job checks the
  worker's uid and HOME in the Railway build).

Rows 12 and 31, the gcc trims §3-D5 accepted (sanitizer runtimes, libc.a and libstdc++.a),
are pinned in test_prd262_worker_d5_trims.py.

scripts/ci/image_worker_toolchain.py proves inside the image that ``gcc -flto`` still
links, that git still commits, branches, clones, pulls and clones over HTTPS, that the
stripped node runs, keeps .dynsym and loads a native N-API addon, and that setpriv drops
root to the worker with the worker's HOME. scripts/ci/image_worker_site_packages.py
loads every library the strip touched, as the worker user, and finds no debug section
and no tests directory, and that psycopg2 links Debian's libpq and queries the compose
stack's Postgres image.

Pure file reads: no Docker, no network.
"""
from __future__ import annotations

import importlib.util
import re
import shlex
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKER = ROOT / "services" / "workspace-worker"
DOCKERFILE = (WORKER / "Dockerfile").read_text()
IMAGES = yaml.safe_load((ROOT / ".github" / "workflows" / "images.yml").read_text())
BUILD_STEPS = IMAGES["jobs"]["build"]["steps"]
TOOLCHAIN_SCRIPT = "scripts/ci/image_worker_toolchain.py"
SITE_PACKAGES_SCRIPT = "scripts/ci/image_worker_site_packages.py"
SITE = "/usr/local/lib/python3.12/site-packages"


def _load(relative: str):
    """A CI script, loaded by path as the other image tests load theirs."""
    name = Path(relative).stem
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


toolchain = _load(TOOLCHAIN_SCRIPT)
site_packages = _load(SITE_PACKAGES_SCRIPT)


def _instructions(text: str) -> list[str]:
    """Instructions with continuation lines joined; comment lines dropped."""
    code = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    return [line.strip() for line in re.sub(r"\\[ \t]*\n", " ", code).splitlines() if line.strip()]


INSTRUCTIONS = _instructions(DOCKERFILE)


def _only(predicate) -> str:
    found = [line for line in INSTRUCTIONS if predicate(line)]
    assert len(found) == 1, f"expected one matching instruction, found {found}"
    return found[0]


def _commands(run: str) -> list[list[str]]:
    """The commands of a RUN, split at && and ;, each as its words."""
    body = run[len("RUN "):] if run.startswith("RUN ") else run
    return [shlex.split(part) for part in re.split(r"&&|;", body) if part.strip()]


APT_RUN = _only(lambda line: line.startswith("RUN") and "build-essential" in line)


def _rm_paths(run: str) -> list[str]:
    return [word for words in _commands(run) if words[:2] == ["rm", "-f"] for word in words[2:]]


def _worker_step(script: str) -> dict:
    steps = [step for step in BUILD_STEPS if script in step.get("run", "")]
    assert len(steps) == 1, f"images.yml's build job runs {script} in {len(steps)} steps"
    return steps[0]


# --------------------------------------------------------------------------- row 16


def test_lto_dump_goes_in_the_layer_that_installs_gcc():
    assert "/usr/bin/*lto-dump*" in _rm_paths(APT_RUN)
    assert toolchain.LTO_DUMP_GLOB == "/usr/bin/*lto-dump*", "the in-image check looks where the Dockerfile deletes"
    # The toolchain stays: gcc, g++ and make all come with build-essential.
    assert "build-essential" in APT_RUN


def test_the_in_image_lto_check_links_two_units_with_flto():
    """Two units, so the link really runs lto-wrapper and lto1 across them."""
    assert set(toolchain.LTO_UNITS) == {"main.c", "answer.c"}
    source = (ROOT / TOOLCHAIN_SCRIPT).read_text()
    assert '["gcc", "-flto", "-O2", "-c", name' in source
    assert '["gcc", "-flto", "-O2", *objects' in source


# --------------------------------------------------------------------------- row 19


GIT_TRIM = {
    "/usr/bin/git-shell", "/usr/bin/scalar",
    "/usr/lib/git-core/git-daemon", "/usr/lib/git-core/git-http-backend",
    "/usr/lib/git-core/git-http-push", "/usr/lib/git-core/git-imap-send",
    "/usr/lib/git-core/git-shell", "/usr/lib/git-core/scalar",
}
# Rows 12 and 31 delete what dpkg -L lists, through these variables (test_prd262_worker_d5_trims.py).
D5_TRIMS = {"$sanitizers", "$libc_a", "$libstdcxx_a"}


def test_the_git_trim_deletes_the_server_side_binaries_and_links_git_core_git():
    removed = set(_rm_paths(APT_RUN)) - {"/usr/bin/*lto-dump*"} - D5_TRIMS
    assert removed == GIT_TRIM
    assert ["ln", "-sf", "/usr/bin/git", "/usr/lib/git-core/git"] in _commands(APT_RUN)
    # git-remote-http(s) clones, pulls and pushes over HTTPS: it stays.
    assert not [path for path in removed if "remote" in path]


def test_the_in_image_check_looks_for_exactly_what_the_trim_deletes():
    assert set(toolchain.GIT_REMOVED) == GIT_TRIM
    assert toolchain.GIT_CORE_GIT == "/usr/lib/git-core/git" and toolchain.GIT_BINARY == "/usr/bin/git"
    assert toolchain.HTTPS_REPO.startswith("https://")


# --------------------------------------------------------------------------- row 21

# Shared with services/media-render/Dockerfile, byte for byte: the same stage on the same
# node digest strips to the same file, and COPY --link makes it the same layer blob.
NODE_STRIPPED_STAGE = (
    "# Node, stripped. The same stage, byte for byte, in services/workspace-worker/Dockerfile\n"
    "# and services/media-render/Dockerfile, so the two images share one node layer on a host.\n"
    "# The binary goes into the /node-layer tree, every directory of it at the binary's upstream\n"
    "# time, and the images copy that tree to /. A COPY of the file alone stamps /usr/local/bin\n"
    "# with the time of the build, so each build made a different layer.\n"
    "FROM node AS node-stripped\n"
    "RUN apt-get update \\\n"
    " && apt-get install -y --no-install-recommends binutils \\\n"
    " && mkdir -p /node-layer/usr/local/bin \\\n"
    " && strip --strip-unneeded --preserve-dates -o /node-layer/usr/local/bin/node /usr/local/bin/node \\\n"
    " && touch -r /usr/local/bin/node /node-layer/usr/local/bin /node-layer/usr/local /node-layer/usr /node-layer \\\n"
    " && rm -rf /var/lib/apt/lists/*\n"
)
NODE_COPY = "COPY --link --from=node-stripped /node-layer/ /"
NODE_DOCKERFILES = ("services/workspace-worker/Dockerfile", "services/media-render/Dockerfile")


def _stage_lines(stage: str) -> list[str]:
    """The instructions of the worker's stage named ``stage``, its FROM line first."""
    start = next(i for i, line in enumerate(INSTRUCTIONS) if re.fullmatch(rf"FROM \S+ AS {stage}", line))
    end = next((i for i in range(start + 1, len(INSTRUCTIONS)) if INSTRUCTIONS[i].startswith("FROM ")),
               len(INSTRUCTIONS))
    return INSTRUCTIONS[start:end]


def test_the_node_stripped_stage_follows_the_node_stage_word_for_word():
    node_at = DOCKERFILE.index(" AS node\n")
    assert DOCKERFILE[node_at:].startswith(" AS node\n\n" + NODE_STRIPPED_STAGE + "\n")


def test_every_dockerfile_with_the_stage_carries_it_byte_for_byte():
    """media-render carries the same stage (its own lane adds it), never a variant of it."""
    for path in NODE_DOCKERFILES:
        text = (ROOT / path).read_text()
        if "AS node-stripped" in text:
            assert text.count(NODE_STRIPPED_STAGE) == 1, f"{path}'s node-stripped stage is not the shared one"
            assert text.count(NODE_COPY) == 1, f"{path} copies the stripped node some other way"


def test_the_worker_copies_node_first_with_link_and_only_from_the_stripped_stage():
    production = _stage_lines("production")
    assert production[0].startswith("FROM python:3.12-slim@sha256:") and production[1] == NODE_COPY
    # The one COPY that brings the binary, by its path or from the stage's /node-layer tree.
    node_copies = [line for line in INSTRUCTIONS
                   if line.startswith("COPY") and ("/usr/local/bin/node " in line or "--from=node-stripped " in line)]
    assert node_copies == [NODE_COPY]
    assert INSTRUCTIONS[-1] == "CMD []" and production[-1] == "CMD []", "production stays the last stage"


def test_the_in_image_node_checks_look_for_the_strip():
    assert toolchain.NODE == "/usr/local/bin/node"
    assert set(toolchain.NODE_STRIPPED_SECTIONS) == {".symtab", ".strtab"}
    assert "napi_register_module_v1" in toolchain.NODE_ADDON_SOURCE
    assert set(toolchain.NODE_TOOLS) == {"npm", "npx", "corepack", "pnpm"}


# --------------------------------------------------------------------------- row 27

PSYCOPG2_VERSION = "2.9.13"  # psycopg2-binary's pin before the swap: same release, same API
PROD_PYTHON = "FROM python:3.12-slim@sha256:"


def test_psycopg2_replaces_psycopg2_binary_at_the_same_release():
    requirements = _requirement_lines()
    assert [line for line in requirements if line.startswith("psycopg2")] == ["psycopg2>=2.9.0,<3.0.0"]
    lock = (WORKER / "requirements.lock").read_text()
    assert re.findall(r"^psycopg2==(\S+) \\$", lock, re.M) == [PSYCOPG2_VERSION]
    for name in ("requirements.lock", "requirements-browser.lock"):
        assert "psycopg2-binary" not in _lock_pins(name), name


def _psycopg2_stage_installs() -> tuple[list[str], list[list[str]]]:
    """The psycopg2 stage's instructions, and every pip or uv install its RUNs make."""
    stage = _stage_lines("psycopg2")
    commands = [words for line in stage if line.startswith("RUN ") for words in _commands(line)]
    installs = [words for words in commands
                if words[:3] == ["uv", "pip", "install"] or (words[:1] in (["pip"], ["pip3"]) and "install" in words)]
    return stage, installs


def _install_of(installs: list[list[str]], lock: str) -> list[str]:
    (install,) = [words for words in installs if words[words.index("-r") + 1] == lock]
    return install


def test_the_psycopg2_stage_builds_the_sdist_against_libpq_on_the_production_python():
    stage, installs = _psycopg2_stage_installs()
    production = _stage_lines("production")
    assert stage[0].split(" AS ")[0] == production[0].split(" AS ")[0], "the same Python, so the same ABI"
    assert "COPY --from=locks /locks/psycopg2.lock /locks/" in stage
    (run,) = [line for line in stage if line.startswith("RUN ")]
    commands = _commands(run)
    assert {"gcc", "libc6-dev", "libpq-dev", "binutils"} <= set(_apt_packages(run))
    install = _install_of(installs, "/locks/psycopg2.lock")
    for flag in ("--require-hashes", "--no-deps", "--no-cache", "--compile-bytecode"):
        assert flag in install, flag
    assert install[install.index("--no-binary") + 1] == "psycopg2", "always the sdist, built here"
    assert install[install.index("--prefix") + 1] == "/install"
    assert ["find", "/install", "-type", "f", "-name", "*.so*", "-exec", "strip", "--strip-debug", "{}", "+"] in commands


def test_the_psycopg2_build_runs_only_on_a_hash_checked_backend():
    """--require-hashes checks the sdist, but an isolated build fetches its backend
    (setuptools) unpinned and unhashed. The stage installs the backend from the hashed
    requirements-build.lock first, as wheels only, and builds without isolation on it."""
    stage, installs = _psycopg2_stage_installs()
    assert "COPY requirements-build.lock /locks/" in stage
    backend = _install_of(installs, "/locks/requirements-build.lock")
    psycopg2 = _install_of(installs, "/locks/psycopg2.lock")
    assert installs.index(backend) < installs.index(psycopg2), "the backend is in place before the build"
    assert "--no-build-isolation" in psycopg2
    assert backend[backend.index("--only-binary") + 1] == ":all:", "the backend itself is never built"
    assert [name for name in ("--no-build-isolation", "--no-binary") if name in backend] == []
    # No other install in the stage, and none that a lock with hashes does not fully pin.
    assert len(installs) == 2, installs
    for words in installs:
        assert words[:3] == ["uv", "pip", "install"], f"an install that bypasses uv: {words}"
        assert {"--require-hashes", "--no-deps"} <= set(words), words
        assert words[words.index("-r") + 1].endswith(".lock"), words
    lock = (WORKER / "requirements-build.lock").read_text()
    assert re.findall(r"^([a-z0-9][a-z0-9._-]*)==", lock, re.M) == ["setuptools"]
    assert re.search(r"^    --hash=sha256:[0-9a-f]{64}", lock, re.M), "the build lock carries hashes"


def test_production_installs_libpq5_and_copies_the_build_without_the_toolchain_stage():
    assert "libpq5" in _apt_packages(APT_RUN)
    assert "libpq-dev" not in _apt_packages(APT_RUN)
    copy = "COPY --from=psycopg2 /install /usr/local"
    assert copy in INSTRUCTIONS
    stable_at = INSTRUCTIONS.index(STABLE_RUN)
    user_at = next(i for i, line in enumerate(INSTRUCTIONS) if "useradd" in line)
    assert stable_at < INSTRUCTIONS.index(copy) < user_at, "its own layer, after the stable lock"


def test_the_in_image_check_proves_debians_libpq_and_a_query():
    assert site_packages.REMOVED["psycopg2-binary"] == "row 27"
    assert site_packages.BUNDLED_LIBS_DIR == "psycopg2_binary.libs"
    assert site_packages.LIBPQ_MIN_VERSION >= 170000
    assert site_packages.SYSTEM_LIBPQ.search("\tlibpq.so.5 => /usr/lib/x86_64-linux-gnu/libpq.so.5 (0x1)")
    assert not site_packages.SYSTEM_LIBPQ.search(
        "\tlibpq-ab12.so.5.17 => /usr/local/lib/python3.12/site-packages/psycopg2_binary.libs/libpq-ab12.so.5.17")


def test_images_starts_the_compose_postgres_for_the_psycopg2_check():
    step = _worker_step(SITE_PACKAGES_SCRIPT)
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    images = {service.get("image") for service in compose["services"].values()}
    assert step["env"]["POSTGRES_IMAGE"] in images, "the database image the worker writes to"
    run = step["run"]
    assert "-e POSTGRES_HOST_AUTH_METHOD=trust" in run, "a throwaway database with no password to leak"
    assert f"-e {site_packages.DATABASE_URL_ENV}=postgresql://postgres@worker-probe-db:5432/postgres" in run
    assert run.index("docker network create worker-probe") < run.index("--network worker-probe \\\n")


# --------------------------------------------------------------------------- row 36

AIOHTTP_FAMILY = {"aiohttp", "aiohappyeyeballs", "aiosignal", "frozenlist", "multidict", "propcache", "yarl"}


def test_aiohttp_and_its_helpers_leave_the_lock_and_starlette_and_uvicorn_are_direct():
    requirements = _requirement_lines()
    assert not [line for line in requirements if line.startswith("aiohttp")]
    assert "starlette>=1.0.1,<2.0.0" in requirements, "1.0.1 fixes GHSA-86qp-5c8j-p5mr"
    assert "uvicorn>=0.50.1,<1.0.0" in requirements
    for name in ("requirements.lock", "requirements-browser.lock"):
        assert not _lock_pins(name) & AIOHTTP_FAMILY, name
    assert {"starlette", "uvicorn", "h11"} <= _lock_pins("requirements.lock")


def test_no_worker_module_imports_aiohttp():
    assert "aiohttp" not in _worker_imports()
    vendored = (WORKER / "automatos_metrics.py").read_text()
    assert "add_aiohttp_metrics" not in vendored and "def add_starlette_metrics" in vendored


def test_the_in_image_check_builds_the_app_without_aiohttp():
    assert {name for name, row in site_packages.REMOVED.items() if row == "row 36"} == AIOHTTP_FAMILY
    routes = re.findall(r'^\s+\("(?:GET|POST|DELETE)", ', (WORKER / "worker_http.py").read_text(), re.M)
    assert site_packages.WORKER_APP_ROUTES == len(routes) + 1, "every route, and /metrics"
    assert any("worker_http.build_app" in code for code in site_packages.REPLACEMENT_IMPORTS)


# --------------------------------------------------------------------------- rows 40 and 47


STABLE_RUN = _only(lambda line: line.startswith("RUN uv pip install") and "stable.lock" in line)
STRIP = ["find", SITE, "-type", "f", "-name", "*.so*", "-not", "-path", "*.libs/*",
         "-exec", "strip", "--strip-debug", "{}", "+"]
PRUNE = ["find", SITE, "-type", "d", "-name", "tests", "-prune", "-exec", "rm", "-rf", "{}", "+"]


def test_the_stable_layer_strips_its_libraries_and_drops_their_test_suites():
    commands = _commands(STABLE_RUN)
    assert commands[0][:3] == ["uv", "pip", "install"], "the install first, in the same layer"
    assert commands[1:] == [STRIP, PRUNE]


def test_the_in_image_check_loads_what_the_strip_touched():
    assert site_packages.LIBRARY_GLOB == "*.so*" and site_packages.AUDITWHEEL_SUFFIX == ".libs"
    assert site_packages.TESTS_DIR == "tests"
    assert {"cryptography.hazmat.bindings._rust", "pydantic_core._pydantic_core"} <= site_packages.MUST_LOAD


def test_the_in_image_check_names_extensions_as_python_imports_them(monkeypatch, tmp_path):
    monkeypatch.setattr(site_packages, "SITE", tmp_path)
    for relative in ("rpds/rpds.cpython-312-x86_64-linux-gnu.so", "_cffi_backend.cpython-312-aarch64-linux-gnu.so",
                     "cryptography/hazmat/bindings/_rust.abi3.so", "pkg.libs/libz-1a2b.so.1.3"):
        (tmp_path / relative).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / relative).write_bytes(b"")
    found = [site_packages.module_name(path) for path in site_packages.stripped_files()]
    assert found == ["_cffi_backend", "cryptography.hazmat.bindings._rust", "rpds.rpds"], "*.libs is skipped"


# --------------------------------------------------------------------------- row 47, the lock


def _lock_pins(name: str) -> set[str]:
    return set(re.findall(r"^([a-z0-9][a-z0-9._-]*)==", (WORKER / name).read_text(), re.M))


def _requirement_lines() -> list[str]:
    lines = [line.split("#", 1)[0].strip() for line in (WORKER / "requirements.in").read_text().splitlines()]
    return [line for line in lines if line]


def _worker_imports() -> set[str]:
    """The top-level module of every import in the worker's own code."""
    found = set()
    for path in WORKER.glob("*.py"):
        for line in path.read_text().splitlines():
            match = re.match(r"\s*(?:from|import)\s+([A-Za-z_][\w]*)", line)
            found |= {match.group(1)} if match else set()
    return found


def test_redis_has_no_hiredis_extra_and_python_dotenv_is_gone():
    requirements = _requirement_lines()
    assert [line for line in requirements if line.startswith("redis")] == ["redis>=5.0.0,<8.0.0"]
    assert not [line for line in requirements if "dotenv" in line or "hiredis" in line]
    for lock in ("requirements.lock", "requirements-browser.lock"):
        assert not _lock_pins(lock) & {"hiredis", "python-dotenv"}, lock
    assert "redis" in _lock_pins("requirements.lock")
    assert not _worker_imports() & {"dotenv", "hiredis"}, "the worker imports what the lock dropped"


def test_the_in_image_check_looks_for_the_removed_distributions():
    assert {"hiredis", "python-dotenv"} <= set(site_packages.REMOVED)
    assert any("HIREDIS_AVAILABLE" in code for code in site_packages.REPLACEMENT_IMPORTS)


# --------------------------------------------------------------------------- images.yml


def test_images_runs_the_toolchain_check_in_the_worker_it_built():
    step = _worker_step(TOOLCHAIN_SCRIPT)
    condition = " ".join(step["if"].split())
    assert "!cancelled()" in condition and "steps.build.outcome == 'success'" in condition
    assert "matrix.name == 'workspace-worker'" in condition
    assert step["env"] == {"IMAGE": "${{ matrix.image }}", "DIGEST": "${{ steps.build.outputs.digest }}"}
    assert 'docker pull --quiet "${IMAGE}@${DIGEST}"' in step["run"]
    assert f'--entrypoint python "${{IMAGE}}@${{DIGEST}}" - < {TOOLCHAIN_SCRIPT}' in step["run"]
    scan_at = next(i for i, other in enumerate(BUILD_STEPS) if other.get("name") == "Vulnerability scan")
    assert BUILD_STEPS.index(step) > scan_at, "after the reports and the scan, so they run first"


# --------------------------------------------------------------------------- row 47


def _apt_packages(run: str) -> list[str]:
    words = next(words for words in _commands(run) if words[:2] == ["apt-get", "install"])
    return [word for word in words[2:] if not word.startswith("-")]


def test_npm_is_copied_without_its_docs_and_man_pages():
    assert _stage_lines("npm") == [
        "FROM node AS npm",
        "RUN rm -rf /usr/local/lib/node_modules/npm/docs /usr/local/lib/node_modules/npm/man",
    ]
    copies = [line for line in INSTRUCTIONS if line.startswith("COPY") and "node_modules/npm" in line]
    assert copies == ["COPY --from=npm /usr/local/lib/node_modules/npm /usr/local/lib/node_modules/npm"]


def test_the_pnpm_install_leaves_no_compile_cache_or_npm_logs():
    run = _only(lambda line: line.startswith("RUN") and "pnpm@" in line)
    commands = _commands(run)
    cleanup = ["rm", "-rf", "/tmp/node-compile-cache", "/root/.npm"]
    assert commands[-1] == cleanup, "last, after npm cache clean, so nothing writes them again"
    assert ["npm", "cache", "clean", "--force"] in commands


def test_the_in_image_check_looks_for_the_leftovers_before_npm_runs():
    assert set(toolchain.NODE_LEFTOVERS) == {
        "/usr/local/lib/node_modules/npm/docs", "/usr/local/lib/node_modules/npm/man",
        "/tmp/node-compile-cache", "/root/.npm",
    }
    assert toolchain.CHECKS[0][1] is toolchain.check_node_leftovers


def test_gosu_is_gone_and_the_in_image_check_drops_as_the_entrypoint_does():
    assert "gosu" not in _apt_packages(APT_RUN)
    assert not [line for line in INSTRUCTIONS if "gosu" in line]
    exec_line = (WORKER / "entrypoint.sh").read_text().rstrip().splitlines()[-1]
    assert exec_line == f'exec {" ".join(toolchain.SETPRIV_DROP)} python -m main "$@"'
    # F452: no_new_privs set, and (security review) no capability left after the drop.
    assert toolchain.WORKER_IDENTITY == "1000 1000 /home/worker"
    assert toolchain.NO_NEW_PRIVS == "1" and toolchain.NO_CAPABILITIES == "0000000000000000 0000000000000000"


def test_images_runs_the_site_packages_check_as_the_worker_user():
    step = _worker_step(SITE_PACKAGES_SCRIPT)
    condition = " ".join(step["if"].split())
    assert "!cancelled()" in condition and "matrix.name == 'workspace-worker'" in condition
    assert "steps.build.outcome == 'success'" in condition
    assert {key: step["env"][key] for key in ("IMAGE", "DIGEST")} == {
        "IMAGE": "${{ matrix.image }}", "DIGEST": "${{ steps.build.outputs.digest }}"}
    assert "docker run --rm -i --user 1000 --network worker-probe" in step["run"]
    assert f'--entrypoint python "${{IMAGE}}@${{DIGEST}}" - < {SITE_PACKAGES_SCRIPT}' in step["run"]
    assert [title.split(":")[0] for title, _ in site_packages.CHECKS] == [
        "row 40", "row 47", "row 27", "row 27", "rows 27, 36 and 47"]


def test_every_toolchain_check_names_its_breakdown_row():
    rows = [title.split(":")[0] for title, _ in toolchain.CHECKS]
    assert rows == ["row 47", "row 16", "row 12", "row 31", "rows 12 and 31",
                    "row 19", "row 19", "row 19", "row 21", "row 21", "row 21", "row 47", "F452"]
