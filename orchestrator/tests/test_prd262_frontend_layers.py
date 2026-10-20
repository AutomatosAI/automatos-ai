"""PRD-262 US-048: unchanged frontend layers keep their digest.

- The builder's build RUN moves the dependencies Next traced out of the standalone output,
  to .next/standalone-node_modules. In the same RUN the move is a rename and adds no layer.
- The production stage's last four filesystem layers are, in order: the traced
  node_modules, public/, .next/static and the server (the rest of the standalone output).
  BuildKit keys a COPY on its files' content, not their mtimes, so a layer whose files and
  every layer below it are unchanged keeps its digest, and a code-only update downloads
  only static and the server.
- public/ comes from the builder: its copy is the build context's public/ plus any file the
  build writes there (PRD-261's theme-sdk shims are generated and never committed), which a
  copy straight from the build context would leave out.
- frontend/.dockerignore keeps tests, docs and tsconfig.tsbuildinfo out of the build
  context, and no shipped source imports one of them.
- test.yml's prod-image-parity job checks that the local build reuses the SaaS build's
  node_modules and public layers (frontend/scripts/check-layer-reuse.sh, run here against a
  stub docker) and that the build context holds no test, doc or tsbuildinfo.

The update delta of a code-only main update (8 MB or less) is measured against `edge` by
images.yml's US-001 report. File reads, plus the check script against a stub docker.
"""
from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
FRONTEND = ROOT / "frontend"
DOCKERFILE = (FRONTEND / "Dockerfile").read_text()
DOCKERIGNORE = (FRONTEND / ".dockerignore").read_text()
TEST_YML = (ROOT / ".github" / "workflows" / "test.yml").read_text()
JOBS = yaml.safe_load(TEST_YML)["jobs"]
CHECK_SCRIPT = FRONTEND / "scripts" / "check-layer-reuse.sh"

TRACED = ".next/standalone-node_modules"
BUILD = (
    f"RUN npm run build && rm -rf .next/cache && mv .next/standalone/node_modules {TRACED}"
    # PRD-262 #47: the same RUN deletes the trace files (test_prd262_frontend_trace_diet.py).
    " && find .next/standalone/.next -name '*.nft.json' -delete"
)
APP_LAYERS = [
    f"COPY --from=builder --chown=nextjs:nodejs /app/{TRACED} ./node_modules",
    "COPY --from=builder --chown=nextjs:nodejs /app/public ./public",
    "COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static",
    "COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./",
]
LAYER_MAKERS = re.compile(r"^(COPY|ADD|RUN|WORKDIR)\s", re.I)
CONFIG_ONLY = {"USER", "EXPOSE", "HEALTHCHECK", "ENV", "CMD", "ENTRYPOINT", "LABEL", "STOPSIGNAL", "ARG"}
LEFT_OUT = ("**/__tests__", "**/*.test.*", "**/*.md", "tsconfig.tsbuildinfo")
SOURCE_SUFFIXES = {".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs"}
NOT_SHIPPED_DIRS = {"node_modules", ".next", "__tests__"}
# An import, dynamic import or require of a test module, anything under __tests__, or a .md file.
LEFT_OUT_IMPORT = re.compile(
    r"""(?:\bfrom\s*|\bimport\s*\(?\s*|\brequire\s*\(\s*)"""
    r"""['"][^'"\n]*(?:/__tests__/[^'"\n]*|\.test(?:\.[cm]?[jt]sx?)?|\.md)['"]"""
)
SAAS_BUILD = "-t parity-fe frontend/"
LOCAL_BUILD = "-t parity-fe-local frontend/"
REUSE_STEP = "Frontend — the local build reuses the SaaS build's node_modules and public layers"
CONTEXT_STEP = "Frontend — the build context leaves out tests, docs and tsconfig.tsbuildinfo"


def _instructions(text: str) -> list[str]:
    """Instructions with continuation lines joined; comment lines dropped."""
    code = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    return [line.strip() for line in code.replace("\\\n", " ").splitlines() if line.strip()]


def _stage(name: str) -> list[str]:
    """The instructions of stage `name`, without its FROM line."""
    match = re.search(rf"^FROM\s+\S+\s+AS\s+{re.escape(name)}\s*$", DOCKERFILE, re.M | re.I)
    assert match, f"no stage named {name}"
    rest = DOCKERFILE[match.end():]
    following = re.search(r"^FROM\s", rest, re.M)
    return _instructions(rest[: following.start()] if following else rest)


# ── the Dockerfile ──────────────────────────────────────────────────────────


def test_the_build_run_moves_the_traced_dependencies_out_of_the_standalone_output():
    builder = _stage("builder")
    assert builder[-1] == BUILD, "the move ends the build's own RUN, the builder's last instruction"
    assert [line for line in _instructions(DOCKERFILE) if TRACED in line and line.startswith("RUN")] == [BUILD]


def test_the_runtime_ends_with_node_modules_public_static_and_the_server():
    production = _stage("production")
    layers = [line for line in production if LAYER_MAKERS.match(line)]
    assert layers[-len(APP_LAYERS):] == APP_LAYERS
    after = production[production.index(APP_LAYERS[-1]) + 1:]
    changes_files = [line for line in after if line.split()[0].upper() not in CONFIG_ONLY]
    assert changes_files == [], "check-layer-reuse.sh reads the last four layers as the app's"


def test_public_comes_from_the_builder_with_the_files_the_build_writes():
    copies = [line for line in _stage("production") if line.startswith("COPY") and re.search(r"\s\./public$", line)]
    assert copies == [APP_LAYERS[1]], "a copy from the build context would leave out generated public files"


def test_the_user_and_node_are_in_place_before_the_app_layers():
    production = _stage("production")
    first_app = production.index(APP_LAYERS[0])
    user = next(i for i, line in enumerate(production) if line.startswith("RUN") and "adduser" in line)
    node = production.index("COPY --from=node-strip /usr/local/bin/node /usr/local/bin/node")
    assert max(user, node) < first_app


# ── the build context ───────────────────────────────────────────────────────


def _ignore_lines() -> list[str]:
    return [line.strip() for line in DOCKERIGNORE.splitlines() if line.strip() and not line.lstrip().startswith("#")]


def _is_left_out(relative: Path) -> bool:
    name = relative.name
    return "__tests__" in relative.parts or ".test." in name or name.endswith(".md") or name == "tsconfig.tsbuildinfo"


def _shipped_sources() -> list[Path]:
    found = []
    for directory, subdirs, files in os.walk(FRONTEND):
        subdirs[:] = [name for name in subdirs if name not in NOT_SHIPPED_DIRS]
        found += [Path(directory, name) for name in files
                  if Path(name).suffix in SOURCE_SUFFIXES and ".test." not in name]
    return found


def test_the_context_leaves_out_tests_docs_and_the_typescript_build_cache():
    lines = _ignore_lines()
    assert [pattern for pattern in LEFT_OUT if pattern not in lines] == []
    assert "*.md" not in lines, "`**/*.md` covers this directory too"


def test_no_shipped_source_imports_a_file_the_context_leaves_out():
    sources = _shipped_sources()
    assert len(sources) > 100, "the walk found no frontend sources"
    importing = [str(path.relative_to(ROOT)) for path in sources
                 if LEFT_OUT_IMPORT.search(path.read_text(encoding="utf-8", errors="replace"))]
    assert importing == []


def test_the_import_pattern_finds_what_the_context_leaves_out():
    for line in ("import { harness } from '../__tests__/looks-harness'", "import notes from './NOTES.md'",
                 "const helper = require('./sum.test')", "await import(\"./a.test.tsx\")"):
        assert LEFT_OUT_IMPORT.search(line), line
    for line in ("a.download = `${title}.md`", "{ 'text/markdown': ['.md'] }", "import { x } from './test-utils'"):
        assert not LEFT_OUT_IMPORT.search(line), line


def test_public_holds_no_file_the_context_leaves_out():
    public = FRONTEND / "public"
    assert [str(path) for path in public.rglob("*") if _is_left_out(path.relative_to(FRONTEND))] == []


# ── CI ──────────────────────────────────────────────────────────────────────


def _parity_step(name: str) -> tuple[int, str]:
    steps = JOBS["prod-image-parity"]["steps"]
    found = [(i, step["run"]) for i, step in enumerate(steps) if step.get("name") == name]
    assert len(found) == 1, f"expected one prod-image-parity step named {name!r}"
    return found[0]


def _building(tag: str) -> int:
    steps = JOBS["prod-image-parity"]["steps"]
    found = [i for i, step in enumerate(steps) if tag in step.get("run", "")]
    assert len(found) == 1, f"expected one prod-image-parity step running {tag!r}"
    return found[0]


def test_parity_checks_that_the_local_build_reuses_the_saas_layers():
    position, run = _parity_step(REUSE_STEP)
    assert position > max(_building(SAAS_BUILD), _building(LOCAL_BUILD)), "both images exist by then"
    assert run.strip() == "bash frontend/scripts/check-layer-reuse.sh parity-fe parity-fe-local"


def test_parity_looks_inside_the_build_context():
    _, run = _parity_step(CONTEXT_STEP)
    assert "docker build --target development -t parity-fe-context frontend/" in run
    assert "docker run --rm -w /app parity-fe-context sh -c 'find . -path ./node_modules -prune" in run
    # The backend ownership check (US-016) is the parity step found by `find /app`.
    assert "find /app" not in run
    for test in ("-name __tests__", '-name "*.test.*"', '-name "*.md"', "-name tsconfig.tsbuildinfo"):
        assert test in run, test
    assert 'if [ -n "$found" ]' in run and "exit 1" in run


def test_a_change_to_the_reuse_check_runs_the_parity_job():
    match = next(step for step in JOBS["changes"]["steps"] if step.get("id") == "match")
    assert re.search(match["env"]["PROD_IMAGE_PATHS"], "frontend/scripts/check-layer-reuse.sh")


# ── check-layer-reuse.sh, against a stub docker ─────────────────────────────

# `docker image inspect … <image>` prints <image>.layers; `docker run … --entrypoint node
# <image> …` prints <image>.traced. Every call is logged.
_STUB_DOCKER = """#!/bin/bash
echo "docker $*" >> "$STUB_DIR/calls.log"
case "$1" in
  image) cat "$STUB_DIR/${*: -1}.layers" ;;
  run)
    while [ "$#" -gt 0 ] && [ "$1" != "--entrypoint" ]; do shift; done
    cat "$STUB_DIR/$3.traced" ;;
  *) exit 1 ;;
esac
"""
BELOW = ["sha256:alpine", "sha256:workdir", "sha256:apk", "sha256:node", "sha256:user"]
SAAS = BELOW + ["sha256:modules", "sha256:public", "sha256:static-saas", "sha256:server-saas"]
LOCAL_REUSING = BELOW + ["sha256:modules", "sha256:public", "sha256:static-local", "sha256:server-local"]
LOCAL_TRACING_OTHERS = BELOW + ["sha256:modules-local", "sha256:public-local", "sha256:static-local",
                                "sha256:server-local"]


def _run_check(tmp_path: Path, local: list[str], traced: dict | None = None) -> tuple[int, str, list[str]]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker = bin_dir / "docker"
    docker.write_text(_STUB_DOCKER, encoding="utf-8")
    docker.chmod(docker.stat().st_mode | stat.S_IXUSR)
    for image, layers in (("parity-fe", SAAS), ("parity-fe-local", local)):
        (tmp_path / f"{image}.layers").write_text("\n".join(layers) + "\n\n", encoding="utf-8")
    for image, listing in (traced or {}).items():
        (tmp_path / f"{image}.traced").write_text(listing, encoding="utf-8")
    (tmp_path / "calls.log").touch()
    env = {"PATH": f"{bin_dir}{os.pathsep}/usr/bin{os.pathsep}/bin", "STUB_DIR": str(tmp_path),
           "GITHUB_STEP_SUMMARY": str(tmp_path / "summary.md")}
    proc = subprocess.run(["bash", str(CHECK_SCRIPT), "parity-fe", "parity-fe-local"], env=env,
                          stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=30)
    return proc.returncode, proc.stdout + proc.stderr, (tmp_path / "calls.log").read_text().splitlines()


needs_bash = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


@needs_bash
def test_reused_node_modules_and_public_pass_and_go_to_the_summary(tmp_path):
    code, out, calls = _run_check(tmp_path, LOCAL_REUSING)
    assert code == 0, out
    for line in ("node_modules: reused", "public: reused", "static: rebuilt", "server: rebuilt"):
        assert line in out, out
    assert not [call for call in calls if call.startswith("docker run")], "no traced-file listing is needed"
    summary = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "parity-fe-local against parity-fe: node_modules reused · public reused · static rebuilt · server rebuilt" in summary


@needs_bash
@pytest.mark.parametrize(
    ("local", "says"),
    [
        (BELOW + ["sha256:modules", "sha256:public-local", "sha256:static-local", "sha256:server-local"],
         "public/ was rebuilt although node_modules below it was reused"),
        (SAAS, "the two builds made the same bundle"),
        (["sha256:alpine-other"] + LOCAL_REUSING[1:], "layer 1 differs, below node_modules"),
        (LOCAL_REUSING[:-1], "expected the same number, at least 4"),
    ],
    ids=["public-rebuilt", "same-bundle", "other-base", "layer-count"],
)
def test_a_broken_reuse_fails(tmp_path, local, says):
    code, out, _ = _run_check(tmp_path, local)
    assert code == 1, out
    assert says in out, out


@needs_bash
def test_different_traced_dependencies_warn_and_list_the_difference(tmp_path):
    traced = {"parity-fe": "/app/node_modules/a 100644 aaa\n", "parity-fe-local": "/app/node_modules/a 100644 bbb\n"}
    code, out, calls = _run_check(tmp_path, LOCAL_TRACING_OTHERS, traced)
    assert code == 0, out
    assert "::warning::the two builds traced different dependencies" in out
    assert "< /app/node_modules/a 100644 aaa" in out and "> /app/node_modules/a 100644 bbb" in out
    runs = [call for call in calls if call.startswith("docker run")]
    assert [call.split(" -e ")[0] for call in runs] == [
        f"docker run --rm --network none --entrypoint node {image}" for image in ("parity-fe", "parity-fe-local")
    ]


@needs_bash
def test_the_same_traced_files_in_a_rebuilt_layer_fail(tmp_path):
    same = "/app/node_modules/a 100644 aaa\n"
    code, out, _ = _run_check(tmp_path, LOCAL_TRACING_OTHERS, {"parity-fe": same, "parity-fe-local": same})
    assert code == 1, out
    assert "BuildKit did not reuse it" in out


def test_the_check_script_is_strict_bash():
    script = CHECK_SCRIPT.read_text()
    assert script.startswith("#!/usr/bin/env bash\n")
    assert "set -euo pipefail" in script
    assert 'exit "$bad"' in script.rstrip().splitlines()[-1]
    assert os.access(CHECK_SCRIPT, os.X_OK), "test.yml runs it with bash, but keep it executable like its sibling"
