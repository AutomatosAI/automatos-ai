"""PRD-262 #29 and #47 (the 19 Oct breakdown): what the standalone server never reads stays out.

The standalone trace reached about 13.7 MB of next's own files (and caniuse-lite) through
requires the production server never runs: next/font's loader and the capsize metrics only
it reads, Babel, browserslist and caniuse-lite, the CSS tools, webpack's crypto polyfill,
Next's file tracer, the dev overlay, and the route runtimes Next picks only when
experimental React or Turbopack is on. A probe of 23 routes and a chat POST read none of
them. React's development builds and Next's React Refresh (0.69 MB, #47) never load
either: the standalone server.js sets NODE_ENV=production, as the Dockerfile does.
next.config.js's STANDALONE_TRACE_EXCLUDES lists them all, which Next applies to the
server's own trace and to every route's. The build's own trace files (43 *.nft.json, 0.39
MB, #47) are read only by deploy tooling, so the build RUN deletes them from the
standalone output before the runtime copies it. And three packages only the browser runs,
loaded on demand in a handler or an effect (Retell's voice client, the LiveKit it brings,
and SheetJS), are resolved to an empty module in the server builds (0.96 MB of server
chunks, #47); the browser bundle keeps them.

This pins:
- every one of those files is matched by a glob, and nothing the server loads is (the
  runtime list holds the files the probe read that sit beside the excluded ones);
- frontend/scripts/check-standalone-image.sh looks for exactly the config's globs, which
  the image's sh can expand (no braces or extglobs);
- the script, run here against a stub docker, fails on a glob that still matches, on a page
  that answers 5xx or nothing, and on a server log that reports a missing module, and
  passes a clean image; it requests every page the build's manifest lists.

In CI the script boots the real images: test.yml's prod-image-parity (both editions) and
images.yml (the published image, both arches).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
FRONTEND = ROOT / "frontend"
NEXT_CONFIG = (FRONTEND / "next.config.js").read_text()
CHECK_SCRIPT = FRONTEND / "scripts" / "check-standalone-image.sh"
NEXT = "node_modules/next/dist/"

# Required only from next/dist/build, the dev server, the dev overlay or each other (#29).
BUILD_ONLY = [
    NEXT + "server/capsize-font-metrics.json",
    NEXT + "server/font-utils.js",
    NEXT + "compiled/@next/font/dist/google/loader.js",
    NEXT + "compiled/babel/bundle.js",
    NEXT + "compiled/babel-packages/packages-bundle.js",
    NEXT + "compiled/browserslist/index.js",
    "node_modules/caniuse-lite/dist/unpacker/agents.js",
    NEXT + "compiled/postcss-preset-env/index.cjs",
    NEXT + "compiled/cssnano-simple/index.js",
    NEXT + "compiled/crypto-browserify/index.js",
    NEXT + "compiled/@vercel/nft/index.js",
    NEXT + "compiled/next-devtools/index.js",
    NEXT + "compiled/next-server/app-page-experimental.runtime.prod.js",
    NEXT + "compiled/next-server/app-page-turbo.runtime.prod.js",
    NEXT + "compiled/next-server/app-page-turbo-experimental.runtime.prod.js",
    NEXT + "compiled/next-server/pages-turbo.runtime.prod.js",
]
# The 8 development files the 19 Oct image held, and React Refresh's production half (#47).
REACT_DEVELOPMENT = [
    "node_modules/react-dom/cjs/react-dom-server-legacy.browser.development.js",
    "node_modules/react-dom/cjs/react-dom-server.browser.development.js",
    "node_modules/react/cjs/react.development.js",
    "node_modules/react/cjs/react-jsx-runtime.development.js",
    NEXT + "compiled/react-refresh/cjs/react-refresh-babel.development.js",
    NEXT + "compiled/react-refresh/cjs/react-refresh-runtime.development.js",
    NEXT + "compiled/react-refresh/cjs/react-refresh-runtime.production.min.js",
    "node_modules/scheduler/cjs/scheduler.development.js",
    NEXT + "compiled/react-is/cjs/react-is.development.js",
]
# What the standalone server loads beside them: it must stay.
RUNTIME = [
    NEXT + "compiled/babel/code-frame.js",
    NEXT + "compiled/babel-code-frame/index.js",
    NEXT + "build/output/log.js",
    NEXT + "build/next-config-ts/transpile-config.js",
    NEXT + "compiled/next-server/app-page.runtime.prod.js",
    NEXT + "compiled/next-server/app-route.runtime.prod.js",
    NEXT + "compiled/next-server/pages.runtime.prod.js",
    NEXT + "compiled/next-server/pages-api.runtime.prod.js",
    NEXT + "compiled/next-server/server.runtime.prod.js",
    NEXT + "compiled/@edge-runtime/primitives/load.js",
    NEXT + "compiled/edge-runtime/index.js",
    NEXT + "shared/lib/modern-browserslist-target.js",
    NEXT + "server/lib/start-server.js",
    NEXT + "server/next-server.js",
    NEXT + "server/route-modules/app-page/module.compiled.js",
    NEXT + "next-devtools/server/shared.js",
    NEXT + "compiled/source-map/source-map.js",
    "node_modules/react-dom/server.browser.js",
    "node_modules/react-dom/cjs/react-dom-server.browser.production.min.js",
    "node_modules/react-dom/cjs/react-dom-server-legacy.browser.production.min.js",
    "node_modules/react/index.js",
    "node_modules/react/cjs/react.production.min.js",
    "node_modules/react/cjs/react-jsx-runtime.production.min.js",
    "node_modules/scheduler/cjs/scheduler.production.min.js",
    NEXT + "compiled/react-is/cjs/react-is.production.js",
    NEXT + "compiled/react-is/index.js",
    "node_modules/@opentelemetry/api/build/src/index.js",
]


CLIENT_ONLY_PACKAGES = ["retell-client-js-sdk", "livekit-client", "xlsx"]
SOURCE_SUFFIXES = {".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs"}
NOT_SHIPPED_DIRS = {"node_modules", ".next", "__tests__"}
PACKAGE_IMPORT = re.compile(
    r"""(?P<how>\bfrom\s*|\bimport\s*\(\s*|\bimport\s*|\brequire\s*\(\s*)"""
    r"""['"](?P<package>retell-client-js-sdk|livekit-client|xlsx)(?:/[^'"]*)?['"]"""
)
USE_CLIENT = re.compile(r"""\s*['"]use client['"]""")


def _config_globs() -> list[str]:
    match = re.search(r"^const STANDALONE_TRACE_EXCLUDES = \[\n(.*?)\n\]$", NEXT_CONFIG, re.M | re.S)
    assert match, "next.config.js defines no STANDALONE_TRACE_EXCLUDES"
    code = "\n".join(line for line in match.group(1).splitlines() if not line.strip().startswith("//"))
    return re.findall(r"'([^']+)'", code)


def _script_globs() -> list[str]:
    match = re.search(r"^TRACE_EXCLUDED=\(\n(.*?)\n\)$", CHECK_SCRIPT.read_text(), re.M | re.S)
    assert match, "check-standalone-image.sh defines no TRACE_EXCLUDED"
    return [line.strip().strip("'") for line in match.group(1).splitlines() if line.strip()]


def _glob_regex(glob: str) -> re.Pattern:
    """picomatch's reading of these globs: ** crosses directories, * and ? do not."""
    parts = re.split(r"(\*\*|\*|\?)", glob)
    wildcards = {"**": ".*", "*": "[^/]*", "?": "[^/]"}
    return re.compile("".join(wildcards.get(part, re.escape(part)) for part in parts))


def _excluded_by(path: str) -> list[str]:
    # Next matches with `contains: true`, against the absolute path in the build (/app).
    return [glob for glob in _config_globs() if _glob_regex(glob).search(f"/app/{path}")]


@pytest.mark.parametrize("path", BUILD_ONLY)
def test_a_build_or_dev_internal_is_traced_out(path):
    assert _excluded_by(path), f"no glob keeps {path} out of the standalone trace"


@pytest.mark.parametrize("path", REACT_DEVELOPMENT)
def test_a_react_development_build_is_traced_out(path):
    assert _excluded_by(path), f"no glob keeps {path} out of the standalone trace"


def test_the_runtime_runs_react_in_production_mode():
    """React's index files pick the development build unless NODE_ENV is production."""
    dockerfile = (FRONTEND / "Dockerfile").read_text()
    production = dockerfile[re.search(r"^FROM\s+\S+\s+AS\s+production\s*$", dockerfile, re.M | re.I).end():]
    assert re.search(r"^ENV NODE_ENV=production$", production, re.M)


@pytest.mark.parametrize("path", RUNTIME)
def test_what_the_server_loads_stays_in_the_trace(path):
    assert _excluded_by(path) == []


def test_the_us_046_globs_stay():
    for glob in ("node_modules/@img/**", "node_modules/sharp/**", "node_modules/typescript/**",
                 NEXT + "compiled/amphtml-validator/**"):
        assert glob in _config_globs(), glob


def test_the_check_script_looks_for_exactly_the_config_globs_and_its_sh_can_expand_each():
    assert _script_globs() == _config_globs()
    for glob in _config_globs():
        assert not re.search(r"[{}\[\]!+@]\(|[{}]", glob), f"{glob}: busybox sh cannot expand braces or extglobs"
        assert glob.startswith("node_modules/"), glob


def test_the_build_deletes_the_trace_files_before_the_runtime_copies_the_standalone_output():
    dockerfile = (FRONTEND / "Dockerfile").read_text()
    builds = [line for line in dockerfile.splitlines() if line.startswith("RUN npm run build ")]
    assert len(builds) == 1, builds
    assert builds[0].endswith(" && find .next/standalone/.next -name '*.nft.json' -delete")
    assert "COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./" in dockerfile
    script = CHECK_SCRIPT.read_text()
    assert """sh -c 'find .next -name "*.nft.json"'""" in script
    assert re.search(r"^check_no_trace_files$", script, re.M)


def test_the_check_script_requests_every_page_and_reads_the_log_for_missing_modules():
    script = CHECK_SCRIPT.read_text()
    assert 'require("/app/.next/app-path-routes-manifest.json")' in script
    assert re.search(r'^expect_status "\$OPTIMISER_PATH" 404 .*\ncheck_every_page$', script, re.M)
    assert "awk '$1 == 0 || $1 >= 500'" in script
    assert 'grep -qE "MODULE_NOT_FOUND|Cannot find module" <<< "$logs"' in script


# ── client-only packages, out of the server bundle ──────────────────────────


def _shipped_sources() -> list[Path]:
    found = []
    for directory, subdirs, files in os.walk(FRONTEND):
        subdirs[:] = [name for name in subdirs if name not in NOT_SHIPPED_DIRS]
        found += [Path(directory, name) for name in files
                  if Path(name).suffix in SOURCE_SUFFIXES and ".test." not in name]
    return found


def test_the_server_builds_resolve_the_client_only_packages_to_an_empty_module():
    match = re.search(r"^const CLIENT_ONLY_PACKAGES = (\[.*\])$", NEXT_CONFIG, re.M)
    assert match, "next.config.js defines no CLIENT_ONLY_PACKAGES"
    assert json.loads(match.group(1).replace("'", '"')) == CLIENT_ONLY_PACKAGES
    assert "  if (!isServer) return config\n" in NEXT_CONFIG
    assert "Object.fromEntries(CLIENT_ONLY_PACKAGES.map((name) => [name, false]))" in NEXT_CONFIG
    assert re.search(r"^  webpack: withoutClientOnlyPackagesOnTheServer,$", NEXT_CONFIG, re.M)
    assert not re.search(r"\bexternals\b", NEXT_CONFIG.split("const nextConfig")[1]), "externals would trace them in"


def _is_dynamic(found: re.Match) -> bool:
    return found.group("how").startswith("import") and "(" in found.group("how")


def test_the_source_loads_each_client_only_package_only_on_demand_in_client_code():
    """A static import would put the package back in the server render, where it is now empty."""
    loads = {}
    for path in _shipped_sources():
        text = path.read_text(encoding="utf-8", errors="replace")
        for match in PACKAGE_IMPORT.finditer(text):
            name = path.relative_to(FRONTEND).as_posix()
            assert _is_dynamic(match), f"{name}: a static import of {match.group('package')}"
            assert USE_CLIENT.match(text), f"{name} is not a client module"
            loads.setdefault(match.group("package"), []).append(name)
    assert loads == {
        # Theme voice: the Live engine's one Retell loader (hooks/use-retell-call.ts is deleted).
        "retell-client-js-sdk": ["lib/voice/live/retell-call.ts"],
        "xlsx": ["components/widgets/FileWidget/FilePreview.tsx"],
    }


def test_the_import_pattern_tells_a_dynamic_import_from_a_static_one():
    for line in ("const XLSX = await import('xlsx')", 'const { RetellWebClient } = await import("retell-client-js-sdk")'):
        found = PACKAGE_IMPORT.search(line)
        assert found and _is_dynamic(found), line
    for line in ("import * as XLSX from 'xlsx'", "import 'livekit-client'", "const x = require('xlsx/dist/x')",
                 "export { utils } from 'xlsx'"):
        found = PACKAGE_IMPORT.search(line)
        assert found and not _is_dynamic(found), line


def test_the_check_script_looks_for_each_client_only_package_in_the_server_output():
    script = CHECK_SCRIPT.read_text()
    block = re.search(r"^CLIENT_ONLY_MARKERS=\(\n(.*?)\n\)$", script, re.M | re.S)
    assert block, "check-standalone-image.sh defines no CLIENT_ONLY_MARKERS"
    entries = [line.strip().strip("'") for line in block.group(1).splitlines()]
    assert [entry.split(":", 1)[0] for entry in entries] == CLIENT_ONLY_PACKAGES
    assert 'grep -rqF -- "${entry#*:}" .next/server' in script
    assert re.search(r"^check_client_only_packages$", script, re.M)
    sources = "\n".join(path.read_text(encoding="utf-8", errors="replace") for path in _shipped_sources())
    for entry in entries:
        assert entry.split(":", 1)[1] not in sources, f"{entry}: our own code carries the marker"


# ── the script, against a stub docker ───────────────────────────────────────

# `docker run --rm <image> sh -c <script> sh <args>` runs <script> with the host's sh in the
# fake /app; `node -v` and the ELF reader print canned output; `docker exec <c> node -e <js>`
# answers the page walk from pages.txt and a single request from status.txt (200 otherwise);
# `docker logs` prints log.txt.
_STUB_DOCKER = r"""#!/bin/bash
echo "docker $1" >> "$STUB_DIR/calls.log"
case "$1" in
  run)
    shift
    if [ "$1" = "-d" ]; then echo started; exit 0; fi
    shift 2
    case "$1 $2" in
      "sh -c") cd "$STUB_DIR/app" && exec sh -c "$3" "${@:4}" ;;
      "node -v") echo v24.22.0 ;;
      "node -e") cat "$STUB_DIR/sections.txt" ;;
      *) exit 9 ;;
    esac ;;
  inspect) echo true ;;
  exec)
    case "$5" in
      *app-path-routes-manifest*) cat "$STUB_DIR/pages.txt" ;;
      *) status=$(awk -v p="$6" '$1 == p {print $2}' "$STUB_DIR/status.txt"); echo "${status:-200}" ;;
    esac ;;
  logs) cat "$STUB_DIR/log.txt" ;;
  rm) ;;
  *) exit 9 ;;
esac
"""
CLEAN_APP = [
    ".next/server/app/chat/page.js",
    ".next/app-path-routes-manifest.json",
    NEXT + "compiled/next-server/app-page.runtime.prod.js",
    NEXT + "compiled/next-server/pages.runtime.prod.js",
    NEXT + "compiled/babel/code-frame.js",
    "server.js",
]
CLEAN_PAGES = "200 /chat\n307 /settings\n404 /_not-found\n"
CLEAN_LOG = "  ▲ Next.js 15.5.27\n ✓ Ready in 120ms\n"

needs_bash = pytest.mark.skipif(shutil.which("bash") is None or shutil.which("sh") is None, reason="needs bash")


def _run_check(tmp_path: Path, app=CLEAN_APP, pages=CLEAN_PAGES, log=CLEAN_LOG) -> tuple[int, str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker = bin_dir / "docker"
    docker.write_text(_STUB_DOCKER, encoding="utf-8")
    docker.chmod(docker.stat().st_mode | stat.S_IXUSR)
    for relative in app:
        (tmp_path / "app" / relative).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "app" / relative).write_text("x", encoding="utf-8")
    (tmp_path / "app").mkdir(exist_ok=True)
    for name, text in (("sections.txt", ""), ("pages.txt", pages), ("log.txt", log),
                       ("status.txt", "/_next/image?url=%2Fbrand%2Fautomatos-mark.png&w=96&q=75 404\n")):
        (tmp_path / name).write_text(text, encoding="utf-8")
    env = {"PATH": f"{bin_dir}{os.pathsep}/usr/bin{os.pathsep}/bin", "STUB_DIR": str(tmp_path)}
    proc = subprocess.run(["bash", str(CHECK_SCRIPT), "parity-fe-local", "/chat"], env=env,
                          stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=60)
    return proc.returncode, proc.stdout + proc.stderr


@needs_bash
def test_a_clean_image_passes_and_every_page_is_requested(tmp_path):
    code, out = _run_check(tmp_path)
    assert code == 0, out
    assert "absent from /app: node_modules/@img/**" in out
    assert "3 pages answered" in out
    assert "no *.nft.json in /app/.next" in out
    assert "no client-only package in /app/.next/server: retell-client-js-sdk livekit-client xlsx" in out
    assert "307 /settings" in out


@needs_bash
def test_a_glob_that_still_matches_fails_once_by_its_glob(tmp_path):
    leftovers = [NEXT + "compiled/next-server/app-page-turbo.runtime.prod.js",
                 NEXT + "compiled/next-server/pages-turbo.runtime.prod.js",
                 "node_modules/caniuse-lite/data/agents.js"]
    code, out = _run_check(tmp_path, app=CLEAN_APP + leftovers)
    assert code == 1, out
    carried = re.findall(r"the standalone build carries (\S+)", out)
    # In the config's order, each glob once, however many files it still matches.
    assert carried == ["node_modules/caniuse-lite/**", NEXT + "compiled/next-server/*-turbo.runtime.prod.js"]


@needs_bash
def test_client_only_package_code_in_the_server_output_fails(tmp_path):
    leaking = tmp_path / "app" / ".next" / "server" / "chunks" / "7752.js"
    leaking.parent.mkdir(parents=True)
    leaking.write_text('var h="<html><head><title>SheetJS Table Export</title></head>";', encoding="utf-8")
    code, out = _run_check(tmp_path)
    assert code == 1, out
    assert "the server output carries xlsx, which only the browser runs" in out
    assert "carries livekit-client" not in out


@needs_bash
def test_a_trace_file_left_in_the_image_fails(tmp_path):
    code, out = _run_check(tmp_path, app=CLEAN_APP + [".next/server/app/chat/page.js.nft.json"])
    assert code == 1, out
    assert "/app/.next holds trace files (*.nft.json): 1, such as .next/server/app/chat/page.js.nft.json" in out


@needs_bash
@pytest.mark.parametrize("pages", ["200 /chat\n500 /settings\n", "200 /chat\n0 /settings\n"], ids=["5xx", "no-answer"])
def test_a_page_that_errors_or_never_answers_fails(tmp_path, pages):
    code, out = _run_check(tmp_path, pages=pages)
    assert code == 1, out
    assert f"pages that answered 5xx or nothing: {pages.splitlines()[1]}" in out


@needs_bash
def test_a_manifest_with_no_page_fails(tmp_path):
    code, out = _run_check(tmp_path, pages="")
    assert code == 1, out
    assert "app-path-routes-manifest.json lists no page" in out


@needs_bash
@pytest.mark.parametrize("line", [
    "Error: Cannot find module 'next/dist/compiled/babel/bundle.js'",
    "  code: 'MODULE_NOT_FOUND',",
])
def test_a_missing_module_in_the_log_fails(tmp_path, line):
    code, out = _run_check(tmp_path, log=CLEAN_LOG + line + "\n")
    assert code == 1, out
    assert "the server log reports a missing module" in out
