"""PRD-262 #7 (19 Oct breakdown): media-render's npm tree holds what renders load.

About 75 MB of the js stage's tree is never loaded by `hyperframes render`, `check`
or `snapshot`, the three commands the service runs: Hyperframes' preview studio,
skills and font-localising CLI, prettier and giget (declared, never imported),
WebDriver BiDi (chromium-bidi, zod, its protocol: Firefox or protocol
'webDriverBiDi' only), devtools-protocol (types only), puppeteer-core's sources and
browser bundle, and every source map, type declaration and Markdown file. This
suite pins:

- js/prune.txt, the list, against what the CLI loads;
- the js stage deleting it in the RUN that runs `npm ci`, with `rm -r` so a path a
  Hyperframes bump moved fails the build, the licences kept, the dangling .bin
  links gone;
- media-render.yml's proof: the fixture rendered (a check, then a render) and
  previewed (a check, then a snapshot) with Node's module loading traced
  (ci/trace_modules.mjs, never in the image), judged by ci/assert_module_trace.py
  (its own tests: test_prd262_media_render_module_trace.py);
- the in-image suite's checks of the tree (services/media-render/tests/test_image_npm.py).

Pure file reads: no Docker, no network.
"""
from __future__ import annotations

import json
import re

from tests.test_prd262_media_render_diet import _run_with
from tests.test_prd262_media_render_stages import MEDIA_RENDER, _stage, _steps

PRUNE_LIST = MEDIA_RENDER / "js" / "prune.txt"
LOCK = json.loads((MEDIA_RENDER / "js" / "package-lock.json").read_text())
TRACE_STEP = "Render, check and snapshot never load the pruned npm paths (Node's module loading traced)"
SIZE_STEP = "The image is at most 1.30 GB on disk (amd64)"

# What render, check and snapshot never load (hyperframes 0.8.62's cli.js, read 19 Oct).
PRUNED = (
    "hyperframes/dist/studio",
    "hyperframes/dist/skills",
    "hyperframes/dist/fontLocalizeCli.js",
    "hyperframes/bin/hyperframes-localize-fonts.mjs",
    "prettier",
    "giget",
    "chromium-bidi",
    "zod",
    "webdriver-bidi-protocol",
    "devtools-protocol",
    "puppeteer-core/src",
    "puppeteer-core/lib/es5-iife",
)
# cli.js's static imports, puppeteer-core's CDP path and GSAP: never pruned whole.
LOADED = (
    "@hono/node-server", "adm-zip", "compare-versions", "css-tree", "esbuild", "fontkit", "hono",
    "ignore", "open", "postcss", "puppeteer-core", "sharp", "@puppeteer/browsers", "ws", "gsap",
)


def _read_prune() -> list[str]:
    lines = (line.split("#", 1)[0].strip() for line in PRUNE_LIST.read_text().splitlines())
    return [line for line in lines if line]


def test_prune_txt_lists_exactly_the_paths_no_render_loads():
    pruned = _read_prune()
    assert tuple(pruned) == PRUNED
    for path in pruned:
        assert not path.startswith("/") and ".." not in path.split("/") and not re.search(r"[*?\[\s]", path), path


def test_every_pruned_package_is_a_production_package_of_the_lock():
    for path in PRUNED:
        name = "/".join(path.split("/")[:2]) if path.startswith("@") else path.split("/")[0]
        entry = LOCK["packages"].get(f"node_modules/{name}")
        assert entry and not entry.get("dev"), f"{name} is not installed by npm ci --omit=dev"


def test_nothing_the_cli_loads_is_pruned_whole():
    assert not {path for path in PRUNED if "/" not in path} & set(LOADED)
    # Of those, only named paths inside hyperframes and puppeteer-core go.
    assert {path.split("/")[0] for path in PRUNED if "/" in path} == {"hyperframes", "puppeteer-core"}


def test_the_js_stage_prunes_strictly_in_the_run_that_installs():
    assert "COPY js/package.json js/package-lock.json js/prune.txt ./" in _stage("js")
    run = _run_with("js", "npm ci")
    install = run.index("npm ci --omit=dev --no-fund --no-audit")
    prune = run.index("sed -e 's/#.*//' -e '/^[[:space:]]*$/d' prune.txt | (cd node_modules && xargs rm -r --)")
    drop = run.index(
        "find node_modules -type f \\( -name '*.map' -o -name '*.d.ts' -o -name '*.d.mts' -o -name '*.d.cts' -o -name '*.md' \\)"
        " ! -iname '*licen[cs]e*' ! -iname '*notice*' -delete"
    )
    links = run.index("find node_modules/.bin -xtype l -delete")
    assert install < prune < drop < links
    # rm -r, not rm -rf: a path a Hyperframes bump moved fails the build.
    assert "rm -rf" not in run


def test_the_trace_hook_observes_and_stays_out_of_the_image():
    assert "ci/" in (MEDIA_RENDER / ".dockerignore").read_text().splitlines()
    hook = (MEDIA_RENDER / "ci" / "trace_modules.mjs").read_text()
    assert 'import { registerHooks } from "node:module";' in hook
    assert 'const LOG = "/trace/modules.log";' in hook
    for logged in ("record({ argv: process.argv.slice(1) });", "record({ file: fileURLToPath(result.url) });",
                   "record({ failed: specifier, parent: context.parentURL ?? null });"):
        assert logged in hook, logged
    # A failure still reaches the caller: the hook never changes what resolves.
    assert "throw error;" in hook and "return result;" in hook


def test_media_render_yml_renders_checks_and_snapshots_with_module_loading_traced():
    (step,) = [step["run"] for step in _steps("media-render") if step.get("name") == TRACE_STEP]
    assert "-e NODE_OPTIONS=--import=/ci/trace_modules.mjs" in step
    assert '-v "$PWD/services/media-render/ci:/ci:ro" -v "$out/trace:/trace"' in step
    assert 'b["preview"] = {"at": [1.5]}' in step
    assert '--bundle "$out/render.json" --out "$out/render" --output render.mp4' in step
    assert '--bundle "$out/preview.json" --out "$out/preview" --output preview-01.png' in step
    assert (
        'python3 services/media-render/ci/assert_module_trace.py --trace "$out/trace/modules.log"'
        " --prune services/media-render/js/prune.txt --commands check,render,snapshot"
    ) in re.sub(r"\\\n\s*", "", step)
    names = [step.get("name") for step in _steps("media-render")]
    assert names.index(TRACE_STEP) < names.index(SIZE_STEP)


def test_the_in_image_suite_checks_the_tree():
    suite = (MEDIA_RENDER / "tests" / "test_image_npm.py").read_text()
    for needle in (
        'PRUNE_LIST = Path(__file__).resolve().parents[1] / "js" / "prune.txt"',
        "assert [path for path in pruned if (JS_TREE / path).exists()] == []",
        "def test_no_source_map_type_declaration_or_markdown_but_the_licences():",
        "def test_no_bin_link_points_at_nothing():",
    ):
        assert needle in suite, needle
