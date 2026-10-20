"""PRD-262 #7: the npm paths render, check and snapshot never load are not in the image.

The js stage deletes the paths js/prune.txt lists, every source map, type
declaration and Markdown file but the licences and notices, and the .bin links
left pointing at nothing. media-render.yml proves nothing loads them (a render, a
check and a snapshot with Node's module loading traced); this checks the image
holds what the Dockerfile says. Runs inside the image (the media-render CI job).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

JS_TREE = Path("/opt/media-render/js/node_modules")
PRUNE_LIST = Path(__file__).resolve().parents[1] / "js" / "prune.txt"
DROPPED_SUFFIXES = (".map", ".d.ts", ".d.mts", ".d.cts", ".md")
KEPT_DOCS = re.compile(r"licen[cs]e|notice", re.I)
# What every render loads: the prune must leave it.
LOADED = ("hyperframes/dist/cli.js", "puppeteer-core/package.json", "sharp/package.json", "esbuild/package.json")


def _pruned() -> List[str]:
    lines = (line.split("#", 1)[0].strip() for line in PRUNE_LIST.read_text().splitlines())
    return [line for line in lines if line]


def test_the_pruned_paths_are_gone_and_what_renders_load_stays():
    pruned = _pruned()
    assert pruned, f"{PRUNE_LIST} lists nothing"
    assert [path for path in pruned if (JS_TREE / path).exists()] == []
    assert [path for path in LOADED if not (JS_TREE / path).is_file()] == []


def test_no_source_map_type_declaration_or_markdown_but_the_licences():
    left = [
        str(path) for path in JS_TREE.rglob("*")
        if path.is_file() and path.name.endswith(DROPPED_SUFFIXES) and not KEPT_DOCS.search(path.name)
    ]
    assert left == []


def test_no_bin_link_points_at_nothing():
    assert [link.name for link in (JS_TREE / ".bin").iterdir() if not link.exists()] == []
    assert (JS_TREE / ".bin" / "hyperframes").exists()
