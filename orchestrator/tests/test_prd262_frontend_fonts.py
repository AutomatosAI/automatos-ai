"""PRD-262: building the frontend image fetches nothing from Google Fonts.

next/font/google downloads each family's CSS and font files from Google while `next build`
runs, and a change on Google's side has failed builds that changed no font (7 Oct 2026). The
core fonts are vendored under frontend/app/fonts/core and loaded with next/font/local
(app/core-fonts.ts), as Play's are (app/play-fonts.ts), so `docker build` needs no Google
fetch. This pins that no shipped frontend source imports next/font/google, and that every
file a next/font/local call names is committed.

Pure file reads: no Node, no network.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FRONTEND = ROOT / "frontend"
SOURCE_SUFFIXES = {".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs"}
NOT_SHIPPED_DIRS = {"node_modules", ".next", "__tests__"}
# An import, a dynamic import or a require of next/font/google (or the old @next/font/google).
GOOGLE_FONT_IMPORT = re.compile(
    r"""(?:\bfrom\s*|\bimport\s*\(?\s*|\brequire\s*\(\s*)['"](?:@next/font|next/font)/google['"]"""
)
LOCAL_FONT_PATH = re.compile(r"""\bpath:\s*['"](\./fonts/[^'"]+\.woff2)['"]""")
FONT_MODULES = ("app/core-fonts.ts", "app/play-fonts.ts")


def _shipped_sources() -> list[Path]:
    found = []
    for directory, subdirs, files in os.walk(FRONTEND):
        subdirs[:] = [name for name in subdirs if name not in NOT_SHIPPED_DIRS]
        found += [Path(directory, name) for name in files
                  if Path(name).suffix in SOURCE_SUFFIXES and ".test." not in name]
    return found


def test_no_shipped_source_loads_a_font_from_google():
    sources = _shipped_sources()
    assert len(sources) > 100, "the walk found no frontend sources"
    importing = [str(path.relative_to(ROOT)) for path in sources
                 if GOOGLE_FONT_IMPORT.search(path.read_text(encoding="utf-8", errors="replace"))]
    assert importing == []


def test_the_import_pattern_finds_a_google_font_import():
    for line in ("import { Geist } from 'next/font/google'", 'const f = require("next/font/google")',
                 "import { Inter } from '@next/font/google'"):
        assert GOOGLE_FONT_IMPORT.search(line), line
    for line in ("import localFont from 'next/font/local'", "// next/font/google failed the build"):
        assert not GOOGLE_FONT_IMPORT.search(line), line


def test_the_root_layout_takes_the_core_fonts_from_the_vendored_module():
    layout = (FRONTEND / "app" / "layout.tsx").read_text(encoding="utf-8")
    assert "from '@/app/core-fonts'" in layout
    assert "CORE_FONT_VARIABLE_CLASSES" in layout


def test_every_file_a_local_font_call_names_is_committed():
    for module in FONT_MODULES:
        source = FRONTEND / module
        paths = LOCAL_FONT_PATH.findall(source.read_text(encoding="utf-8"))
        assert paths, f"{module} names no vendored font file"
        missing = [path for path in paths if not (source.parent / path).is_file()]
        assert missing == [], f"{module}: {missing}"
