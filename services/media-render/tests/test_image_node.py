"""PRD-262 #22: the image's node binary is stripped, and native addons still load.

The node-stripped stage strips Node's symbol tables (about 20 MB). Native addons
resolve Node-API through the dynamic symbol table, which strip keeps: sharp's
addon, which the Hyperframes CLI imports at every start, must still load. Runs
inside the image (the media-render CI job).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from elf import section_names

JS_TREE = Path("/opt/media-render/js/node_modules")
SHARP_VERSIONS = "const sharp = require(process.argv[1]); console.log(JSON.stringify(sharp.versions));"


def test_node_carries_no_symbol_table_and_keeps_its_dynamic_symbols():
    sections = section_names(Path(shutil.which("node") or "").resolve())
    assert ".symtab" not in sections and ".dynsym" in sections


def test_node_runs_and_sharps_native_addon_loads():
    run = subprocess.run(["node", "-e", SHARP_VERSIONS, str(JS_TREE / "sharp")], capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stderr
    assert "vips" in run.stdout
