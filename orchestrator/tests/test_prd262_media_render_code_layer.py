"""PRD-262 (19 Oct breakdown, the note under the table): media-render's code is its last layer.

The versions record (2.3 MB of stdlib .pyc written at build), Node's compile cache
(4.2 MB) and the Chrome smoke test need none of the app's code, so they run before
it: a code-only change rebuilds the one `COPY . .` at the end, about 6.6 MB less
per rebuild. The versions RUN gets media_render/__init__.py and versions.py alone,
so versions.py must import nothing but the standard library.

Pure file reads: no Docker, no network.
"""
from __future__ import annotations

import ast
import sys

from tests.test_prd262_media_render_stages import MEDIA_RENDER, _stage


def test_the_code_is_the_last_layer_of_the_runtime():
    runtime = [line for line in _stage("runtime").splitlines() if line.strip()]
    versions_copy = runtime.index("COPY media_render/__init__.py media_render/versions.py /app/media_render/")
    versions_run = next(i for i, line in enumerate(runtime) if line.startswith("RUN python -m media_render.versions"))
    seed = runtime.index("RUN hyperframes --version")
    smoke = next(i for i, line in enumerate(runtime) if line.startswith("RUN timeout 120 /opt/chrome/chrome-headless-shell"))
    code = runtime.index("COPY . .")
    assert versions_copy < versions_run < seed < smoke < code
    assert not [line for line in runtime[code + 1 :] if line.startswith(("RUN ", "COPY "))]


def test_versions_py_needs_nothing_but_the_standard_library():
    tree = ast.parse((MEDIA_RENDER / "media_render" / "versions.py").read_text())
    imported = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    imported |= {node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
    assert not [node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.level]
    assert imported - {"__future__"} <= set(sys.stdlib_module_names), imported
    package = ast.parse((MEDIA_RENDER / "media_render" / "__init__.py").read_text())
    imports = [node for node in ast.walk(package) if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert imports == [], "the package's __init__ runs beside versions.py alone"
