#!/usr/bin/env python3
"""Record the PDF table fixtures' cell grids with camelot's stream mode (PRD-262 US-023).

Until US-023 the API extracted PDF tables with camelot-py 0.11.0: lattice mode first,
which failed on every PDF in the image (its ghostscript backend has no Python binding
there), then stream mode. Stream output is therefore the only table output production
ever stored, and the reference pdfplumber is judged against. This script runs stream
mode, with the call the API made (``pages="all"``), on the fixtures that
``orchestrator/tests/fixtures/pdf_tables.py`` draws, and writes each table's cell grid,
page by page, as JSON. The committed copy is
``orchestrator/tests/fixtures/pdf_table_grids.json``; the permanent test
(``orchestrator/tests/test_prd262_pdf_tables.py``) reads only that file.

test.yml's ``reference-goldens`` job installs ``requirements-pdf-tables.lock`` here
(camelot-py 0.11.0, opencv-python-headless 5.0.0.93 and the rest at the versions the
API's W2 lock installed with them) into a throwaway venv, runs this script and diffs
its output against the committed copy, so the reference is proven in the run that
ships the swap. The script refuses to run against any other camelot or opencv.

By hand, from the repository root, with Python 3.11:

    python3 -m venv /tmp/pdf-tables
    /tmp/pdf-tables/bin/pip install --no-deps --require-hashes \\
        -r scripts/ci/reference/requirements-pdf-tables.lock
    /tmp/pdf-tables/bin/python scripts/ci/reference/pdf_tables.py \\
        --out orchestrator/tests/fixtures/pdf_table_grids.json

Exit 0 after writing the grids, 2 when the installed versions are not the pinned ones.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
import warnings
from importlib import metadata
from pathlib import Path
from types import ModuleType
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / "orchestrator" / "tests" / "fixtures" / "pdf_tables.py"
# The extractor and the opencv it imports, at the versions production ran.
PINNED = {"camelot-py": "0.11.0", "opencv-python-headless": "5.0.0.93"}


def load_fixtures() -> ModuleType:
    """orchestrator/tests/fixtures/pdf_tables.py, loaded from its file (the venv has no app)."""
    spec = importlib.util.spec_from_file_location("pdf_table_fixtures", FIXTURES)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # its dataclasses look their module up by name
    spec.loader.exec_module(module)
    return module


def version_errors() -> List[str]:
    """Each pinned distribution that is missing or at another version."""
    errors = []
    for name, wanted in PINNED.items():
        try:
            found = metadata.version(name)
        except metadata.PackageNotFoundError:
            found = "not installed"
        if found != wanted:
            errors.append(f"{name}: wanted {wanted}, found {found}")
    return errors


def record(directory: Path) -> Dict[str, Dict[str, List[dict]]]:
    """Every fixture's tables as stream mode finds them: page number and cell grid."""
    import camelot  # only the reference venv has it

    grids = {}
    for name, path in load_fixtures().write_fixtures(directory).items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # camelot warns on every page it reads
            tables = camelot.read_pdf(str(path), pages="all", flavor="stream", suppress_stdout=True)
        grids[name] = [{"page": int(table.page), "rows": table.df.values.tolist()} for table in tables]
    return {"fixtures": grids}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, help="write the grids here (default: standard output)")
    args = parser.parse_args(argv)
    errors = version_errors()
    if errors:
        print("::error::the reference needs the pinned extractor: " + "; ".join(errors))
        return 2
    with tempfile.TemporaryDirectory() as directory:
        grids = record(Path(directory))
    text = json.dumps(grids, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
