#!/usr/bin/env python3
"""Record the text splitter fixtures' chunks with langchain (PRD-262 US-026).

Until US-026 the API split text for RAG ingestion with langchain-text-splitters 1.0.0
on langchain-core 1.6.10, the versions its W2 lock installed: a
``RecursiveCharacterTextSplitter`` for text and a ``PythonCodeTextSplitter`` for
Python. US-026 vendored the two classes into
``orchestrator/modules/rag/chunking/text_splitters.py``. This script runs every case
that ``orchestrator/tests/fixtures/text_splitter_cases.py`` defines through langchain's
``create_documents`` and writes each chunk's ``page_content`` and ``metadata`` as JSON.
The committed copy is ``orchestrator/tests/fixtures/text_splitter_chunks.json``; the
permanent test (``orchestrator/tests/test_prd262_text_splitters.py``) reads only that
file.

test.yml's ``reference-goldens`` job installs ``requirements-text-splitters.lock`` here
(the two langchain packages and the rest at the versions the W2 lock installed with
them) into a throwaway venv, runs this script and diffs its output against the
committed copy, so the reference is proven in the run that ships the swap. The script
refuses to run against any other langchain-text-splitters or langchain-core.

By hand, from the repository root, with Python 3.11:

    python3 -m venv /tmp/text-splitters
    /tmp/text-splitters/bin/pip install --no-deps --require-hashes \\
        -r scripts/ci/reference/requirements-text-splitters.lock
    /tmp/text-splitters/bin/python scripts/ci/reference/text_splitters.py \\
        --out orchestrator/tests/fixtures/text_splitter_chunks.json

Exit 0 after writing the chunks, 2 when the installed versions are not the pinned ones.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from importlib import metadata
from pathlib import Path
from types import ModuleType
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[3]
CASES = ROOT / "orchestrator" / "tests" / "fixtures" / "text_splitter_cases.py"
# The splitters and the core they build on, at the versions production ran.
PINNED = {"langchain-text-splitters": "1.0.0", "langchain-core": "1.6.10"}


def load_cases() -> ModuleType:
    """orchestrator/tests/fixtures/text_splitter_cases.py, loaded from its file (the venv has no app)."""
    spec = importlib.util.spec_from_file_location("text_splitter_cases", CASES)
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


def record() -> Dict[str, Dict[str, List[dict]]]:
    """Every case's chunks as langchain's create_documents returns them."""
    import langchain_text_splitters  # only the reference venv has it

    chunks = {}
    for name, case in load_cases().CASES.items():
        splitter = getattr(langchain_text_splitters, case.splitter)(**case.settings())
        documents = splitter.create_documents(case.texts(), metadatas=case.metadata_list())
        chunks[name] = [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in documents]
    return {"cases": chunks}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, help="write the chunks here (default: standard output)")
    args = parser.parse_args(argv)
    errors = version_errors()
    if errors:
        print("::error::the reference needs the pinned splitters: " + "; ".join(errors))
        return 2
    text = json.dumps(record(), indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
