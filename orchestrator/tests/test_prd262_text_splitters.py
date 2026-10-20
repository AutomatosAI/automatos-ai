"""PRD-262 US-026: the text splitters RAG ingestion uses are vendored, chunk for chunk.

When the semantic chunker is unavailable or fails, ``DocumentProcessor.chunk_document``
splits Python with ``PythonCodeTextSplitter`` and everything else with
``RecursiveCharacterTextSplitter``. Both came from langchain-text-splitters 1.0.0 until
US-026 moved them into modules/rag/chunking/text_splitters.py, restructured into small
functions. This suite holds them to langchain's output without langchain installed:

- for every case tests/fixtures/text_splitter_cases.py defines, the vendored splitter
  returns the committed golden chunks (tests/fixtures/text_splitter_chunks.json),
  page_content and metadata, exactly. test.yml's reference-goldens job re-records the
  goldens with langchain, pinned and hashed, whenever they, the cases or the documents
  change;
- the goldens name every case, and the reference script runs the same cases;
- ``DocumentProcessor`` builds the vendored splitters, and ``chunk_document``'s
  basic-splitter path (the semantic chunker off) returns the golden chunks of the
  Python and plain-text documents, less those its own filter drops. The retrieval eval
  lane never imports the splitters, so it cannot prove this;
- ``create_documents`` gives every chunk its own deep copy of its text's metadata, the
  sizes are checked, ``length_function`` measures the pieces, and a piece no separator
  can cut stays whole;
- the module carries langchain's MIT notice and copies only Python's separators.

text_splitters.py is loaded from its file (standard library only), so sibling tests'
stubs of modules.rag.* in sys.modules cannot get in the way of the parity tests.
"""
from __future__ import annotations

import dataclasses
import importlib
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest
import yaml

from tests.fixtures.reference_locks import pins, unhashed
from tests.fixtures.text_splitter_cases import CASES, CODE, TEXT, read_document

ROOT = Path(__file__).resolve().parents[2]
SPLITTERS = ROOT / "orchestrator" / "modules" / "rag" / "chunking" / "text_splitters.py"
GOLDEN = json.loads((ROOT / "orchestrator" / "tests" / "fixtures" / "text_splitter_chunks.json")
                    .read_text(encoding="utf-8"))["cases"]
REFERENCE = ROOT / "scripts" / "ci" / "reference"
WORKFLOW = ROOT / ".github" / "workflows" / "test.yml"
VENDORED_MODULE = "modules.rag.chunking.text_splitters"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses look their module up by name
    spec.loader.exec_module(module)
    return module


splitters = _load("prd262_text_splitters", SPLITTERS)


def _split(case) -> list:
    splitter = getattr(splitters, case.splitter)(**case.settings())
    chunks = splitter.create_documents(case.texts(), metadatas=case.metadata_list())
    return [{"page_content": chunk.page_content, "metadata": chunk.metadata} for chunk in chunks]


# --------------------------------------------------------------------------- parity


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_vendored_splitters_return_the_golden_chunks_exactly(name):
    assert _split(CASES[name]) == GOLDEN[name], (
        f"{name}: the vendored splitter no longer returns the chunks langchain-text-splitters "
        "1.0.0 records (PRD-262 US-026); the restructure must not change a chunk"
    )


def test_the_goldens_cover_every_case_and_nothing_else():
    assert set(GOLDEN) == set(CASES)
    assert all(GOLDEN[name] for name in CASES), "every case has chunks"
    # The Python splitter cuts the module differently, so chunk_document's code path is
    # told apart from its text path; with no class or def it falls back to the text ladder.
    assert GOLDEN["inventory.py.txt/code"] != GOLDEN["inventory.py.txt/text"]
    assert GOLDEN["guide.md/code"] == GOLDEN["guide.md/text"]
    assert GOLDEN["notes.txt/code"] == GOLDEN["notes.txt/text"]


def test_every_golden_chunk_fits_and_comes_from_its_document():
    for name, case in CASES.items():
        texts = case.texts()
        size = case.settings()["chunk_size"]
        for chunk in GOLDEN[name]:
            content = chunk["page_content"]
            assert content == content.strip() and 0 < len(content) <= size, (name, content[:40])
            assert any(content in text for text in texts), (name, content[:40])


# --------------------------------------------------------------------------- the ingest


@pytest.fixture
def basic_path(monkeypatch):
    """The ingest manager with the semantic chunker off: chunk_document's basic-splitter path."""
    manager = importlib.import_module("modules.rag.ingestion.manager")
    monkeypatch.setattr(manager, "SEMANTIC_CHUNKER_AVAILABLE", False)
    return manager


def test_the_document_processor_builds_the_vendored_splitters(basic_path):
    processor = basic_path.DocumentProcessor()
    built = {name: (type(getattr(processor, name)).__module__, type(getattr(processor, name)).__name__)
             for name in ("text_splitter", "code_splitter")}
    assert built == {"text_splitter": (VENDORED_MODULE, TEXT), "code_splitter": (VENDORED_MODULE, CODE)}
    assert not hasattr(processor, "markdown_splitter"), "it was never read (PRD-262 US-026)"


# chunk_document then drops, with its own filter, a chunk under 50 characters or of fewer
# than 5 words: in the notes, the dock line's 28-character lead-in and the two halves of
# the lot ID run, one word each.
@pytest.mark.parametrize("document, file_type, case, kept", [
    ("inventory.py.txt", "PYTHON", "inventory.py.txt/code", (0, 1, 2, 3)),
    ("notes.txt", "TEXT", "notes.txt/text", (0, 1, 2, 3, 7)),
])
def test_chunk_document_serves_the_golden_chunks_on_its_basic_path(basic_path, document, file_type, case, kept):
    kind = basic_path.DocumentType[file_type]
    chunks = basic_path.DocumentProcessor().chunk_document(read_document(document), kind, {"document_id": 7})
    golden = [chunk["page_content"] for chunk in GOLDEN[case]]
    assert [chunk.content for chunk in chunks] == [golden[index] for index in kept]
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(kept)))
    assert [chunk.metadata for chunk in chunks] == [
        {"file_type": kind.value, "chunk_size": len(golden[index]), "document_id": 7} for index in kept
    ]
    dropped = [text for index, text in enumerate(golden) if index not in kept]
    assert all(len(text) < 50 or len(text.split()) < 5 for text in dropped), dropped


# --------------------------------------------------------------------------- what the vendored code does


def test_every_chunk_gets_its_own_deep_copy_of_its_texts_metadata():
    metadata = {"origin": "Brazil", "tags": ["natural"]}
    splitter = splitters.RecursiveCharacterTextSplitter(chunk_size=10, chunk_overlap=0)
    chunks = splitter.create_documents(["aaaa bbbb cccc", "dddd"], metadatas=[metadata, {"origin": "Colombia"}])
    assert [chunk.page_content for chunk in chunks] == ["aaaa bbbb", "cccc", "dddd"]
    assert [chunk.metadata for chunk in chunks] == [metadata, metadata, {"origin": "Colombia"}]
    first, second = chunks[0].metadata, chunks[1].metadata
    assert first is not metadata and first is not second and first["tags"] is not metadata["tags"]
    plain = splitter.create_documents(["aaaa bbbb cccc"])
    assert [chunk.metadata for chunk in plain] == [{}, {}] and plain[0].metadata is not plain[1].metadata
    with pytest.raises(dataclasses.FrozenInstanceError):
        chunks[0].page_content = "changed"


@pytest.mark.parametrize("size, overlap, message", [
    (0, 0, "chunk_size must be > 0"),
    (10, -1, "chunk_overlap must be >= 0"),
    (10, 11, "larger chunk overlap"),
])
def test_impossible_sizes_are_refused(size, overlap, message):
    for splitter in (splitters.RecursiveCharacterTextSplitter, splitters.PythonCodeTextSplitter):
        with pytest.raises(ValueError, match=message):
            splitter(chunk_size=size, chunk_overlap=overlap)


def test_length_function_measures_the_pieces():
    splitter = splitters.RecursiveCharacterTextSplitter(chunk_size=3, chunk_overlap=1, length_function=lambda piece: 1)
    assert splitter.split_text("a b c d e") == ["a b c", "c d e"]


def test_a_piece_no_separator_can_cut_stays_whole():
    splitter = splitters.RecursiveCharacterTextSplitter(separators=["\n"], chunk_size=5, chunk_overlap=0)
    assert splitter.split_text("abcdefgh\nij") == ["abcdefgh", "ij"]
    assert splitter.split_text("abcdefgh") == ["abcdefgh"], "no separator found: the last one is used"


def test_the_module_carries_the_mit_notice_and_only_pythons_separators():
    source = SPLITTERS.read_text(encoding="utf-8")
    assert "# MIT License" in source and "# Copyright (c) LangChain, Inc." in source
    assert splitters.PYTHON_SEPARATORS == ("\nclass ", "\ndef ", "\n\tdef ", "\n\n", "\n", " ", "")
    assert not hasattr(splitters.RecursiveCharacterTextSplitter, "get_separators_for_language")
    assert not hasattr(splitters, "Language")


# --------------------------------------------------------------------------- the reference and CI


def test_the_reference_lock_is_its_input_pinned_and_hashed():
    wanted = pins(REFERENCE / "requirements-text-splitters.in")
    locked = pins(REFERENCE / "requirements-text-splitters.lock")
    assert wanted and locked == wanted, "every package the reference runs is pinned in the .in"
    text = (REFERENCE / "requirements-text-splitters.lock").read_text(encoding="utf-8")
    assert text.splitlines()[1] == (
        "#    uv pip compile scripts/ci/reference/requirements-text-splitters.in --universal "
        "--python-version 3.11 --only-binary :all: --generate-hashes "
        "-o scripts/ci/reference/requirements-text-splitters.lock"
    )
    assert unhashed(text) == []


def test_the_reference_script_runs_only_the_pinned_splitters(capsys):
    script = _load("prd262_text_splitter_reference", REFERENCE / "text_splitters.py")
    wanted = pins(REFERENCE / "requirements-text-splitters.in")
    assert script.PINNED and all(wanted[name] == version for name, version in script.PINNED.items())
    assert set(script.load_cases().CASES) == set(CASES)
    assert script.version_errors(), "this suite's environment has no langchain"
    assert script.main([]) == 2, "it refuses to record with anything but the pinned versions"
    assert "::error::the reference needs the pinned splitters" in capsys.readouterr().out


def test_ci_re_records_the_chunks_whenever_they_or_their_inputs_change():
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    match = next(step for step in workflow["jobs"]["changes"]["steps"] if step.get("id") == "match")
    paths = re.compile(match["env"]["REFERENCE_PATHS"])
    for path in ("scripts/ci/reference/text_splitters.py", "scripts/ci/reference/requirements-text-splitters.lock",
                 "orchestrator/tests/fixtures/text_splitter_cases.py",
                 "orchestrator/tests/fixtures/text_splitter_chunks.json",
                 "orchestrator/tests/fixtures/text_splitters/notes.txt"):
        assert paths.search(path), path
    step = next(step for step in workflow["jobs"]["reference-goldens"]["steps"]
                if "text_splitters.py" in step.get("run", ""))
    assert step["if"] == "${{ !cancelled() }}", "it runs even when the PDF reference before it failed"
    assert step["env"]["LOCK"] == "scripts/ci/reference/requirements-text-splitters.lock"
    assert step["env"]["CHUNKS"] == "orchestrator/tests/fixtures/text_splitter_chunks.json"
    run = step["run"]
    assert 'python -m venv "$VENV"' in run
    assert '"$VENV/bin/pip" install --no-deps --require-hashes --only-binary :all: -r "$LOCK"' in run
    assert '"$VENV/bin/python" scripts/ci/reference/text_splitters.py --out' in run
    assert 'diff -u <(jq -S . "$CHUNKS")' in run and "exit 1" in run
