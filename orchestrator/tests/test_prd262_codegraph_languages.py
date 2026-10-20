"""PRD-262 US-028: the code-graph parser reads every language it lists, off the network.

modules/codegraph/parsers/treesitter_parser.py parses each language in SUPPORTED_LANGUAGES
with the grammar of its own installed tree_sitter_* package (TreeSitterParser.GRAMMARS).
tree-sitter-language-pack, which since 1.0.0 ships no grammars and downloaded each one the
first time it was used, is gone, and so is the fallback that covered three languages and
called a language() that tree_sitter_typescript does not have. Ruby's definitions are
named after their keywords (`class`, `method`), and Swift names a function with a
simple_identifier, so both now yield symbols.

One fixture per SUPPORTED_LANGUAGES extension, in tests/fixtures/codegraph_languages,
parses without a syntax error under its grammar and yields the symbols it defines.

CI runs this file in the backend shards and, inside the API image, in the throwaway test
image with ``--network none`` (scripts/ci/image_test_image.py, test.yml's
prod-image-parity): a grammar the image lacked could not be fetched there, so it would
fail here. test_prd262_codegraph_grammars.py checks the requirements and that CI step.
Nothing here reads outside orchestrator/tests, which is all the test image mounts.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest

# CI collection-order safety net: a sibling's stub of a modules.* package must not stand
# in for the real one when this module imports the parser.
import tests.conftest as _conftest  # noqa: E402

_conftest._restore_real_app_modules()

import modules.codegraph.parsers.treesitter_parser as treesitter_parser  # noqa: E402
from modules.codegraph.parsers.treesitter_parser import TreeSitterParser  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "codegraph_languages"

# Extension -> the (name, symbol type) pairs its fixture defines. A grammar may yield more
# than these (Java's and C#'s interface methods, Swift's bow), never fewer.
EXPECTED = {
    ".py": {("Greeter", "class"), ("greet", "method"), ("make_greeter", "function")},
    ".js": {("Greeter", "class"), ("greet", "method"), ("makeGreeter", "function")},
    ".jsx": {("Greeting", "function"), ("Panel", "class"), ("render", "method")},
    ".ts": {("Greeting", "interface"), ("Greeter", "class"), ("greet", "method"), ("makeGreeter", "function")},
    ".tsx": {("GreetingProps", "interface"), ("Greeting", "function"), ("Panel", "class"), ("render", "method")},
    ".java": {("Greeter", "class"), ("greet", "method"), ("Polite", "interface")},
    ".go": {("Greet", "function"), ("MakeGreeting", "function")},
    ".rs": {("Greeter", "class"), ("greet", "method"), ("Polite", "interface"), ("make_greeter", "function")},
    ".cpp": {("Greeter", "class"), ("add", "function")},
    ".c": {("add", "function"), ("greet", "function")},
    ".rb": {("Greeter", "class"), ("greet", "method"), ("make_greeter", "function")},
    ".php": {("Greeter", "class"), ("greet", "method"), ("make_greeter", "function")},
    ".swift": {("Polite", "interface"), ("Greeter", "class"), ("greet", "method"), ("makeGreeter", "function")},
    ".kt": {("Greeter", "class"), ("greet", "method"), ("makeGreeter", "function")},
    ".cs": {("IPolite", "interface"), ("Greeter", "class"), ("Greet", "method")},
}


class _UnloadableGrammars(TreeSitterParser):
    """Go's package is missing, and TypeScript's is asked for the language() it lacks."""

    GRAMMARS = {
        **TreeSitterParser.GRAMMARS,
        "go": ("tree_sitter_prd262_missing", "language"),
        "typescript": ("tree_sitter_typescript", "language"),
    }


@pytest.fixture(scope="module")
def parser() -> TreeSitterParser:
    return TreeSitterParser()


def _fixture(extension: str) -> Path:
    return FIXTURES / f"sample{extension}"


def test_every_supported_extension_has_one_fixture():
    assert set(EXPECTED) == set(TreeSitterParser.SUPPORTED_LANGUAGES)
    assert sorted(path.name for path in FIXTURES.glob("sample.*")) == sorted(
        _fixture(extension).name for extension in EXPECTED
    )


def test_every_language_has_a_grammar_package_of_its_own():
    """TypeScript and TSX come from one package with a function each; PHP from language_php()."""
    grammars = TreeSitterParser.GRAMMARS
    assert set(grammars) == set(TreeSitterParser.SUPPORTED_LANGUAGES.values())
    assert all(module.startswith("tree_sitter_") for module, _ in grammars.values())
    assert grammars["typescript"] == ("tree_sitter_typescript", "language_typescript")
    assert grammars["tsx"] == ("tree_sitter_typescript", "language_tsx")
    assert grammars["php"] == ("tree_sitter_php", "language_php")


def test_tree_sitter_is_available(parser):
    assert parser.available


@pytest.mark.parametrize("extension", sorted(EXPECTED))
def test_the_fixture_parses_and_yields_its_symbols(parser, extension):
    path = _fixture(extension)
    content = path.read_text(encoding="utf-8")
    language = TreeSitterParser.SUPPORTED_LANGUAGES[extension]

    result = parser.parse_file(str(path), content, repo_root=str(FIXTURES))

    assert result.language == language
    assert result.errors == [], result.errors
    found = {(symbol.name, symbol.symbol_type) for symbol in result.symbols}
    assert EXPECTED[extension] <= found, f"{path.name}: {sorted(found)}"
    assert all(symbol.file_path == path.name for symbol in result.symbols)
    tree = parser._get_parser(language).parse(content.encode("utf-8"))
    assert not tree.root_node.has_error, f"{path.name} does not parse cleanly under the {language} grammar"


def test_rubys_keyword_nodes_count_in_ruby_only(parser):
    """JavaScript's `class` node is a class expression: it stays out of the symbols, as before."""
    result = parser.parse_file("expression.js", "const Named = class Inner {\n  run() {}\n};\n")
    assert result.errors == []
    assert "Inner" not in {symbol.name for symbol in result.symbols}


def test_a_grammar_that_does_not_load_fails_its_files_and_is_logged_once(monkeypatch):
    log = Mock()
    monkeypatch.setattr(treesitter_parser, "logger", log)
    parser = _UnloadableGrammars()

    for name in ("first.go", "second.go"):
        result = parser.parse_file(name, "package main\n")
        assert (result.errors, result.symbols) == (["No parser available for go"], [])
    result = parser.parse_file("third.ts", "let answer = 42;\n")
    assert result.errors == ["No parser available for typescript"]

    assert [call.args[1:] for call in log.exception.call_args_list] == [
        ("go", "tree_sitter_prd262_missing", "language"),
        ("typescript", "tree_sitter_typescript", "language"),
    ], "each failure is logged once, and the second Go file does not import again"
    assert parser.parse_file("fourth.py", "def ok():\n    return 1\n").errors == [], "other grammars still load"
