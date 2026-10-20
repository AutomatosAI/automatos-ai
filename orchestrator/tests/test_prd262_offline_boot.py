"""PRD-262 US-033: the API boots with no internet, and counts tokens with none.

tiktoken downloads an encoding it has no cached copy of, through ``requests.get`` with
no timeout, into a temp directory no new container keeps. Three modules loaded an
encoder at import (``modules/rag/service.py``, ``context_optimizer.py`` and
``core/context_guard.py``), two of them with the fallback outside any ``try``. So
refused egress killed ``import main``, and egress that drops packets hung it for about
two minutes per attempt. Now:

* ``core.context_guard.get_tokenizer()`` is the one cached getter: the encoding loads on
  the first count, once, and a failure is logged once and estimated (~4 characters per
  token) instead of retried on every count;
* the RAG service and the context optimizer count through ``core.context_guard``, and
  nothing else in orchestrator/ imports tiktoken;
* nothing in orchestrator/ loads an encoding at import (this file's AST walk);
* the image fetches every encoding the code requests into ``TIKTOKEN_CACHE_DIR`` at
  build time, in the stage both servers start from, and proves the cache serves them
  through a proxy that refuses every connection.

``scripts/ci/smoke-egress.sh`` boots the backend with its egress cut
(test_prd262_smoke_egress.py holds it). Pure file reads and a fake tiktoken: no Docker,
no network.
"""
from __future__ import annotations

import ast
import importlib.util
import logging
import re
import sys
import types
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple

import pytest

ORCH = Path(__file__).resolve().parents[1]
DOCKERFILE = (ORCH / "Dockerfile").read_text(encoding="utf-8")
CONTEXT_GUARD = ORCH / "core" / "context_guard.py"
# The modules whose import-time encoders US-033 replaced with the shared counter.
COUNTER_USERS = (ORCH / "modules" / "rag" / "service.py",
                 ORCH / "modules" / "search" / "optimization" / "context_optimizer.py")
# tiktoken's calls that load an encoding, and the getter and counters that call them.
LOADS_AN_ENCODING = frozenset({
    "get_encoding", "encoding_for_model",
    "get_tokenizer", "count_tokens", "count_message_tokens", "count_tool_tokens", "truncate_to_token_budget",
})
CACHE_DIR = "/opt/tiktoken"
REFUSING_PROXY = "http://127.0.0.1:9"


# --------------------------------------------------------------------------- AST helpers


def _called_name(call: ast.Call) -> Optional[str]:
    """``get_encoding`` for ``tiktoken.get_encoding(...)`` and for ``get_encoding(...)``."""
    func = call.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def _is_main_guard(node: ast.AST) -> bool:
    """``if __name__ == "__main__":``, whose body never runs on import."""
    if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
        return False
    left = node.test.left
    return isinstance(left, ast.Name) and left.id == "__name__"


def _visit(node: ast.AST, deferred: bool, names: frozenset, found: List[Tuple[int, str]]) -> None:
    """Walk ``node``; ``deferred`` is True inside code that does not run on import."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        # Decorators and default values run when the def runs; the body runs when called.
        defaults = list(node.args.defaults) + [d for d in node.args.kw_defaults if d is not None]
        for child in defaults + list(getattr(node, "decorator_list", [])):
            _visit(child, deferred, names, found)
        for child in node.body if isinstance(node.body, list) else [node.body]:
            _visit(child, True, names, found)
        return
    if _is_main_guard(node):
        for child in node.body:
            _visit(child, True, names, found)
        for child in node.orelse:
            _visit(child, deferred, names, found)
        return
    if not deferred and isinstance(node, ast.Call) and _called_name(node) in names:
        found.append((node.lineno, _called_name(node)))
    for child in ast.iter_child_nodes(node):
        _visit(child, deferred, names, found)


def import_time_calls(source: str, names: frozenset = LOADS_AN_ENCODING) -> List[Tuple[int, str]]:
    """(line, name) of each call to one of ``names`` that runs when the module is imported.

    Module level, a class body, an ``if``/``try``/``with`` at module level, a decorator
    or a default argument all run on import. A function or lambda body does not, nor
    does an ``if __name__ == "__main__":`` block.
    """
    found: List[Tuple[int, str]] = []
    _visit(ast.parse(source), False, names, found)
    return sorted(found)


def _orchestrator_sources() -> Iterator[Tuple[Path, str]]:
    """Every Python file of the orchestrator except its tests."""
    for path in sorted(ORCH.rglob("*.py")):
        relative = path.relative_to(ORCH).as_posix()
        if relative.startswith("tests/") or "/tests/" in relative or "__pycache__" in relative:
            continue
        yield path, path.read_text(encoding="utf-8")


def _string_constants(tree: ast.Module) -> Dict[str, str]:
    """The module-level ``NAME = "text"`` (or ``NAME: str = "text"``) assignments."""
    constants: Dict[str, str] = {}
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        value = getattr(node, "value", None)
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            constants.update({target.id: value.value for target in targets if isinstance(target, ast.Name)})
    return constants


def _encoding_argument(call: ast.Call) -> Optional[ast.expr]:
    if call.args:
        return call.args[0]
    return next((keyword.value for keyword in call.keywords if keyword.arg == "encoding_name"), None)


def requested_encodings(source: str) -> Tuple[Set[str], List[str]]:
    """The encodings a module asks tiktoken for, and what makes the answer unknowable.

    ``get_encoding`` must name its encoding with a string, or with a module-level string
    constant. ``encoding_for_model`` is refused: the encoding it loads depends on the
    model name and on the tiktoken version (gpt-4o maps to o200k_base), so the image
    could not know which ones to fetch.
    """
    tree = ast.parse(source)
    constants = _string_constants(tree)
    names: Set[str] = set()
    problems: List[Tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        called = _called_name(node)
        if called == "encoding_for_model":
            problems.append((node.lineno, "encoding_for_model() loads an encoding the image cannot know"))
        elif called == "get_encoding":
            argument = _encoding_argument(node)
            if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                names.add(argument.value)
            elif isinstance(argument, ast.Name) and argument.id in constants:
                names.add(constants[argument.id])
            else:
                problems.append((node.lineno, "get_encoding() names its encoding with no string constant"))
    return names, [f"line {line}: {problem}" for line, problem in sorted(problems)]


# --------------------------------------------------------------------------- the walker itself

WALKER_CASES = {
    "module level": ("import tiktoken\nENC = tiktoken.get_encoding('cl100k_base')\n", [(2, "get_encoding")]),
    "inside a module-level try": (
        "try:\n    import tiktoken\n    ENC = tiktoken.encoding_for_model('gpt-4')\n"
        "except Exception:\n    ENC = tiktoken.get_encoding('cl100k_base')\n",
        [(3, "encoding_for_model"), (5, "get_encoding")],
    ),
    "a class body": ("class A:\n    TOKENS = count_tokens('x')\n", [(2, "count_tokens")]),
    "a default argument": ("def f(enc=get_tokenizer()):\n    return enc\n", [(1, "get_tokenizer")]),
    "a decorator": ("@cache(count_tokens('x'))\ndef f():\n    pass\n", [(1, "count_tokens")]),
    "a function body": ("def f():\n    return tiktoken.get_encoding('cl100k_base')\n", []),
    "a method body": ("class A:\n    def f(self):\n        return count_tokens('x')\n", []),
    "a lambda body": ("F = lambda text: count_tokens(text)\n", []),
    "a __main__ block": ("if __name__ == '__main__':\n    print(count_tokens('x'))\n", []),
    "an unrelated call": ("X = encode('x')\n", []),
}


@pytest.mark.parametrize("case", sorted(WALKER_CASES))
def test_the_walker_finds_only_calls_that_run_on_import(case):
    source, expected = WALKER_CASES[case]
    assert import_time_calls(source) == expected


def test_the_walker_reads_the_encoding_a_call_names():
    source = ("NAME = 'cl100k_base'\nOTHER: str = 'o200k_base'\n"
              "def a():\n    tiktoken.get_encoding(NAME)\n"
              "def b():\n    tiktoken.get_encoding(encoding_name=OTHER)\n"
              "def c(name):\n    tiktoken.get_encoding(name)\n"
              "def d():\n    tiktoken.encoding_for_model('gpt-4o')\n")
    names, problems = requested_encodings(source)
    assert names == {"cl100k_base", "o200k_base"}
    assert problems == [
        "line 8: get_encoding() names its encoding with no string constant",
        "line 10: encoding_for_model() loads an encoding the image cannot know",
    ]


# --------------------------------------------------------------------------- the code


def test_nothing_in_the_orchestrator_loads_an_encoding_at_import():
    """A module-level tiktoken load hung or killed ``import main`` with no egress."""
    offenders = [f"{path.relative_to(ORCH)}:{line} {name}()"
                 for path, source in _orchestrator_sources()
                 for line, name in import_time_calls(source)]
    assert not offenders, (
        "these load a tiktoken encoding when their module is imported; call them inside a "
        f"function (core.context_guard.get_tokenizer() loads it on first use): {offenders}"
    )


def _imports_tiktoken(source: str) -> bool:
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import) and any(alias.name.split(".")[0] == "tiktoken" for alias in node.names):
            return True
        if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "tiktoken":
            return True
    return False


def test_only_the_context_guard_imports_tiktoken():
    """One tokenizer: the three import-time encoders became core.context_guard.get_tokenizer()."""
    importers = sorted(path.relative_to(ORCH).as_posix()
                       for path, source in _orchestrator_sources() if _imports_tiktoken(source))
    assert importers == ["core/context_guard.py"]


@pytest.mark.parametrize("path", COUNTER_USERS, ids=lambda path: path.name)
def test_the_rag_service_and_the_optimizer_count_through_the_context_guard(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported = {alias.name for node in tree.body if isinstance(node, ast.ImportFrom)
                and node.module == "core.context_guard" for alias in node.names}
    assert "count_tokens" in imported, f"{path.name} must count with core.context_guard.count_tokens"


def test_every_encoding_the_code_requests_is_fetched_into_the_image():
    requested: Set[str] = set()
    problems: List[str] = []
    for path, source in _orchestrator_sources():
        names, found = requested_encodings(source)
        requested |= names
        problems += [f"{path.relative_to(ORCH)} {problem}" for problem in found]
    assert not problems, problems
    assert "cl100k_base" in requested, "the scan no longer finds the context guard's encoding"
    missing = requested - set(_baked_encodings())
    assert not missing, f"orchestrator/Dockerfile does not fetch {sorted(missing)} into {CACHE_DIR}"


def test_the_context_guard_names_the_encoding_it_loads():
    tree = ast.parse(CONTEXT_GUARD.read_text(encoding="utf-8"))
    assert _string_constants(tree)["TOKENIZER_ENCODING"] in _baked_encodings()
    getter = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "get_tokenizer")
    decorators = [ast.unparse(decorator) for decorator in getter.decorator_list]
    assert decorators == ["functools.lru_cache(maxsize=1)"], decorators


# --------------------------------------------------------------------------- the image


def _instructions(text: str) -> List[str]:
    """The Dockerfile's instructions: continuation lines joined, comments dropped, spaces collapsed."""
    joined = re.sub(r"\\[ \t]*\n", " ", text)
    return [" ".join(line.split()) for line in joined.splitlines()
            if line.strip() and not line.lstrip().startswith("#")]


def _stage(name: str) -> List[str]:
    """A stage's instructions after its FROM."""
    instructions = _instructions(DOCKERFILE)
    start = next(i for i, line in enumerate(instructions) if re.fullmatch(rf"FROM \S+ as {name}", line, re.I))
    end = next((i for i in range(start + 1, len(instructions)) if instructions[i].upper().startswith("FROM ")),
               len(instructions))
    return instructions[start + 1:end]


def _bake_run() -> str:
    (run,) = [line for line in _stage("base") if line.startswith("RUN ") and "tiktoken" in line]
    return run


def _baked_encodings() -> List[str]:
    match = re.search(r'\bencodings="([^"]*)"', _bake_run())
    assert match, "the base stage's tiktoken RUN must list its encodings as encodings=\"...\""
    return match.group(1).split()


def test_the_base_stage_fetches_the_encodings_into_the_cache_dir():
    base = _stage("base")
    env = f"ENV TIKTOKEN_CACHE_DIR={CACHE_DIR}"
    run = _bake_run()
    assert env in base, "the image must point tiktoken at the cache it fetched"
    # tiktoken is installed before the fetch, and the cache dir is set before it. Since
    # US-022, pybuild installs into two layers (stable and rest) and base copies both.
    installs = [i for i, line in enumerate(base) if line.startswith("COPY --from=pybuild /install")]
    assert installs, "the base stage must copy pybuild's installed packages"
    copied = {base[i] for i in installs}
    for layer in ("stable", "rest"):
        assert f"COPY --from=pybuild /install/{layer} /usr/local" in copied, \
            f"the base stage must copy pybuild's /install/{layer} layer"
    assert max(installs) < base.index(env) < base.index(run)
    assert "tiktoken.get_encoding(name) for name in sys.argv[1:]" in run
    assert run.count("$encodings") == 2, "fetched once, then loaded again offline"
    # The second load goes through a proxy where nothing listens: the cache serves it.
    assert f"HTTPS_PROXY={REFUSING_PROXY} HTTP_PROXY={REFUSING_PROXY} python -B -c" in run
    assert "--mount=type=cache" not in run  # Railway builds this file


@pytest.mark.parametrize("stage", ["development", "production"])
def test_both_servers_start_from_the_stage_that_holds_the_cache(stage):
    instructions = _instructions(DOCKERFILE)
    assert f"FROM base as {stage}" in instructions
    assert not any(line.startswith("ENV TIKTOKEN_CACHE_DIR=") for line in _stage(stage)), \
        "the cache dir is set once, in the base stage"


# --------------------------------------------------------------------------- the getter


class _FakeEncoding:
    """Splits on spaces: one token per word."""

    name = "cl100k_base"

    def encode(self, text: str) -> List[str]:
        return text.split()

    def decode(self, tokens: List[str]) -> str:
        return " ".join(tokens)


def _fake_tiktoken(error: Optional[Exception] = None) -> types.ModuleType:
    module = types.ModuleType("tiktoken")
    module.calls = []

    def get_encoding(name: str) -> _FakeEncoding:
        module.calls.append(name)
        if error is not None:
            raise error
        return _FakeEncoding()

    module.get_encoding = get_encoding
    return module


def _fresh_context_guard() -> types.ModuleType:
    """core/context_guard.py loaded as a new module, so its getter's cache starts empty."""
    spec = importlib.util.spec_from_file_location("prd262_context_guard_probe", CONTEXT_GUARD)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_tokenizer_loads_on_the_first_count_and_only_once(monkeypatch):
    fake = _fake_tiktoken()
    monkeypatch.setitem(sys.modules, "tiktoken", fake)
    guard = _fresh_context_guard()
    assert fake.calls == [], "importing the module must not load an encoding"

    assert guard.count_tokens("") == 0
    assert fake.calls == [], "an empty text needs no tokenizer"
    assert guard.count_tokens("one two three") == 3
    assert guard.truncate_to_token_budget("a b c d", 2, suffix="!") == "a b!"
    assert guard.count_message_tokens([{"role": "user", "content": "x y"}]) == 4 + 2 + 2
    assert guard.get_tokenizer() is guard.get_tokenizer()
    assert fake.calls == [guard.TOKENIZER_ENCODING] == ["cl100k_base"]


@pytest.mark.parametrize("error", [ConnectionError("egress refused"), TimeoutError("egress dropped")])
def test_a_tokenizer_that_cannot_load_is_logged_once_and_estimated(monkeypatch, caplog, error):
    fake = _fake_tiktoken(error)
    monkeypatch.setitem(sys.modules, "tiktoken", fake)
    guard = _fresh_context_guard()

    with caplog.at_level(logging.ERROR, logger=guard.logger.name):
        assert guard.get_tokenizer() is None
        assert guard.count_tokens("x" * 40) == 10
        assert guard.count_tokens("y" * 3) == 1
        assert guard.truncate_to_token_budget("z" * 20, 2, suffix="…") == "zzzzzzzz…"

    assert fake.calls == ["cl100k_base"], "a failed load is not retried on every count"
    logged = [record for record in caplog.records if "tiktoken unavailable" in record.getMessage()]
    assert len(logged) == 1 and logged[0].exc_info is not None


def test_no_tiktoken_installed_estimates(monkeypatch):
    monkeypatch.setitem(sys.modules, "tiktoken", None)  # `import tiktoken` raises ImportError
    guard = _fresh_context_guard()
    assert guard.get_tokenizer() is None
    assert guard.count_tokens("abcdefgh") == 2
