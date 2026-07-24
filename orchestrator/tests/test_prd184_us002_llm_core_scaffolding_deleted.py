"""PRD-184 US-002 — the llm-core dead scaffolding is deleted, not orphaned.

~1,700 LOC of zero-external-caller scaffolding survived ONLY because the
``core/llm/__init__.py`` barrel re-exported it. No live code ever imported the
symbols; the barrel was the sole reference. Deleted here:

* core/llm/function_executor.py       (FunctionExecutor, FunctionResult)
* core/llm/function_registry.py       (FunctionRegistry, FunctionSpec, ...)
* core/llm/response_parser.py         (ResponseParser, ParsedResponse)
* core/llm/semantic_skill_matcher.py  (SemanticSkillMatcher, get_skill_matcher)
* core/global_function_registry.py    (GlobalFunctionRegistry — 0 importers)
* api/anthropic_client.py             (stranded dead DUPLICATE, see below)

``api/anthropic_client.py`` is NOT the live Anthropic provider — it is a stranded
byte-similar copy in the wrong package: it does ``from .base import ...`` but
``api/base.py`` does not exist, so it is not even importable, and nothing imports
it. The LIVE provider is ``core/llm/clients/anthropic_client.py`` (used by
``core/llm/manager.py`` + the clients barrel + tests) — that one SURVIVES. This
story owns the api/ copy (PRD-212 defers to it).

BOUNDARY (this guard proves both sides): the trimmed barrel still re-exports the
KEPT infra (create_llm_manager, EmbeddingManager, RerankManager), and the live
clients/anthropic_client.py provider is intact.

Pure/static — file reads only, imports no app package.
"""
from __future__ import annotations

import pathlib
import re
import sys

_ORCH = pathlib.Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

_SOURCE_DIRS = ("modules", "services", "core", "api", "consumers", "evals")

_GONE_FILES = (
    "core/llm/function_executor.py",
    "core/llm/function_registry.py",
    "core/llm/response_parser.py",
    "core/llm/semantic_skill_matcher.py",
    "core/global_function_registry.py",
    "api/anthropic_client.py",
)

# Dead dotted-module paths / snake-case module names. Deliberately precise so
# they do NOT collide with live code that shares a word:
#   - ``core.llm.clients.anthropic_client`` (LIVE provider) does NOT contain
#     ``api.anthropic_client``.
#   - The tool_registry docstring prose says CamelCase ``GlobalFunctionRegistry``;
#     the snake-case module token ``global_function_registry`` (case-sensitive)
#     does not match it.
#   - ``rerank_manager`` (kept) is not listed.
_GONE_TOKENS = (
    "core.llm.function_executor",
    "core.llm.function_registry",
    "core.llm.response_parser",
    "core.llm.semantic_skill_matcher",
    "core.global_function_registry",
    "global_function_registry",
    "api.anthropic_client",
)
_GONE_TOKEN_PATTERNS = tuple(
    (t, re.compile(rf"\b{re.escape(t)}\b")) for t in _GONE_TOKENS
)

# Barrel symbols that must no longer be re-exported by core/llm/__init__.py.
_DEAD_BARREL_SYMBOLS = (
    "function_executor",
    "function_registry",
    "response_parser",
    "semantic_skill_matcher",
    "FunctionExecutor",
    "FunctionResult",
    "FunctionRegistry",
    "FunctionSpec",
    "ResponseParser",
    "ParsedResponse",
    "SemanticSkillMatcher",
    "get_skill_matcher",
)
# Infra the trimmed barrel MUST keep exporting (proves a precise, not blunt, cut).
_KEPT_BARREL_SYMBOLS = (
    "create_llm_manager",
    "EmbeddingManager",
    "RerankManager",
    "get_rerank_manager",
)


def test_llm_core_dead_files_deleted():
    for rel in _GONE_FILES:
        assert not (_ORCH / rel).exists(), (
            f"{rel} must stay deleted (PRD-184 US-002) — zero-caller scaffolding, no shim"
        )


def test_llm_core_no_dead_scaffolding():
    """No live source imports/references the deleted modules (no dangling imports)."""
    offenders = []
    for d in _SOURCE_DIRS:
        root = _ORCH / d
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            text = path.read_text(errors="ignore")
            for token, pattern in _GONE_TOKEN_PATTERNS:
                if pattern.search(text):
                    offenders.append(f"{path.relative_to(_ORCH)}: {token}")
    for extra in ("main.py", "config.py"):
        text = (_ORCH / extra).read_text(errors="ignore")
        for token, pattern in _GONE_TOKEN_PATTERNS:
            if pattern.search(text):
                offenders.append(f"{extra}: {token}")
    assert not offenders, f"dangling llm-core scaffolding references: {offenders}"


def test_barrel_drops_dead_symbols_keeps_infra():
    """core/llm/__init__.py no longer re-exports the razed scaffolding, but STILL
    exports the live LLM infra (precise trim, not a blunt gutting)."""
    src = (_ORCH / "core" / "llm" / "__init__.py").read_text()
    present = [s for s in _DEAD_BARREL_SYMBOLS if re.search(rf"\b{re.escape(s)}\b", src)]
    assert not present, f"core/llm barrel still re-exports deleted scaffolding: {present}"
    missing = [s for s in _KEPT_BARREL_SYMBOLS if s not in src]
    assert not missing, f"core/llm barrel must keep live infra exports: {missing}"


def test_live_anthropic_provider_survives():
    """Boundary proof: the api/ copy dies, the REAL provider lives. Deleting the
    live one would break core/llm/manager.py — exactly what the gate forbids."""
    live = _ORCH / "core" / "llm" / "clients" / "anthropic_client.py"
    assert live.exists(), "core/llm/clients/anthropic_client.py (live provider) must survive"
    manager = (_ORCH / "core" / "llm" / "manager.py").read_text()
    assert "from .clients.anthropic_client import AnthropicProvider" in manager
