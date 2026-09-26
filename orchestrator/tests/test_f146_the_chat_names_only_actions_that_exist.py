"""F146 follow-up (25 Sep, the cleanup batch): the chat names only actions that exist.

870cd13d7 removed the legacy ``platform_*_recipe`` aliases, but four places still
named them: AutoBrain's phrase map (its log said "Platform query
(platform_execute_recipe)"), the intent classifier's platform hints (a hint to a
tool that does not exist does nothing, so "show my workflows" hinted nothing),
the chat's workflow-update prefixes (which could never match), and F133's
driver-aware tuple. Each now names the Playbook action, or nothing.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from modules.tools.discovery.action_registry import get_action_registry

ORCHESTRATOR = Path(__file__).resolve().parents[1]
LEGACY = re.compile(r"\bplatform_(?:list|create|update|delete|execute|get|add)_recipe(?:s|_step|_execution)?\b")


def _missing(names):
    registry = get_action_registry()
    return [name for name in names if registry.get(name) is None]


def test_autobrains_phrase_map_names_real_actions():
    from consumers.chatbot.auto import _PLATFORM_KEYWORDS

    assert _missing(_PLATFORM_KEYWORDS) == []


@pytest.mark.parametrize("question, hint", [
    ("show me my playbooks", "platform_list_playbooks"),
    ("what workflows do we have", "platform_list_playbooks"),
    ("create a recipe for the monday reorder", "platform_create_playbook"),
])
def test_a_playbook_question_hints_a_playbook_action(question, hint):
    from consumers.chatbot.intent_classifier import SmartIntentClassifier

    hints = SmartIntentClassifier.__new__(SmartIntentClassifier)._get_platform_tool_hints(question)
    assert hint in hints and _missing(hints) == []


def test_the_driver_aware_actions_all_exist():
    from modules.tools.discovery.platform_executor import _DRIVER_AWARE_ACTIONS

    assert _missing(_DRIVER_AWARE_ACTIONS) == []


def test_no_production_code_names_a_removed_recipe_action():
    """The refusal test lists them on purpose, and the utterance-corpus generator
    maps old names to new ones; nothing that runs may name them."""
    allowed = {ORCHESTRATOR / "scripts" / "generate_utterance_corpus.py"}
    found = []
    for path in ORCHESTRATOR.rglob("*.py"):
        if "tests" in path.relative_to(ORCHESTRATOR).parts or path in allowed:
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if LEGACY.search(line):
                found.append(f"{path.relative_to(ORCHESTRATOR)}:{number}")
    assert found == []
