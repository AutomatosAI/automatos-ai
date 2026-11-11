"""The dead PromptAnalyzer helpers stay gone (#806).

``extract_search_terms`` and ``parse_explicit_tool_call`` were helpers for a
retired retrieval path; the live chat path (consumers/chatbot/service.py)
never called them. If one creeps back, this fails.

Pure unit tests: no app boot, no DB, no LLM.
"""

import consumers.chatbot.prompt_analyzer as prompt_analyzer
from consumers.chatbot.prompt_analyzer import PromptAnalyzer


def test_dead_methods_are_gone():
    assert not hasattr(PromptAnalyzer, "extract_search_terms")
    assert not hasattr(PromptAnalyzer, "parse_explicit_tool_call")
    assert not hasattr(prompt_analyzer, "EXPLICIT_TOOL_CALL_RE")


def test_live_methods_remain():
    assert hasattr(PromptAnalyzer, "is_fresh_start_request")
    assert hasattr(PromptAnalyzer, "convert_to_llm_messages")
    assert hasattr(PromptAnalyzer, "extract_latest_user_text")
