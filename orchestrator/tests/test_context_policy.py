from __future__ import annotations

from orchestrator.context_policy import ContextPolicy, assemble_context


def test_default_policy_assembly():
    policy = ContextPolicy.default("code_assistant")
    out = assemble_context(
        q="How do I paginate SQLAlchemy results?",
        policy=policy,
        slot_values={
            "INSTRUCTION": "You are a helpful code assistant.",
            "MEMORY": "User prefers concise answers and Python 3.11.",
            "RETRIEVAL": "Doc: Use limit/offset or keyset pagination.",
            "CODE": "models.py shows Query on Agent table.",
            "TOOLS": "Available tools: db_query, web_search.",
            "CONSTRAINTS": "Output under 200 tokens; return code first.",
        },
    )
    assert "USER_QUERY" in out.prompt
    assert out.stats["TOTAL"] > 0
    assert "INSTRUCTION" in out.slots_used


def test_budget_and_truncation():
    policy = ContextPolicy.default("tiny")
    policy.max_total_chars = 120
    out = assemble_context(
        q="X",
        policy=policy,
        slot_values={
            "INSTRUCTION": "I" * 200,
            "MEMORY": "M" * 200,
            "RETRIEVAL": "R" * 200,
            "CODE": "C" * 200,
            "TOOLS": "T" * 200,
            "CONSTRAINTS": "Z" * 200,
        },
    )
    assert out.stats["TOTAL"] <= 120


