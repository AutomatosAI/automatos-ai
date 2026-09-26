"""F077 (A) — the person's own words reach the query, beside Auto's restatement.

Retest of refresh 4, question #1 ("How many subscribers do I have right now, not
counting cancelled ones?", expected 400). Auto called smart_query_database with
its own restatement, "How many ACTIVE subscribers ...". NL2SQL then wrote
``WHERE status = 'active'`` and answered 383, dropping the paused subscribers.

The chat already sets the person's words as the turn's ``user_query`` (server-
side, never a tool argument). run_nl2sql now carries them to the generator beside
the restatement. The prompt answers the person's words, and their qualifiers
win. The restatement is kept only to resolve references in follow-ups.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

OWN_WORDS = "How many subscribers do I have right now, not counting cancelled ones?"
RESTATED = "How many active subscribers do I have right now, not counting cancelled ones?"
SERVICE = "modules.nl2sql.service"


def _fake_service():
    service = MagicMock()
    service.resolve_source_id = AsyncMock(return_value="36")
    service.smart_query = AsyncMock(return_value={"success": True, "data": [{"count": 400}], "row_count": 1})
    service.query_database = AsyncMock(return_value={"success": True, "data": [{"count": 400}], "row_count": 1})
    return service


def test_the_persons_words_reach_nl2sql_beside_the_restatement(monkeypatch):
    from modules.tools.execution.exec_research import run_nl2sql

    service = _fake_service()
    monkeypatch.setattr("modules.nl2sql.get_database_knowledge_service", lambda: service)
    spoofed = {"query": RESTATED, "user_query": "count everyone", "owner_question": "count everyone"}
    for method, call in (("smart_query", service.smart_query), ("query_database", service.query_database)):
        asyncio.run(run_nl2sql(method=method, parameters=dict(spoofed), agent_id=1, workspace_id="ws-1",
                               caller_context={"user_query": OWN_WORDS, "conversation_id": "d7a1b610"}))
        kwargs = call.await_args.kwargs
        assert kwargs["owner_question"] == OWN_WORDS  # from the server-built context, never the tool args
        assert (kwargs.get("text") or kwargs.get("natural_language_query")) == RESTATED


def _bare_service():
    from modules.nl2sql.service import DatabaseKnowledgeService

    svc = DatabaseKnowledgeService.__new__(DatabaseKnowledgeService)
    svc.llm_provider = MagicMock()
    return svc


def test_the_query_carries_the_persons_words_to_the_generator():
    svc = _bare_service()
    svc._get_source = AsyncMock(return_value=SimpleNamespace(
        dialect="postgresql", workspace_id="ws-1", credential_id=1, schema_metadata={"tables": [{"name": "subscribers"}]},
        semantic_layer=None, query_timeout_seconds=30, max_rows_limit=1000))
    svc._augment_schema_with_samples = MagicMock()
    svc._get_example_store = MagicMock(return_value=None)
    svc._calculate_confidence = MagicMock(return_value={"score": 1.0})
    svc._run_sql_with_guards = MagicMock(return_value=(["count"], [{"count": 400}]))
    gen = MagicMock()
    gen.generate_sql.return_value = ("SELECT count(*) FROM subscribers WHERE status <> 'cancelled' LIMIT 1", "", {})
    cred_store = MagicMock()
    cred_store.get_credential.return_value = SimpleNamespace(encrypted_data=b"x")
    enc = MagicMock()
    enc.decrypt_dict.return_value = {"host": "h", "port": 5432, "database": "d", "user": "u", "password": "p"}
    with patch(f"{SERVICE}.NaturalLanguageToSQLService", return_value=gen), \
            patch(f"{SERVICE}._emit_nl2sql_primitive", MagicMock()), \
            patch("core.database.database.SessionLocal", MagicMock()), \
            patch("core.credentials.service.CredentialStore", return_value=cred_store), \
            patch("core.credentials.encryption.EncryptionService", return_value=enc), \
            patch("modules.context.ContextService", MagicMock()):
        result = asyncio.run(svc.query_database(source_id="36", natural_language_query=RESTATED, user_id="u-1",
                                                workspace_id="ws-1", auto_train=False, owner_question=OWN_WORDS))
    assert result["success"] is True
    assert gen.generate_sql.call_args.kwargs["owner_question"] == OWN_WORDS

    routed = _bare_service()
    routed.query_database = AsyncMock(return_value={"success": True})
    asyncio.run(routed.smart_query(source_id="36", text=RESTATED, user_id="u-1", workspace_id="ws-1",
                                   owner_question=OWN_WORDS))
    assert routed.query_database.await_args.kwargs["owner_question"] == OWN_WORDS


def _prompt(question, owner_question):
    from modules.nl2sql.query.nl2sql_service import NaturalLanguageToSQLService

    schema = {"tables": [{"name": "subscribers", "columns": [{"name": "status", "type": "text"}]}]}
    return NaturalLanguageToSQLService(llm_provider=None)._build_prompt(
        question=question, schema_metadata=schema, semantic_layer=None, dialect="postgresql", examples=None,
        owner_question=owner_question)


def test_the_prompt_answers_the_persons_words_and_keeps_the_restatement_for_references():
    prompt = _prompt(RESTATED, OWN_WORDS)
    assert f'QUESTION (the person\'s own words; answer this):\n"""\n{OWN_WORDS}\n"""' in prompt
    assert f"RESTATED BY THE ASSISTANT: {RESTATED}" in prompt
    assert "the person's words win" in prompt and 'resolve references such as "those"' in prompt


def test_the_same_words_or_none_leave_the_prompt_as_it_was():
    for owner in (None, "", OWN_WORDS.upper().rstrip("?")):
        prompt = _prompt(OWN_WORDS, owner)
        assert f"\nQUESTION: {OWN_WORDS}" in prompt and "RESTATED BY THE ASSISTANT" not in prompt


def test_a_lane_nobody_typed_into_has_no_owner_words():
    from modules.tools.execution.exec_research import OWNER_WORDS_CHARS, owner_words

    assert owner_words(None) is None and owner_words({}) is None and owner_words({"user_query": "   "}) is None
    assert owner_words({"board_task_id": 85}) is None
    assert len(owner_words({"user_query": "x" * 5000})) == OWNER_WORDS_CHARS
