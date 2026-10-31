"""PRD-268 US-C22a (§6.11.5, §12-Q22, Q26, K15) — ``Calibration@1`` from titles and kinds only, against Postgres.

``services/calibration/build.py`` ``build_calibration`` over ``sources.py`` (the reads),
``prompt.py`` (the one model call's messages), ``answer.py`` (what of the answer is kept) and
``strings.py`` (the words). The model is a fake that reads the refs off the prompt it is given.
Proven here:

* the reads name titles and kinds only: a spy on every statement the build executes finds four
  SELECTs and no body, summary, description, content, chunk, file path or prompt column; no
  Deliverable read is called;
* the prompt holds the Deliverables and Knowledge the person may see (a document whose
  ``is_shared`` is NULL is shared) and their own conversations, and never another member's
  conversation or private document, another workspace's rows, a deleted or archived Deliverable,
  an agent's heartbeat report or output, a switched-off skill, Auto's thread, a widget visitor's
  chat, a chat still called "New Chat" or a blank title; each kind is cut at ``PER_KIND``;
* provenance: a line citing a source it was not given is dropped whole, and so is one citing
  only a skill; a line with "85%", or with a web address, is dropped; a line longer than 160
  characters is dropped; at most 3 lines are kept; with every line dropped the text part says
  nothing was clear enough (not that there was nothing to read);
* the read transaction the build opened is ended before the model call;
* each kept line counts its sources ("from 1 Deliverable and 1 Conversation"), in both languages;
* the offers name an installed skill and are topped up with the starters, always 3;
* nothing to read (a skill alone is not the person's work): empty, the 3 starters, no model call;
  otherwise exactly one model call, booked as ``calibration`` (customer work);
* an answer with no JSON raises ``CalibrationFailed``; a failed call raises;
* every strings key is in both tables.

The ``v_workspace_outputs`` view is migration-managed and CI's create_all database has none: the
reading session makes a TEMP view over ``deliverables`` (which shadows the real one where it
exists) and never commits it. The seed commits, and the fixture sweeps it.
"""
from __future__ import annotations

import json
import re
import uuid
from types import SimpleNamespace as NS

import pytest
from sqlalchemy import create_engine, event, text

from services.calibration import build as build_mod
from services.calibration.answer import carries_a_link, carries_a_number_for_sureness, kept_lines, kept_offers
from services.calibration.build import CALIBRATION_PURPOSE, CalibrationFailed, build_calibration
from services.calibration.model import MAX_LINE_CHARS, MAX_LINES, OFFERS, SOURCE_TYPES, CalibrationSource
from services.calibration.sources import PER_KIND, Read
from services.calibration.strings import (
    STARTER_KEYS, STRINGS, calibration_language, calibration_text, consent_line, source_count, starters,
)
from services.credits.classification import CUSTOMER_PURPOSES

# The real model builder, before any test puts the fake in its place.
REAL_LLM = build_mod.calibration_llm
# The reading session's stand-in for the migration-managed view: the columns the reads touch,
# and the summary, so a read that selected it would show.
OUTPUTS_VIEW = """
CREATE OR REPLACE TEMP VIEW v_workspace_outputs AS
SELECT d.id, d.workspace_id, d.source_type, d.artifact_type, d.title, d.summary, d.file_path, d.status,
       d.deleted_at, d.created_at
FROM deliverables d
"""
# A column a calibration read must never name.
FORBIDDEN = re.compile(
    r"\b(summary|content|description|doc_metadata|excerpt|body|file_path|prompt_template|implementation|"
    r"last_context|parameters|chunk\w*|document_chunks)\b",
    re.IGNORECASE,
)
# What the person may not see, or is not their work: none of it reaches the model.
HIDDEN_MARKS = ("OTHER-PERSON", "OTHER-WORKSPACE", "SECRET", "Old deck", "Agent status", "Agent output",
                "New Chat", "Widget visitor", "AUTO-THREAD", "ARCHIVED-MARK", "INACTIVE-SKILL")
VISIBLE = ("Q3 pricing review", "Launch plan", "Onboarding notes.pdf", "Brand guide.docx", "Shared by default.pdf",
           "Pricing for Lisbon", "Social writer")


# --------------------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def engine():
    from core.database.database import get_database_url

    try:
        eng = create_engine(get_database_url(), pool_pre_ping=True)
        with eng.connect() as c:
            c.execute(text("SELECT kind, widget_key_id FROM chats LIMIT 1"))
            c.execute(text("SELECT is_shared, source_type, created_by_user_id FROM documents LIMIT 1"))
            c.execute(text("SELECT title, deleted_at FROM deliverables LIMIT 1"))
            c.execute(text("SELECT skill_id FROM workspace_enabled_skills LIMIT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"the calibration tests need Postgres with the chats, documents and deliverables tables: {exc}")
    yield eng
    eng.dispose()


def _person(s, tag):
    clerk = f"user_prd268w3cal_{tag}_{uuid.uuid4().hex[:10]}"
    pk = s.execute(text("INSERT INTO users (username, email, clerk_user_id) VALUES (:u, :e, :c) RETURNING id"),
                   {"u": clerk, "e": f"{clerk}@t.test", "c": clerk}).scalar()
    return NS(clerk=clerk, pk=int(pk))


def _workspace(s, owner):
    ws = str(uuid.uuid4())
    s.execute(text("INSERT INTO workspaces (id, name, owner_id, settings) "
                   "VALUES (CAST(:id AS uuid), 'prd268w3 calibration', :o, CAST('{}' AS jsonb))"), {"id": ws, "o": owner})
    return ws


def _deliverable(s, ws, title, *, kind="report", origin="chat", deleted=False, status="ready"):
    s.execute(text(
        "INSERT INTO deliverables (workspace_id, source_type, artifact_type, title, summary, file_path, status, "
        "deleted_at) VALUES (CAST(:w AS uuid), :o, :k, :t, 'SECRET-SUMMARY of the report', :p, :st, "
        "CASE WHEN :d THEN now() ELSE NULL END)"),
        {"w": ws, "o": origin, "k": kind, "t": title, "p": f"outputs/{uuid.uuid4().hex}.md", "d": deleted,
         "st": status})


def _document(s, ws, name, *, shared=True, owner=None, origin=None):
    s.execute(text(
        "INSERT INTO documents (filename, original_filename, file_type, workspace_id, is_shared, created_by_user_id, "
        "description, source_type, file_path) "
        "VALUES (:f, :f, 'pdf', CAST(:w AS uuid), :sh, :o, 'SECRET-BODY of the document', :src, '/data/SECRET-PATH')"),
        {"f": name, "w": ws, "sh": shared, "o": owner, "src": origin})


def _chat(s, ws, user, title, *, kind="user", widget=False):
    s.execute(text(
        "INSERT INTO chats (id, user_id, workspace_id, title, visibility, kind, widget_key_id, summary) "
        "VALUES (CAST(:id AS uuid), :u, CAST(:w AS uuid), :t, 'private', :k, CAST(:wk AS uuid), "
        "CAST('{\"last_summary\": \"SECRET-CHECKPOINT\"}' AS jsonb))"),
        {"id": str(uuid.uuid4()), "u": user, "w": ws, "t": title, "k": kind,
         "wk": str(uuid.uuid4()) if widget else None})


def _skill(s, ws, name, *, active=None):
    pk = int(s.execute(text(
        "INSERT INTO skills (name, skill_type, description, prompt_template, is_active) "
        "VALUES (:n, 'communication', 'SECRET-SKILL-DESCRIPTION', 'SECRET-SKILL-PROMPT', :a) RETURNING id"),
        {"n": name, "a": active}).scalar())
    s.execute(text("INSERT INTO workspace_enabled_skills (workspace_id, skill_id) VALUES (CAST(:w AS uuid), :s)"),
              {"w": ws, "s": pk})
    return pk


def _seed_a(s, ws, person, other):
    """The person's workspace: what they may see, and what they may not."""
    _deliverable(s, ws, "Q3 pricing review")
    _deliverable(s, ws, "Launch plan", kind="document")
    _deliverable(s, ws, "Old deck", deleted=True)
    _deliverable(s, ws, "Agent status", origin="heartbeat")
    _deliverable(s, ws, "Failed run ARCHIVED-MARK", status="archived")
    _deliverable(s, ws, "   ")
    _document(s, ws, "Onboarding notes.pdf")
    _document(s, ws, "Shared by default.pdf", shared=None)
    _document(s, ws, "Brand guide.docx", shared=False, owner=person)
    _document(s, ws, "OTHER-PERSON private.pdf", shared=False, owner=other)
    _document(s, ws, "Agent output.md", origin="agent_output")
    _chat(s, ws, person, "Pricing for Lisbon")
    _chat(s, ws, person, "New Chat")
    _chat(s, ws, person, "AUTO-THREAD", kind="auto")
    _chat(s, ws, person, "Widget visitor question", widget=True)
    _chat(s, ws, other, "OTHER-PERSON chat")


@pytest.fixture
def world(engine, new_session):
    """Pia's workspace A (with Oto, another member), Oto's workspace B where Pia also has a chat,
    Lu's workspace C with nothing in it but an installed skill, and Lu's workspace D with one
    Deliverable more than a kind is cut at. Swept on teardown."""
    s = new_session()
    pia, oto, lu = _person(s, "pia"), _person(s, "oto"), _person(s, "lu")
    ws_a, ws_b, ws_c, ws_d = (_workspace(s, owner) for owner in (pia.pk, oto.pk, lu.pk, lu.pk))
    _seed_a(s, ws_a, pia.pk, oto.pk)
    _deliverable(s, ws_b, "OTHER-WORKSPACE deliverable")
    _document(s, ws_b, "OTHER-WORKSPACE.pdf")
    _chat(s, ws_b, pia.pk, "OTHER-WORKSPACE chat")
    for n in range(PER_KIND + 1):
        _deliverable(s, ws_d, f"Bulk {n}")
    skills = [_skill(s, ws_a, "Social writer"), _skill(s, ws_a, "INACTIVE-SKILL", active=False),
              _skill(s, ws_b, "OTHER-WORKSPACE skill"), _skill(s, ws_c, "Lone skill")]
    s.commit()
    yield NS(pia=pia, oto=oto, lu=lu, a=ws_a, b=ws_b, c=ws_c, d=ws_d)
    s = new_session.sweep()
    workspaces = [ws_a, ws_b, ws_c, ws_d]
    for table in ("deliverables", "documents", "workspace_enabled_skills", "chats"):
        s.execute(text(f"DELETE FROM {table} WHERE CAST(workspace_id AS text) = ANY(:w)"), {"w": workspaces})
    s.execute(text("DELETE FROM skills WHERE id = ANY(:ids)"), {"ids": skills})
    s.execute(text("DELETE FROM workspaces WHERE CAST(id AS text) = ANY(:w)"), {"w": workspaces})
    s.execute(text("DELETE FROM users WHERE clerk_user_id = ANY(:c)"), {"c": [pia.clerk, oto.clerk, lu.clerk]})
    s.commit()


@pytest.fixture
def reading(new_session):
    """A session that sees the TEMP outputs view; rolled back (the view with it) after."""
    s = new_session()
    s.execute(text(OUTPUTS_VIEW))
    yield s
    s.rollback()


class _Model:
    """The model: records each call and answers what ``answer(refs)`` says, ``refs`` being each
    title's ref as the prompt gave it."""

    def __init__(self, answer):
        self.answer, self.calls, self.workspaces = answer, [], []

    def generate_response_sync(self, messages):
        self.calls.append(messages)
        return NS(content=self.answer(_refs_by_title(messages[1]["content"])))


def _refs_by_title(prompt):
    rows = (json.loads(line) for line in prompt.splitlines() if line.startswith("{"))
    return {row["title"]: row["ref"] for row in rows}


@pytest.fixture
def model(monkeypatch):
    """Install a model whose answer the test gives; Deliverable reads refuse."""
    from services.deliverable_service import DeliverableService

    def refuse(*args, **kwargs):
        raise AssertionError("a calibration never reads a Deliverable's content")

    monkeypatch.setattr(DeliverableService, "get_deliverable", refuse)

    def install(answer):
        fake = _Model(answer)

        def llm(workspace_id):
            fake.workspaces.append(str(workspace_id))
            return fake

        monkeypatch.setattr(build_mod, "calibration_llm", llm)
        return fake

    return install


def _answer(refs):
    return json.dumps({
        "lines": [
            {"text": "You review pricing for clients in Lisbon",
             "sources": [refs["Q3 pricing review"], refs["Pricing for Lisbon"]]},
            {"text": "You run hiring", "sources": ["D99"]},
            {"text": "You plan launches", "sources": [refs["Launch plan"], "C42"]},
            {"text": "You are 85% focused on sales", "sources": [refs["Launch plan"]]},
            {"text": "You keep onboarding and brand notes",
             "sources": [refs["Onboarding notes.pdf"], refs["Brand guide.docx"]]},
            {"text": "You write about nothing", "sources": []},
            {"text": "You use social writing", "sources": [refs["Social writer"]]},
            {"text": "Verify your pricing at evil-example.com/login", "sources": [refs["Q3 pricing review"]]},
        ],
        "offers": [{"text": "Draft a LinkedIn post", "skill": refs["Social writer"]},
                   {"text": "Write code", "skill": "S99"},
                   {"text": "Review the launch plan", "skill": refs["Launch plan"]}],
    })


class _Spy:
    """Every statement executed on the engine while it is open."""

    def __init__(self, engine):
        self.engine, self.statements = engine, []

    def _seen(self, conn, cursor, statement, parameters, context, executemany):
        self.statements.append(statement)

    def __enter__(self):
        event.listen(self.engine, "before_cursor_execute", self._seen)
        return self

    def __exit__(self, *exc):
        event.remove(self.engine, "before_cursor_execute", self._seen)


# --------------------------------------------------------------------------- the reads


def test_the_reads_name_titles_and_kinds_only(engine, world, reading, model):
    fake = model(_answer)
    with _Spy(engine) as spy:
        build_calibration(reading, world.a, world.pia.pk, "en")
    assert len(spy.statements) == 4
    for statement in spy.statements:
        assert statement.lstrip().upper().startswith("SELECT"), statement
        assert not FORBIDDEN.search(statement), statement
    assert "SECRET" not in json.dumps(fake.calls[0], ensure_ascii=False)


def test_the_model_sees_what_the_person_may_see_and_nothing_else(world, reading, model):
    fake = model(_answer)
    build_calibration(reading, world.a, world.pia.pk, "en")
    prompt = fake.calls[0][1]["content"]
    for title in VISIBLE:
        assert json.dumps(title) in prompt, title
    for mark in HIDDEN_MARKS:
        assert mark not in prompt, mark
    assert fake.workspaces == [world.a]


def test_another_person_reads_only_their_own_conversations(world, reading, model):
    fake = model(lambda refs: json.dumps({"lines": [], "offers": []}))
    build_calibration(reading, world.a, world.oto.pk, "en")
    prompt = fake.calls[0][1]["content"]
    assert json.dumps("OTHER-PERSON chat") in prompt and json.dumps("OTHER-PERSON private.pdf") in prompt
    assert "Pricing for Lisbon" not in prompt and "Brand guide.docx" not in prompt


def test_another_workspace_never_reaches_the_prompt(world, reading, model):
    fake = model(lambda refs: json.dumps({"lines": [], "offers": []}))
    build_calibration(reading, world.b, world.pia.pk, "en")
    prompt = fake.calls[0][1]["content"]
    assert "OTHER-WORKSPACE chat" in prompt
    for title in VISIBLE:
        assert title not in prompt, title


def test_each_kind_is_cut_at_per_kind_and_a_blank_title_is_left_out(world, reading, model):
    fake = model(lambda refs: json.dumps({"lines": [], "offers": []}))
    build_calibration(reading, world.d, world.lu.pk, "en")
    refs = _refs_by_title(fake.calls[0][1]["content"])
    assert len([ref for ref in refs.values() if ref.startswith("D")]) == PER_KIND
    assert sorted(refs.values()) == sorted(f"D{n}" for n in range(1, PER_KIND + 1))
    seen_a = model(lambda refs: json.dumps({"lines": [], "offers": []}))
    build_calibration(reading, world.a, world.pia.pk, "en")
    assert '"title": ""' not in seen_a.calls[0][1]["content"]


def test_the_read_transaction_ends_before_the_model_call(engine, world, model):
    from sqlalchemy.orm import Session

    seen = []
    with engine.connect() as conn:
        conn.execute(text(OUTPUTS_VIEW))
        conn.commit()
        session = Session(bind=conn)
        try:
            fake = model(lambda refs: (seen.append(session.in_transaction()), json.dumps({"lines": []}))[1])
            build_calibration(session, world.a, world.pia.pk, "en")
        finally:
            session.close()
            conn.execute(text("DROP VIEW IF EXISTS pg_temp.v_workspace_outputs"))
            conn.commit()
    assert seen == [False] and len(fake.calls) == 1


# --------------------------------------------------------------------------- what is kept


def test_provenance_and_percentages_drop_lines_and_the_counts_read_from_n(world, reading, model):
    model(_answer)
    calibration = build_calibration(reading, world.a, world.pia.pk, "en", calibration_id="cal-1")
    assert calibration.id == "cal-1" and calibration.v == 1 and calibration.empty is False
    assert [line.text for line in calibration.lines] == [
        "You review pricing for clients in Lisbon", "You keep onboarding and brand notes"]
    assert [line.id for line in calibration.lines] == ["line-1", "line-2"]
    first, second = calibration.lines
    assert [source.type for source in first.sources] == ["deliverable", "conversation"]
    assert [source.label for source in first.sources] == ["Q3 pricing review", "Pricing for Lisbon"]
    assert source_count("en", first.sources) == "from 1 Deliverable and 1 Conversation"
    assert source_count("en", second.sources) == "from 2 documents"
    assert not any("85" in line.text for line in calibration.lines)
    assert calibration.offers == ["Draft a LinkedIn post (with Social writer)", *starters("en")[:2]]
    dumped = calibration.model_dump(mode="json")
    assert set(dumped) == {"v", "id", "lines", "offers", "empty"}
    assert set(dumped["lines"][0]["sources"][0]) == {"type", "id", "label"}


def test_portuguese_brazil_reads_its_own_table(world, reading, model):
    fake = model(_answer)
    calibration = build_calibration(reading, world.a, world.pia.pk, "pt-BR")
    assert "Portuguese (Brazil)" in fake.calls[0][0]["content"]
    assert source_count("pt-BR", calibration.lines[0].sources) == "de 1 Entregável e 1 Conversa"
    assert source_count("pt-BR", calibration.lines[1].sources) == "de 2 documentos"
    assert calibration.offers == ["Draft a LinkedIn post (com Social writer)", *starters("pt-BR")[:2]]
    text_part = calibration_text("pt-BR", calibration)
    assert "- You review pricing for clients in Lisbon (de 1 Entregável e 1 Conversa)" in text_part
    assert STRINGS["pt-BR"]["text_close"] in text_part


def test_the_text_part_lists_the_lines_with_their_counts_and_the_offers(world, reading, model):
    model(_answer)
    calibration = build_calibration(reading, world.a, world.pia.pk, "en")
    text_part = calibration_text("en", calibration)
    assert text_part.startswith(STRINGS["en"]["text_lines"])
    assert "- You review pricing for clients in Lisbon (from 1 Deliverable and 1 Conversation)" in text_part
    assert "- Draft a LinkedIn post (with Social writer)" in text_part
    assert "%" not in text_part


def test_every_line_dropped_says_nothing_was_clear_enough_not_nothing_to_read(world, reading, model):
    model(lambda refs: json.dumps({"lines": [{"text": "You are 90% in sales", "sources": [refs["Launch plan"]]}]}))
    calibration = build_calibration(reading, world.a, world.pia.pk, "en")
    assert calibration.empty is False and calibration.lines == [] and calibration.offers == starters("en")
    text_part = calibration_text("en", calibration)
    assert text_part.startswith(STRINGS["en"]["text_no_lines"])
    assert STRINGS["en"]["text_empty"] not in text_part and STRINGS["en"]["text_close"] not in text_part


def test_nothing_to_read_is_empty_with_three_starters_and_no_model_call(world, reading, model):
    fake = model(_answer)
    with_skill_only = build_calibration(reading, world.c, world.lu.pk, "en")
    assert with_skill_only.empty is True and with_skill_only.lines == []
    assert with_skill_only.offers == starters("en") and len(with_skill_only.offers) == OFFERS
    assert fake.calls == []
    empty_pt = build_calibration(reading, world.c, world.lu.pk, "pt-BR")
    assert empty_pt.offers == starters("pt-BR") and fake.calls == []
    assert calibration_text("en", with_skill_only).startswith(STRINGS["en"]["text_empty"])


def test_exactly_one_model_call_booked_as_calibration(world, reading, model, monkeypatch):
    fake = model(_answer)
    build_calibration(reading, world.a, world.pia.pk, "en")
    assert len(fake.calls) == 1
    assert CALIBRATION_PURPOSE == "calibration" and CALIBRATION_PURPOSE in CUSTOMER_PURPOSES
    built = []

    class _Manager:
        def __init__(self, **kwargs):
            built.append(kwargs)

    monkeypatch.setattr("core.llm.manager.LLMManager", _Manager)
    REAL_LLM(world.a)
    assert built == [{"service_name": "orchestrator", "workspace_id": world.a, "request_type": "calibration"}]


def test_an_answer_with_no_json_fails_and_a_failed_call_raises(world, reading, model):
    model(lambda refs: "Sorry, I can't help with that.")
    with pytest.raises(CalibrationFailed):
        build_calibration(reading, world.a, world.pia.pk, "en")

    def boom(refs):
        raise RuntimeError("provider down")

    model(boom)
    with pytest.raises(RuntimeError):
        build_calibration(reading, world.a, world.pia.pk, "en")


# --------------------------------------------------------------------------- pure rules


def _reads():
    sources = [CalibrationSource(type="deliverable", id=str(n), label=f"Deliverable {n}") for n in range(1, 6)]
    return {f"D{n}": Read(source, "report") for n, source in enumerate(sources, 1)}


def test_at_most_three_lines_and_a_long_line_is_dropped():
    refs = _reads()
    too_long = "x" * (MAX_LINE_CHARS + 1)
    answer = {"lines": [{"text": too_long, "sources": ["D1"]},
                        *({"text": f"Line {n}", "sources": [f"D{n}"]} for n in range(1, 6))]}
    kept = kept_lines(answer, refs)
    assert [line.text for line in kept] == ["Line 1", "Line 2", "Line 3"] and len(kept) == MAX_LINES


def test_a_line_or_offer_with_a_number_for_sureness_is_dropped():
    for words in ("85%", "85\uff05", "about 85 % of it", "9/10", "8 out of 10", "nine out of ten", "9 de 10",
                  "confidence 0.8", "70 por cento", "certeza de 90"):
        assert carries_a_number_for_sureness(words), words
    for words in ("You plan Q4 marketing for 2 brands", "Você escreve 3 relatórios por semana"):
        assert not carries_a_number_for_sureness(words), words
    refs = {**_reads(), "S1": Read(CalibrationSource(type="skill", id="9", label="Writer"), "communication")}
    offers = kept_offers({"offers": [{"text": "Write 100% of your posts", "skill": "S1"}]}, refs, "en")
    assert offers == starters("en")


def test_a_line_or_offer_that_points_anywhere_is_dropped():
    for words in ("verify at evil-example.com/login", "mail ops@acme.co", "call +351 912 345 678", "https://x.test",
                  "www.example"):
        assert carries_a_link(words), words
    for words in ("You keep onboarding notes.pdf", "You plan the launch on 2026-10-31", "Q3 2026 pricing for 10 clients"):
        assert not carries_a_link(words), words
    refs = {**_reads(), "S1": Read(CalibrationSource(type="skill", id="9", label="Writer"), "communication")}
    assert kept_lines({"lines": [{"text": "See acme.io for it", "sources": ["D1"]}]}, refs) == []
    assert kept_offers({"offers": [{"text": "Post to acme.io", "skill": "S1"}]}, refs, "en") == starters("en")
    risky = {**refs, "S2": Read(CalibrationSource(type="skill", id="8", label="100% Writer"), "communication")}
    assert kept_offers({"offers": [{"text": "Draft a post", "skill": "S2"}]}, risky, "en") == starters("en")
    assert kept_lines({"lines": [{"text": "You use Writer", "sources": ["S1"]}]}, refs) == []


def test_duplicate_offers_and_lines_are_kept_once():
    refs = {**_reads(), "S1": Read(CalibrationSource(type="skill", id="9", label="Writer"), "communication")}
    offers = kept_offers({"offers": [{"text": "Use Writer for a post", "skill": "S1"}] * 2}, refs, "en")
    assert offers == ["Use Writer for a post", *starters("en")[:2]]
    lines = kept_lines({"lines": [{"text": "Same", "sources": ["D1"]}, {"text": "same", "sources": ["D2"]}]}, refs)
    assert [line.text for line in lines] == ["Same"]


def test_every_key_is_in_both_tables_and_the_chooser_falls_back():
    assert set(STRINGS["en"]) == set(STRINGS["pt-BR"])
    for language in ("en", "pt-BR"):
        for kind in SOURCE_TYPES:
            assert f"{kind}_one" in STRINGS[language] and f"{kind}_many" in STRINGS[language]
        assert all(key in STRINGS[language] for key in STARTER_KEYS)
        assert all(value.strip() for value in STRINGS[language].values())
    assert consent_line("en") == "Auto reads the titles of your recent work to say what it knows."
    assert calibration_language("pt-PT") == "pt-BR" and calibration_language("pt") == "pt-BR"
    assert calibration_language("fr-FR") == "en" and calibration_language(None) == "en"
    assert len(starters("en")) == len(starters("pt-BR")) == OFFERS
