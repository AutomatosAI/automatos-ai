"""PRD-268 W2 US-C19c (§6.11.2, K7, K11, K16) — ``Brief@1``: the structured brief, its sections, counts, fold and strings.

``services.brief.build.build_brief`` assembles one person's brief for one local date: Auto's
Read as ``prose`` (``generate_digest``, unchanged), and five sections, each from its own
reader, folded at 5 and never at 6, with exact counts. On Postgres with committing sessions;
the schedule is faked (the calendar's own read, ``ActivityService.get_schedule``, recorded),
the readiness check is held (``services.readiness.service.held``), and Auto's Read runs
through the real ``generate_digest`` with a fixed snapshot, no cache, and the model faked so
each model call books one ``llm_usage`` row, as the real manager does. Proven here:

* every section from seeded rows: Needs you is ``needs_you()``'s rows with W1's state and next
  step; Suggested is up to 3 of the person's open suggestions (a lapsed one, another person's
  and another workspace's left out); Today is the schedule on the local date in the person's
  zone; Since is the person's started suggestions and the tickets and Missions finished since
  their last brief; Couldn't check is the brief's readiness items that are not ready;
* the counts are exact, and ``counts.needs_you`` is ``needs_you_counts()['total']`` for the same
  caller; a viewer's brief has no question and no approval grant;
* each section folds at 5 and never at 6 (5, 6 and 7 rows); an empty brief carries the empty
  sentence; ``spoken`` is the greeting and the numbered titles, never a why;
* the strings have every key in English and Portuguese (Brazil), and K7 chooses the table;
* one ``generate_digest`` call and one ``llm_usage`` row at most, none when the Read is given or
  cached; building a brief writes nothing; another workspace's rows never appear.
"""
from __future__ import annotations

import json
import uuid
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace as NS
from uuid import UUID

import pytest
from sqlalchemy import create_engine, text

from services.brief import strings as brief_strings
from services.brief.model import ACTION_VERBS, SECTIONS, Brief, BriefRow
from services.brief.rows import FOLD_AT, fold

NOW = datetime.now(timezone.utc)
TODAY = NOW.date()
TOMORROW = TODAY + timedelta(days=1)
PROSE = "Two things need you: the price list and a question about Sundays. The menu photos are done."
DIGEST = {"text": PROSE, "state_hash": "state-hash-given", "generated_at": NOW.isoformat()}
SNAPSHOT = {"counts": {}, "needs_attention": [], "active": [], "recent_completions": [],
            "needs_attention_count": 0, "state_hash": "state-hash-read"}
SAO_PAULO = "America/Sao_Paulo"
BRIEF_KEYS = {"v", "date", "greeting", "prose", "spoken", "sections", "counts", "labels", "state_hash",
              "generated_at"}


def _schedule_items():
    """The calendar's items: three on the local date in UTC; a social post tomorrow; a task at
    23:30 in São Paulo, which is tomorrow in UTC."""
    return [
        {"id": "routine-7", "name": "SCOUT Routine", "type": "routine", "next_run_at": f"{TODAY}T09:00:00+00:00"},
        {"id": "board-9", "board_task_id": 9, "name": "Rota due", "type": "task_due",
         "next_run_at": f"{TODAY}T12:00:00+00:00"},
        {"id": "mission-1", "mission_id": "00000000-0000-0000-0000-000000000001", "name": "Autumn blend deadline",
         "type": "mission", "next_run_at": f"{TODAY}T16:00:00+00:00"},
        {"id": "social-3", "name": "Autumn post", "type": "social", "next_run_at": f"{TOMORROW}T09:00:00+00:00"},
        {"id": "task-4", "name": "Late call", "type": "task", "next_run_at": f"{TODAY}T23:30:00-03:00"},
    ]


# --------------------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def engine():
    from core.database.database import get_database_url

    try:
        eng = create_engine(get_database_url(), pool_pre_ping=True)
        with eng.connect() as c:
            c.execute(text("SELECT status FROM proactive_suggestions LIMIT 1"))
            c.execute(text("SELECT posted_at FROM mobile_digest_posts LIMIT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"the brief tests need Postgres with the PRD-266 and PRD-268 tables: {exc}")
    yield eng
    eng.dispose()


def _workspace(s, name):
    ws = str(uuid.uuid4())
    s.execute(text("INSERT INTO workspaces (id, name) VALUES (CAST(:id AS uuid), :n)"), {"id": ws, "n": name})
    return ws


def _person(s, tag, name=None):
    clerk = f"user_prd268w2bb_{tag}_{uuid.uuid4().hex[:10]}"
    pk = s.execute(text(
        "INSERT INTO users (username, email, clerk_user_id, name) VALUES (:u, :e, :c, :n) RETURNING id"),
        {"u": clerk, "e": f"{clerk}@t.test", "c": clerk, "n": name}).scalar()
    return clerk, int(pk)


def _ticket(s, ws, title, status, *, source="user", done=None):
    return s.execute(text(
        "INSERT INTO board_tasks (workspace_id, title, status, source_type, completed_at) "
        "VALUES (CAST(:w AS uuid), :t, :s, :src, :done) RETURNING id"),
        {"w": ws, "t": title, "s": status, "src": source, "done": done}).scalar()


def _run(s, ws, goal, state, state_type, *, completed=None):
    return str(s.execute(text(
        "INSERT INTO orchestration_runs (workspace_id, goal, created_by, state, state_type, completed_at) "
        "VALUES (CAST(:w AS uuid), :g, 'owner', :st, :sty, :done) RETURNING id"),
        {"w": ws, "g": goal, "st": state, "sty": state_type, "done": completed}).scalar())


def _grant(s, ws, kind, subject_id, *, question=None, reason=None):
    return s.execute(text(
        "INSERT INTO approval_grants (workspace_id, subject_type, subject_id, status, kind, question_md, reason, "
        "details, requested_at) VALUES (CAST(:w AS uuid), 'board_task', :sid, 'pending', :k, :q, :r, "
        "CAST('{}' AS jsonb), :at) RETURNING id"),
        {"w": ws, "sid": str(subject_id), "k": kind, "q": question, "r": reason, "at": NOW}).scalar()


def _claim(s, ws, user_pk, local_date, posted_at):
    s.execute(text(
        "INSERT INTO mobile_digest_posts (workspace_id, user_id, local_date, posted_at) "
        "VALUES (CAST(:w AS uuid), :u, :d, :at)"), {"w": ws, "u": user_pk, "d": local_date, "at": posted_at})


def _suggestion(s, ws, user_pk, title, *, age, severity="can_wait", status="open", lifetime=timedelta(days=7),
                decided=None, ref=None):
    from core.models.proactive import ProactiveSuggestion

    made = NOW - age
    row = ProactiveSuggestion(
        workspace_id=UUID(ws), user_id=user_pk, source_type="check_in", title=title,
        detail=f"Why: {title.lower()}, as the check-in read it.", evidence=[],
        proposed_action={"kind": "board_task", "payload": {}}, subject_key=f"title:{title}", severity=severity,
        status=status, last_seen_at=made, created_at=made, expires_at=made + lifetime, decided_at=decided,
        started_ref=ref)
    s.add(row)
    s.flush()
    return str(row.id)


def _sweep(new_session, workspaces, clerks):
    s = new_session.sweep()
    for table in ("llm_usage", "proactive_suggestions", "mobile_digest_posts", "approval_grants", "board_tasks",
                  "orchestration_runs"):
        s.execute(text(f"DELETE FROM {table} WHERE CAST(workspace_id AS text) = ANY(:w)"), {"w": workspaces})
    s.execute(text("DELETE FROM users WHERE clerk_user_id = ANY(:c)"), {"c": clerks})
    s.execute(text("DELETE FROM workspaces WHERE CAST(id AS text) = ANY(:w)"), {"w": workspaces})
    s.commit()


def _seed_other_workspace(s, other, person_pk):
    """What another workspace holds, each title marked OTHER: none of it may reach the brief."""
    review = _ticket(s, other, "OTHER review", "review", done=NOW - timedelta(hours=1))
    _grant(s, other, "question", review, question="OTHER question?")
    _ticket(s, other, "OTHER done", "done", done=NOW - timedelta(hours=1))
    _run(s, other, "OTHER mission", "completed", "terminal", completed=NOW - timedelta(hours=1))
    _suggestion(s, other, person_pk, "OTHER suggestion", age=timedelta(minutes=5))
    _suggestion(s, other, person_pk, "OTHER started", age=timedelta(hours=1), status="started",
                decided=NOW - timedelta(minutes=10))
    # A later claim elsewhere: "since" is this workspace's last brief, not this one.
    _claim(s, other, person_pk, TODAY - timedelta(days=1), NOW - timedelta(minutes=30))


@pytest.fixture
def floor(engine, new_session):
    """One café's workspace with a row in every section for Sam, and another workspace."""
    s = new_session()
    ws, other = _workspace(s, "prd268w2-brief"), _workspace(s, "prd268w2-brief-other")
    (clerk, pk), (someone, someone_pk) = _person(s, "sam", "Sam Rivera"), _person(s, "jo")
    hour = timedelta(hours=1)
    review = _ticket(s, ws, "Price list for the cafés", "review", done=NOW - hour)
    blocked = _ticket(s, ws, "Rota for October", "blocked")
    t = NS(ws=ws, other=other, pk=pk, review=review, stuck=_ticket(s, ws, "Order the cups", "assigned"),
           failed=_ticket(s, ws, "Weekly numbers", "failed", done=NOW - hour),
           plan=_run(s, ws, "Launch the autumn blend", "awaiting_approval", "blocked"),
           question=_grant(s, ws, "question", blocked, question="Which café opens first on Sundays?"),
           approval=_grant(s, ws, "approval", review, reason="Send the price list to 40 cafés"))
    _claim(s, ws, pk, TODAY - timedelta(days=1), NOW - timedelta(hours=20))
    t.finished = _ticket(s, ws, "Menu photos", "done", done=NOW - 2 * hour)
    _ticket(s, ws, "Old invoice run", "done", done=NOW - timedelta(days=3))
    t.filed = _ticket(s, ws, "Refresh the sitemap", "done", source="suggestion", done=NOW - timedelta(minutes=30))
    t.mission = _run(s, ws, "Autumn newsletter", "completed", "terminal", completed=NOW - 3 * hour)
    t.oat = _suggestion(s, ws, pk, "Restock oat milk", age=3 * hour, severity="today")
    _suggestion(s, ws, pk, "Book the Lisbon flights", age=2 * hour)
    t.invoice = _suggestion(s, ws, pk, "Chase the October invoice", age=hour)
    t.domain = _suggestion(s, ws, pk, "Renew the domain", age=timedelta(minutes=30))
    t.lapsed = _suggestion(s, ws, pk, "Water filters", age=timedelta(days=8))
    _suggestion(s, ws, someone_pk, "Jo's own suggestion", age=timedelta(minutes=10))
    t.started = _suggestion(s, ws, pk, "Fix the sitemap", age=2 * hour, status="started",
                            decided=NOW - timedelta(minutes=40), ref={"board_task_id": t.filed, "watch_id": "w"})
    _suggestion(s, ws, pk, "Old start", age=timedelta(days=3), status="started", decided=NOW - timedelta(days=2))
    _seed_other_workspace(s, other, pk)
    s.commit()
    yield t
    _sweep(new_session, [ws, other], [clerk, someone])


@pytest.fixture
def bare(engine, new_session):
    """A factory of workspaces holding only ``failed`` failed tickets and ``done`` done ones."""
    made = NS(workspaces=[], clerks=[])

    def build(failed=0, done=0):
        s = new_session()
        ws = _workspace(s, "prd268w2-brief-bare")
        clerk, pk = _person(s, "bare")
        for n in range(failed):
            _ticket(s, ws, f"Failed run {n}", "failed", done=NOW - timedelta(minutes=n + 1))
        for n in range(done):
            _ticket(s, ws, f"Finished run {n}", "done", done=NOW - timedelta(minutes=n + 1))
        s.commit()
        made.workspaces.append(ws)
        made.clerks.append(clerk)
        return ws, pk

    yield build
    _sweep(new_session, made.workspaces, made.clerks)


@pytest.fixture
def schedule(monkeypatch):
    """The calendar's read, recorded: each workspace it was built for and each range it read."""
    import services.brief.today as today

    rig = NS(items=_schedule_items(), built_for=[], ranges=[])

    class Recorded:
        def __init__(self, db, workspace_id):
            rig.built_for.append(str(workspace_id))

        def get_schedule(self, *, range_days=7):
            rig.ranges.append(range_days)
            return {"scheduled": [dict(item) for item in rig.items], "scheduler_active": True}

    monkeypatch.setattr(today, "ActivityService", Recorded)
    return rig


@pytest.fixture
def checks(monkeypatch):
    """Hold a readiness check per workspace: ``hold(ws)`` the default (Auto's model needs setting
    up; web search, needed by voice only, too; push, needed by push only, unanswered), or
    ``hold(ws, ready=True)`` with every item ready."""
    from services.readiness import items as i
    from services.readiness import service
    from services.readiness.probes import ITEMS

    held = service.HeldChecks(600)
    monkeypatch.setattr(service, "held", held)
    default = {i.CORE: i.Finding(i.READY, i.CORE_READY),
               i.MODEL_ROUTE: i.Finding(i.NEEDS_SETUP, i.MODEL_UNSET, fix=i.CHOOSE_MODEL),
               i.WEB_SEARCH: i.Finding(i.NEEDS_SETUP, i.SEARCH_NONE, fix=i.ADD_KEY),
               i.PUSH: i.Finding(i.NOT_CHECKED, i.NOT_ANSWERED)}

    def hold(ws, ready=False):
        findings = {item.id: i.Finding(i.READY, i.CORE_READY) if ready else default[item.id] for item in ITEMS}
        held.put(UUID(ws), service.Checked(NOW, tuple(service.Result(item, findings[item.id], 3) for item in ITEMS)))

    return hold


class _BookedLLM:
    """The digest's model: each answer books one llm_usage row, as the real manager does."""

    def __init__(self, new_session, workspace_id):
        self.new_session, self.workspace_id = new_session, workspace_id

    async def generate_response(self, messages):
        s = self.new_session()
        s.execute(text(
            "INSERT INTO llm_usage (workspace_id, model_id, provider, tier, request_type, input_tokens, "
            "output_tokens, total_tokens, input_cost, output_cost, total_cost) VALUES (CAST(:w AS uuid), 'm', "
            "'openrouter', 'direct', 'digest', 10, 5, 15, 0, 0, 0)"), {"w": str(self.workspace_id)})
        s.commit()
        return NS(content=PROSE)


@pytest.fixture
def read(monkeypatch, new_session):
    """Auto's Read through the real ``generate_digest``: a fixed snapshot, no cache, the model
    faked (``_BookedLLM``). Each model manager made is recorded by its workspace."""
    import core.llm as core_llm
    import services.digest_service as digest_service

    made = []
    monkeypatch.setattr(digest_service, "build_digest_snapshot", lambda db, ws, period="1d": dict(SNAPSHOT))
    monkeypatch.setattr(digest_service, "_default_redis", lambda: None)

    def factory(**kwargs):
        made.append(str(kwargs["workspace_id"]))
        return _BookedLLM(new_session, kwargs["workspace_id"])

    monkeypatch.setattr(core_llm, "create_llm_manager", factory)
    return made


def _brief(new_session, ws, pk, *, may_answer=True, language=None, zone=None, digest=DIGEST):
    from services.brief.build import build_brief

    return build_brief(new_session(), ws, pk, TODAY, language, may_answer=may_answer, zone=zone, digest=digest)


def _ids(brief, section):
    return [row["id"] for row in brief["sections"][section]]


def _usage_rows(new_session, ws):
    return new_session().execute(text(
        "SELECT count(*) FROM llm_usage WHERE workspace_id = CAST(:w AS uuid)"), {"w": ws}).scalar()


# --------------------------------------------------------------------------- the shape and the strings


def test_the_brief_is_brief_at_one_with_every_id_a_string_and_unknown_fields_ignored(floor, new_session,
                                                                                     schedule, checks):
    checks(floor.ws)
    brief = _brief(new_session, floor.ws, floor.pk)

    assert set(brief) == BRIEF_KEYS and brief["v"] == 1 and brief["date"] == TODAY.isoformat()
    assert Brief.model_validate({**brief, "a_later_field": 1}).model_dump(mode="json", exclude_none=True) == brief
    rows = [row for section in SECTIONS for row in brief["sections"][section]]
    assert rows and all(isinstance(row["id"], str) and isinstance(row.get("opens", {}).get("id", ""), str)
                        for row in rows)
    read_back = BriefRow.model_validate({"id": 7, "title": "t", "why": "w", "opens": {"type": "ticket", "id": 9},
                                         "colour": "red"})
    assert (read_back.id, read_back.opens.id) == ("7", "9") and not hasattr(read_back, "colour")


@pytest.mark.parametrize("language", ["en", "pt-BR"])
def test_the_strings_have_every_key_the_brief_uses_in_both_languages(language):
    table = brief_strings.STRINGS[language]

    assert set(brief_strings.STRINGS) == {"en", "pt-BR"}
    assert set(table) == set(brief_strings.STRINGS["en"]) == set(brief_strings.STRINGS["pt-BR"])
    used = {f"section_{name}" for name in SECTIONS} | {f"action_{verb}" for verb in ACTION_VERBS}
    used |= {"empty", "more", "untitled", "action_start_done", "action_hide_done", "today_other"}
    used |= {f"greeting_{part}{anon}" for _, part in brief_strings.GREETING_HOURS for anon in ("", "_anon")}
    used |= {"today_routine", "today_playbook", "today_task", "today_mission", "today_task_due", "today_social",
             "since_ready", "since_failed", "since_working", "since_started"}
    assert used <= set(table) and all(sentence.strip() for sentence in table.values())


def test_a_cut_holds_a_title_to_its_characters_and_its_bytes():
    from services.brief.rows import BYTES_PER_CHAR, TITLE_CHARS, cut

    assert cut("  Price   list\nfor the cafés ", TITLE_CHARS) == "Price list for the cafés"
    assert len(cut("x" * 500, TITLE_CHARS)) == TITLE_CHARS and cut("x" * 500, TITLE_CHARS).endswith("…")
    for wide in ("😀" * 500, "語" * 500, "é" * 500):
        assert len(cut(wide, TITLE_CHARS).encode("utf-8")) <= TITLE_CHARS * BYTES_PER_CHAR
        assert cut(wide, TITLE_CHARS).endswith("…")                     # never a split character


def test_a_brief_of_the_most_rows_fits_its_message_part():
    """Every section at its most (6, 3, 6, 6 and both readiness items the brief needs), each title
    and reason of 3-byte characters cut at its limit, the spoken words, and 3 KB of Auto's Read:
    the part ``post_background_message`` saves still passes ``checked_parts``."""
    from services.brief.build import spoken_words
    from services.brief.model import PART_TYPE, BriefCounts, BriefLabels, BriefOpens, BriefSections
    from services.brief.rows import WHY_CHARS, action, cut, title_of
    from services.chat_parts import CHAT_PART_MAX_BYTES, checked_parts

    def row(section):
        return BriefRow(id=f"{section}:{uuid.uuid4()}", title=title_of("語" * 500, "pt-BR"),
                        why=cut("語" * 900, WHY_CHARS), at=NOW.isoformat(), number="#0001.12",
                        display_state="needs_you", opens=BriefOpens(type="ticket", id=str(uuid.uuid4())),
                        action=action("start", "pt-BR", done=True))

    most = {"needs_you": FOLD_AT + 1, "suggestions": 3, "today": FOLD_AT + 1, "since": FOLD_AT + 1, "unchecked": 2}
    sections = BriefSections(**{name: [row(name) for _ in range(rows)] for name, rows in most.items()})
    greeted = brief_strings.greeting("pt-BR", cut("語" * 100, 40), 7)
    brief = Brief(v=1, date=TODAY.isoformat(), greeting=greeted, prose="語" * 1000,
                  spoken=spoken_words(greeted, sections, None), sections=sections,
                  counts=BriefCounts(needs_you=999, suggestions=999, today=999, since=999),
                  labels=BriefLabels(sections={name: name for name in SECTIONS},
                                     more={name: brief_strings.more("pt-BR", 999) for name in SECTIONS}),
                  state_hash="a" * 64, generated_at=NOW.isoformat()).model_dump(mode="json", exclude_none=True)

    (saved,) = checked_parts([{"type": PART_TYPE, "brief": brief}])
    assert saved["brief"] == brief
    assert len(json.dumps(saved, separators=(",", ":"), ensure_ascii=False).encode("utf-8")) < CHAT_PART_MAX_BYTES


def test_portuguese_is_its_own_copy_and_the_language_follows_k7():
    en, pt = brief_strings.STRINGS["en"], brief_strings.STRINGS["pt-BR"]

    assert all(pt[key] != en[key] for key in en)
    choose = brief_strings.brief_language
    assert [choose(tag) for tag in ("pt-BR", "pt_br", "pt", "pt-PT", "en", "en-GB", "fr-FR", "", None, 7)] == [
        "pt-BR", "pt-BR", "pt-BR", "pt-BR", "en", "en", "en", "en", "en", "en"]
    assert brief_strings.more("en", 2) == "Show 2 more" and brief_strings.more("pt-BR", 2) == "Mostrar mais 2"


# --------------------------------------------------------------------------- each section


def test_needs_you_is_needs_yous_rows_with_their_state_and_next_step(floor, new_session, schedule, checks):
    from services.proactive.strings import APPROVAL_PLAN, STRINGS

    checks(floor.ws)
    rows = {row["id"]: row for row in _brief(new_session, floor.ws, floor.pk)["sections"]["needs_you"]}
    en = STRINGS["en"]

    # needs_you()'s kinds in its order; its two approvals newest first, as it lists them.
    assert [key.split(":", 1)[0] for key in rows] == ["review", "question", "approval", "approval", "stuck", "failed"]
    assert set(rows) == {f"review:{floor.review}", f"question:{floor.question}", f"approval:{floor.approval}",
                         f"approval:{floor.plan}", f"stuck:{floor.stuck}", f"failed:{floor.failed}"}
    assert rows[f"question:{floor.question}"]["opens"] == {"type": "question", "id": str(floor.question)}
    assert rows[f"question:{floor.question}"]["action"]["verb"] == "answer"
    assert rows[f"question:{floor.question}"]["title"] == "Which café opens first on Sundays?"
    assert rows[f"approval:{floor.plan}"]["opens"] == {"type": "mission", "id": floor.plan}
    assert rows[f"approval:{floor.plan}"]["why"] == en[APPROVAL_PLAN]
    assert rows[f"review:{floor.review}"]["opens"] == {"type": "ticket", "id": str(floor.review)}
    assert rows[f"stuck:{floor.stuck}"]["why"] == en["no_agent"]
    assert rows[f"failed:{floor.failed}"]["display_state"] == "failed"
    assert {row["display_state"] for key, row in rows.items() if not key.startswith("failed")} == {"needs_you"}


def test_suggested_is_up_to_three_of_the_persons_open_suggestions(floor, new_session, schedule, checks):
    checks(floor.ws)
    brief = _brief(new_session, floor.ws, floor.pk)
    rows = brief["sections"]["suggestions"]

    # Due today first, then the newest; a lapsed one and Jo's are left out.
    assert _ids(brief, "suggestions") == [f"suggestion:{s}" for s in (floor.oat, floor.domain, floor.invoice)]
    assert all(row["opens"] == {"type": "suggestion", "id": row["id"].split(":", 1)[1]} for row in rows)
    assert all(row["action"] == {"verb": "start", "label": "Start", "done_label": "Started"} for row in rows)
    assert rows[0]["why"] == "Why: restock oat milk, as the check-in read it."
    assert brief["counts"]["suggestions"] == 4 and brief["labels"]["more"]["suggestions"] == "Show 1 more"


def test_today_is_the_schedule_on_the_local_date_in_the_persons_zone(floor, new_session, schedule, checks):
    checks(floor.ws)
    utc = _brief(new_session, floor.ws, floor.pk)
    sao_paulo = _brief(new_session, floor.ws, floor.pk, zone=SAO_PAULO)
    rows = {row["id"]: row for row in utc["sections"]["today"]}

    assert list(rows) == ["schedule:routine-7", "schedule:board-9", "schedule:mission-1"]
    assert rows["schedule:board-9"]["opens"] == {"type": "ticket", "id": "9"}
    assert rows["schedule:mission-1"]["opens"] == {"type": "mission", "id": "00000000-0000-0000-0000-000000000001"}
    assert "opens" not in rows["schedule:routine-7"] and rows["schedule:routine-7"]["why"] == "An agent's routine runs."
    assert rows["schedule:board-9"]["at"] == f"{TODAY}T12:00:00+00:00"
    assert utc["counts"]["today"] == 3 and sao_paulo["counts"]["today"] == 4
    assert _brief(new_session, floor.ws, floor.pk, zone="Not/A_Zone")["counts"]["today"] == 3   # read as UTC
    assert "schedule:task-4" in _ids(sao_paulo, "today") and "schedule:social-3" not in _ids(sao_paulo, "today")
    assert set(schedule.built_for) == {floor.ws} and set(schedule.ranges) == {1}


def test_since_is_what_started_and_finished_since_the_last_brief(floor, new_session, schedule, checks):
    checks(floor.ws)
    brief = _brief(new_session, floor.ws, floor.pk)
    rows = {row["id"]: row for row in brief["sections"]["since"]}

    # Newest first: the started suggestion (its ticket finished), the menu photos, the newsletter.
    assert list(rows) == [f"suggestion:{floor.started}", f"ticket:{floor.finished}", f"mission:{floor.mission}"]
    assert rows[f"suggestion:{floor.started}"]["opens"] == {"type": "ticket", "id": str(floor.filed)}
    assert {row["display_state"] for row in rows.values()} == {"ready"}
    assert {row["why"] for row in rows.values()} == {"Finished since your last brief."}
    assert rows[f"mission:{floor.mission}"]["opens"] == {"type": "mission", "id": floor.mission}
    assert brief["counts"]["since"] == 3


def test_a_started_suggestion_says_what_its_ticket_became(bare, new_session, schedule, checks):
    ws, pk = bare()
    checks(ws, ready=True)
    s = new_session()
    started = {}
    for status, title in (("in_progress", "Draft the menu"), ("failed", "Send the rota"), ("review", "Price list")):
        ticket = _ticket(s, ws, title, status, source="suggestion")
        started[status] = (_suggestion(s, ws, pk, title, age=timedelta(hours=2), status="started",
                                       decided=NOW - timedelta(minutes=20), ref={"board_task_id": ticket}), ticket)
    chat = _suggestion(s, ws, pk, "Talk it through", age=timedelta(hours=2), status="started",
                       decided=NOW - timedelta(minutes=10), ref=None)
    s.commit()
    rows = {row["id"]: row for row in _brief(new_session, ws, pk)["sections"]["since"]}
    en = brief_strings.STRINGS["en"]

    def said(status):
        row = rows[f"suggestion:{started[status][0]}"]
        return row["display_state"] if "display_state" in row else None, row["why"], row["opens"]["id"]

    assert said("in_progress") == ("working", en["since_working"], str(started["in_progress"][1]))
    assert said("failed") == ("failed", en["since_failed"], str(started["failed"][1]))
    # Review the owner judges is needs_you()'s to say: the board's read draws nothing for it.
    assert said("review") == (None, en["since_started"], str(started["review"][1]))
    # A chat start filed nothing: the row opens the suggestion.
    assert rows[f"suggestion:{chat}"]["opens"] == {"type": "suggestion", "id": chat}
    assert "display_state" not in rows[f"suggestion:{chat}"]


def test_couldnt_check_is_the_briefs_readiness_items_that_are_not_ready(floor, new_session, schedule, checks):
    from services.readiness.items import CHOOSE_MODEL, MODEL_UNSET
    from services.readiness.strings import say

    checks(floor.ws)
    (row,) = _brief(new_session, floor.ws, floor.pk)["sections"]["unchecked"]

    # Web search (voice only) and push (push only) are not the brief's.
    assert row["id"] == "readiness:model_route" and row["display_state"] == "needs_you"
    # The local edition adds how long the probe took after the sentence; SaaS shows the sentence alone.
    assert row["title"] == say("en", "model_route") and row["why"].startswith(say("en", MODEL_UNSET))
    assert row["opens"] == {"type": "settings", "id": "/settings?tab=orchestrator"}
    assert row["action"] == {"verb": "connect", "label": say("en", CHOOSE_MODEL)}


# --------------------------------------------------------------------------- counts, the viewer, the fold


def test_the_counts_are_exact_and_needs_you_is_the_badge(floor, new_session, schedule, checks):
    from services.needs_you import needs_you_counts

    checks(floor.ws)
    brief = _brief(new_session, floor.ws, floor.pk)

    assert needs_you_counts(new_session(), UUID(floor.ws))["total"] == brief["counts"]["needs_you"] == 6
    assert brief["counts"] == {"needs_you": 6, "suggestions": 4, "today": 3, "since": 3}
    assert len(brief["sections"]["needs_you"]) == 6 and "needs_you" not in brief["labels"]["more"]  # 6 never folds


def test_a_viewers_brief_has_no_question_and_no_approval_grant(floor, new_session, schedule, checks):
    from services.needs_you import needs_you_counts

    checks(floor.ws)
    brief = _brief(new_session, floor.ws, floor.pk, may_answer=False)
    rows = brief["sections"]["needs_you"]

    assert needs_you_counts(new_session(), UUID(floor.ws), may_answer=False)["total"] == brief["counts"]["needs_you"]
    assert not [row for row in rows if row["id"].startswith("question:") or row["opens"]["type"] == "question"]
    assert f"approval:{floor.approval}" not in _ids(brief, "needs_you")
    # A Mission waiting for its plan's approval is counted for a viewer, as needs_you() counts it (R5).
    assert [row["id"] for row in rows if row["id"].startswith("approval:")] == [f"approval:{floor.plan}"]


@pytest.mark.parametrize("listed, shown, more", [(5, 5, None), (6, 6, None), (7, 5, "Show 2 more")])
def test_each_section_folds_at_five_and_never_at_six(bare, new_session, schedule, checks, listed, shown, more):
    ws, pk = bare(failed=listed, done=listed)
    checks(ws, ready=True)
    schedule.items = [{**_schedule_items()[0], "id": f"routine-{n}"} for n in range(listed)]
    brief = _brief(new_session, ws, pk)

    for section in ("needs_you", "today", "since"):
        assert len(brief["sections"][section]) == shown and brief["counts"][section] == listed
        assert brief["labels"]["more"].get(section) == more
    assert [len(fold([BriefRow(id=str(n), title="t", why="w") for n in range(rows)], rows)[0])
            for rows in (5, 6, 7)] == [FOLD_AT, FOLD_AT + 1, FOLD_AT]


def test_an_empty_brief_carries_the_empty_sentence(bare, new_session, schedule, checks):
    ws, pk = bare()
    checks(ws, ready=True)
    schedule.items = []
    brief = _brief(new_session, ws, pk)
    empty = brief_strings.STRINGS["en"]["empty"]

    assert all(brief["sections"][section] == [] for section in SECTIONS)
    assert brief["counts"] == {"needs_you": 0, "suggestions": 0, "today": 0, "since": 0}
    assert brief["labels"]["empty"] == empty and brief["labels"]["more"] == {}
    assert brief["spoken"].startswith(brief["greeting"].rstrip(".")) and brief["spoken"].endswith(empty)


# --------------------------------------------------------------------------- copy, spoken, language


def test_the_labels_and_the_greeting_are_the_servers_in_the_persons_language(floor, new_session, schedule, checks):
    from services.proactive.strings import STRINGS as NEXT_STEPS
    from services.readiness.strings import say as readiness_say

    checks(floor.ws)
    en, pt = _brief(new_session, floor.ws, floor.pk), _brief(new_session, floor.ws, floor.pk, language="pt-BR")
    english_greetings = {brief_strings.greeting("en", "Sam", hour) for hour in (6, 13, 20)}

    assert en["greeting"] in english_greetings and "Sam" in en["greeting"]
    assert pt["greeting"] in {brief_strings.greeting("pt-BR", "Sam", hour) for hour in (6, 13, 20)}
    assert en["labels"]["sections"] == {name: brief_strings.STRINGS["en"][f"section_{name}"] for name in SECTIONS}
    assert pt["labels"]["sections"]["since"] == "Desde o seu último resumo"
    assert pt["labels"]["more"]["suggestions"] == "Mostrar mais 1"
    stuck = next(row for row in pt["sections"]["needs_you"] if row["id"] == f"stuck:{floor.stuck}")
    assert stuck["why"] == NEXT_STEPS["pt-BR"]["no_agent"]
    (unchecked,) = pt["sections"]["unchecked"]
    assert unchecked["title"] == readiness_say("pt-BR", "model_route") != readiness_say("en", "model_route")


def test_spoken_is_the_greeting_and_the_numbered_titles_never_the_why(floor, new_session, schedule, checks):
    from modules.voice.spoken_style import speechify

    checks(floor.ws)
    brief = _brief(new_session, floor.ws, floor.pk)
    titles = [row["title"] for section in SECTIONS for row in brief["sections"][section]]
    said = " ".join([brief["greeting"], *(f"{n}. {title}." for n, title in enumerate(titles, 1))])

    assert brief["spoken"] == speechify(said)
    assert "1. Price list for the cafés" in brief["spoken"] and f"{len(titles)}. " in brief["spoken"]
    assert "as the check-in read it" not in brief["spoken"] and "Finished since" not in brief["spoken"]


# --------------------------------------------------------------------------- Auto's Read, writes, tenancy


def test_one_generate_digest_call_and_one_llm_usage_row(floor, new_session, schedule, checks, read):
    checks(floor.ws)
    brief = _brief(new_session, floor.ws, floor.pk, digest=None)

    assert (brief["prose"], brief["state_hash"]) == (PROSE, SNAPSHOT["state_hash"])
    assert read == [floor.ws] and _usage_rows(new_session, floor.ws) == 1


def test_a_read_given_or_cached_makes_no_model_call(floor, new_session, schedule, checks, read, monkeypatch):
    import services.digest_service as digest_service

    checks(floor.ws)
    given = _brief(new_session, floor.ws, floor.pk)
    cached = {"text": "From the cache.", "state_hash": SNAPSHOT["state_hash"], "generated_at": NOW.isoformat()}
    monkeypatch.setattr(digest_service, "_default_redis", lambda: NS(get=lambda key: json.dumps(cached)))
    from_cache = _brief(new_session, floor.ws, floor.pk, digest=None)

    assert (given["prose"], given["state_hash"]) == (PROSE, DIGEST["state_hash"])
    assert from_cache["prose"] == "From the cache."
    assert read == [] and _usage_rows(new_session, floor.ws) == 0


def test_building_a_brief_writes_nothing(floor, new_session, schedule, checks):
    checks(floor.ws)
    count = "SELECT {} FROM {} WHERE workspace_id = CAST(:w AS uuid)"
    tables = {"proactive_suggestions": "count(*) FILTER (WHERE status = 'open')", "mobile_digest_posts": "count(*)",
              "messages": "count(*)", "board_tasks": "count(*)"}
    before = {t: new_session().execute(text(count.format(c, t)), {"w": floor.ws}).scalar() for t, c in tables.items()}
    _brief(new_session, floor.ws, floor.pk)

    # The lapsed suggestion is left out of the brief, and still open: the brief moved nothing.
    assert {t: new_session().execute(text(count.format(c, t)), {"w": floor.ws}).scalar()
            for t, c in tables.items()} == before


def test_another_workspaces_rows_never_appear(floor, new_session, schedule, checks):
    checks(floor.ws)
    brief = _brief(new_session, floor.ws, floor.pk)

    assert "OTHER" not in json.dumps(brief)
    assert set(schedule.built_for) == {floor.ws}
    # Its later claim does not move "since": this workspace's last brief was 20 hours ago.
    assert f"ticket:{floor.finished}" in _ids(brief, "since")
