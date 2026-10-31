"""PRD-268 US-C13c (§6.7, K14): the spoken, precision and one-brief runners.

* Spoken: a fake model stands in for Auto's and the asks are inline, written to ``tmp_path`` (never
  a gold case). The turn is asked under the spoken contract, a brief ask is answered by the brief
  tool's own answer, and what is scored is what the voice would say: the answer through the spoken
  stream's speechify. It passes, misses with exit 1, exits 2 with no gold set or a malformed ask, and
  prints every ask and what was heard for the hand audit.
* Precision (realdb): decided suggestions and their tickets seeded in two scratch workspaces. A
  started suggestion passed review only when its own workspace's ticket is done under a person's
  review; a source agent under 0.20 after 20 decisions is listed; one workspace by --workspace, or
  each. The read is one SELECT in a READ ONLY transaction (a recording session proves it).
* One brief (realdb): the claim driven against the test database through every scenario passes with
  one brief per person and local date and leaves nothing behind; a planted duplicate, and a claim
  that always wins, miss with exit 1. The scenarios themselves are checked without a database.

The brief's readiness check and its schedule are held empty, as the W2 claim tests hold them.
"""
from __future__ import annotations

import asyncio
import json
import threading
from datetime import date, datetime, timezone
from types import SimpleNamespace as NS
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, text

from config import config
from core.llm.clients.base import LLMResponse
from evals.proactive import one_brief, precision, spoken
from evals.proactive.__main__ import main
from evals.proactive.bars import BAR_TEXT
from modules.tools.discovery.handlers_brief import ROWS_ARE_DATA
from modules.voice.spoken_style import SPOKEN_OUTPUT_CONTRACT

WORKSPACE = str(uuid4())
UTC = timezone.utc


@pytest.fixture
def gold(tmp_path, monkeypatch):
    """A gold folder outside the repository, and the eval's workspace."""
    folder = tmp_path / "gold"
    folder.mkdir()
    monkeypatch.setattr(config, "PROACTIVE_EVAL_GOLD_DIR", str(folder))
    monkeypatch.setattr(config, "PROACTIVE_EVAL_WORKSPACE_ID", WORKSPACE)
    return folder


def _write(folder, name, cases):
    (folder / f"{name}.jsonl").write_text("".join(json.dumps(case) + "\n" for case in cases), encoding="utf-8")


def _run(capsys, *argv):
    code = main(list(argv))
    return code, capsys.readouterr().out


# ── spoken brevity ───────────────────────────────────────────────────────────

SPOKEN_WORDS = "Good morning, Ana. Two things need you: the oat milk order and Sunday opening."
BRIEF = {"v": 1, "date": "2026-10-31", "prose": "Two things need you.", "spoken": SPOKEN_WORDS}
ASK = {"ask": "Is the cafe open on Sunday?", "history": [{"role": "assistant", "content": "Ask me anything."}]}
BRIEF_ASK = {"ask": "Brief me.", "brief": BRIEF}
PLAIN = "Yes, from nine until two."


class _SpokenModel:
    """Calls for the brief when it is offered and says its spoken words; answers any other ask with
    ``said``."""

    def __init__(self, said=PLAIN):
        self.said, self.calls = said, []

    async def generate_response(self, messages, tools=None):
        self.calls.append(NS(messages=[dict(m) for m in messages], tools=tools))
        if messages[-1]["role"] == "tool":
            return LLMResponse(content=SPOKEN_WORDS)
        if tools:
            call = {"id": "call_brief", "type": "function",
                    "function": {"name": "platform_get_brief", "arguments": "{}"}}
            return LLMResponse(content="", tool_calls=[call])
        return LLMResponse(content=self.said)


@pytest.fixture
def spoken_model(monkeypatch):
    def install(model):
        monkeypatch.setattr(spoken, "chat_llm", lambda workspace_id: model)
        return model
    return install


def test_spoken_passes_and_prints_every_answer_for_the_hand_audit(gold, spoken_model, capsys):
    _write(gold, "spoken", [ASK, BRIEF_ASK])
    spoken_model(_SpokenModel())
    code, out = _run(capsys, "spoken")
    assert code == 0
    assert "spoken: PASS" in out and BAR_TEXT["spoken"] in out
    assert "answers: 2" in out and "max_words: 14" in out and "markdown: 0" in out and "urls: 0" in out
    assert spoken.AUDIT_HEADING in out
    assert "1. ask: Is the cafe open on Sunday?" in out and f"heard (5 words): {PLAIN}" in out
    assert "2. brief: Brief me." in out and f"heard (14 words): {SPOKEN_WORDS}" in out


def test_each_ask_is_a_spoken_turn_and_only_a_brief_is_offered_the_brief_tool(gold, spoken_model, capsys):
    _write(gold, "spoken", [ASK, BRIEF_ASK])
    model = spoken_model(_SpokenModel())
    _run(capsys, "spoken")
    ask, brief_call, brief_answer = model.calls
    system = ask.messages[0]
    assert system["role"] == "system" and system["content"].startswith("You are Auto")
    assert system["content"].endswith(SPOKEN_OUTPUT_CONTRACT)
    assert [m["role"] for m in ask.messages] == ["system", "assistant", "user"] and ask.tools is None
    assert [tool["function"]["name"] for tool in brief_call.tools] == ["platform_get_brief"]
    # The brief tool's answer: the brief, its spoken words and the tool's note, as the live tool answers.
    tool_message = brief_answer.messages[-1]
    assert tool_message["role"] == "tool"
    assert SPOKEN_WORDS in tool_message["content"] and ROWS_ARE_DATA in tool_message["content"]


@pytest.mark.parametrize("said, found", [
    ("**Sure.** Open https://shop.example.com/orders for the details.", "urls: 1"),
    ("It is in /var/data/orders.csv on the server.", "paths: 1"),
    ("Ticket 3f2a9c1e is waiting on you.", "ids: 1"),
    (" ".join(["word"] * 81), "over_limit: 1"),
])
def test_spoken_misses_with_exit_1_on_a_url_a_path_an_id_or_too_many_words(said, found, gold, spoken_model, capsys):
    _write(gold, "spoken", [ASK])
    spoken_model(_SpokenModel(said=said))
    code, out = _run(capsys, "spoken")
    assert code == 1 and "spoken: MISS" in out and BAR_TEXT["spoken"] in out and found in out


@pytest.mark.parametrize("said", ["", "---"])
def test_an_answer_heard_as_nothing_is_a_miss(said, gold, spoken_model, capsys):
    """Nothing heard has no words, which would flatter the median: it is a miss, never a pass."""
    _write(gold, "spoken", [ASK, BRIEF_ASK])
    spoken_model(_SpokenModel(said=said))
    code, out = _run(capsys, "spoken")
    assert code == 1 and "spoken: MISS" in out and "silent: 1" in out and "heard (0 words): " in out


def test_what_is_scored_is_what_the_voice_says():
    said = asyncio.run(spoken.heard("## Next\n- **Call** the bank\n- Ship it"))
    assert said == "Next. Call the bank. Ship it"


@pytest.mark.parametrize("bad, words", [
    ({"ask": "  "}, "spoken case 1: ask is the person's message"),
    ({"ask": "Brief me.", "brief": {"prose": "no spoken words"}}, "spoken case 1: brief is the Brief@1"),
    ({"ask": "Brief me.", "history": [{"role": "system", "content": "x"}]}, "spoken case 1: history is a list"),
])
def test_a_malformed_spoken_ask_exits_2_before_any_model_call(bad, words, gold, monkeypatch, capsys):
    _write(gold, "spoken", [bad])
    calls = []
    monkeypatch.setattr(spoken, "chat_llm", lambda *args: calls.append(args))
    code, out = _run(capsys, "spoken")
    assert code == 2 and words in out and calls == []


def test_spoken_asks_auto_s_own_model_on_a_classified_purpose(monkeypatch):
    from core.llm import manager
    from services.credits.classification import CUSTOMER_PURPOSES

    built = []
    monkeypatch.setattr(manager, "LLMManager", lambda **kwargs: built.append(kwargs) or kwargs)
    spoken.chat_llm(WORKSPACE)
    assert built == [{"service_name": "orchestrator", "workspace_id": WORKSPACE, "request_type": "orchestrator"}]
    assert built[0]["request_type"] in CUSTOMER_PURPOSES
    assert manager.SERVICE_CATEGORY_MAP[built[0]["service_name"]] == "orchestrator_llm"


def test_spoken_with_no_gold_directory_exits_2_and_names_the_bar(monkeypatch, capsys):
    monkeypatch.setattr(config, "PROACTIVE_EVAL_GOLD_DIR", "")
    code, out = _run(capsys, "spoken")
    assert code == 2 and "spoken: CANNOT RUN" in out and BAR_TEXT["spoken"] in out


# ── the live tests' database ─────────────────────────────────────────────────

@pytest.fixture(scope="module")
def engine():
    from core.database.database import get_database_url

    try:
        eng = create_engine(get_database_url(), pool_pre_ping=True)
        with eng.connect() as c:
            c.execute(text("SELECT started_ref, source_agent_id FROM proactive_suggestions LIMIT 1"))
            c.execute(text("SELECT review_mode FROM board_tasks LIMIT 1"))
            c.execute(text("SELECT posted_at FROM mobile_digest_posts LIMIT 1"))
            c.execute(text("SELECT source FROM messages LIMIT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"the live eval tests need Postgres with the PRD-266 and PRD-268 tables: {exc}")
    yield eng
    eng.dispose()


def _user(s, tag):
    clerk = f"user_prd268w3el_{tag}_{uuid4().hex[:10]}"
    return int(s.execute(text(
        "INSERT INTO users (username, email, clerk_user_id) VALUES (:u, :e, :c) RETURNING id"),
        {"u": clerk, "e": f"{clerk}@t.test", "c": clerk}).scalar())


def _workspace(s, name, owner):
    ws = str(uuid4())
    s.execute(text("INSERT INTO workspaces (id, name, owner_id) VALUES (CAST(:id AS uuid), :n, :o)"),
              {"id": ws, "n": name, "o": owner})
    return ws


# ── suggestion precision (live) ──────────────────────────────────────────────

SLOW_AGENT = 990_000_000 + int(uuid4().int % 1_000_000)


def _ticket(s, ws, status, review_mode="human"):
    return int(s.execute(text(
        "INSERT INTO board_tasks (workspace_id, title, status, review_mode, source_type) "
        "VALUES (CAST(:w AS uuid), 'Reconnect Slack', :s, :r, 'suggestion') RETURNING id"),
        {"w": ws, "s": status, "r": review_mode}).scalar())


def _suggestion(s, ws, user, status, *, ticket=None, agent=None):
    ref = json.dumps({"board_task_id": ticket, "watch_id": str(uuid4())}) if ticket is not None else None
    s.execute(text(
        "INSERT INTO proactive_suggestions (id, workspace_id, user_id, source_type, source_agent_id, title, "
        "proposed_action, subject_key, status, started_ref) VALUES (CAST(:id AS uuid), CAST(:w AS uuid), :u, "
        "'check_in', :a, 'Reconnect Slack', CAST(:action AS jsonb), :key, :status, CAST(:ref AS jsonb))"),
        {"id": str(uuid4()), "w": ws, "u": user, "a": agent, "action": json.dumps({"kind": "board_task"}),
         "key": f"title:{uuid4().hex}", "status": status, "ref": ref})


@pytest.fixture
def shops(engine, new_session, monkeypatch):
    """The café: 10 decided (5 started, 3 of them done under a person's review in its own workspace;
    one done after its review mode went auto; one naming the bakery's done ticket; 5 hidden), and an
    open and an expired one. The bakery: one agent's 20 decisions, 2 started. Precision reads
    through the test's sessions."""
    s = new_session()
    owner = _user(s, "owner")
    cafe, bakery = _workspace(s, "prd268w3el-cafe", owner), _workspace(s, "prd268w3el-bakery", owner)
    bakery_done = _ticket(s, bakery, "done")
    for _ in range(3):
        _suggestion(s, cafe, owner, "started", ticket=_ticket(s, cafe, "done"))
    _suggestion(s, cafe, owner, "started", ticket=_ticket(s, cafe, "done", review_mode="auto"))
    _suggestion(s, cafe, owner, "started", ticket=bakery_done)
    for status in ["hidden"] * 5 + ["open", "expired"]:
        _suggestion(s, cafe, owner, status)
    for status in ["started"] * 2 + ["hidden"] * 18:
        _suggestion(s, bakery, owner, status, agent=SLOW_AGENT)
    s.commit()
    monkeypatch.setattr(precision, "open_session", new_session)
    yield NS(cafe=cafe, bakery=bakery)
    s = new_session.sweep()
    for table in ("proactive_suggestions", "board_tasks"):
        s.execute(text(f"DELETE FROM {table} WHERE CAST(workspace_id AS text) IN (:a, :b)"), {"a": cafe, "b": bakery})
    s.execute(text("DELETE FROM workspaces WHERE CAST(id AS text) IN (:a, :b)"), {"a": cafe, "b": bakery})
    s.execute(text("DELETE FROM users WHERE id = :u"), {"u": owner})
    s.commit()


def test_precision_reads_started_and_passed_review_from_the_board(shops, capsys):
    code, out = _run(capsys, "precision", "--workspace", shops.cafe)
    assert code == 0 and "precision: PASS" in out and BAR_TEXT["precision"] in out
    assert "workspaces: 1" in out and "decided: 10" in out and "started: 5" in out and "started_share: 0.500" in out
    # Only the café's own tickets done under a person's review passed it.
    assert "passed_review: 3" in out and "passed_review_share: 0.300" in out
    assert f"workspace {shops.cafe}: PASS, decided 10, started 5 (0.500), started and passed review 3 (0.300)" in out


def test_a_source_agent_under_the_slow_share_after_20_decisions_is_listed(shops, capsys):
    code, out = _run(capsys, "precision", "--workspace", shops.bakery)
    assert code == 1 and "precision: MISS" in out and "started_share: 0.100" in out
    assert f"workspace {shops.bakery}: MISS" in out
    assert f"slow source agent {SLOW_AGENT}: 0.100 of its 20 decisions started" in out


def test_precision_reads_each_workspace_and_misses_when_one_does(shops, capsys):
    code, out = _run(capsys, "precision")
    assert code == 1 and "precision: MISS" in out
    assert f"workspace {shops.cafe}: PASS" in out and f"workspace {shops.bakery}: MISS" in out


class _RecordingSession:
    """Records each statement; answers no rows."""

    def __init__(self):
        self.statements, self.rolled_back, self.closed = [], False, False

    def execute(self, statement, params=None):
        self.statements.append(str(statement).strip())
        return NS(fetchall=lambda: [])

    def rollback(self):
        self.rolled_back = True

    def close(self):
        self.closed = True


def test_precision_reads_in_a_read_only_transaction_and_writes_nothing(monkeypatch, capsys):
    session = _RecordingSession()
    monkeypatch.setattr(precision, "open_session", lambda: session)
    code, out = _run(capsys, "precision")
    assert session.statements[0] == "SET TRANSACTION READ ONLY"
    assert len(session.statements) == 2 and session.statements[1].startswith("SELECT")
    assert session.rolled_back and session.closed
    # No decided suggestion is no evidence, never a pass.
    assert code == 1 and "workspaces: 0" in out and "decided: 0" in out


def test_workspace_is_precision_s_argument_only(capsys):
    with pytest.raises(SystemExit) as stopped:
        main(["spoken", "--workspace", WORKSPACE])
    assert stopped.value.code == 2
    code, out = _run(capsys, "precision", "--workspace", "the-cafe")
    assert code == 2 and "precision: CANNOT RUN" in out and "not a workspace id" in out


# ── one brief ────────────────────────────────────────────────────────────────

# Saturday 24 Oct 2026, 07:30 UTC: Lisbon's clocks go back on Sunday 25 Oct at 01:00 UTC.
SAT = datetime(2026, 10, 24, 7, 30, tzinfo=UTC)


def test_the_scenarios_open_devices_and_tabs_at_once_a_phone_away_and_across_a_dst_change():
    devices, away, dst = one_brief.scenarios(SAT)
    (wave,) = devices.waves
    assert [o.device for o in wave] == ["phone 1", "phone 2", "phone 3", "tab 1", "tab 2"]
    assert {(o.zone, o.at) for o in wave} == {("Europe/Lisbon", datetime(2026, 10, 24, 7, tzinfo=UTC))}
    assert len(away.waves) == 9 and all({o.zone for o in w} == {"Pacific/Auckland", "Europe/Lisbon"} for w in away.waves)
    # For part of the day the phone's date is ahead of the tab's; for part they agree.
    pairs = [tuple(o.local_date for o in w) for w in away.waves]
    assert any(a != b for a, b in pairs) and any(a == b for a, b in pairs)
    assert dst.name == "a DST change (Europe/Lisbon, 2026-10-25T01:00:00+00:00)"
    assert dst.waves[0][0].at == datetime(2026, 10, 24, 19, tzinfo=UTC) and len(dst.waves) == 25
    assert dst.local_dates() == {date(2026, 10, 24), date(2026, 10, 25)}


def test_the_next_dst_change_is_found_in_the_zone():
    after = datetime(2026, 10, 1, tzinfo=UTC)
    assert one_brief.next_offset_change("Europe/Lisbon", after) == datetime(2026, 10, 25, 1, tzinfo=UTC)
    assert one_brief.next_offset_change("America/New_York", after) == datetime(2026, 11, 1, 6, tzinfo=UTC)
    assert one_brief.next_offset_change("UTC", after) is None


def test_no_dst_change_to_open_across_exits_2_before_anything_is_written(monkeypatch, capsys):
    monkeypatch.setattr(one_brief, "DST_ZONES", ("UTC",))
    monkeypatch.setattr(one_brief, "open_session", lambda: pytest.fail("nothing may be written"))
    code, out = _run(capsys, "one_brief")
    assert code == 2 and "one_brief: CANNOT RUN" in out and "changes its clocks" in out


@pytest.fixture
def held_brief(engine, new_session, monkeypatch):
    """The brief's readiness check and schedule held empty; the eval runs on the test's sessions.
    Yields how many scratch people and workspaces exist, to compare after the run."""
    import services.brief.today as today
    import services.brief.unchecked as unchecked

    class NoSchedule:
        def __init__(self, db, workspace_id):
            pass

        def get_schedule(self, *, range_days=7):
            return {"scheduled": [], "scheduler_active": True}

    monkeypatch.setattr(today, "ActivityService", NoSchedule)
    monkeypatch.setattr(unchecked, "current", lambda workspace_id, language: {"items": []})
    monkeypatch.setattr(one_brief, "open_session", new_session)
    return lambda: _scratch_rows(new_session)


def _scratch_rows(new_session):
    with new_session() as s:
        like = {"p": f"{one_brief.SCRATCH_PREFIX}-%"}
        return (s.execute(text("SELECT count(*) FROM users WHERE clerk_user_id LIKE :p"), like).scalar(),
                s.execute(text("SELECT count(*) FROM workspaces WHERE name LIKE :p"), like).scalar())


def test_one_brief_passes_with_one_brief_per_person_and_local_date(held_brief, capsys, caplog):
    before = held_brief()
    with caplog.at_level("WARNING", logger=one_brief.__name__):
        code, out = _run(capsys, "one_brief")
    # It says which database it writes its scratch rows to before the first write.
    assert "writing a scratch workspace and 3 people to database" in caplog.text
    assert code == 0, out
    assert "one_brief: PASS" in out and BAR_TEXT["one_brief"] in out
    assert "duplicates: 0" in out and "missing: 0" in out and "unexpected: 0" in out
    assert "three devices and two tabs: 5 opens on 1 local dates, 1 briefs" in out
    assert "a phone in another timezone: 18 opens on " in out and "a DST change (" in out
    assert held_brief() == before  # the scratch workspace and its people are gone


def test_a_planted_duplicate_is_a_miss_with_exit_1(held_brief, monkeypatch, capsys):
    real, lock, planted = one_brief.post_brief, threading.Lock(), []

    def twice_once(db, person, brief, via):
        message = real(db, person, brief, via)
        with lock:
            plant = not planted
            planted.append(person)
        if plant:
            real(db, person, brief, via)
        return message

    monkeypatch.setattr(one_brief, "post_brief", twice_once)
    before = held_brief()
    code, out = _run(capsys, "one_brief")
    assert code == 1 and "one_brief: MISS" in out and "duplicates: 1" in out
    assert f"   {planted[0].local_date.isoformat()}: 2 briefs" in out
    assert held_brief() == before


def test_a_claim_that_always_wins_is_counted_as_duplicates(held_brief, monkeypatch, capsys):
    """The eval measures the claim: without it, every open posts its own brief."""
    monkeypatch.setattr(one_brief, "claim_brief", lambda db, workspace_id, user_id, local_date: True)
    code, out = _run(capsys, "one_brief")
    assert code == 1 and "one_brief: MISS" in out
    assert "three devices and two tabs: 5 opens on 1 local dates, 5 briefs" in out
