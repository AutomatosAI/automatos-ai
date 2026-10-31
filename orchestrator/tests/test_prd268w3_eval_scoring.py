"""PRD-268 US-C13a (§6.7, K14): the proactive evals' bars, scorers and gold-set reader.

Inline synthetic cases only, never a gold case: each scorer is checked at its bar, just under it
and just over it. The bars are pinned to the PRD's numbers so a run can never be passed by
tuning a bar. The gold reader refuses an empty setting, a folder that is not there and one
inside the repository, and reads one JSON object per line.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from config import config
from evals.proactive import bars, gold
from evals.proactive.scoring import (
    ACT_TODAY,
    CAN_WAIT,
    MISS,
    PASS,
    ChipsTurn,
    Decision,
    SilenceRun,
    Suggested,
    TriageVerdict,
    score_one_brief,
    score_precision,
    score_reply_chips,
    score_silence,
    score_spoken,
    score_triage,
)

REASON = "Vector's check-ins failed 401 three times today."
CHIPS = ("Send now", "Tomorrow")
QUESTION = "Shall I send it now, or first thing tomorrow?"


# ── the bars ─────────────────────────────────────────────────────────────────


def test_the_bars_are_the_prds():
    assert (bars.TRIAGE_MIN_RECALL, bars.TRIAGE_MIN_PRECISION, bars.TRIAGE_REASON_MAX_CHARS) == (0.95, 0.90, 120)
    assert (bars.SILENCE_MIN_QUIET_SILENT, bars.SILENCE_MIN_PLANTED_NAMED, bars.SILENCE_MAX_DUPLICATES) == (
        0.95, 0.85, 0)
    assert (bars.CHIPS_MIN_ON_CHOICE, bars.CHIPS_MAX_OUT_OF_LIMITS, bars.CHIPS_MAX_REPLACED,
            bars.CHIPS_MAX_ON_OPEN) == (16 / 20, 0, 0, 1 / 10)
    assert (bars.SPOKEN_MAX_MEDIAN_WORDS, bars.SPOKEN_MAX_WORDS) == (40, 80)
    assert bars.BRIEFS_PER_PERSON_PER_DATE == 1
    assert (bars.PRECISION_MIN_STARTED, bars.PRECISION_MIN_PASSED_REVIEW, bars.PRECISION_SLOW_AGENT_SHARE,
            bars.PRECISION_SLOW_AGENT_DECISIONS) == (0.40, 0.30, 0.20, 20)


def test_every_score_names_its_bar_in_words():
    scores = [score_triage([]), score_silence([]), score_reply_chips([]), score_spoken([]),
              score_one_brief([]), score_precision([])]
    assert {score.name for score in scores} == set(bars.BAR_TEXT)
    for score in scores:
        as_dict = score.as_dict()
        assert as_dict["bar"] == bars.BAR_TEXT[score.name] and as_dict["eval"] == score.name
        assert as_dict["verdict"] == score.verdict
    assert "0.95" in bars.BAR_TEXT["triage"] and "120" in bars.BAR_TEXT["triage"]
    assert "40" in bars.BAR_TEXT["spoken"] and "80" in bars.BAR_TEXT["spoken"]


def test_no_evidence_is_never_a_pass():
    for score in (score_triage([]), score_silence([]), score_reply_chips([]), score_spoken([]),
                  score_one_brief([]), score_precision([])):
        assert score.verdict == MISS, score.name


# ── triage ───────────────────────────────────────────────────────────────────


def _triage(hits: int, missed: int, false_alarms: int, quiet: int = 5, reason: str = REASON):
    return ([TriageVerdict(ACT_TODAY, ACT_TODAY, reason)] * hits
            + [TriageVerdict(ACT_TODAY, CAN_WAIT, reason)] * missed
            + [TriageVerdict(CAN_WAIT, ACT_TODAY, reason)] * false_alarms
            + [TriageVerdict(CAN_WAIT, CAN_WAIT, reason)] * quiet)


@pytest.mark.parametrize("hits, missed, passes", [(19, 1, True), (18, 1, False), (39, 1, True)])
def test_triage_recall_at_under_and_over_its_bar(hits, missed, passes):
    score = score_triage(_triage(hits, missed, 0))
    assert score.numbers["recall"] == hits / (hits + missed)
    assert score.passed is passes


@pytest.mark.parametrize("hits, false_alarms, passes", [(9, 1, True), (8, 1, False), (19, 1, True)])
def test_triage_precision_at_under_and_over_its_bar(hits, false_alarms, passes):
    score = score_triage(_triage(hits, 0, false_alarms))
    assert score.numbers["precision"] == hits / (hits + false_alarms)
    assert score.passed is passes


@pytest.mark.parametrize("length, passes", [(120, True), (121, False), (119, True), (0, False)])
def test_every_triage_verdict_carries_a_reason_of_at_most_120_characters(length, passes):
    verdicts = _triage(20, 0, 0) + [TriageVerdict(CAN_WAIT, CAN_WAIT, "r" * length)]
    score = score_triage(verdicts)
    assert score.passed is passes
    assert score.numbers["bad_reasons"] == (0 if passes else 1)


def test_a_case_with_no_verdict_misses_and_counts_as_none():
    score = score_triage(_triage(20, 0, 0) + [TriageVerdict(CAN_WAIT, None, REASON)])
    assert score.numbers["no_verdict"] == 1 and score.verdict == MISS


def test_a_set_with_no_act_today_case_cannot_pass():
    score = score_triage(_triage(0, 0, 0, quiet=10))
    assert score.numbers["recall"] is None and score.verdict == MISS


def test_a_case_labelled_otherwise_is_refused():
    with pytest.raises(ValueError):
        TriageVerdict("urgent", ACT_TODAY, REASON)


# ── check-in silence ─────────────────────────────────────────────────────────

NEED = ("vector", "401")
NAMED = Suggested("Vector's check-ins fail", "Three 401s in a row since 09:00: reconnect its key.")


def _silence(quiet_silent: int, quiet_said: int, named: int, unnamed: int):
    return ([SilenceRun((), ())] * quiet_silent
            + [SilenceRun((), (Suggested("Review the Q3 report"),))] * quiet_said
            + [SilenceRun(NEED, (NAMED,))] * named
            + [SilenceRun(NEED, (Suggested("Tidy the board"),))] * unnamed)


@pytest.mark.parametrize("silent, said, passes", [(19, 1, True), (37, 2, False), (39, 2, True)])
def test_quiet_snapshots_end_silent_at_under_and_over_the_bar(silent, said, passes):
    score = score_silence(_silence(silent, said, 1, 0))
    assert score.numbers["quiet_silent"] == silent / (silent + said)
    assert score.passed is passes


@pytest.mark.parametrize("named, unnamed, passes", [(17, 3, True), (16, 3, False), (18, 3, True)])
def test_planted_needs_are_named_at_under_and_over_the_bar(named, unnamed, passes):
    score = score_silence(_silence(1, 0, named, unnamed))
    assert score.numbers["planted_named"] == named / (named + unnamed)
    assert score.passed is passes


def test_a_need_is_named_only_when_every_word_is_there_title_or_detail():
    half = SilenceRun(NEED, (Suggested("Vector needs a look"),))
    in_detail = SilenceRun(NEED, (Suggested("Vector needs a look", "It hit 401 three times."),))
    inside_a_number = SilenceRun(NEED, (Suggested("Vector needs a look", "Run 14012 finished."),))
    assert score_silence([half]).numbers["planted_named"] == 0
    assert score_silence([in_detail]).numbers["planted_named"] == 1
    assert score_silence([inside_a_number]).numbers["planted_named"] == 0


def test_a_suggestion_repeating_an_open_or_hidden_title_is_a_duplicate():
    clean = _silence(1, 0, 1, 0)
    repeat = SilenceRun(NEED, (Suggested("VECTOR's  check-ins FAIL!", "401 again."),),
                        taken_titles=("vector's check-ins fail",))
    assert score_silence(clean).numbers["duplicates"] == 0 and score_silence(clean).passed
    score = score_silence(clean + [repeat])
    assert score.numbers["duplicates"] == 1 and score.verdict == MISS


def test_a_planted_need_that_says_nothing_is_refused():
    with pytest.raises(ValueError):
        SilenceRun(("  ",), ())


# ── reply chips ──────────────────────────────────────────────────────────────


def _chips(on_choice: int, off_choice: int, on_open: int, off_open: int):
    return ([ChipsTurn(True, QUESTION, CHIPS)] * on_choice
            + [ChipsTurn(True, QUESTION)] * off_choice
            + [ChipsTurn(False, "Here's how the brief is built.", CHIPS)] * on_open
            + [ChipsTurn(False, "Here's how the brief is built.")] * off_open)


@pytest.mark.parametrize("on, off, passes", [(16, 4, True), (15, 4, False), (17, 4, True)])
def test_chips_on_choice_turns_at_under_and_over_the_bar(on, off, passes):
    score = score_reply_chips(_chips(on, off, 0, 10))
    assert score.numbers["on_choice"] == on / (on + off)
    assert score.passed is passes


@pytest.mark.parametrize("on, off, passes", [(1, 9, True), (2, 17, False), (1, 10, True)])
def test_chips_on_open_questions_at_under_and_over_the_bar(on, off, passes):
    score = score_reply_chips(_chips(20, 0, on, off))
    assert score.numbers["on_open"] == on / (on + off)
    assert score.passed is passes


@pytest.mark.parametrize("replies", [("Yes",), ("Yes", "No", "Maybe", "Later"), ("Yes", "**No**"),
                                     ("Yes", "Send it to everyone now"), ("Yes", "yes")])
def test_chips_outside_the_tools_limits_miss(replies):
    score = score_reply_chips(_chips(20, 0, 0, 10) + [ChipsTurn(True, QUESTION, replies)])
    assert score.numbers["out_of_limits"] == 1 and score.verdict == MISS


def test_chips_in_place_of_the_answer_miss():
    score = score_reply_chips(_chips(20, 0, 0, 10) + [ChipsTurn(True, "   ", CHIPS)])
    assert score.numbers["replaced"] == 1 and score.verdict == MISS


# ── spoken brevity ───────────────────────────────────────────────────────────


def _say(words: int) -> str:
    return " ".join(["word"] * words)


@pytest.mark.parametrize("words, passes", [(40, True), (41, False), (39, True)])
def test_spoken_median_at_under_and_over_the_bar(words, passes):
    score = score_spoken([_say(words)] * 3)
    assert score.numbers["median_words"] == words
    assert score.passed is passes


@pytest.mark.parametrize("words, passes", [(80, True), (81, False), (79, True)])
def test_no_spoken_answer_over_80_words(words, passes):
    score = score_spoken([_say(10), _say(10), _say(words)])
    assert score.numbers["max_words"] == words
    assert score.passed is passes


@pytest.mark.parametrize("answer, found", [
    ("Here is **one** thing to do.", "markdown"),
    ("- one item\n- another", "markdown"),
    ("Open https://example.com to see it.", "urls"),
    ("It's on automatos.app now.", "urls"),
    ("I saved it in /var/data/report.md for you.", "paths"),
    ("It's in ~/notes/today.", "paths"),
    ("I wrote it to docs/prd/notes.md earlier.", "paths"),
    ("Ticket #0846 is done.", "ids"),
    ("The run 3f2b8c4e-1a2b-4c3d-8e9f-0a1b2c3d4e5f stopped.", "ids"),
    ("Commit 3f9a2c1d7e landed.", "ids"),
])
def test_a_spoken_answer_carrying_what_a_listener_must_not_hear_misses(answer, found):
    score = score_spoken([answer])
    assert score.numbers[found] == 1 and score.verdict == MISS


def test_plain_spoken_answers_pass():
    answers = ["I've put it in the chat. Want me to run through it?",
               "Vector's been failing since nine, and it's the same key error. I can reconnect it.",
               "It runs 24/7, and you can stop it and/or pause it whenever you like.",
               "The facade work is done, about twenty quid all in."]
    score = score_spoken(answers)
    assert score.verdict == PASS, score.numbers


# ── one brief ────────────────────────────────────────────────────────────────

PEOPLE_DAYS = [("ana", "2026-10-25"), ("ana", "2026-10-26"), ("bo", "2026-10-26")]


@pytest.mark.parametrize("copies, passes", [(1, True), (0, False), (2, False)])
def test_one_brief_per_person_per_local_date_at_under_and_over(copies, passes):
    briefs = PEOPLE_DAYS[1:] + [PEOPLE_DAYS[0]] * copies
    score = score_one_brief(briefs, expected=PEOPLE_DAYS)
    assert score.passed is passes
    assert score.numbers["duplicates"] == max(0, copies - 1)
    assert score.numbers["missing"] == (1 if copies == 0 else 0)


def test_a_brief_on_a_date_nobody_expected_misses():
    score = score_one_brief(PEOPLE_DAYS + [("bo", "2026-10-27")], expected=PEOPLE_DAYS)
    assert score.numbers["unexpected"] == 1 and score.verdict == MISS


def test_without_an_expectation_only_duplicates_count():
    assert score_one_brief(PEOPLE_DAYS).verdict == PASS
    score = score_one_brief(PEOPLE_DAYS + [PEOPLE_DAYS[2]])
    assert score.numbers["duplicates"] == 1 and score.numbers["missing"] is None and score.verdict == MISS


# ── suggestion precision (live) ──────────────────────────────────────────────


def _decisions(passed: int, started_only: int, hidden: int, agent: str = "a1"):
    return ([Decision(agent, True, True)] * passed + [Decision(agent, True, False)] * started_only
            + [Decision(agent, False, True)] * hidden)


@pytest.mark.parametrize("started_only, hidden, passes", [(1, 6, True), (0, 7, False), (2, 5, True)])
def test_started_share_at_under_and_over_the_bar(started_only, hidden, passes):
    score = score_precision(_decisions(3, started_only, hidden))
    assert score.numbers["started_share"] == (3 + started_only) / 10
    assert score.passed is passes


@pytest.mark.parametrize("passed, started_only, passes", [(3, 2, True), (2, 3, False), (4, 1, True)])
def test_started_and_passed_review_share_at_under_and_over_the_bar(passed, started_only, passes):
    score = score_precision(_decisions(passed, started_only, 5))
    assert score.numbers["passed_review_share"] == passed / 10
    assert score.passed is passes


def test_a_hidden_suggestion_never_counts_as_passing_review():
    score = score_precision(_decisions(0, 0, 10))
    assert score.numbers["passed_review"] == 0


def test_a_source_agent_under_the_share_after_20_decisions_is_listed():
    decisions = (_decisions(0, 3, 17, "slow") + _decisions(0, 4, 16, "at_bar")
                 + _decisions(0, 0, 19, "too_few") + _decisions(10, 0, 0, "good"))
    score = score_precision(decisions)
    assert score.numbers["slow_agents"] == (("slow", 20, 0.15),)


# ── the gold reader ──────────────────────────────────────────────────────────


@pytest.fixture
def gold_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PROACTIVE_EVAL_GOLD_DIR", str(tmp_path))
    return tmp_path


@pytest.mark.parametrize("value", ["", "   "])
def test_an_empty_setting_is_refused(monkeypatch, value):
    monkeypatch.setattr(config, "PROACTIVE_EVAL_GOLD_DIR", value)
    with pytest.raises(gold.GoldSetUnavailable, match="PROACTIVE_EVAL_GOLD_DIR is not set"):
        gold.read_gold("triage")


def test_a_folder_that_is_not_there_is_refused(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "PROACTIVE_EVAL_GOLD_DIR", str(tmp_path / "nowhere"))
    with pytest.raises(gold.GoldSetUnavailable, match="not a folder"):
        gold.gold_dir()


def test_a_file_is_not_a_folder(monkeypatch, tmp_path):
    (tmp_path / "triage.jsonl").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(config, "PROACTIVE_EVAL_GOLD_DIR", str(tmp_path / "triage.jsonl"))
    with pytest.raises(gold.GoldSetUnavailable, match="not a folder"):
        gold.gold_dir()


@pytest.mark.parametrize("inside", [Path(__file__).resolve().parent, gold.repository_root()])
def test_a_folder_inside_the_repository_is_refused(monkeypatch, inside):
    monkeypatch.setattr(config, "PROACTIVE_EVAL_GOLD_DIR", str(inside))
    with pytest.raises(gold.GoldSetUnavailable, match="inside this repository"):
        gold.gold_dir()


def test_a_link_from_outside_into_the_repository_is_refused(monkeypatch, tmp_path):
    link = tmp_path / "gold"
    link.symlink_to(Path(__file__).resolve().parent, target_is_directory=True)
    monkeypatch.setattr(config, "PROACTIVE_EVAL_GOLD_DIR", str(link))
    with pytest.raises(gold.GoldSetUnavailable, match="inside this repository"):
        gold.gold_dir()


def test_the_repository_holds_the_evals():
    evals = Path(gold.__file__).resolve().parent
    assert gold.repository_root() in evals.parents


def test_a_gold_set_is_one_json_object_per_line(gold_folder):
    cases = [{"expected": ACT_TODAY, "title": "401 streak"}, {"expected": CAN_WAIT, "title": "weekly report"}]
    text = json.dumps(cases[0]) + "\n\n" + json.dumps(cases[1]) + "\n"
    (gold_folder / "triage.jsonl").write_text(text, encoding="utf-8")
    assert gold.read_gold("triage") == cases


def test_a_missing_gold_set_is_refused(gold_folder):
    with pytest.raises(gold.GoldSetUnavailable, match="silence.jsonl is missing"):
        gold.read_gold("silence")


@pytest.mark.parametrize("text, message", [
    ('{"a": 1}\n{not json\n', "line 2 is not JSON"),
    ('["a list"]\n', "line 1 is not a JSON object"),
    ("\n  \n", "has no cases"),
])
def test_a_malformed_or_empty_gold_set_is_refused(gold_folder, text, message):
    (gold_folder / "spoken.jsonl").write_text(text, encoding="utf-8")
    with pytest.raises(gold.GoldSetUnavailable, match=message):
        gold.read_gold("spoken")


def test_an_eval_with_no_gold_set_is_not_read(gold_folder):
    with pytest.raises(ValueError):
        gold.read_gold("precision")
    with pytest.raises(ValueError):
        gold.read_gold("../triage")
