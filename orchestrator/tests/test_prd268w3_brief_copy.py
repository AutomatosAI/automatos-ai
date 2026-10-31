"""PRD-268 W3 US-C23 (§6.11, W2's K16) — ``Brief@1`` carries its copy, in the person's language.

``copy`` is every word a client needs to draw the brief, whatever its rows: each section's
label, the empty sentence, and "Show {n} more" unformatted, from ``services.brief.strings`` in
English and Portuguese (Brazil), chosen by the person's locale as the rest of the brief is
(``brief_language``, W2's K7). So a phone that folds the rows its own way, or a section with no
rows, never makes a client write its own words (Go #31 drew English labels until this landed).
Proven here, with the section readers faked (no database):

* both languages carry every key the copy reads, and Portuguese is its own;
* ``build_brief`` always carries ``copy`` (the empty sentence too, when there are rows), and the
  labels it folds with read the same words; a locale picks the copy exactly as it picks the rest;
* ``copy`` is read and written by that name, a brief saved before it still reads, and the
  largest brief with its copy still fits its message part;
* the default card (``frontend/components/proactive``) reads the section labels and the empty
  sentence from ``copy``, keeps the server's own "Show N more" line, and writes no section word
  of its own (the card's render is proven in ``brief-copy.test.tsx``).
"""
from __future__ import annotations

import json
import re
import uuid
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from services.brief import build
from services.brief import strings as brief_strings
from services.brief.model import (
    NEEDS_YOU,
    PART_TYPE,
    SECTIONS,
    SINCE,
    SUGGESTIONS,
    TODAY,
    UNCHECKED,
    Brief,
    BriefCounts,
    BriefLabels,
    BriefOpens,
    BriefRow,
    BriefSections,
)
from services.brief.needs import Needs
from services.brief.rows import FOLD_AT, WHY_CHARS, action, cut, title_of

ORCH = Path(__file__).resolve().parents[1]
PROACTIVE = ORCH.parent / "frontend" / "components" / "proactive"
WIRE = ORCH.parent / "frontend" / "lib" / "proactive" / "wire.ts"
# The default card's files: the card, its sections and its rows.
CARD_FILES = ("brief-card.tsx", "brief-sections.tsx", "brief-row.tsx")
LANGUAGES = ("en", "pt-BR")
COPY_KEYS = {"section_labels", "empty_sentence", "show_more_template"}
WORKSPACE = uuid.uuid4()
NOW = datetime(2026, 10, 30, 8, 0, tzinfo=timezone.utc)
DIGEST = {"text": "Two things moved overnight.", "state_hash": "state-hash", "generated_at": NOW.isoformat()}


class _Frozen(datetime):
    """``build_brief``'s clock, held at ``NOW``."""

    @classmethod
    def now(cls, tz=None):
        return NOW if tz is None else NOW.astimezone(tz)


def _row(name: str, n: int) -> BriefRow:
    return BriefRow(id=f"{name}:{n}", title=f"{name} {n}", why=f"Why {name} {n}.")


def _rows(name: str, many: int) -> list:
    return [_row(name, n) for n in range(many)]


@pytest.fixture
def faked(monkeypatch):
    """The section readers faked: ``rows`` maps a section to how many rows it reads. Needs you's
    fold counts ``seen`` (its rows and ``unseen_more``, rows past those read) and its badge counts
    ``total`` (``seen`` and ``hidden_from_them``, the rows this person may not see, as a viewer)."""
    def given(rows, *, unseen_more=0, hidden_from_them=0):
        many = {name: rows.get(name, 0) for name in SECTIONS}
        seen = many[NEEDS_YOU] + unseen_more
        needs = Needs(seen + hidden_from_them, seen, _rows(NEEDS_YOU, many[NEEDS_YOU]))
        rest = {name: (many[name], _rows(name, many[name])) for name in (SUGGESTIONS, TODAY, SINCE, UNCHECKED)}
        monkeypatch.setattr(build, "needs_rows", lambda *args, **kwargs: needs)
        monkeypatch.setattr(build, "_sections", lambda *args, **kwargs: rest)
        monkeypatch.setattr(build, "_first_name", lambda *args, **kwargs: "Sam")
        monkeypatch.setattr(build, "datetime", _Frozen)
    return given


def _brief(language):
    return build.build_brief(object(), WORKSPACE, 7, date(2026, 10, 30), language, may_answer=True, digest=DIGEST)


# A TypeScript string literal ('…', "…" or `…`), or the text between two JSX tags.
_LITERAL = re.compile(r"'(?:[^'\\\n]|\\.)*'|\"(?:[^\"\\\n]|\\.)*\"|`(?:[^`\\]|\\.)*`|>[^<>{}]*<")


def _code_of(source: str) -> str:
    """A TypeScript file without its comments, so a word in a docstring is not read as drawn."""
    return re.sub(r"//[^\n]*", "", re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL))


def _server_words() -> list:
    """The copy's words in both languages as patterns: each section label, the empty sentence,
    and "Show {n} more" with any number (or N) in place of ``{n}``."""
    words = [brief_strings.say(language, key) for language in LANGUAGES
             for key in [f"section_{name}" for name in SECTIONS] + ["empty"]]
    templates = [re.escape(brief_strings.say(language, "more")).replace(re.escape("{n}"), r"\S+")
                 for language in LANGUAGES]
    return [re.compile(re.escape(word)) for word in words] + [re.compile(template) for template in templates]


# --------------------------------------------------------------------------- the strings


@pytest.mark.parametrize("language", LANGUAGES)
def test_both_languages_carry_every_key_the_copy_reads(language):
    table = brief_strings.STRINGS[language]
    words = build.brief_copy(language)

    assert {f"section_{name}" for name in SECTIONS} | {"empty", "more"} <= set(table)
    assert set(words.section_labels) == set(SECTIONS)
    assert all(words.section_labels[name] == table[f"section_{name}"] and words.section_labels[name].strip()
               for name in SECTIONS)
    assert words.empty_sentence == table["empty"] and words.empty_sentence.strip()
    assert words.show_more_template == table["more"] and words.show_more_template.count("{n}") == 1
    assert words.show_more_template.format(n=3) == brief_strings.more(language, 3)


def test_portuguese_copy_is_its_own():
    en, pt = (build.brief_copy(language).model_dump() for language in LANGUAGES)

    assert all(pt["section_labels"][name] != en["section_labels"][name] for name in SECTIONS)
    assert pt["empty_sentence"] != en["empty_sentence"] and pt["show_more_template"] != en["show_more_template"]
    assert pt["show_more_template"] == "Mostrar mais {n}" and en["show_more_template"] == "Show {n} more"


# --------------------------------------------------------------------------- the brief carries it


@pytest.mark.parametrize("tag, language", [("pt-BR", "pt-BR"), ("pt-PT", "pt-BR"), ("pt_br", "pt-BR"),
                                           ("en-GB", "en"), ("fr-FR", "en"), (None, "en")])
def test_the_brief_carries_its_copy_in_the_persons_language_whatever_its_rows(faked, tag, language):
    faked({NEEDS_YOU: FOLD_AT + 2, SUGGESTIONS: 1}, unseen_more=1, hidden_from_them=2)
    brief = _brief(tag)

    assert brief["copy"] == build.brief_copy(language).model_dump(mode="json")
    assert set(brief["copy"]) == COPY_KEYS and "copy_" not in brief
    assert brief["labels"]["sections"] == brief["copy"]["section_labels"]
    assert "empty" not in brief["labels"] and brief["copy"]["empty_sentence"]       # rows, and still the sentence
    shown = len(brief["sections"][NEEDS_YOU])
    folded = (FOLD_AT + 2 + 1) - shown                                          # the fold counts what they may see
    assert shown == FOLD_AT and brief["counts"][NEEDS_YOU] == FOLD_AT + 2 + 1 + 2   # the badge counts the lot
    assert brief["labels"]["more"] == {NEEDS_YOU: brief["copy"]["show_more_template"].format(n=folded)}
    assert brief["counts"][NEEDS_YOU] - shown != folded           # so a client never fills the template from counts
    assert brief["greeting"] == brief_strings.greeting(language, "Sam", NOW.hour)     # one language for the lot


def test_an_empty_brief_says_the_copys_sentence_and_its_labels_say_the_same(faked):
    faked({})
    brief = _brief("pt-BR")

    sentence = brief_strings.say("pt-BR", "empty")
    assert brief["copy"]["empty_sentence"] == brief["labels"]["empty"] == sentence
    assert all(brief["sections"][name] == [] for name in SECTIONS) and brief["labels"]["more"] == {}
    assert brief["copy"]["section_labels"] == {name: brief_strings.say("pt-BR", f"section_{name}") for name in SECTIONS}


# --------------------------------------------------------------------------- reading and size


def test_copy_is_read_and_written_by_its_name_and_a_brief_saved_before_it_still_reads(faked):
    faked({TODAY: 2})
    brief = _brief("en")

    assert Brief.model_validate({**brief, "a_later_field": 1}).model_dump(mode="json", exclude_none=True) == brief
    with_extra = {**brief, "copy": {**brief["copy"], "a_later_word": "x"}}
    assert Brief.model_validate(with_extra).model_dump(mode="json", exclude_none=True) == brief
    older = {key: value for key, value in brief.items() if key != "copy"}
    read = Brief.model_validate(older)
    assert read.copy_ is None and read.model_dump(mode="json", exclude_none=True) == older


def test_the_largest_brief_with_its_copy_fits_its_message_part():
    """W2's largest brief (every section at its most, 3-byte titles and reasons cut at their
    limits, 3 KB of Auto's Read) with the Portuguese copy beside it still passes ``checked_parts``."""
    from services.chat_parts import CHAT_PART_MAX_BYTES, checked_parts

    def row(section):
        return BriefRow(id=f"{section}:{uuid.uuid4()}", title=title_of("語" * 500, "pt-BR"),
                        why=cut("語" * 900, WHY_CHARS), at=NOW.isoformat(), number="#0001.12",
                        display_state="needs_you", opens=BriefOpens(type="ticket", id=str(uuid.uuid4())),
                        action=action("start", "pt-BR", done=True))

    most = {NEEDS_YOU: FOLD_AT + 1, SUGGESTIONS: 3, TODAY: FOLD_AT + 1, SINCE: FOLD_AT + 1, UNCHECKED: 2}
    sections = BriefSections(**{name: [row(name) for _ in range(rows)] for name, rows in most.items()})
    greeted = brief_strings.greeting("pt-BR", cut("語" * 100, 40), 7)
    brief = Brief(v=1, date="2026-10-30", greeting=greeted, prose="語" * 1000,
                  spoken=build.spoken_words(greeted, sections, None), sections=sections,
                  counts=BriefCounts(needs_you=999, suggestions=999, today=999, since=999),
                  labels=BriefLabels(sections={name: name for name in SECTIONS},
                                     more={name: brief_strings.more("pt-BR", 999) for name in SECTIONS}),
                  copy_=build.brief_copy("pt-BR"), state_hash="a" * 64,
                  generated_at=NOW.isoformat()).model_dump(mode="json", exclude_none=True)

    (saved,) = checked_parts([{"type": PART_TYPE, "brief": brief}])
    assert saved["brief"] == brief and set(saved["brief"]["copy"]) == COPY_KEYS
    assert len(json.dumps(saved, separators=(",", ":"), ensure_ascii=False).encode("utf-8")) < CHAT_PART_MAX_BYTES


# --------------------------------------------------------------------------- the default card


def test_the_default_card_reads_the_servers_labels_from_copy_and_writes_no_section_word():
    sections = (PROACTIVE / "brief-sections.tsx").read_text(encoding="utf-8")
    code = _code_of(sections)

    assert "copy?.section_labels" in code and "copy?.empty_sentence" in code
    assert "labels?.more" in code                                 # the server's own fold line, its exact number
    assert "show_more_template" not in code                      # the card never fills the template from counts
    assert re.search(r"copyOf\(\s*fields\.copy\s*\)", WIRE.read_text(encoding="utf-8"))
    patterns = _server_words()
    assert len(patterns) == 2 * (len(SECTIONS) + 2)
    for name in CARD_FILES:
        literals = _LITERAL.findall(_code_of((PROACTIVE / name).read_text(encoding="utf-8")))
        found = sorted({literal for literal in literals for pattern in patterns if pattern.search(literal)})
        assert literals and not found, f"{name} writes the server's words itself: {found}"
