"""F368 (night 10c): the brand board's type samples fit, it says the currency and dates, and its name is its own.

Every board render cut the type specimens mid-word at the column's edge ("The quick |"),
left out the kit's currency and date style, and the gallery had a "Brand board" (the
social card) beside a "Brand Board" (the PDF). Now each sample is the whole words that
fit its column, under its own label line; the board prints "Money and dates"; and the
two starters are "Brand board (PDF)" and "Brand board (social)", the platform's rows
renamed in place, a person's own row left as it is.

The PDF test is real (WeasyPrint prints, pdfplumber reads back), as the board's own tests.
"""
from __future__ import annotations

import io
from types import SimpleNamespace
from typing import Any, Dict
from uuid import UUID

import pdfplumber

from modules.documents import seed_templates
from modules.documents.blocks import brand_board as bb
from modules.documents.blocks import render_document_html, validate_blocks
from modules.documents.blocks.design_tokens import design
from modules.documents.brand_kit import get_brand_kit
from modules.documents.presets import BRAND_BOARD
from modules.documents.social_starters import SOCIAL_BRAND_STARTER_SLUGS, social_starters
from modules.documents.template_summary import STARTER_CREATOR

WS = UUID("00000000-0000-0000-0000-0000000368a1")
WORDS = bb.TYPE_SAMPLE.split()
# A kit with the type pushed up: the samples must still be whole words.
BIG = {"type_scale": {"display": {"size_pt": 44, "line_pt": 50}, "h1": {"size_pt": 30, "line_pt": 36}}}


def _kit(**raw: Any) -> Dict[str, Any]:
    return get_brand_kit({"brand_kit": {"name": "Harbourline", **raw}})


def _board(kit: Dict[str, Any]) -> str:
    return render_document_html(validate_blocks(BRAND_BOARD["blocks"]), {}, kit, data={}).html


def test_every_sample_is_whole_words_that_fit_its_column():
    for kit in (_kit(), _kit(**BIG), _kit(page_margin_mm=30)):
        width = bb.type_column_pt(kit)
        for sample in bb.type_samples(kit):
            assert bb.TYPE_SAMPLE.startswith(sample.text), sample
            assert sample.text == bb.TYPE_SAMPLE or bb.TYPE_SAMPLE[len(sample.text)] == " ", sample  # never mid-word
            assert sample.text == WORDS[0] or len(sample.text) * sample.size_pt * bb.AVG_CHAR_EM <= width, sample
    body = next(s for s in bb.type_samples(_kit()) if s.step == "body")
    assert body.text == bb.TYPE_SAMPLE  # the body size takes the whole line


def test_each_label_is_its_own_line_above_its_sample():
    page = _board(_kit())
    display = next(s for s in bb.type_samples(_kit()) if s.step == "display")
    assert f'<p class="board-type-label">{display.label}</p><p class="board-type-sample' in page
    assert f'height:{display.line_pt:g}pt">{display.text}</p>' in page


def test_the_printed_samples_stay_inside_their_column():
    from weasyprint import HTML

    kit = _kit(**BIG)
    pdf = HTML(string=_board(kit)).write_pdf()
    right_edge = design(kit).page_margin_mm * bb.MM_TO_PT + bb.type_column_pt(kit)
    sample_sizes = {round(s.size_pt) for s in bb.type_samples(kit) if s.size_pt >= 12}
    with pdfplumber.open(io.BytesIO(pdf)) as document:
        words = document.pages[0].extract_words(extra_attrs=["size"])
    printed = [w for w in words if round(w["size"]) in sample_sizes and w["text"] in WORDS]
    assert printed, words
    assert max(w["x1"] for w in printed) <= right_edge + 1, [(w["text"], w["x1"]) for w in printed]


def test_the_board_says_the_currency_and_the_date_style():
    lines = [line.text for line in bb.locale_lines(_kit(currency="GBP", date_style="MMMM d, yyyy"))]
    assert lines == ["Currency: GBP (£): amounts print as £1234.50.", "Dates: month first, as in October 5, 2026."]
    (none, day_first) = bb.locale_lines(_kit())
    assert none.note and none.text == bb.NO_CURRENCY_NOTE
    assert day_first.text == "Dates: day first, as in 5 October 2026."

    page = _board(_kit(currency="EUR"))
    assert bb.LOCALE_LABEL in page and "Currency: EUR (€): amounts print as €1234.50." in page


def test_the_two_board_starters_have_names_of_their_own():
    (social,) = [s for s in social_starters() if s["slug"] in SOCIAL_BRAND_STARTER_SLUGS]
    assert (BRAND_BOARD["name"], social["name"]) == ("Brand board (PDF)", "Brand board (social)")
    assert BRAND_BOARD["name"].casefold() != social["name"].casefold()


class _Session:
    """Keeps the template rows; answers the seeder's lookup by workspace and name."""

    def __init__(self, rows):
        self.rows, self._criteria = list(rows), {}

    def query(self, _model):
        self._criteria = {}
        return self

    def filter(self, *criteria):
        for criterion in criteria:
            self._criteria[criterion.left.key] = criterion.right.value
        return self

    def first(self):
        return next((r for r in self.rows if all(getattr(r, k, None) == v for k, v in self._criteria.items())), None)

    def add(self, row):
        self.rows.append(row)

    def commit(self):
        pass


def _old_board(created_by: str = STARTER_CREATOR, is_active: bool = True) -> SimpleNamespace:
    return SimpleNamespace(workspace_id=WS, name="Brand Board", created_by=created_by, is_active=is_active,
                           blocks={"version": 1, "blocks": []}, description="old", format="pdf", category="brand",
                           sample_data={})


def _named(db: _Session, name: str):
    return [row for row in db.rows if row.name == name]


def test_the_platforms_old_board_row_is_renamed_in_place():
    old = _old_board()
    db = _Session([old])
    seed_templates.seed_starter_templates(db, WS)

    assert old.name == "Brand board (PDF)" and old.blocks == BRAND_BOARD["blocks"]
    assert _named(db, "Brand board (PDF)") == [old] and _named(db, "Brand Board") == []


def test_a_persons_own_brand_board_keeps_its_name_and_the_starter_is_seeded_beside_it():
    mine = _old_board(created_by="user_7")
    db = _Session([mine])
    seed_templates.seed_starter_templates(db, WS)

    assert mine.name == "Brand Board" and mine.blocks == {"version": 1, "blocks": []}
    assert len(_named(db, "Brand board (PDF)")) == 1


def test_a_board_the_person_deleted_stays_deleted_under_the_new_name_too():
    gone = _old_board(is_active=False)
    db = _Session([gone])
    seed_templates.seed_starter_templates(db, WS)

    assert gone.name == "Brand Board" and _named(db, "Brand board (PDF)") == []


def test_the_social_cards_old_row_is_renamed_too():
    old = SimpleNamespace(workspace_id=WS, name="Brand board", created_by=STARTER_CREATOR, is_active=True, blocks={},
                          description="", format="social_image", category="social", sample_data={})
    db = _Session([old])
    seed_templates.seed_social_starters(db, WS, commit=False)

    assert old.name == "Brand board (social)" and _named(db, "Brand board") == []
