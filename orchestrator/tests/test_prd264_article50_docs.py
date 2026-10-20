"""PRD-264 EUAI-14 and EUAI-12: the Article 50 compliance docs exist and say only what is true.

Pure file reads, no services. They lock four things the stories ask for:

* ``docs/compliance/ARTICLE-50-GAP-ANALYSIS.md`` (EUAI-14) covers each Code of Practice
  commitment (signed metadata, watermark, detection, interoperability, the 2 Feb 2027
  date), names the watermark gap and decision D11, and dates the revisit after Web
  Summit (10-12 Nov 2026);
* ``docs/compliance/ARTICLE-50.md`` (EUAI-12, the automatos-ai guide) lists, surface by
  surface, what Automatos provides and what the deployer does;
* both are in ``docs/compliance/README.md``'s table, and every relative link in the
  compliance docs resolves;
* neither claims compliance or certification, and both say a part is live only once it
  is merged and deployed (PRD-264 §7, EUAI-15).
"""
from __future__ import annotations

import datetime
import pathlib
import re
from typing import List

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_COMPLIANCE = _REPO_ROOT / "docs" / "compliance"
_GAP_ANALYSIS = _COMPLIANCE / "ARTICLE-50-GAP-ANALYSIS.md"
_GUIDE = _COMPLIANCE / "ARTICLE-50.md"
_README = _COMPLIANCE / "README.md"
_DETECTION = _COMPLIANCE / "DETECTION.md"
_ARTICLE_50_DOCS = (_GAP_ANALYSIS, _GUIDE)

# The Code commitments EUAI-14 must answer one by one (PRD-264 §1, the 50(7) row).
_CODE_COMMITMENT_HEADINGS = (
    "### Signed metadata",
    "### Watermark",
    "### Detection",
    "### Interoperability",
    "### The 2 Feb 2027 date",
)
# The last day of Web Summit Lisbon; the revisit of the signing decision comes after it.
_WEB_SUMMIT_LAST_DAY = datetime.date(2026, 11, 12)
_REVISIT_HEADING = "## Revisit"
# "revisited on Mon 16 Nov 2026": the date the signing decision is taken again.
_REVISIT_DATE = re.compile(r"revisited on (?:[A-Z][a-z]{2} )?(\d{1,2}) (Nov|Dec) (2026)\b")
_MONTHS = {"Nov": 11, "Dec": 12}

# Every story whose surface or mark the guide must place (PRD-264 EUAI-12).
_GUIDE_STORIES = (
    "EUAI-02a", "EUAI-02b", "EUAI-03", "EUAI-04", "EUAI-05", "EUAI-06a", "EUAI-06b",
    "EUAI-07", "EUAI-08a", "EUAI-08b", "EUAI-08c", "EUAI-09", "EUAI-10", "EUAI-11a",
)
# What the deployer does, in the guide's words (EUAI-12's first criterion).
_GUIDE_DEPLOYER_DUTIES = ("Keep the notices", "Don't strip the marks", "50(3)", "50(4) ¶1", "50(4) ¶2")

# Public copy never claims compliance or certification (PRD-264 §7).
_OVERCLAIM = re.compile(r"\b(compliant|certified|conformant|aligned by design)\b", re.IGNORECASE)
_LIVE_ONLY_WHEN = "merged and deployed"
_RELATIVE_LINK = re.compile(r"\]\((\./[^)#\s]+)(?:#[^)]*)?\)")


def _text(doc: pathlib.Path) -> str:
    return doc.read_text(encoding="utf-8")


def _plain(doc: pathlib.Path) -> str:
    """The doc's text without Markdown emphasis, so "merged **and** deployed" reads plainly."""
    return _text(doc).replace("*", "")


def _section(text: str, heading: str) -> str:
    """The body under ``heading`` up to the next heading of the same or a higher level."""
    start = text.index(heading) + len(heading)
    level = heading.split(" ", 1)[0]
    following = re.search(rf"^#{{1,{len(level)}}} ", text[start:], re.MULTILINE)
    return text[start:start + following.start()] if following else text[start:]


def _revisit_dates(text: str) -> List[datetime.date]:
    return [
        datetime.date(int(year), _MONTHS[month], int(day))
        for day, month, year in _REVISIT_DATE.findall(text)
    ]


@pytest.mark.parametrize("doc", _ARTICLE_50_DOCS, ids=lambda d: d.name)
def test_article_50_doc_exists(doc: pathlib.Path) -> None:
    assert doc.is_file(), f"{doc.relative_to(_REPO_ROOT)} is missing (PRD-264 EUAI-12 / EUAI-14)"


def test_gap_analysis_answers_each_code_commitment() -> None:
    text = _text(_GAP_ANALYSIS)
    missing = [heading for heading in _CODE_COMMITMENT_HEADINGS if heading not in text]
    assert not missing, f"the gap analysis has no section for {missing}"
    for heading in _CODE_COMMITMENT_HEADINGS:
        body = _section(text, heading)
        assert "**Does:**" in body or "**Applies to:**" in body, f"{heading}: what Automatos does"
        assert "**Doesn't:**" in body or "**Where Automatos stands" in body, f"{heading}: what it doesn't"


def test_gap_analysis_says_why_the_marking_meets_the_50_2_test() -> None:
    text = _text(_GAP_ANALYSIS)
    for quality in ("**Effective.**", "**Interoperable.**", "**Robust.**", "**Reliable.**"):
        assert quality in text, f"the gap analysis does not say why the marking is {quality}"


def test_gap_analysis_names_the_watermark_gap_and_d11() -> None:
    watermark = _section(_text(_GAP_ANALYSIS), "### Watermark")
    assert "D11" in watermark
    assert "main gap" in watermark
    assert "EUAI-13" in watermark


def test_gap_analysis_dates_the_revisit_after_web_summit() -> None:
    revisit = _section(_plain(_GAP_ANALYSIS), _REVISIT_HEADING)
    assert "Web Summit" in revisit
    dates = _revisit_dates(revisit)
    assert dates, "the revisit section does not say the day signing is revisited on"
    assert all(day > _WEB_SUMMIT_LAST_DAY for day in dates), f"the revisit {dates} is not after Web Summit"


def test_guide_places_every_surface_story() -> None:
    text = _text(_GUIDE)
    missing = [story for story in _GUIDE_STORIES if story not in text]
    assert not missing, f"ARTICLE-50.md does not place {missing}"


def test_guide_says_what_the_deployer_does() -> None:
    text = _text(_GUIDE)
    missing = [duty for duty in _GUIDE_DEPLOYER_DUTIES if duty not in text]
    assert not missing, f"ARTICLE-50.md does not state the deployer's {missing}"
    assert "| The deployer |" in text, "the surface tables need a deployer column"


@pytest.mark.parametrize("doc", _ARTICLE_50_DOCS, ids=lambda d: d.name)
def test_doc_claims_no_compliance_or_certification(doc: pathlib.Path) -> None:
    claims = sorted({match.group(0) for match in _OVERCLAIM.finditer(_text(doc))})
    assert not claims, f"{doc.name} claims {claims}; say what is built instead (PRD-264 §7)"


@pytest.mark.parametrize("doc", _ARTICLE_50_DOCS, ids=lambda d: d.name)
def test_doc_says_a_part_is_live_only_once_merged_and_deployed(doc: pathlib.Path) -> None:
    assert _LIVE_ONLY_WHEN in _plain(doc), f"{doc.name} must say what is live only once {_LIVE_ONLY_WHEN}"


@pytest.mark.parametrize("doc", _ARTICLE_50_DOCS, ids=lambda d: d.name)
def test_readme_table_links_the_doc(doc: pathlib.Path) -> None:
    rows = [line for line in _text(_README).splitlines() if line.startswith("|")]
    assert any(f"(./{doc.name})" in row for row in rows), f"README.md's table does not link {doc.name}"


@pytest.mark.parametrize("doc", (*_ARTICLE_50_DOCS, _README, _DETECTION), ids=lambda d: d.name)
def test_every_relative_link_resolves(doc: pathlib.Path) -> None:
    targets = _RELATIVE_LINK.findall(_text(doc))
    broken = [target for target in targets if not (doc.parent / target).exists()]
    assert not broken, f"{doc.name} links to missing files {broken}"
