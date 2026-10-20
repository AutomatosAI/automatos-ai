#!/usr/bin/env python3
"""PRD-262 US-019: a brand kit that names Helvetica prints as it did before.

Debian's ghostscript brought fonts-urw-base35 into the API image, and with it the faces
fontconfig gives a stack naming Helvetica, Times, Courier, Palatino or Bookman (Nimbus
Sans, Nimbus Roman, Nimbus Mono PS, P052, URW Bookman). US-019 removes ghostscript and
names the font package itself, so a brand kit that names Helvetica with no font file
keeps those metrics instead of falling back to DejaVu. CI proves it on a real layout in
the published ``edge`` image and in the image the change builds. Both lay out the same
page, so only what US-019 changes, the fonts and libraries, can move a line:

    html                         inside the new image, on its code:
        docker run --rm -i -w /app --entrypoint python IMAGE - html < scripts/ci/image_pdf_fixture.py
        Prints ``FIXTURE_HTML <json>``: a block template whose brand kit names Helvetica,
        written the way generation_service writes a Branded PDF's page (the kit made
        render-ready, then the block renderer).

    layout FILE                  inside each image, FILE being html's output mounted in:
        docker run --rm -i -v OUT:/tmp/page.txt:ro --entrypoint python IMAGE - layout /tmp/page.txt < …
        Lays the page out with the image's WeasyPrint and fonts, reads it back with
        pdfplumber, and prints ``FIXTURE_PDF <json>``: each page's lines and the fonts
        its text is set in.

    compare EDGE_OUT NEW_OUT     on the runner:
        Fails unless the new image printed the same pages and lines in the same fonts as
        ``edge``, and set the text in Nimbus Sans.

Standard library only outside ``html`` (the image's app modules) and ``layout``
(WeasyPrint and pdfplumber).
"""
from __future__ import annotations

import difflib
import io
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

HTML_MARKER = "FIXTURE_HTML "
PDF_MARKER = "FIXTURE_PDF "
BRAND_FONT = "Helvetica"
# fonts-urw-base35's Helvetica, as a PDF names it once ``flat`` has run.
EXPECTED_FACE = "NimbusSans"
TITLE = "Harbourline wholesale terms"
SENTENCES = (
    "Harbourline roasts its beans in small batches every Tuesday and Thursday morning.",
    "Orders placed before noon leave the roastery the same afternoon by courier.",
    "Wholesale customers receive a price list each quarter, printed in the brand's own font.",
    "Every invoice states the blend, the weight in kilos, the unit price and the delivery day.",
    "Returns are accepted within fourteen days when the bags are sealed and stored dry.",
)
# Enough text for two pages or more, so the page count is a real check too.
PARAGRAPHS = 18
MAX_DIFF_LINES = 40
USAGE = "usage: image_pdf_fixture.py html | layout FILE | compare EDGE_OUT NEW_OUT"


def paragraph(number: int) -> str:
    """Paragraph ``number``: the sentences, each paragraph starting one further along. Pure."""
    turn = number % len(SENTENCES)
    return f"{number}. " + " ".join(SENTENCES[turn:] + SENTENCES[:turn])


def page_blocks() -> Dict[str, Any]:
    """The fixture as a block template: a title and ``PARAGRAPHS`` text blocks. Pure."""
    title = {"type": "heading", "id": "title", "level": 1, "content": [{"type": "text", "text": TITLE}]}
    texts = [{"type": "text", "id": f"p{n}", "content": [{"type": "text", "text": paragraph(n)}]}
             for n in range(1, PARAGRAPHS + 1)]
    return {"version": 1, "blocks": [title, *texts]}


def lines(text: str) -> List[str]:
    """A page's printed lines, spacing normalised, blank lines dropped. Pure."""
    return [" ".join(line.split()) for line in text.splitlines() if line.strip()]


def flat(fontname: str) -> str:
    """A PDF font name without its subset prefix, spaces or hyphens ("ABCDEF+NimbusSans-Regular"). Pure."""
    return fontname.split("+")[-1].replace(" ", "").replace("-", "")


def fixture_html() -> str:
    """The fixture's page, as the image's own code writes it (generation_service's block path):
    a kit whose font is ``BRAND_FONT``, made render-ready, through the block renderer."""
    from modules.documents.blocks import render_document_html, validate_blocks
    from modules.documents.brand_fonts import brand_kit_for_media_render
    from modules.documents.brand_kit import get_brand_kit

    kit = brand_kit_for_media_render(get_brand_kit({"brand_kit": {"font_family": BRAND_FONT}}))
    return render_document_html(validate_blocks(page_blocks()), {}, kit, title=TITLE).html


def render_pdf(html: str) -> bytes:
    """``html`` laid out as a PDF by the image's WeasyPrint and fonts, as generation_service does."""
    from weasyprint import HTML

    return HTML(string=html).render().write_pdf()


def read_pdf(data: bytes) -> Dict[str, Any]:
    """Each page's lines and the sorted fonts the text is set in, read with pdfplumber."""
    import pdfplumber

    with pdfplumber.open(io.BytesIO(data)) as document:
        pages = [lines(page.extract_text() or "") for page in document.pages]
        fonts = sorted({flat(char["fontname"]) for page in document.pages for char in page.chars})
    return {"pages": pages, "fonts": fonts}


def marked(output: str, marker: str) -> Any:
    """The JSON on the last line of ``output`` that starts with ``marker``. Pure; ValueError without one."""
    found = [line[len(marker):] for line in output.splitlines() if line.startswith(marker)]
    if not found:
        raise ValueError(f"no {marker.strip()} line in the output")
    return json.loads(found[-1])


def _numbered(facts: Dict[str, Any]) -> List[str]:
    return [f"page {page}: {line}" for page, printed in enumerate(facts["pages"], 1) for line in printed]


def problems(edge: Dict[str, Any], new: Dict[str, Any]) -> List[str]:
    """Why the new image's render differs from edge's, or does not wear Nimbus Sans. Pure."""
    found = []
    if not new["pages"] or not any(new["pages"]):
        found.append("the new image's PDF has no text")
    if len(new["pages"]) != len(edge["pages"]):
        found.append(f"the PDF has {len(new['pages'])} pages; edge printed {len(edge['pages'])}")
    if new["pages"] != edge["pages"]:
        diff = list(difflib.unified_diff(_numbered(edge), _numbered(new), "edge", "new", lineterm="", n=0))
        found.append("the line breaks differ from edge's:\n" + "\n".join(diff[:MAX_DIFF_LINES]))
    if new["fonts"] != edge["fonts"]:
        found.append(f"the text is set in {new['fonts']}; edge set it in {edge['fonts']}")
    if not any(EXPECTED_FACE in font for font in new["fonts"]):
        found.append(f"{BRAND_FONT} did not print in {EXPECTED_FACE} (fonts-urw-base35): {new['fonts']}")
    return found


def summary(name: str, facts: Dict[str, Any]) -> str:
    total = sum(len(printed) for printed in facts["pages"])
    return f"{name}: {len(facts['pages'])} pages, {total} lines, set in {', '.join(facts['fonts'])}"


def compare(edge_path: str, new_path: str) -> int:
    edge = marked(Path(edge_path).read_text(encoding="utf-8"), PDF_MARKER)
    new = marked(Path(new_path).read_text(encoding="utf-8"), PDF_MARKER)
    print(summary("edge", edge))
    print(summary("new", new))
    found = problems(edge, new)
    for problem in found:
        print(f"::error title=Helvetica PDF::{problem.splitlines()[0]}")
        print(problem)
    if not found:
        print(f"a brand kit naming {BRAND_FONT} prints the same pages and line breaks as edge, in {EXPECTED_FACE}")
    return 1 if found else 0


def main(argv: List[str]) -> int:
    if argv == ["html"]:
        print(HTML_MARKER + json.dumps(fixture_html()))
        return 0
    if argv[:1] == ["layout"] and len(argv) == 2:
        page = marked(Path(argv[1]).read_text(encoding="utf-8"), HTML_MARKER)
        print(PDF_MARKER + json.dumps(read_pdf(render_pdf(page))))
        return 0
    if argv[:1] == ["compare"] and len(argv) == 3:
        return compare(argv[1], argv[2])
    print(USAGE, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
