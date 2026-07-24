"""How Auto sounds when she is HEARD rather than read.

Two halves of one contract:

* ``SPOKEN_OUTPUT_CONTRACT`` — the directive appended to the system prompt on
  spoken turns. Auto's brain is the same brain (ONE AUTO); this only tells it
  which medium the answer lands in. Without it she writes her normal chat
  reply — markdown bullets, ``**bold**``, headers, numbered lists — and the
  TTS reads the punctuation out loud.
* ``speechify()`` — the belt-and-braces sanitizer. Even a well-steered model
  emits the occasional asterisk or bullet, and a stray ``**`` costs a whole
  sentence of credibility when spoken. Never trust the prompt alone for
  something the user HEARS.

Both are pure and dependency-free so they unit-test without a DB, a socket or
a model.
"""
from __future__ import annotations

import re

# The directive. Written to be read by a model mid-conversation: concrete,
# example-led, and short enough to survive prompt budgeting.
SPOKEN_OUTPUT_CONTRACT = """
## YOU ARE SPEAKING OUT LOUD
This reply is converted to speech and played to a person in a room. They HEAR
it; they cannot see it. Write for the ear.

- Keep it to one to three short sentences, then stop and let them answer. A
  spoken paragraph is unbearable; a spoken list is worse.
- NEVER use markdown. No bullets, numbered lists, headings, bold, italics,
  code blocks, tables, links or emoji — they are read aloud literally and
  sound broken.
- Say it the way you'd say it to someone across a desk: contractions, plain
  words, one idea per sentence.
- Speak numbers, money, dates and times as words — "about twenty quid",
  "half three", "the twelfth" — never symbols or digits-with-punctuation.
- Don't read out URLs, file paths, IDs, hashes or code. Say "I've put it in
  the chat" and carry on.
- If the honest answer is long or structured, give the headline out loud and
  offer the rest: "there are four of them — want me to run through it?"
- While working, say so in a handful of words. Don't narrate each step.
- If you didn't catch them, say so plainly and ask them to say it again.
""".strip()


# ── speechify ──────────────────────────────────────────────────────────────
# Order matters: fences before inline code, images before links, emphasis
# after structure (so "**- item**" is fully unwrapped).

_CODE_FENCE_RE = re.compile(r"```[\s\S]*?(?:```|\Z)")
_INLINE_CODE_RE = re.compile(r"`+([^`]*)`+")
_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_HEADER_RE = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]*", re.M)
_BLOCKQUOTE_RE = re.compile(r"^[ \t]{0,3}>[ \t]?", re.M)
_HR_RE = re.compile(r"^[ \t]{0,3}([-*_])(?:[ \t]*\1){2,}[ \t]*$", re.M)
_TABLE_ROW_RE = re.compile(r"^[ \t]*\|.*$", re.M)
_BULLET_RE = re.compile(r"^[ \t]*(?:[-*+•‣▪]|\d{1,2}[.)])[ \t]+", re.M)
# Paired emphasis only — a lone asterisk in prose ("2 * 3") is left alone.
_EMPHASIS_RE = re.compile(r"(\*\*\*|\*\*|\*|___|__|_)(\S[\s\S]*?\S|\S)\1")
_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0001F1E6-\U0001F1FF"
    "\U00002190-\U000021FF"
    "\U0000FE00-\U0000FE0F"
    "\U00002B00-\U00002BFF"
    "]+"
)
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
_MULTI_STOP_RE = re.compile(r"(?:\s*\.){2,}")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.;:!?])")
# A line ending in ":" (a lead-in to a list) that we then terminated becomes
# ":." — keep the colon, drop the stop.
_PUNCT_THEN_STOP_RE = re.compile(r"([,;:!?])\s*\.")


def speechify(text: str) -> str:
    """Strip anything that a text-to-speech engine would mangle.

    Markdown structure becomes plain clauses: a bullet list turns into
    sentences, a link into its label, a code span into its contents. Returns
    "" when nothing speakable survives (e.g. a lone code fence), which the
    caller treats as "emit no frame".
    """
    if not text:
        return ""

    out = _CODE_FENCE_RE.sub(" ", text)
    out = _IMAGE_RE.sub(" ", out)
    out = _LINK_RE.sub(r"\1", out)
    out = _INLINE_CODE_RE.sub(r"\1", out)
    out = _HR_RE.sub(" ", out)
    out = _TABLE_ROW_RE.sub(" ", out)
    out = _HEADER_RE.sub("", out)
    out = _BLOCKQUOTE_RE.sub("", out)
    # A bullet ends the previous clause: "one\n- two" → "one. two"
    out = _BULLET_RE.sub("", out)
    # Emphasis can nest (***x***) — unwrap repeatedly, bounded.
    for _ in range(3):
        new = _EMPHASIS_RE.sub(r"\2", out)
        if new == out:
            break
        out = new
    out = _EMOJI_RE.sub(" ", out)

    # Newlines become sentence boundaries so list items don't run together.
    out = re.sub(r"\n{2,}", ". ", out)
    out = out.replace("\n", ". ")
    out = _MULTI_STOP_RE.sub(".", out)
    out = _PUNCT_THEN_STOP_RE.sub(r"\1", out)
    out = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", out)
    out = _MULTI_SPACE_RE.sub(" ", out).strip()
    # A unit that reduced to punctuation only says nothing.
    if not re.search(r"[A-Za-z0-9]", out):
        return ""
    return out


# ── speech units ───────────────────────────────────────────────────────────
# TTS wants whole clauses, not tokens. Buffering to a boundary is also what
# makes speechify() correct: markdown markers routinely straddle two stream
# chunks ("**" arrives as "*" then "*"), so sanitizing raw tokens can't work.

_BOUNDARY_RE = re.compile(r"[.!?…](?=[\s\"')\]]|$)|[\n]")


def split_speech_unit(buffer: str, max_chars: int) -> tuple[str, str]:
    """Split off one speakable unit; returns ``(unit, remainder)``.

    ``unit`` is "" when the buffer holds no complete unit yet. A buffer past
    ``max_chars`` with no terminator is cut at the last space so one runaway
    sentence can't hold the whole reply hostage.
    """
    if not buffer:
        return "", ""
    match = _BOUNDARY_RE.search(buffer)
    if match:
        cut = match.end()
        return buffer[:cut], buffer[cut:]
    if max_chars > 0 and len(buffer) >= max_chars:
        window = buffer[:max_chars]
        cut = window.rfind(" ")
        if cut <= 0:
            cut = max_chars
        return buffer[:cut], buffer[cut:]
    return "", buffer
