"""PRD-261 — every rule in a theme's ``dist/theme.css`` sits under the theme's own class.

PRD §4.3 item 8: a theme's CSS is wrapped in its ``html.<htmlClass>``, so it styles nothing
once another theme (or none) is active. This is the server's check of that, written by hand
with no new dependency (D1-f):

* each selector of a top-level rule (or of one inside ``@media``, ``@supports``, ``@layer`` or
  ``@container``) starts with ``html.<class>``, or with ``:where(...)``/``:is(...)`` whose every
  argument does (PA-15 leaves the wrap form to D2-c, so both forms pass);
* rules nested inside a scoped rule, ``@font-face`` and ``@keyframes`` are already contained,
  except a nested selector that names ``&`` anywhere but at its start: CSS Nesting reads it as
  written, so ``:not(&)`` or ``:is(&, body)`` reaches every element outside the theme's class;
* the names those two declare are global, though: every ``@keyframes`` (and
  ``@-webkit-keyframes``) name and every ``@font-face`` ``font-family`` (quoted or not) starts
  with ``<class>-``, so ``@keyframes spin`` can't replace the host's own ``animate-spin``;
* ``!important`` is refused (any ``!`` outside a string or a comment), and so are ``@import``
  and every other at-rule;
* anything the tokenizer could read differently from a browser is refused rather than
  guessed: an unclosed comment, string or block, a brace or semicolon inside brackets, a
  backslash escape of a letter or digit (``u\\72l(`` spells ``url(``), an unquoted ``url()``
  holding anything a browser would end it on.

Refusing more than a browser would is safe; reading a rule as scoped when a browser would
apply it everywhere is not, so every ambiguity above is a refusal.
"""

from __future__ import annotations

import re
from typing import List, NamedTuple, Optional

from services.themes.contract import ThemeError, echo, theme_error

STYLE_PATH = "dist/theme.css"
CSS_ERROR_CODES = ("CSS_UNSCOPED", "CSS_IMPORTANT", "CSS_IMPORT", "CSS_AT_RULE", "CSS_SYNTAX")
MAX_ERRORS = 20
MAX_WRAP_DEPTH = 3
# Brackets and blocks nest at most this deep; deeper is refused, so the scan stays linear.
MAX_NESTING = 64

GROUP_AT_RULES = frozenset({"media", "supports", "layer", "container"})
FONT_FACE = "font-face"
KEYFRAMES_AT_RULES = frozenset({"keyframes", "-webkit-keyframes"})
CONTAINED_AT_RULES = frozenset({FONT_FACE, *KEYFRAMES_AT_RULES})
FONT_FAMILY = "font-family"
DECLARATION_COLON = ":"
QUOTES = frozenset("\"'")
STATEMENT_AT_RULES = frozenset({"charset", "layer"})
WRAPPING_PSEUDOS = (":where(", ":is(")
NESTING_SELECTOR = "&"
OPAQUE_SELECTOR_START = frozenset("\"'\\")  # a string or an escape: no & inside counts
SELECTOR_BOUNDARY = frozenset(" \t\n\r\f>+~.:[#")
BRACKET_STEP = {"(": 1, "[": 1, ")": -1, "]": -1}
OPENING_BRACKET = {")": "(", "]": "["}

# A backslash may escape punctuation only: an escaped letter or digit can spell a keyword.
REFUSED_ESCAPE = re.compile(r"[A-Za-z0-9\n\r\f]")
IDENT_CHAR = re.compile(r"[A-Za-z0-9_\-\u0080-\U0010ffff\\]")
AT_NAME = re.compile(r"@(-?[A-Za-z][A-Za-z0-9-]*)")
URL_NAME = "url"
QUOTED_URL = re.compile(r"\([ \t\n\r\f]*[\"']")
# Written so no two runs can match the same characters: the match is linear, never backtracking
# over a long whitespace run (a ReDoS shape the plain "ws* body* ws*" form had).
PLAIN_URL = re.compile(r"\([ \t\n\r\f]*(?:[^ \t\n\r\f\"'()\\\x00-\x08\x0b\x0e-\x1f\x7f]+[ \t\n\r\f]*)?\)")

TOKEN = re.compile(
    r"(?P<comment>/\*.*?\*/)"
    r"|(?P<bad_comment>/\*)"
    r"|(?P<string>\"(?:[^\"\\\n\r\f]|\\[\s\S])*\"|'(?:[^'\\\n\r\f]|\\[\s\S])*')"
    r"|(?P<bad_string>[\"'])"
    r"|(?P<escape>\\[\s\S]?)"
    r"|(?P<ident>-{0,2}[A-Za-z_\u0080-\U0010ffff][A-Za-z0-9_\-\u0080-\U0010ffff]*)"
    r"|(?P<punct>[{};()\[\]!])"
    r"|(?P<other>[^{};()\[\]!\"'/\\A-Za-z_\-\u0080-\U0010ffff]+|[/\-])",
    re.S,
)
# A selector's pieces, for splitting a list at its top-level commas.
SELECTOR_TOKEN = re.compile(
    r"\\[\s\S]?|\"(?:[^\"\\]|\\[\s\S])*\"?|'(?:[^'\\]|\\[\s\S])*'?|[()\[\],]|[^()\[\],\"'\\]+"
)


class _Frame(NamedTuple):
    """An open block: what kind it is, and whether rules inside it are already contained."""

    kind: str  # "top", "group", "style", "contained" or "font-face"
    scoped: bool


def _split_top_level(text: str) -> List[str]:
    """``text`` split at commas outside brackets and strings."""
    parts: List[str] = []
    depth, start = 0, 0
    for match in SELECTOR_TOKEN.finditer(text):
        piece = match.group(0)
        depth += BRACKET_STEP.get(piece, 0)
        if piece == "," and depth == 0:
            parts.append(text[start:match.start()])
            start = match.end()
    return [*parts, text[start:]]


def _closing_bracket(text: str, start: int) -> Optional[int]:
    """Where the bracket opened just before ``start`` closes, or None."""
    depth = 1
    for match in SELECTOR_TOKEN.finditer(text, start):
        depth += BRACKET_STEP.get(match.group(0), 0)
        if depth == 0:
            return match.start()
    return None


def is_scoped_selector(selector: str, html_class: str, level: int = 0) -> bool:
    """True when the selector can only match inside ``html.<html_class>``."""
    text = selector.strip()
    wrap = f"html.{html_class}"
    if text.startswith(wrap):
        rest = text[len(wrap):]
        return not rest or rest[0] in SELECTOR_BOUNDARY
    pseudo = next((name for name in WRAPPING_PSEUDOS if text.lower().startswith(name)), None)
    if pseudo is None or level >= MAX_WRAP_DEPTH:
        return False
    close = _closing_bracket(text, len(pseudo))
    if close is None:
        return False
    return all(is_scoped_selector(part, html_class, level + 1) for part in _split_top_level(text[len(pseudo):close]))


def is_scoped_selector_list(prelude: str, html_class: str) -> bool:
    """True when every selector of the list is scoped."""
    return all(is_scoped_selector(part, html_class) for part in _split_top_level(prelude))


def _names_nesting_selector(selector: str) -> bool:
    """True when ``&`` appears in the selector outside a string or an escape."""
    pieces = (match.group(0) for match in SELECTOR_TOKEN.finditer(selector))
    return any(NESTING_SELECTOR in piece for piece in pieces if piece[0] not in OPAQUE_SELECTOR_START)


def is_anchored_nested_selector(selector: str) -> bool:
    """True when a selector nested in a scoped rule can only match inside that rule's reach.

    CSS Nesting reads a nested selector with no ``&`` as ``& <selector>``, and one with ``&`` as
    written. One that starts with ``&`` stays on the parent's elements, their descendants and
    their siblings; ``&`` anywhere else (``:not(&)``, ``:is(&, body)``, ``body:not(&)``) is refused.
    """
    text = selector.strip()
    return text.startswith(NESTING_SELECTOR) or not _names_nesting_selector(text)


def is_anchored_nested_selector_list(prelude: str) -> bool:
    """True when every selector of a nested rule's list is anchored."""
    return all(is_anchored_nested_selector(part) for part in _split_top_level(prelude))


def _at_name(prelude: str) -> str:
    match = AT_NAME.match(prelude)
    return match.group(1).lower() if match else ""


def _unquoted(text: str) -> str:
    """``text`` without one pair of matching quotes around it (a CSS string), else as given."""
    if len(text) >= 2 and text[0] in QUOTES and text[-1] == text[0]:
        return text[1:-1]
    return text


def is_theme_name(text: str, html_class: str) -> bool:
    """True for a global name (an animation, a font family) the theme owns: ``<class>-`` then more."""
    pattern = rf"{re.escape(html_class)}-[A-Za-z0-9_-]+"
    return re.fullmatch(pattern, _unquoted(text.strip())) is not None


def _font_family_value(declaration: str) -> Optional[str]:
    """The value of a ``font-family`` declaration, or None for any other declaration."""
    name, colon, value = declaration.partition(DECLARATION_COLON)
    if not colon or name.strip().lower() != FONT_FAMILY:
        return None
    return value


class _Scanner:
    """One pass over the stylesheet's tokens, keeping the open blocks on a stack."""

    def __init__(self, css: str, html_class: str) -> None:
        self.css = css
        self.html_class = html_class
        self.stack: List[_Frame] = [_Frame("top", False)]
        self.prelude: List[str] = []
        self.brackets: List[str] = []
        self.errors: List[ThemeError] = []
        self.after_escape = False

    def error(self, code: str, at: int, message: str) -> None:
        line = self.css.count("\n", 0, at) + 1
        self.errors.append(theme_error(code, f"{STYLE_PATH}:{line}", f"{STYLE_PATH} line {line}: {message}"))

    def take_prelude(self) -> str:
        text = "".join(self.prelude).strip()
        self.prelude = []
        return text

    def run(self) -> List[ThemeError]:
        at = 0
        while at < len(self.css) and len(self.errors) < MAX_ERRORS:
            match = TOKEN.match(self.css, at)
            if match is None:  # every character matches one branch; kept as a guard
                self.error("CSS_SYNTAX", at, "this character can't be read.")
                break
            at = self.take(match)
        if len(self.errors) < MAX_ERRORS:
            self.finish()
        return self.errors[:MAX_ERRORS]

    def take(self, match: "re.Match[str]") -> int:
        """Handle one token; return where the next one starts."""
        kind, text, at = match.lastgroup or "", match.group(0), match.start()
        after_escape, self.after_escape = self.after_escape, kind == "escape"
        if kind in ("bad_comment", "bad_string"):
            what = "comment" if kind == "bad_comment" else "quoted string"
            self.error("CSS_SYNTAX", at, f"a {what} is never closed.")
            return len(self.css)
        if kind == "escape" and (len(text) < 2 or REFUSED_ESCAPE.match(text[1])):
            self.error("CSS_SYNTAX", at, "a backslash may escape punctuation only, never a letter or a digit.")
        if kind == "ident" and text.lower() == URL_NAME and not after_escape:
            return self.take_url(match)
        if kind == "punct":
            self.take_punct(text, at)
            return match.end()
        self.prelude.append(" " if kind == "comment" else text)
        return match.end()

    def take_url(self, match: "re.Match[str]") -> int:
        """``url(`` unquoted is one opaque token up to its ``)``, as a browser reads it."""
        end = match.end()
        glued = match.start() > 0 and IDENT_CHAR.match(self.css[match.start() - 1]) is not None
        if glued or not self.css.startswith("(", end) or QUOTED_URL.match(self.css, end):
            self.prelude.append(match.group(0))
            return end
        plain = PLAIN_URL.match(self.css, end)
        if plain is None:
            self.error("CSS_SYNTAX", match.start(), "an unquoted url() may hold no spaces, quotes, brackets or backslashes.")
            return len(self.css)
        self.prelude.append(self.css[match.start():plain.end()])
        return plain.end()

    def take_punct(self, char: str, at: int) -> None:
        if char == "!":
            self.error("CSS_IMPORTANT", at, "!important (or any other !) is not allowed in a theme's CSS.")
        elif char in BRACKET_STEP:
            self.take_bracket(char, at)
        else:
            self.take_block_punct(char, at)

    def take_bracket(self, char: str, at: int) -> None:
        """Brackets nest by kind, as a browser reads them: '( ]' is not a closed pair."""
        self.prelude.append(char)
        if char not in OPENING_BRACKET:
            if len(self.brackets) >= MAX_NESTING:
                self.error("CSS_SYNTAX", at, f"brackets nest deeper than {MAX_NESTING}.")
                return
            self.brackets.append(char)
            return
        if not self.brackets or self.brackets[-1] != OPENING_BRACKET[char]:
            self.error("CSS_SYNTAX", at, f"a {char} closes nothing.")
            self.brackets = []
            return
        self.brackets.pop()

    def take_block_punct(self, char: str, at: int) -> None:
        if self.brackets:
            self.error("CSS_SYNTAX", at, f"a {char} sits inside an unclosed bracket.")
            self.brackets = []
        handler = {"{": self.open_block, "}": self.close_block, ";": self.end_statement}[char]
        handler(at)

    def open_block(self, at: int) -> None:
        prelude, parent = self.take_prelude(), self.stack[-1]
        if len(self.stack) > MAX_NESTING:  # the stack also holds the top frame
            self.error("CSS_SYNTAX", at, f"blocks nest deeper than {MAX_NESTING}.")
            return
        frame = self.at_rule_frame(prelude, parent, at) if prelude.startswith("@") else self.rule_frame(prelude, parent, at)
        self.stack.append(frame)

    def at_rule_frame(self, prelude: str, parent: _Frame, at: int) -> _Frame:
        name = _at_name(prelude)
        if name in GROUP_AT_RULES:
            return _Frame("group", parent.scoped)
        if name in KEYFRAMES_AT_RULES:
            self.check_keyframes_name(prelude[len(name) + 1:], at)
        elif name not in CONTAINED_AT_RULES:
            self.refuse_at_rule(name, prelude, at)
        return _Frame(FONT_FACE if name == FONT_FACE else "contained", True)

    def check_keyframes_name(self, keyframes_name: str, at: int) -> None:
        if not is_theme_name(keyframes_name, self.html_class):
            self.error(
                "CSS_UNSCOPED", at,
                f'the animation "{echo(keyframes_name.strip())}" must be named {self.html_class}-something: '
                "animation names are shared with the whole app, so this one could replace the app's own.",
            )

    def check_font_family(self, declaration: str, at: int) -> None:
        family = _font_family_value(declaration)
        if family is not None and not is_theme_name(family, self.html_class):
            self.error(
                "CSS_UNSCOPED", at,
                f'the font "{echo(family.strip())}" must be named {self.html_class}-something: '
                "font names are shared with the whole app, so this one could replace the app's own.",
            )

    def rule_frame(self, prelude: str, parent: _Frame, at: int) -> _Frame:
        contained = parent.kind == "contained" or parent.scoped
        if contained and not is_anchored_nested_selector_list(prelude):
            self.error(
                "CSS_UNSCOPED", at,
                f'the nested rule "{echo(prelude)}" names & other than at the start of a selector, '
                f"so it can reach outside html.{self.html_class}.",
            )
        elif not contained and not is_scoped_selector_list(prelude, self.html_class):
            self.error(
                "CSS_UNSCOPED", at,
                f'the rule "{echo(prelude)}" is not under html.{self.html_class} (or :where(html.{self.html_class})).',
            )
        return _Frame("style", True)

    def refuse_at_rule(self, name: str, prelude: str, at: int) -> None:
        if name == "import":
            self.error("CSS_IMPORT", at, "@import is not allowed: a theme ships one stylesheet.")
            return
        self.error("CSS_AT_RULE", at, f'"{echo(prelude)}" is an at-rule a theme\'s CSS may not use.')

    def close_block(self, at: int) -> None:
        leftover, frame = self.take_prelude(), self.stack[-1]
        if frame.kind == "top":
            self.error("CSS_SYNTAX", at, "a } closes nothing.")
            return
        if leftover and frame.kind == "group":
            self.error("CSS_SYNTAX", at, f'"{echo(leftover)}" has no block.')
        if leftover and frame.kind == FONT_FACE:
            self.check_font_family(leftover, at)
        self.stack.pop()

    def end_statement(self, at: int) -> None:
        prelude = self.take_prelude()
        if prelude.startswith("@"):
            name = _at_name(prelude)
            if name not in STATEMENT_AT_RULES:
                self.refuse_at_rule(name, prelude, at)
            return
        if prelude and self.stack[-1].kind in ("top", "group"):
            self.error("CSS_SYNTAX", at, f'"{echo(prelude)}" is not inside a rule.')
        elif self.stack[-1].kind == FONT_FACE:
            self.check_font_family(prelude, at)

    def finish(self) -> None:
        end = len(self.css)
        if len(self.stack) > 1:
            self.error("CSS_SYNTAX", end, "a { is never closed.")
        leftover = self.take_prelude()
        if leftover and len(self.stack) == 1:
            self.error("CSS_SYNTAX", end, f'"{echo(leftover)}" has no block.')
        if self.brackets:
            self.error("CSS_SYNTAX", end, "a bracket is never closed.")


def css_scope_errors(css: str, html_class: str) -> List[ThemeError]:
    """Every scope, ``!important``, at-rule and syntax refusal in ``css``, at most ``MAX_ERRORS``."""
    return _Scanner(css, html_class).run()
