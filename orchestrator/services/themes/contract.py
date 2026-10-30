"""PRD-261 — what the theme host promises a theme, as the server holds it.

One place for the host and SDK versions a package's ``engines`` is checked against, the
names no package may take, the slots and pages a package may fill, and the files a cloned
page is tracked against, and the lists theme.json's ``mobile`` block is checked against (M-01).
It also holds the error shape every check returns and the JSON and plain-text rules they share.

This module uses the standard library only, on purpose:

* the drift test's regenerate one-liner (``tests/test_prd261_core_pages_drift.py``) loads
  it by file path with a bare ``python3``;
* the TypeScript parity test (``frontend/lib/theme-host/__tests__/validator-parity.test.ts``,
  A-06b) reads the version and the lists below as text, so each stays on one line in
  double quotes.

The size caps' config names live here too; ``services/themes/package.py`` reads them
through ``config`` and falls back to the defaults below until B2-wire adds them.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, NamedTuple, Optional, Tuple

# PA-2: the host contract's version (slots, pages, routes, boot data) and the SDK's. They equal
# frontend/lib/theme-host/shared-modules.json "hostVersion" and "sdkVersion"; the parity tests on
# both sides (tests/test_prd261_theme_package.py, validator-parity.test.ts) hold them together.
# Theme voice (C5): the SDK is 1.2.0 for its voice area; the host contract stays 1.1.0.
THEME_HOST_VERSION = "1.1.0"
SDK_VERSION = "1.2.0"

# PRD-261 M-01 (host 1.1.0, plan §2.2.11): theme.json's optional "mobile" block, the data a phone draws a
# theme from (services/themes/mobile_manifest.py checks it; frontend/lib/theme-host/mobile-manifest.ts is
# its twin, and validator-parity.test.ts reads every MOBILE_* name below as text). The tokens are the 24
# Studio tokens lib/look-kit/studio-tokens.ts lookToStudioTokens writes as 'H S% L%' triplets, never its
# eight pre-wrapped aliases; a scheme needs MOBILE_REQUIRED_TOKENS. MOBILE_CONTRAST_PAIRS are
# (text or mark, fill, kind): a "text" pair is measured against MOBILE_AA_TEXT, a "ui" pair against
# MOBILE_AA_UI (WCAG 2.x). A theme with a block must not admit a host below MOBILE_MIN_HOST.
MOBILE_TOKEN_NAMES: Tuple[str, ...] = (
    "--background", "--foreground", "--card", "--card-foreground", "--popover", "--popover-foreground",
    "--primary", "--primary-foreground", "--secondary", "--secondary-foreground", "--muted", "--muted-foreground",
    "--accent", "--accent-foreground", "--destructive", "--destructive-foreground", "--border", "--input", "--ring",
    "--background-gradient-from", "--background-gradient-to", "--glass-border-hover", "--cc-hover", "--cc-hover-border",
)
MOBILE_REQUIRED_TOKENS: Tuple[str, ...] = ("--background", "--foreground")
MOBILE_FONT_ROLES: Tuple[str, ...] = ("sans", "serif", "mono")
MOBILE_MASCOT_MODES: Tuple[str, ...] = ("clips", "stills", "none")
MOBILE_CONTRAST_PAIRS: Tuple[Tuple[str, str, str], ...] = (
    ("--foreground", "--background", "text"), ("--card-foreground", "--card", "text"),
    ("--popover-foreground", "--popover", "text"), ("--secondary-foreground", "--secondary", "text"),
    ("--muted-foreground", "--muted", "text"), ("--primary-foreground", "--primary", "text"),
    ("--accent-foreground", "--accent", "text"), ("--destructive-foreground", "--destructive", "text"),
    ("--primary", "--background", "ui"), ("--ring", "--background", "ui"),
)
MOBILE_AA_TEXT = 4.5
MOBILE_AA_UI = 3.0
MOBILE_MIN_HOST = "1.1.0"

MANIFEST_SCHEMA = "automatos-theme/v1"
LOOK_SCHEMA_SCHEMA = "automatos-look-schema/v1"
MANIFEST_PATH = "theme.json"

# PRD §4.2: a theme's own class on <html>.
HTML_CLASS_PATTERN = re.compile(r"[a-z][a-z0-9-]{2,31}")
# Classes the core owns on <html>: the Style axis, Classic, and next-themes' tones (and its
# "system" name, which it never puts on <html> but a theme must not take either). No theme's id
# keeps one (A-20, QB-11): any other class that fits the pattern is any theme's.
RESERVED_HTML_CLASSES: Tuple[str, ...] = ("studio", "classic", "dark", "light", "system")
# C-01a: only an admin may submit under these publishers. Private uploads are not checked here.
RESERVED_PUBLISHERS: Tuple[str, ...] = ("automatos",)

KNOWN_SLOTS: Tuple[str, ...] = ("frame@1", "lookEditor@1", "styleSheet@1", "notFound@1", "asleep@1", "page-header.mascot@1", "page-header.art@1", "empty-state.mascot@1")
# PA-14: every theme ships a look editor and a style sheet.
REQUIRED_SLOTS: Tuple[str, ...] = ("lookEditor@1", "styleSheet@1")

THEME_BASES: Tuple[str, ...] = ("studio", "classic")
THEME_TONES: Tuple[str, ...] = ("from-look", "user", "light", "dark")
PAGE_MODES: Tuple[str, ...] = ("own", "clone")

# Gerard's Q3 list (ii), as the app routes it (D0, 16 Oct): knowledge is /documents, missions
# are /assignments and /missions/[id], home is /.
PAGE_SCOPE: Tuple[str, ...] = ("/chat", "/deliverables", "/assignments", "/missions/[id]", "/agents", "/documents", "/playbooks", "/command-center", "/")
PAGE_AREAS: Dict[str, str] = {
    "/chat": "chat",
    "/deliverables": "deliverables",
    "/assignments": "missions",
    "/missions/[id]": "missions",
    "/agents": "agents",
    "/documents": "knowledge",
    "/playbooks": "playbooks",
    "/command-center": "command-center",
    "/": "home",
}
# The core files a clone of each page copies, relative to frontend/. A clone's clonedFrom.path
# is one of its route's files; data/core-pages.json holds each file's sha256 (PA-18), so a
# host release that changes one flags the clone "needs update".
PAGE_SOURCES: Dict[str, Tuple[str, ...]] = {
    "/chat": ("app/chat/page.tsx", "components/chatbot/studio-chat-shell.tsx"),
    "/deliverables": ("app/deliverables/page.tsx", "components/deliverables/studio/deliverables-studio.tsx"),
    "/assignments": ("app/assignments/page.tsx",),
    "/missions/[id]": ("app/missions/[id]/page.tsx",),
    "/agents": ("app/agents/page.tsx", "components/agents/studio/agent-management-studio.tsx"),
    "/documents": ("app/documents/page.tsx", "components/documents/document-management.tsx"),
    "/playbooks": ("app/playbooks/page.tsx", "components/playbooks/PlaybooksPanel.tsx"),
    "/command-center": ("app/command-center/page.tsx", "components/command-center/command-center-shell.tsx"),
    "/": ("app/page.tsx",),
}
# PRD §4.4: pages that are always the host's (sign-in, security and roles, admin, the
# Marketplace, the theme menu and look pages, safe mode, the theme-only route root, the API,
# and PRD-264 EUAI-10's public AI-content check, /verify).
HOST_OWNED_ROUTES: Tuple[str, ...] = ("/sign-in", "/sign-up", "/auth", "/sso-callback", "/reset-password", "/accept-invitation", "/settings", "/team", "/admin", "/marketplace", "/looks", "/safe-mode", "/x", "/api", "/dev", "/verify")

# PA-18: the sha256 of every PAGE_SOURCES file in this host release, {"files": {path: sha256}}.
# The running image has no frontend/ folder, so the drift check (A-07b) reads this file;
# tests/test_prd261_core_pages_drift.py holds it to the checkout and has the regenerate command.
CORE_PAGES_FILE = Path(__file__).resolve().parent / "data" / "core-pages.json"


def page_source_files() -> Tuple[str, ...]:
    """Every PAGE_SOURCES file once, in path order."""
    return tuple(sorted({path for paths in PAGE_SOURCES.values() for path in paths}))


def core_page_hashes() -> Dict[str, str]:
    """``{path: sha256}`` of the core page files this release ships (``data/core-pages.json``)."""
    document = json.loads(CORE_PAGES_FILE.read_text(encoding="utf-8"))
    return dict(document["files"])


KB = 1024
MB = 1024 * KB


class CapSetting(NamedTuple):
    """One size cap: the config name B2-wire adds, its default, and the bytes in one unit."""

    config_name: str
    default: int
    unit: int


# PRD §4.2 and Q25: the package as uploaded, JS plus CSS, the entry module, each mascot clip. The package
# cap holds every screenshot a theme may list (24 web and 24 phone at 512 KB, 24 MB) beside its code.
CAP_SETTINGS: Dict[str, CapSetting] = {
    "package": CapSetting("THEME_MAX_PACKAGE_MB", 32, MB),
    "code": CapSetting("THEME_MAX_CODE_KB", 3072, KB),
    "entry": CapSetting("THEME_MAX_ENTRY_KB", 300, KB),
    "clip": CapSetting("THEME_MAX_CLIP_KB", 1024, KB),
}

# PRD-261 A-19c (host 1.1.0, plan §2.2.3): boot/prepaint.json may carry a tone rule under this key,
# {"knob": <a choice knob of the table>, "values": {<every choice of it>: "light" | "dark"}}, so a
# generic look runtime sets a look's tone from data (css_value.py checks it; css-value.ts is its twin).
TONE_RULE_KEY = "tone"
TONE_RULE_FIELDS: Tuple[str, ...] = ("knob", "values")
TONE_RULE_TONES: Tuple[str, ...] = ("light", "dark")
# The running boot's looks (serve.py boot_looks): at most this many presets, each file at most
# BOOT_PRESET_MAX_BYTES (4 KB). frontend/lib/theme-host/boot-sanitize.ts holds the same two numbers.
BOOT_PRESETS_MAX = 12
BOOT_PRESET_MAX_BYTES = 4096

# ---------------------------------------------------------------------------
# The error shape every check returns
# ---------------------------------------------------------------------------

ThemeError = Dict[str, str]
# Messages never echo more than this many characters of the input back.
ECHO_MAX = 40
ECHO_REPLACEMENT = "?"
# Format characters a script or an emoji sequence needs: zero-width non-joiner and joiner.
JOINERS = frozenset({"‌", "‍"})
# Categories that break a line, hide text or carry hidden data (lib/text/plain-line.ts).
# Unassigned (Cn) is left out on purpose: this Python's Unicode tables are older than the
# browser's, so a character added since (U+1FA77 PINK HEART) would be refused here and
# accepted there. The server is never stricter than the client; Cf and the hidden variation
# selectors below still cover hidden data.
UNSAFE_CATEGORIES = frozenset({"Cc", "Zl", "Zp", "Co"})
HIDDEN_VARIATION_SELECTORS = range(0xE0100, 0xE01F0)


def theme_error(code: str, path: str, message: str) -> ThemeError:
    """One refusal: a code, where it is (a file or a dotted field) and a sentence."""
    return {"code": code, "path": path, "message": message}


def is_unsafe_char(char: str) -> bool:
    """True for a character that doesn't belong in one line of plain text."""
    category = unicodedata.category(char)
    if category == "Cf":
        return char not in JOINERS
    return category in UNSAFE_CATEGORIES or ord(char) in HIDDEN_VARIATION_SELECTORS


def echo(text: str) -> str:
    """``text`` as a message may quote it: unsafe characters as '?', at most ECHO_MAX characters."""
    shown = "".join(ECHO_REPLACEMENT if is_unsafe_char(char) else char for char in text[:ECHO_MAX + 1])
    return f"{shown[:ECHO_MAX]}…" if len(text) > ECHO_MAX else shown


def plain_text_problem(value: Any, field: str, max_chars: int) -> Optional[str]:
    """Why ``value`` is not one line of plain text of 1 to ``max_chars`` characters, or None.

    Nothing is trimmed: the value is stored as given, so the two validators can't disagree
    about what trimming removes.
    """
    if not isinstance(value, str):
        return f'"{field}" must be text.'
    if not value or all(char.isspace() for char in value):
        return f'"{field}" can\'t be empty.'
    if len(value) > max_chars:
        return f'"{field}" must be at most {max_chars} characters.'
    if any(is_unsafe_char(char) for char in value):
        return f'"{field}" must be one line of plain text.'
    return None


def is_whole_number(value: Any) -> bool:
    """A JSON number with no fraction (8 and 8.0 alike, as JavaScript reads them); never a bool."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return isinstance(value, int) or value.is_integer()


def js_number(value: Any) -> Optional[float]:
    """A JSON number as JavaScript's ``JSON.parse`` reads it, when that is finite; else None (a bool is no number).

    Python keeps an integer literal whole however long it is, where JavaScript reads the nearest
    double: past 2**53 a rounded one, past the double range Infinity. So the value is compared as
    ``float(value)``, and one that overflows (a 310-digit literal) or is infinite (``1e400``) is not
    finite, as ``Number.isFinite`` says of what ``JSON.parse`` gives for it.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    return number if math.isfinite(number) else None


def is_finite_number(value: Any) -> bool:
    """A JSON number that JavaScript reads as finite (``js_number``); never a bool. The parser below refuses NaN."""
    return js_number(value) is not None


# ---------------------------------------------------------------------------
# JSON files inside a package
# ---------------------------------------------------------------------------

# A theme's JSON nests a few levels (pages./chat.clonedFrom.path); far deeper is refused before
# anything walks it, whatever the running Python's parser would accept.
JSON_MAX_DEPTH = 32


def _refuse_constant(name: str) -> Any:
    raise ValueError(f"{name} is not JSON")


def _nests_deeper_than(data: Any, limit: int) -> bool:
    """True when ``data`` has arrays or objects nested past ``limit`` (walked without recursion)."""
    stack = [(data, 1)]
    while stack:
        node, depth = stack.pop()
        if not isinstance(node, (dict, list)):
            continue
        if depth > limit:
            return True
        stack.extend((child, depth + 1) for child in (node.values() if isinstance(node, dict) else node))
    return False


def parse_json(raw: bytes, path: str) -> Tuple[Any, Optional[ThemeError]]:
    """A package file's JSON document, or BAD_JSON saying why it can't be one.

    Refused: text that isn't UTF-8 JSON, NaN and Infinity (JavaScript's parser has neither),
    nesting past ``JSON_MAX_DEPTH`` and a lone surrogate escape no UTF-8 writer can carry.
    """
    try:
        data = json.loads(raw.decode("utf-8-sig"), parse_constant=_refuse_constant)
    except (UnicodeDecodeError, ValueError, RecursionError):
        return None, theme_error("BAD_JSON", path, f"{path} is not valid JSON text.")
    if _nests_deeper_than(data, JSON_MAX_DEPTH):
        return None, theme_error("BAD_JSON", path, f"{path} nests deeper than {JSON_MAX_DEPTH} levels.")
    try:
        json.dumps(data, ensure_ascii=False).encode("utf-8")
    except (UnicodeEncodeError, RecursionError):
        return None, theme_error("BAD_JSON", path, f"{path} holds text that isn't valid Unicode.")
    return data, None
