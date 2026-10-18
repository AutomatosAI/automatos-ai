"""PRD-261 — what the theme host promises a theme, as the server holds it.

One place for the host and SDK versions a package's ``engines`` is checked against, the
names no package may take, the slots and pages a package may fill, and the files a cloned
page is tracked against. It also holds the error shape every check returns and the JSON
and plain-text rules they share.

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
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, NamedTuple, Optional, Tuple

# PA-2: the host contract's version (slots, pages, routes, boot data) and the SDK's.
# The SDK's equals frontend/lib/theme-host/shared-modules.json "sdkVersion".
THEME_HOST_VERSION = "1.0.0"
SDK_VERSION = "1.0.0"

MANIFEST_SCHEMA = "automatos-theme/v1"
LOOK_SCHEMA_SCHEMA = "automatos-look-schema/v1"
MANIFEST_PATH = "theme.json"

# PRD §4.2: a theme's own class on <html>.
HTML_CLASS_PATTERN = re.compile(r"[a-z][a-z0-9-]{2,31}")
# Classes the core owns on <html>: the Style axis, Classic, and next-themes' tones (and its
# "system" name, which it never puts on <html> but a theme must not take either).
RESERVED_HTML_CLASSES: Tuple[str, ...] = ("studio", "classic", "play", "dark", "light", "system")
# PA-8: Play's package keeps html.play until A-20; no other id may take it.
HTML_CLASS_OWNERS: Dict[str, str] = {"play": "automatos/play"}
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
# Marketplace, the theme menu and look pages, safe mode, the theme-only route root, the API).
HOST_OWNED_ROUTES: Tuple[str, ...] = ("/sign-in", "/sign-up", "/auth", "/sso-callback", "/reset-password", "/accept-invitation", "/settings", "/team", "/admin", "/marketplace", "/looks", "/safe-mode", "/x", "/api", "/dev")

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


# PRD §4.2 and Q25: the package as uploaded, JS plus CSS, the entry module, each mascot clip.
CAP_SETTINGS: Dict[str, CapSetting] = {
    "package": CapSetting("THEME_MAX_PACKAGE_MB", 10, MB),
    "code": CapSetting("THEME_MAX_CODE_KB", 3072, KB),
    "entry": CapSetting("THEME_MAX_ENTRY_KB", 300, KB),
    "clip": CapSetting("THEME_MAX_CLIP_KB", 1024, KB),
}

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
UNSAFE_CATEGORIES = frozenset({"Cc", "Zl", "Zp", "Co", "Cn"})
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


def is_finite_number(value: Any) -> bool:
    """A JSON number that is not a bool (the parser below refuses NaN and the infinities)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and abs(value) != float("inf")


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
