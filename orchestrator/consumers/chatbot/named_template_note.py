"""F351 (night 10b): the studio's "Use with Auto" prompt, built for the template the owner named.

``consumers/chatbot/named_template.py`` finds the template a turn names; this module reads it
(off the event loop, through ``platform_get_template_schema``'s own handler) and writes the
note the model reads last, in the studio's shape (``promptSnippets.ts``): the exact name and
id, every field, each table's columns, and the rules. A name this workspace hasn't got gets
"not here" and the closest names, never a substitute.

F369 (night 10c, chat 423d83ca): "Just make it, no questions." on the Branded Invoice, and Auto
asked for the invoice number, the terms and the due date, as the rules told it to. When the
owner says not to ask, the rules say so: make it now, put a plain default where a number,
a date or the terms are missing, say which in one line, and still invent no name, address,
price or quantity.
"""
from __future__ import annotations

import asyncio
import functools
import logging
import re
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Sequence, Tuple
from uuid import UUID

from consumers.chatbot.named_template import NamedTemplate, named_in_conversation, owner_turns
from consumers.chatbot.paperwork_to_the_team import team_note  # F337(c) (night 10)
from consumers.chatbot.brand_to_the_designer import designer_note  # PRD-255 US-014: Auto delegates the brand

logger = logging.getLogger(__name__)

SYSTEM_ROLE = "system"
MAX_FIELDS = 30              # fields listed in the note; the rest are counted
MAX_COLUMNS = 12             # columns listed per table field
SCHEMA_TOOL = "platform_get_template_schema"
DATA_PREFIX = "data."
REQUIRED_MARK = " (required)"

FOUND_HEAD = ('The owner named their document template "{name}" (template_id {id}){when}. Use THIS template: '
              "call generate_document with template_id {id}. Never make it on another template, even one with "
              "the same fields, and never call another template this one.")
EARLIER = " earlier in this conversation"
FIELDS_LINE = "Fill these fields: {fields}."
MORE = ", and {count} more (read them with " + SCHEMA_TOOL + ")"
LIST_LINE = "{field} is a list of rows, each with {columns}."
NO_FIELDS = "The template has no fill-in fields."
RULES = ("Map the owner's words onto these fields. Before generating, ask in one short message for every field "
         "marked required, and every invoice number, address, term, price or date the document needs, that "
         "the owner didn't give; never invent one. Pass data as an object keyed by these field names.")
# F369: the owner said not to ask. The defaults are the plain ones an owner would expect.
DEFAULT_TERMS_DAYS = 30
NO_QUESTIONS_RULES = (
    "Map the owner's words onto these fields. The owner said not to ask them anything: don't ask, make the "
    "document now. Where a value the document needs is missing, put a plain default instead: today's date for "
    f"its date, payment terms of {DEFAULT_TERMS_DAYS} days and a due date {DEFAULT_TERMS_DAYS} days after it, "
    "and a number made from today's date for an invoice or quote (INV-YYYYMMDD, QUO-YYYYMMDD). Never make up a "
    "name, an address, a price or a quantity: leave that field out. Under the document, say in one line which "
    "values you filled in, so the owner can change them. Pass data as an object keyed by these field names."
)
NO_QUESTIONS = re.compile(
    r"\bno (?:more )?questions\b|\b(?:don['’]?t|do not|no need to)\s+ask\b|\bwithout (?:asking|questions)\b"
    r"|\bjust (?:make|do|generate|create|send) it\b|\bfill in (?:the )?(?:rest|blanks|gaps)\b"
    r"|\buse (?:your )?(?:best )?judge?ment\b", re.I)
NOT_FOUND = ('The owner named a document template "{asked}", and this workspace has no template by that name or '
             "id. Don't make the document on another template in its place: tell the owner it isn't there and "
             "ask which one they mean.")
CLOSEST = "The closest names: {names}."
NO_TEMPLATES = "This workspace has no document templates yet."

Field = Tuple[str, bool]


def _bare(name: object) -> str:
    text = str(name or "").strip()
    return text[len(DATA_PREFIX):] if text.startswith(DATA_PREFIX) else text


def _field(item: Any, required: frozenset) -> Optional[Field]:
    """One data field as (name, required). An item is a ``data.*`` path today; a dict with its
    flags (``required``, ``fallback``) once the schema carries them."""
    if isinstance(item, dict):
        name = _bare(item.get("name") or item.get("field") or item.get("path"))
        flagged = bool(item.get("required")) and not item.get("fallback")
        return (name, flagged or name in required) if name else None
    name = _bare(item)
    return (name, name in required) if name else None


def _legacy_fields(data_schema: Any) -> Tuple[List[Any], frozenset]:
    """A legacy template's fields: its ``data_schema`` properties and required list."""
    schema = data_schema if isinstance(data_schema, dict) else {}
    properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
    return list(properties), frozenset(_bare(n) for n in schema.get("required") or [])


def schema_fields(schema: Dict[str, Any]) -> List[Field]:
    """Every field the schema gives, in its order, with whether it's required."""
    items = list(schema.get("data_fields") or [])
    listed = schema.get("required_fields") or schema.get("required")
    required = frozenset(_bare(n) for n in listed if isinstance(n, str)) if isinstance(listed, list) else frozenset()
    if not items:
        items, required = _legacy_fields(schema.get("data_schema"))
    fields = [_field(item, required) for item in items]
    return [f for f in fields if f]


def _column(item: Any) -> str:
    return str(item.get("key") or item.get("name") or "") if isinstance(item, dict) else str(item or "")


def _legacy_lists(data_schema: Any) -> List[Dict[str, Any]]:
    properties = (data_schema or {}).get("properties") if isinstance(data_schema, dict) else None
    lists = []
    for name, spec in (properties or {}).items():
        items = spec.get("items") if isinstance(spec, dict) and spec.get("type") == "array" else None
        if isinstance(items, dict) and isinstance(items.get("properties"), dict):
            lists.append({"field": name, "columns": list(items["properties"])})
    return lists


def schema_lists(schema: Dict[str, Any], row: Any) -> List[Tuple[str, List[str]]]:
    """Each table field with its columns: the schema's own when it gives them, else the studio's
    (``template_summary.list_fields_of``, what the "Use with Auto" prompt lists)."""
    from modules.documents.template_summary import list_fields_of

    lists = schema.get("list_fields")
    if lists is None:
        lists = list_fields_of(getattr(row, "blocks", None)) if schema.get("uses_blocks") else []
    lists = lists or _legacy_lists(schema.get("data_schema"))
    out = []
    for entry in lists:
        if isinstance(entry, dict) and entry.get("field"):
            columns = [c for c in (_column(c) for c in entry.get("columns") or []) if c]
            out.append((_bare(entry["field"]), columns))
    return out


def _capped(names: Sequence[str], cap: int) -> str:
    shown = ", ".join(names[:cap])
    return shown + (MORE.format(count=len(names) - cap) if len(names) > cap else "")


def no_questions(texts: Sequence[str]) -> bool:
    """Whether the owner, in this turn or the few before it, said not to ask ("Just make it, no questions")."""
    return any(NO_QUESTIONS.search(str(text or "")) for text in texts)


def found_note(named: NamedTemplate, schema: Dict[str, Any], told_not_to_ask: bool = False) -> str:
    """The studio's prompt for the named template, with the rules: the no-questions ones when the owner
    said not to ask (F369)."""
    template_id = str(named.row.id)
    head = FOUND_HEAD.format(name=named.row.name, id=template_id, when=EARLIER if named.earlier else "")
    fields = [name + (REQUIRED_MARK if required else "") for name, required in schema_fields(schema)]
    lines = [head, FIELDS_LINE.format(fields=_capped(fields, MAX_FIELDS)) if fields else NO_FIELDS]
    lines += [LIST_LINE.format(field=field, columns=_capped(columns, MAX_COLUMNS))
              for field, columns in schema_lists(schema, named.row) if columns]
    return " ".join([*lines, NO_QUESTIONS_RULES if told_not_to_ask else RULES])


def not_found_note(named: NamedTemplate) -> str:
    """"Not here", and the names nearest the one the owner gave: never a swap."""
    names = ", ".join(f'"{n}"' for n in named.closest)
    return " ".join([NOT_FOUND.format(asked=named.asked), CLOSEST.format(names=names) if names else NO_TEMPLATES])


def _schema(db: Any, workspace_id: UUID, row: Any) -> Dict[str, Any]:
    """The template's schema from the platform_get_template_schema handler itself (it awaits
    nothing; this runs it to completion in the worker thread)."""
    from modules.tools.discovery.handlers_documents import get_template_schema

    return asyncio.run(get_template_schema(db, workspace_id, {"template_id": str(row.id)}))


def read_note(db: Any, workspace_id: UUID, texts: Sequence[str]) -> Optional[str]:
    """The note for the template these owner turns name; with none named, the hand-to-the-team
    note when the latest asks for paperwork (F337(c)); else None. Brand work (the kit, the look,
    the templates) goes to the Brand designer first (PRD-255 US-014). Blocking: run off the loop."""
    from modules.documents.template_service import DocumentTemplateService

    try:
        with db.begin_nested():
            brand = designer_note(db, workspace_id, texts[0] if texts else "", opening=len(texts) == 1)  # F362
            if brand:
                return brand
            named = named_in_conversation(texts, DocumentTemplateService(db).list_templates(workspace_id))
            if named is None:   # F337(c): paperwork with no template named goes to the team
                return team_note(texts[0] if texts else "")
            if named.row is None:
                return not_found_note(named)
            return found_note(named, _schema(db, workspace_id, named.row), no_questions(texts))
    except Exception:
        logger.exception("[F351] the named template could not be read for this turn")
        return None


def _workspace(raw: Any) -> Optional[UUID]:
    try:
        return UUID(str(raw))
    except (TypeError, ValueError):
        logger.warning("[F351] no workspace id on this turn (%r): no template note", raw)
        return None


async def template_note(chat: Any, latest_text: str, llm_messages: List[Dict[str, Any]]) -> Optional[str]:
    """The note for this turn, or None (a widget visitor's turn, or no template named)."""
    if getattr(chat, "widget_mode", False) or not str(latest_text or "").strip():
        return None
    workspace_id = _workspace(getattr(chat, "workspace_id", None))
    if workspace_id is None:
        return None
    note = await asyncio.to_thread(read_note, chat.db, workspace_id, owner_turns(llm_messages, latest_text))
    if note:
        logger.info("[F351/F337] the turn gets a document note: a named template's fields, paperwork to the team, "
                    "or brand work to the Brand designer")
    return note


Turn = Callable[..., AsyncGenerator[Any, None]]


def fills_the_named_template(retrieval_first: Turn) -> Turn:
    """Wrap ``StreamingChatService._retrieval_first``: a turn that names a template gets the
    studio's prompt for it (or "not here" and the closest names) after the document passages."""
    @functools.wraps(retrieval_first)
    async def wrapped(chat: Any, latest_text: str, llm_messages: List[Dict[str, Any]], *args: Any,
                      **kwargs: Any) -> AsyncGenerator[Any, None]:
        note = await template_note(chat, latest_text, llm_messages)
        async for frame in retrieval_first(chat, latest_text, llm_messages, *args, **kwargs):
            yield frame
        if note:
            llm_messages.append({"role": SYSTEM_ROLE, "content": note})
    return wrapped


__all__ = ["FOUND_HEAD", "NOT_FOUND", "NO_QUESTIONS_RULES", "RULES", "fills_the_named_template", "found_note",
           "no_questions", "not_found_note", "read_note", "schema_fields", "schema_lists", "template_note"]
