"""PRD-270 W2 (K8) — the item bundle format, schema 1: its caps, its patterns and its specs.

One JSON document per approved item travels from Automatos Cloud to a local install. The specs
below are what ``validate.py`` walks. An object spec names every key it allows, so a key outside
the spec, at any depth, is refused: only the spec's keys can ever cross. A key an object spec does
not mark optional is required. An optional key is absent rather than null, except where its spec
allows null (a step's ``agent`` and ``pass_to``). A Playbook's agent step names an agent of the
bundle and carries its prompt_template; its document step names none (validate.py checks both).

The caps come from the columns the local install writes (core.py: Agent.name String(255),
marketplace_category String(100), version String(50); composio_cache.py: app_name String(100);
the personas, plugins and packages tables' slugs). Standard library only: the hosted export and
the local import share this package.
"""
from __future__ import annotations

import re
from types import MappingProxyType
from typing import Any, Mapping, Optional, Tuple

Spec = Mapping[str, Any]

BUNDLE_SCHEMA = 1

# The envelope, every type.
EXTERNAL_ID_LENGTH = 32
EXTERNAL_ID_PATTERN = r"^[0-9a-f]{32}$"
VERSION_MAX = 32
VERSION_PATTERN = r"^[0-9A-Za-z][0-9A-Za-z.-]{0,31}$"
NAME_MAX = 255
DESCRIPTION_MAX = 4000
CATEGORY_MAX = 100
ICON_MAX = 16
# An emoji or a short name, never an image URL, which a card would load from anywhere: of ASCII only
# letters, digits, space, '_' and '-' (so no '/', '\\', ':', '.', '%' or '@'), and any character past
# the C1 controls, which emoji and their joiners are.
ICON_PATTERN = r"^(?:[A-Za-z0-9 _-]|[^\x00-\x9f])*$"
TAGS_MAX = 20
TAG_MAX = 64
PUBLISHER_NAME_MAX = 100

# An agent.
AGENT_TYPE_MAX = 100
PROMPT_TEXT_MAX = 32768
PERSONA_SLUG_MAX = 100
PERSONA_SLUG_PATTERN = r"^[a-z0-9][a-z0-9-]{0,99}$"
MODEL_ID_MAX = 255
MODEL_ID_PATTERN = r"^[A-Za-z0-9._:/-]{1,255}$"
PROVIDER_MAX = 50
PROVIDER_PATTERN = r"^[a-z0-9_.-]{1,50}$"
SKILLS_MAX = 50
SKILL_NAME_MAX = 255
APPS_MAX = 50
APP_MAX = 100
APP_PATTERN = r"^[A-Z0-9_]{1,100}$"
PLUGINS_MAX = 20
PLUGIN_SLUG_MAX = 100
PLUGIN_SLUG_PATTERN = r"^[a-z0-9][a-z0-9-]{0,99}$"

# The targets a reference names (refs.py): a seeded agent's slug, a seeded Playbook's template id,
# a display name.
AGENT_SLUG_MAX = 255
AGENT_SLUG_PATTERN = r"^[a-z0-9][a-z0-9_-]{0,254}$"
TEMPLATE_ID_MAX = 100
TEMPLATE_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$"
REF_NAME_MAX = 255

# A Playbook. STEP_KEYS are the keys a stored step carries (core.py PLAYBOOK_STEP_FIELDS, the
# document step's, seed_socials_package.py) less agent_id, which names a row of the submitter's
# workspace: a bundle's step names its agent by a local id of the bundle's `agents` instead.
LOCAL_ID_PATTERN = r"^a[0-9]{1,3}$"
AGENTS_MAX = 50
STEPS_MAX = 50
STEP_KEYS = ("step_id", "order", "type", "prompt_template", "output_key", "error_handling", "pass_to",
             "config", "name", "title", "optional", "until", "instructions")
DOCUMENT_STEP_TYPE = "generate_document"  # core.py PLAYBOOK_DOCUMENT_STEP: renders a template, runs no agent
STEP_TYPES = ("agent", DOCUMENT_STEP_TYPE)
STEP_ERROR_HANDLING = ("stop", "skip", "retry", "fallback")
STEP_TOKEN_MAX = 100
STEP_TOKEN_PATTERN = r"^[A-Za-z0-9_.-]{1,100}$"
STEP_ORDER_MAX = 10000
PROMPT_TEMPLATE_MAX = 16384
STEP_TEXT_MAX = 255
STEP_UNTIL_MAX = 4096
STEP_CONFIG_KEYS = ("title", "format", "template_name", "data")
STEP_CONFIG_VALUE_MAX = 4096
IO_MAX = 50
IO_NAME_PATTERN = r"^[a-z][a-z0-9_]{0,63}$"
IO_FIELD_KEYS = ("type", "required", "description", "default", "enum")
IO_TYPE_MAX = 32
IO_DESCRIPTION_MAX = 1000
SCALAR_TEXT_MAX = 4096
SCALAR_NUMBER_MAX = 10 ** 15  # either side of zero; under JavaScript's largest safe integer
SCALAR_LIST_MAX = 50
# WorkflowTemplate.validate_execution_config's keys (core.py).
EXECUTION_KEYS = ("mode", "max_retries", "timeout_per_step", "total_timeout", "parallel_limit", "wait_for_me")
EXECUTION_MODES = ("sequential", "parallel")
EXECUTION_NUMBER_MAX = 1_000_000

# The fields whose value is a reference; refs.py says which kinds each takes.
PLAYBOOK_AGENT_REF = "playbook.agents"


def is_plain_text(value: str) -> bool:
    """Whether ``value`` is text a row can hold: no NUL (Postgres refuses it) and valid UTF-8 (no
    lone surrogate, which a JSON escape can carry and the canonical bytes could not encode)."""
    if "\x00" in value:
        return False
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def text(max_len: int, *, min_len: int = 0, pattern: Optional[str] = None, nullable: bool = False) -> Spec:
    """A string of ``min_len``..``max_len`` characters, matching ``pattern`` when one is given."""
    compiled = re.compile(pattern) if pattern else None
    return MappingProxyType({"kind": "text", "min": min_len, "max": max_len, "pattern": compiled, "nullable": nullable})


def number(max_value: int, *, min_value: int = 0, integer: bool = False) -> Spec:
    """A number (an integer when ``integer``) from ``min_value`` to ``max_value``."""
    return MappingProxyType({"kind": "number", "min": min_value, "max": max_value, "integer": integer})


def flag() -> Spec:
    """True or false."""
    return MappingProxyType({"kind": "flag"})


def choice(values: Tuple[str, ...]) -> Spec:
    """One of ``values``."""
    return MappingProxyType({"kind": "choice", "values": frozenset(values)})


def listing(items: Spec, max_items: int, *, min_items: int = 0) -> Spec:
    """A list of ``min_items``..``max_items`` entries, each matching ``items``."""
    return MappingProxyType({"kind": "list", "items": items, "min": min_items, "max": max_items})


def obj(fields: Mapping[str, Spec], *, optional: Tuple[str, ...] = ()) -> Spec:
    """An object with only ``fields``' keys; each is required unless named in ``optional``."""
    required = frozenset(key for key in fields if key not in optional)
    return MappingProxyType({"kind": "object", "fields": MappingProxyType(dict(fields)), "required": required})


def named(key_pattern: str, values: Spec, max_entries: int) -> Spec:
    """An object used as a map: up to ``max_entries`` names matching ``key_pattern``, each value ``values``."""
    return MappingProxyType({"kind": "map", "key": re.compile(key_pattern), "values": values, "max": max_entries})


def ref(field: str) -> Spec:
    """A reference, of a kind ``field`` takes (refs.FIELD_REFS)."""
    return MappingProxyType({"kind": "ref", "field": field})


def local_id() -> Spec:
    """A local id of the Playbook's ``agents``, or null; validate.py checks it names one."""
    return MappingProxyType({"kind": "local_id"})


def scalars(*, lists: bool) -> Spec:
    """A string (up to SCALAR_TEXT_MAX), a number (up to SCALAR_NUMBER_MAX either side of zero), true,
    false or null; with ``lists``, also a list of them."""
    return MappingProxyType({"kind": "scalars", "lists": lists, "max_items": SCALAR_LIST_MAX})


def _envelope(type_name: str, body: Spec) -> Spec:
    """Every bundle's fields, and the type's own object under the type's name."""
    return obj({
        "bundle": number(BUNDLE_SCHEMA, min_value=BUNDLE_SCHEMA, integer=True),
        "type": choice((type_name,)),
        "externalId": text(EXTERNAL_ID_LENGTH, pattern=EXTERNAL_ID_PATTERN),
        "version": text(VERSION_MAX, pattern=VERSION_PATTERN),
        "name": text(NAME_MAX, min_len=1),
        "description": text(DESCRIPTION_MAX),
        "category": text(CATEGORY_MAX),
        "icon": text(ICON_MAX, pattern=ICON_PATTERN),
        "tags": listing(text(TAG_MAX, min_len=1), TAGS_MAX),
        "publisher": obj({"name": text(PUBLISHER_NAME_MAX, min_len=1)}),
        type_name: body,
    })


APP = text(APP_MAX, pattern=APP_PATTERN)
PLUGIN_SLUG = text(PLUGIN_SLUG_MAX, pattern=PLUGIN_SLUG_PATTERN)
STEP_TOKEN = text(STEP_TOKEN_MAX, pattern=STEP_TOKEN_PATTERN)

AGENT_SPEC = obj({
    "agentType": text(AGENT_TYPE_MAX, min_len=1),
    "prompt": obj({"custom": flag(), "text": text(PROMPT_TEXT_MAX)}),
    "persona": text(PERSONA_SLUG_MAX, pattern=PERSONA_SLUG_PATTERN, nullable=True),
    "model": obj({
        "modelId": text(MODEL_ID_MAX, pattern=MODEL_ID_PATTERN, nullable=True),
        "provider": text(PROVIDER_MAX, pattern=PROVIDER_PATTERN, nullable=True),
    }),
    "skills": listing(obj({
        "name": text(SKILL_NAME_MAX, min_len=1),
        "package": text(PLUGIN_SLUG_MAX, pattern=PLUGIN_SLUG_PATTERN, nullable=True),
    }), SKILLS_MAX),
    "apps": listing(APP, APPS_MAX),
    "plugins": listing(PLUGIN_SLUG, PLUGINS_MAX),
})

STEP_SPEC = obj({
    "step_id": STEP_TOKEN,
    "order": number(STEP_ORDER_MAX, integer=True),
    "agent": local_id(),
    "type": choice(STEP_TYPES),
    "prompt_template": text(PROMPT_TEMPLATE_MAX),
    "output_key": STEP_TOKEN,
    "error_handling": choice(STEP_ERROR_HANDLING),
    "pass_to": text(STEP_TOKEN_MAX, pattern=STEP_TOKEN_PATTERN, nullable=True),
    "config": obj({key: text(STEP_CONFIG_VALUE_MAX) for key in STEP_CONFIG_KEYS}, optional=STEP_CONFIG_KEYS),
    "name": text(STEP_TEXT_MAX),
    "title": text(STEP_TEXT_MAX),
    "optional": flag(),
    "until": text(STEP_UNTIL_MAX),
    "instructions": text(PROMPT_TEMPLATE_MAX),
}, optional=tuple(key for key in STEP_KEYS if key not in ("step_id", "order")))

IO_FIELD_SPEC = obj({
    "type": text(IO_TYPE_MAX, min_len=1),
    "required": flag(),
    "description": text(IO_DESCRIPTION_MAX),
    "default": scalars(lists=True),
    "enum": listing(scalars(lists=False), SCALAR_LIST_MAX),
}, optional=IO_FIELD_KEYS)

EXECUTION_SPEC = obj({
    "mode": choice(EXECUTION_MODES),
    **{key: number(EXECUTION_NUMBER_MAX) for key in EXECUTION_KEYS if key not in ("mode", "wait_for_me")},
    "wait_for_me": flag(),
}, optional=EXECUTION_KEYS)

PLAYBOOK_SPEC = obj({
    "agents": named(LOCAL_ID_PATTERN, ref(PLAYBOOK_AGENT_REF), AGENTS_MAX),
    "steps": listing(STEP_SPEC, STEPS_MAX),
    "inputs": named(IO_NAME_PATTERN, IO_FIELD_SPEC, IO_MAX),
    "outputs": named(IO_NAME_PATTERN, IO_FIELD_SPEC, IO_MAX),
    "execution": EXECUTION_SPEC,
    "apps": listing(APP, APPS_MAX),
})

# One spec per bundle type; the types a reader knows are the ones with a spec.
TYPE_SPECS: Mapping[str, Spec] = MappingProxyType({
    "agent": _envelope("agent", AGENT_SPEC),
    "playbook": _envelope("playbook", PLAYBOOK_SPEC),
})
BUNDLE_TYPES: Tuple[str, ...] = tuple(TYPE_SPECS)
