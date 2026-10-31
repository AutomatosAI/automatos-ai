"""PRD-270 W2 (K8) — the reference grammar of an item bundle.

A reference is an object with exactly one key, its kind, naming one target:

* ``{"slug": s}`` an agent the local edition seeds, by its slug; a package's plugin member names
  its plugin by the plugin's slug;
* ``{"templateId": s}`` a Playbook the local edition seeds, by its template id;
* ``{"item": externalId}`` another item of the same catalogue, of the type the field needs;
* ``{"name": s}`` a name: in a Playbook's agents, shown to the user only, never resolved to a row;
  a package's skill member names its skill by it, beside an optional ``package``, the slug of the
  plugin the skill comes from (the one key a reference may carry beside its kind);
* ``{"modelId": s}`` a model, by its model id;
* ``{"app": APP}`` a Composio app, by its name.

Each field that holds a reference takes only some kinds (``FIELD_REFS``); a package member's
field is its type's (schema.member_ref_field). A reference of another kind, of another form, or
with a target off its kind's pattern is refused as 'ref'. Standard library only: the hosted export
and the local import share this package.
"""
from __future__ import annotations

import re
from types import MappingProxyType
from typing import Any, Mapping, Optional, Tuple

from services.marketplace_bundles.errors import BundleError
from services.marketplace_bundles.schema import (
    AGENT_SLUG_MAX, AGENT_SLUG_PATTERN, APP_MAX, APP_PATTERN, EXTERNAL_ID_LENGTH, EXTERNAL_ID_PATTERN,
    MODEL_ID_MAX, MODEL_ID_PATTERN, PLAYBOOK_AGENT_REF, PLUGIN_SLUG_MAX, PLUGIN_SLUG_PATTERN, REF_NAME_MAX,
    SKILL_REF_NAME_PATTERN, TEMPLATE_ID_MAX, TEMPLATE_ID_PATTERN, is_plain_text, member_ref_field,
)

# A target's form: its longest length, and its pattern or None for any text.
Form = Tuple[int, Optional[re.Pattern[str]]]

REF_KINDS: Mapping[str, Form] = MappingProxyType({
    "slug": (AGENT_SLUG_MAX, re.compile(AGENT_SLUG_PATTERN)),
    "templateId": (TEMPLATE_ID_MAX, re.compile(TEMPLATE_ID_PATTERN)),
    "item": (EXTERNAL_ID_LENGTH, re.compile(EXTERNAL_ID_PATTERN)),
    "name": (REF_NAME_MAX, None),
    "modelId": (MODEL_ID_MAX, re.compile(MODEL_ID_PATTERN)),
    "app": (APP_MAX, re.compile(APP_PATTERN)),
})
PLUGIN_SLUG_FORM: Form = (PLUGIN_SLUG_MAX, re.compile(PLUGIN_SLUG_PATTERN))

# A package member's type -> the kinds its reference takes (K8): a seeded agent or an agent of the
# catalogue; a seeded Playbook or a Playbook of the catalogue; a skill by its name; a plugin by its
# slug; a model by its id; a tool by its Composio app.
MEMBER_REFS: Mapping[str, Tuple[str, ...]] = MappingProxyType({
    "agent": ("slug", "item"),
    "playbook": ("templateId", "item"),
    "skill": ("name",),
    "plugin": ("slug",),
    "llm": ("modelId",),
    "tool": ("app",),
})

# field -> the kinds it takes. A Playbook's agent is a seeded agent, an agent of the catalogue,
# or, when it is neither (an agent of the submitter's own workspace), a name to show.
FIELD_REFS: Mapping[str, Tuple[str, ...]] = MappingProxyType({
    PLAYBOOK_AGENT_REF: ("slug", "item", "name"),
    **{member_ref_field(member_type): kinds for member_type, kinds in MEMBER_REFS.items()},
})

# (field, kind) -> the target's form where it is narrower than the kind's own: a plugin member's
# slug is a plugin's (String(100), no '_'), not an agent's; a skill member's name, which the import
# looks up, holds no control character.
FIELD_FORMS: Mapping[Tuple[str, str], Form] = MappingProxyType({
    (member_ref_field("plugin"), "slug"): PLUGIN_SLUG_FORM,
    (member_ref_field("skill"), "name"): (REF_NAME_MAX, re.compile(SKILL_REF_NAME_PATTERN)),
})

# field -> the keys a reference there may carry beside its kind, each optional, with its form.
REF_QUALIFIERS: Mapping[str, Mapping[str, Form]] = MappingProxyType({
    member_ref_field("skill"): MappingProxyType({"package": PLUGIN_SLUG_FORM}),
})


def check_ref(value: Any, field: str, where: str) -> None:
    """Refuse ``value`` with 'ref' unless it is one reference of a kind ``field`` takes."""
    qualifiers = REF_QUALIFIERS.get(field, {})
    kinds = [key for key in value if key not in qualifiers] if isinstance(value, dict) else []
    if len(kinds) != 1:
        raise BundleError("ref", f"{where} is not a reference in the bundle format.")
    kind = kinds[0]
    if kind not in FIELD_REFS[field]:
        raise BundleError("ref", f"{where} is a kind of reference this field does not take.")
    form = FIELD_FORMS.get((field, kind), REF_KINDS[kind])
    if not _fits(form, value[kind]) or not all(_fits(qualifiers[key], value[key]) for key in value if key != kind):
        raise BundleError("ref", f"{where} names its target in a form the bundle format does not allow.")


def _fits(form: Form, target: Any) -> bool:
    """Whether ``target`` is text of ``form``: within its length and on its pattern."""
    longest, pattern = form
    if not isinstance(target, str) or not 0 < len(target) <= longest or not is_plain_text(target):
        return False
    return pattern is None or pattern.fullmatch(target) is not None
