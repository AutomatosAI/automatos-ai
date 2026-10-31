"""PRD-270 W2 (K8) — the reference grammar of an item bundle.

A reference is an object with exactly one key, its kind, naming one target:

* ``{"slug": s}`` an agent the local edition seeds, by its slug;
* ``{"templateId": s}`` a Playbook the local edition seeds, by its template id;
* ``{"item": externalId}`` another item of the same catalogue, of the type the field needs;
* ``{"name": s}`` a name shown to the user only: it never resolves to a row;
* ``{"modelId": s}`` a model, by its model id;
* ``{"app": APP}`` a Composio app, by its name.

Each field that holds a reference takes only some kinds (``FIELD_REFS``). A reference of another
kind, of another form, or with a target off its kind's pattern is refused as 'ref'. Standard
library only: the hosted export and the local import share this package.
"""
from __future__ import annotations

import re
from types import MappingProxyType
from typing import Any, Mapping, Optional, Tuple

from services.marketplace_bundles.errors import BundleError
from services.marketplace_bundles.schema import (
    AGENT_SLUG_MAX, AGENT_SLUG_PATTERN, APP_MAX, APP_PATTERN, EXTERNAL_ID_LENGTH, EXTERNAL_ID_PATTERN,
    MODEL_ID_MAX, MODEL_ID_PATTERN, PLAYBOOK_AGENT_REF, REF_NAME_MAX, TEMPLATE_ID_MAX, TEMPLATE_ID_PATTERN,
    is_plain_text,
)

# kind -> (the target's longest length, its pattern or None for any text)
REF_KINDS: Mapping[str, Tuple[int, Optional[re.Pattern[str]]]] = MappingProxyType({
    "slug": (AGENT_SLUG_MAX, re.compile(AGENT_SLUG_PATTERN)),
    "templateId": (TEMPLATE_ID_MAX, re.compile(TEMPLATE_ID_PATTERN)),
    "item": (EXTERNAL_ID_LENGTH, re.compile(EXTERNAL_ID_PATTERN)),
    "name": (REF_NAME_MAX, None),
    "modelId": (MODEL_ID_MAX, re.compile(MODEL_ID_PATTERN)),
    "app": (APP_MAX, re.compile(APP_PATTERN)),
})

# field -> the kinds it takes. A Playbook's agent is a seeded agent, an agent of the catalogue,
# or, when it is neither (an agent of the submitter's own workspace), a name to show.
FIELD_REFS: Mapping[str, Tuple[str, ...]] = MappingProxyType({
    PLAYBOOK_AGENT_REF: ("slug", "item", "name"),
})


def check_ref(value: Any, field: str, where: str) -> None:
    """Refuse ``value`` with 'ref' unless it is one reference of a kind ``field`` takes."""
    if not isinstance(value, dict) or len(value) != 1:
        raise BundleError("ref", f"{where} is not a reference in the bundle format.")
    kind, target = next(iter(value.items()))
    if kind not in FIELD_REFS[field]:
        raise BundleError("ref", f"{where} is a kind of reference this field does not take.")
    if not _fits(kind, target):
        raise BundleError("ref", f"{where} names its target in a form the bundle format does not allow.")


def _fits(kind: str, target: Any) -> bool:
    """Whether ``target`` is text its kind allows: within its length and on its pattern."""
    longest, pattern = REF_KINDS[kind]
    if not isinstance(target, str) or not 0 < len(target) <= longest or not is_plain_text(target):
        return False
    return pattern is None or pattern.fullmatch(target) is not None
