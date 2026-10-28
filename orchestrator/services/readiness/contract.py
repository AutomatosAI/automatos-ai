"""PRD-268 §6.11.3 — ``Readiness@1``, the shape every client reads.

```ts
type Readiness = { v: 1; checked_at: string; items: ReadinessItem[] }
type ReadinessItem = {
  id: string; label: string; detail: string
  state: 'ready' | 'not_checked' | 'needs_setup'
  fix?: { label: string; href: string }          // an app path, never an outside URL
  needed_by: Array<'brief' | 'voice' | 'push' | 'check_in'>
}
```

A fix whose href is not an app path (``/…``, never ``//…`` or ``/\\…``, which a browser
reads as another host) cannot be built.
"""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, field_validator

APP_PATH_ROOT = "/"
_OTHER_HOST_PREFIXES = ("//", "/\\")

State = Literal["ready", "not_checked", "needs_setup"]
NeededBy = Literal["brief", "voice", "push", "check_in"]


class ReadinessFix(BaseModel):
    """What a person can do about an item, and the app path that does it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    label: str
    href: str

    @field_validator("href")
    @classmethod
    def _an_app_path(cls, value: str) -> str:
        if not value.startswith(APP_PATH_ROOT) or value.startswith(_OTHER_HOST_PREFIXES):
            raise ValueError("a fix opens an app path")
        return value


class ReadinessItem(BaseModel):
    """One thing Auto depends on, and whether it could be checked."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    label: str
    detail: str
    state: State
    fix: Optional[ReadinessFix] = None
    needed_by: List[NeededBy]


class Readiness(BaseModel):
    """``Readiness@1``: when the items were checked, and each item."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    v: Literal[1]
    checked_at: str
    items: List[ReadinessItem]
