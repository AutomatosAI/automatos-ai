from __future__ import annotations

from typing import Dict, List, Optional, Literal
from pydantic import BaseModel


SlotName = Literal["INSTRUCTION", "MEMORY", "RETRIEVAL", "CODE", "TOOLS", "CONSTRAINTS"]


class Slot(BaseModel):
    name: SlotName
    content: str = ""
    weight: float = 1.0
    enabled: bool = True
    max_chars: Optional[int] = None


class ContextPolicy(BaseModel):
    policy_id: str
    slots: Dict[SlotName, Slot]
    max_total_chars: int = 12000
    version: int = 1

    @classmethod
    def default(cls, policy_id: str) -> "ContextPolicy":
        return cls(
            policy_id=policy_id,
            slots={
                "INSTRUCTION": Slot(name="INSTRUCTION", weight=1.0, enabled=True),
                "MEMORY": Slot(name="MEMORY", weight=0.8, enabled=True, max_chars=1500),
                "RETRIEVAL": Slot(name="RETRIEVAL", weight=1.0, enabled=True, max_chars=5000),
                "CODE": Slot(name="CODE", weight=0.9, enabled=True, max_chars=3000),
                "TOOLS": Slot(name="TOOLS", weight=0.6, enabled=True, max_chars=800),
                "CONSTRAINTS": Slot(name="CONSTRAINTS", weight=0.7, enabled=True, max_chars=600),
            },
        )


class AssembledContext(BaseModel):
    policy_id: str
    prompt: str
    stats: Dict[str, int]
    slots_used: List[SlotName]


def _truncate_to(s: str, n: Optional[int]) -> str:
    if n is None or len(s) <= n:
        return s
    return s[: max(0, n - 3)] + "..."


def assemble_context(
    q: str,
    policy: ContextPolicy,
    slot_values: Dict[SlotName, str],
    header_fmt: str = "## {name}\n",
) -> AssembledContext:
    for k, v in slot_values.items():
        if k in policy.slots:
            policy.slots[k].content = v or ""
    ordered = sorted(
        [s for s in policy.slots.values() if s.enabled], key=lambda s: s.weight, reverse=True
    )
    parts: List[str] = []
    char_budget = policy.max_total_chars
    stats: Dict[str, int] = {}
    for s in ordered:
        text = _truncate_to(s.content, s.max_chars)
        block = (header_fmt.format(name=s.name) + text).strip()
        if char_budget - len(block) < 0:
            block = block[: max(0, char_budget)]
        used = len(block)
        char_budget -= used
        parts.append(block)
        stats[s.name] = used
        if char_budget <= 0:
            break
    uq = f"\n\n## USER_QUERY\n{q}".strip()
    if char_budget - len(uq) > 0:
        parts.append(uq)
        stats["USER_QUERY"] = len(uq)
    else:
        parts.append(uq[: max(0, char_budget)])
        stats["USER_QUERY"] = max(0, char_budget)
    assembled = "\n\n".join([p for p in parts if p])
    stats["TOTAL"] = sum(stats.values())
    slots_used = [s.name for s in ordered if stats.get(s.name, 0) > 0]
    return AssembledContext(policy_id=policy.policy_id, prompt=assembled, stats=stats, slots_used=slots_used)


