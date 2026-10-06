"""Split the seeded templates' renders across CI jobs and processes (scripts/ci).

Gerard, 6 Oct: "why is CI taking 45 mins". ``social_template_previews.py`` renders
about 180 bundles one after another against one media-render (about 15 s each, 43
minutes). The CI job now runs it as several shards, each against its own
media-render: ``--shard i/N`` keeps every N-th render unit, starting at the i-th.

A unit is what must render together: a video (its preview, its probe, its
lengths), an image template (every size, then its probe), one Infographic
binding (every size), the heading-font Title card, one night-kit render. Every
shard walks the same units in the same order, so each unit lands in exactly one
shard and the N shards together render every unit once: no check is skipped.
"""
from __future__ import annotations

from typing import Iterable, Iterator, Tuple, TypeVar

T = TypeVar("T")


def parse(text: str) -> Tuple[int, int]:
    """``"i/N"`` as ``(i, N)``: 0 <= i < N."""
    try:
        index, count = (int(part) for part in str(text).split("/"))
    except ValueError as exc:
        raise ValueError(f"--shard takes i/N (like 0/4), not {text!r}") from exc
    if count < 1 or not 0 <= index < count:
        raise ValueError(f"--shard {text}: needs 0 <= i < N")
    return index, count


class Shard:
    """Which render units this run takes; every unit by default (``0/1``)."""

    def __init__(self, index: int = 0, count: int = 1) -> None:
        self.index, self.count, self.seen = index, count, 0
        self.taken = 0

    def configure(self, text: str) -> None:
        self.index, self.count = parse(text)
        self.seen = self.taken = 0

    def take(self) -> bool:
        """The next unit, in the walk every shard shares: whether it is this shard's."""
        mine = self.seen % self.count == self.index
        self.seen += 1
        self.taken += int(mine)
        return mine

    def mine(self, units: Iterable[T]) -> Iterator[T]:
        """This shard's units out of ``units``, in order."""
        for unit in units:
            if self.take():
                yield unit

    def label(self) -> str:
        return f"shard {self.index + 1} of {self.count}: {self.taken} of {self.seen} render units"


__all__ = ["Shard", "parse"]
