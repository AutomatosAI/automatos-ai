"""FIFO admission with an overall limit and a per-workspace limit (owner, 2026-09-23).

A render lane runs at most two jobs at once overall and one per workspace;
every other job waits. Waiting jobs start in arrival order, except that a job
whose workspace is already at its limit is passed over for the next one that
can start, so one busy workspace never holds the others up.
"""

from __future__ import annotations

import asyncio
from collections import Counter, deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple


@dataclass
class _Ticket:
    key: str
    workspace: str
    granted: "asyncio.Future[None]"


class Lane:
    def __init__(self, max_running: int, max_per_workspace: int) -> None:
        if max_running < 1 or max_per_workspace < 1:
            raise ValueError("a lane needs room for at least one job")
        self._max_running = max_running
        self._max_per_workspace = max_per_workspace
        self._waiting: Deque[_Ticket] = deque()
        self._by_workspace: Counter = Counter()
        self._running = 0

    @property
    def running(self) -> int:
        return self._running

    def waiting(self) -> Tuple[str, ...]:
        return tuple(ticket.key for ticket in self._waiting)

    def position(self, key: str) -> Optional[int]:
        """1 for the next job in line; None when the job is not waiting."""
        for index, ticket in enumerate(self._waiting, start=1):
            if ticket.key == key:
                return index
        return None

    def submit(self, key: str, workspace: str) -> "asyncio.Future[None]":
        """Join the line. The future resolves when the job may start; it may already have."""
        ticket = _Ticket(key, workspace, asyncio.get_running_loop().create_future())
        self._waiting.append(ticket)
        self._pump()
        return ticket.granted

    def withdraw(self, key: str) -> None:
        """Leave the line without starting (the waiter was cancelled)."""
        self._waiting = deque(ticket for ticket in self._waiting if ticket.key != key)
        self._pump()

    def release(self, workspace: str) -> None:
        """A started job finished: free its slot and start whoever can go next."""
        self._running -= 1
        self._by_workspace[workspace] -= 1
        if self._by_workspace[workspace] <= 0:
            del self._by_workspace[workspace]
        self._pump()

    def _pump(self) -> None:
        still_waiting: Deque[_Ticket] = deque()
        for ticket in self._waiting:
            if ticket.granted.done():
                continue  # cancelled while waiting
            if self._running < self._max_running and self._by_workspace[ticket.workspace] < self._max_per_workspace:
                self._running += 1
                self._by_workspace[ticket.workspace] += 1
                ticket.granted.set_result(None)
            else:
                still_waiting.append(ticket)
        self._waiting = still_waiting
