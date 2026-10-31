"""PRD-268 US-C22b — the stand-ins the calibration route tests install (``test_prd268w3_calibration_routes.py``).

* ``FakeModel``: the calibration's one model call (installed at ``build.calibration_llm``). It
  answers one line citing the Deliverable and Pia's chat by the refs its prompt gave them, unless
  the test gives it a failure, a content or a delay.
* ``FakeMemory``: the memory service (installed at ``memory.memory_service``). It records each
  ``store_long_term`` call and answers ``answer``, or each of ``answers`` in turn.
"""
from __future__ import annotations

import json
import time
from types import SimpleNamespace as NS

LINE = "You review pricing for clients in Lisbon"
STORED = {"success": True, "id": "point-1"}


def answer_citing(prompt: str) -> str:
    """The model's answer: ``LINE``, citing the Deliverable and Pia's chat by their refs in ``prompt``."""
    refs = {row["title"]: row["ref"] for row in (json.loads(line) for line in prompt.splitlines()
                                                 if line.startswith("{"))}
    return json.dumps({"lines": [{"text": LINE, "sources": [refs["Q3 pricing review"], refs["Pricing for Lisbon"]]}],
                       "offers": []})


class FakeModel:
    """The model: records each call and answers ``answer_citing``, or the test's failure or content."""

    def __init__(self, failure=None, content=None, delay=0.0):
        self.failure, self.content, self.delay, self.calls = failure, content, delay, []

    def generate_response_sync(self, messages):
        self.calls.append(messages)
        time.sleep(self.delay)
        if self.failure is not None:
            raise self.failure
        return NS(content=self.content if self.content is not None else answer_citing(messages[1]["content"]))


class FakeMemory:
    """The memory service: records each ``store_long_term`` call and answers ``answer``, or each
    of ``answers`` in turn."""

    def __init__(self, answer=None, answers=()):
        self.answer, self.answers, self.calls = answer or STORED, list(answers), []

    async def store_long_term(self, **kwargs):
        self.calls.append(kwargs)
        return self.answers[len(self.calls) - 1] if len(self.calls) <= len(self.answers) else self.answer
