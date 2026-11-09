"""Lean-turn eval: a lean agent's prompt overhead and time to first token (Wave 2, §5.3).

PRD-274 ("Automatos for products") sets two targets for a lean turn:
  * **prompt overhead within 2,000 tokens** of the agent's own context: what the
    platform adds beyond the agent's persona, description, recalled memory and the
    descriptions of the tools it is granted;
  * **time to first token within 1.5 s** of the model's own on the same request.

Offline (default, CI): the real ``LEAN`` context build for a sample agent on the
database the orchestrator is configured with, its granted tools described as a turn
describes them, and the same agent's ``CHATBOT`` build for comparison. It measures
tokens and the build's own time; no model is called.

Live (``--live --model <OpenRouter model id>``) adds two things:
  * **memory recall**: the sample customer's facts are filed in a fresh workspace on the
    configured Qdrant with real embeddings (``evals.memory_pipeline.live_service``, a
    collection made for the run and dropped after it), so the lean build recalls through
    the Context Router as a turn does: its build time includes the embedding and the
    search, and the eval checks the two facts the question needs were recalled;
  * the model's time to first token, streamed straight from OpenRouter
    (``OPENROUTER_API_KEY``; no platform usage is recorded), for the bare request (the
    agent's own context and the question) and for the lean turn's request, ``--runs`` times
    each, alternating. The lean turn's time to first token is its build time plus the
    model's on its request.

Exit 1 when a measured target is missed, so CI can hold the overhead target.

Run (from orchestrator/):
    python -m evals.lean_turn
    python -m evals.lean_turn --live --model <id> --runs 5 --json
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import statistics
import sys
import time
import uuid
from types import SimpleNamespace
from typing import Any, AsyncIterator, Dict, List, Optional

OVERHEAD_TARGET_TOKENS = 2_000
FIRST_TOKEN_TARGET_MS = 1_500

PERSONA = (
    "You are Quill, the support agent for Acme Billing. You answer customers' questions about their "
    "invoices, payment dates and plan changes, in two or three short sentences, and you say so plainly "
    "when you do not know."
)
DESCRIPTION = "Answers Acme Billing's customers inside the billing portal."
QUESTION = "When is my next invoice due, and can I move it to the first of the month?"
GROUPS = ["reports", "playbooks"]
# The customer's facts, filed for --live; the question needs the first two.
MEMORIES = (
    ("due-date", "The customer's invoices fall due on the 15th of each month."),
    ("move-date", "A customer may move their invoice due date once a year, to the 1st or the 15th."),
    ("plan", "The customer is on the Team plan, billed monthly in GBP."),
    ("contact", "The customer's billing contact is Priya in their finance team."),
    ("vat", "The customer updated their VAT number in March."),
    ("portal", "The billing portal's dark mode shipped in October."),
)
RECALL_GOLD = ("due-date", "move-date")


def sample_agent(workspace_id: str) -> SimpleNamespace:
    """A lean support agent with a custom persona and two tool groups (no database row)."""
    return SimpleNamespace(
        id=None, name="Quill", agent_type="support", description=DESCRIPTION, workspace_id=workspace_id,
        is_system_agent=False, slug="quill", use_custom_persona=True, custom_persona_prompt=PERSONA,
        persona=None, configuration={"runtime_profile": "lean", "session_tool_groups": GROUPS},
    )


def _tokens(text: str) -> int:
    from core.context_guard import count_tokens

    return count_tokens(text or "")


async def _build(mode: Any, agent: Any) -> Any:
    from core.database.database import SessionLocal
    from modules.context import ContextService

    db = SessionLocal()
    try:
        return await ContextService(db).build_context(
            mode=mode, agent=agent, workspace_id=agent.workspace_id,
            messages=[{"role": "user", "content": QUESTION}], query=QUESTION,
        )
    finally:
        db.close()


async def lean_request(agent: Any) -> Dict[str, Any]:
    """The lean turn's request for ``agent``: messages, the agent's own text, build time."""
    from consumers.chatbot import lean_turn
    from modules.context import ContextMode

    started = time.perf_counter()
    built = await _build(ContextMode.LEAN, agent)
    surface = lean_turn.lean_tool_surface(lean_turn.granted_names(agent), is_super_admin=False)
    messages = lean_turn.with_granted_actions([{"role": "system", "content": built.system_prompt}, *built.messages], surface)
    prep_ms = (time.perf_counter() - started) * 1000
    tools_block = messages[0]["content"][len(built.system_prompt):]
    own = "\n".join([PERSONA, DESCRIPTION, built.memory_context or "", tools_block])
    return {"messages": messages, "own": own, "prep_ms": prep_ms, "surface": surface,
            "memory": built.memory_context or ""}


@contextlib.asynccontextmanager
async def live_memory() -> AsyncIterator[str]:
    """A fresh workspace holding ``MEMORIES`` on the configured Qdrant with real embeddings,
    installed as the process's memory service for the block; the collection is dropped and
    the service put back after it."""
    from evals.memory_pipeline import FACT_IMPORTANCE, LIVE_DIMENSION, LIVE_EMBEDDING_MODEL, live_service
    from modules.memory.unified_memory_service import UnifiedMemoryService

    service, drop = live_service(LIVE_EMBEDDING_MODEL, LIVE_DIMENSION)
    workspace = str(uuid.uuid4())
    before = UnifiedMemoryService._instance
    UnifiedMemoryService._instance = service
    try:
        for memory_id, text in MEMORIES:
            stored = await service.store_long_term(
                workspace_id=workspace, content=text,
                metadata={"memory_id": memory_id, "type": "fact", "importance": FACT_IMPORTANCE},
            )
            if not stored.get("success"):
                raise RuntimeError(f"the store refused {memory_id}: {stored.get('error')}")
        yield workspace
    finally:
        UnifiedMemoryService._instance = before
        await drop()


def recalled(memory_context: str) -> List[str]:
    """The ``MEMORIES`` ids whose text the build's memory block carries."""
    return [memory_id for memory_id, text in MEMORIES if text in (memory_context or "")]


def measure_offline(workspace_id: str) -> Dict[str, Any]:
    """Token counts and build times for the sample agent's lean and full turns."""
    return asyncio.run(measure(workspace_id))


async def measure_with_live_memory() -> Dict[str, Any]:
    """``measure`` on a workspace holding the customer's facts, on one event loop (the
    store's client belongs to the loop that made it)."""
    async with live_memory() as workspace:
        return await measure(workspace)


async def measure(workspace_id: str) -> Dict[str, Any]:
    """Token counts and build times for the sample agent's lean and full turns."""
    from modules.context import ContextMode

    agent = sample_agent(workspace_id)
    lean = await lean_request(agent)
    started = time.perf_counter()
    full = await _build(ContextMode.CHATBOT, agent)
    full_ms = (time.perf_counter() - started) * 1000
    system = lean["messages"][0]["content"]
    own_tokens = _tokens(lean["own"])
    return {
        "lean_system_tokens": _tokens(system),
        "own_tokens": own_tokens,
        "overhead_tokens": _tokens(system) - own_tokens,
        "full_system_tokens": _tokens(full.system_prompt),
        "lean_prep_ms": round(lean["prep_ms"]),
        "full_prep_ms": round(full_ms),
        "lean_tools": len(lean["surface"]),
        "recalled": recalled(lean["memory"]),
        "messages": lean["messages"],
    }


async def _first_token_ms(client: Any, model: str, messages: List[Dict[str, Any]]) -> float:
    started = time.perf_counter()
    stream = await client.chat.completions.create(model=model, messages=messages, stream=True, max_tokens=64)
    elapsed: Optional[float] = None
    async for chunk in stream:
        delta = chunk.choices[0].delta if chunk.choices else None
        if elapsed is None and delta is not None and (delta.content or getattr(delta, "reasoning", None)):
            elapsed = (time.perf_counter() - started) * 1000
    return elapsed if elapsed is not None else (time.perf_counter() - started) * 1000


async def measure_live(model: str, runs: int, lean_messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The model's median time to first token on the bare and the lean request."""
    from openai import AsyncOpenAI

    from config import config

    if not config.OPENROUTER_API_KEY:
        raise SystemExit("--live needs OPENROUTER_API_KEY")
    client = AsyncOpenAI(api_key=config.OPENROUTER_API_KEY, base_url=config.OPENROUTER_BASE_URL)
    bare = [{"role": "system", "content": f"{PERSONA}\n\n{DESCRIPTION}"}, {"role": "user", "content": QUESTION}]
    times: Dict[str, List[float]] = {"bare": [], "lean": []}
    for _ in range(runs):
        times["bare"].append(await _first_token_ms(client, model, bare))
        times["lean"].append(await _first_token_ms(client, model, lean_messages))
    return {f"model_first_token_ms_{name}": round(statistics.median(values)) for name, values in times.items()}


def run(live: bool, model: Optional[str], runs: int, workspace_id: str) -> Dict[str, Any]:
    """The measurements and each target's verdict."""
    if live:
        result = asyncio.run(measure_with_live_memory())
        result["recall_met"] = set(RECALL_GOLD) <= set(result["recalled"])
    else:
        result = measure_offline(workspace_id)
    messages = result.pop("messages")
    result["overhead_target_tokens"] = OVERHEAD_TARGET_TOKENS
    result["overhead_met"] = result["overhead_tokens"] <= OVERHEAD_TARGET_TOKENS
    if live:
        result.update(asyncio.run(measure_live(model, runs, messages)))
        lean_ttft = result["lean_prep_ms"] + result["model_first_token_ms_lean"]
        result["lean_first_token_ms"] = lean_ttft
        result["first_token_over_model_ms"] = lean_ttft - result["model_first_token_ms_bare"]
        result["first_token_target_ms"] = FIRST_TOKEN_TARGET_MS
        result["first_token_met"] = result["first_token_over_model_ms"] <= FIRST_TOKEN_TARGET_MS
    return result


def render(result: Dict[str, Any]) -> str:
    lines = [
        f"lean prompt        {result['lean_system_tokens']:>6} tokens (full: {result['full_system_tokens']})",
        f"agent's own        {result['own_tokens']:>6} tokens",
        f"overhead           {result['overhead_tokens']:>6} tokens (target <= {OVERHEAD_TARGET_TOKENS}): "
        f"{'met' if result['overhead_met'] else 'MISSED'}",
        f"build              {result['lean_prep_ms']:>6} ms (full: {result['full_prep_ms']} ms)",
    ]
    if "recall_met" in result:
        lines.append(f"memory recalled    {', '.join(result['recalled']) or 'nothing'} "
                     f"(needs {', '.join(RECALL_GOLD)}): {'met' if result['recall_met'] else 'MISSED'}")
    if "first_token_met" in result:
        lines.append(
            f"first token        {result['lean_first_token_ms']:>6} ms, {result['first_token_over_model_ms']} ms over "
            f"the model's {result['model_first_token_ms_bare']} ms (target <= {FIRST_TOKEN_TARGET_MS}): "
            f"{'met' if result['first_token_met'] else 'MISSED'}"
        )
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    from config import config

    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--live", action="store_true", help="also time the model's first token on OpenRouter")
    parser.add_argument("--model", help="the OpenRouter model id for --live")
    parser.add_argument("--runs", type=int, default=5, help="requests per side for --live")
    parser.add_argument("--workspace", default=config.DEFAULT_WORKSPACE_ID, help="the workspace the build reads")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)
    if args.live and not args.model:
        parser.error("--live needs --model")
    result = run(args.live, args.model, args.runs, args.workspace)
    sys.stdout.write((json.dumps(result, indent=2) if args.json else render(result)) + "\n")
    met = result["overhead_met"] and result.get("first_token_met", True) and result.get("recall_met", True)
    return 0 if met else 1


if __name__ == "__main__":
    raise SystemExit(main())
