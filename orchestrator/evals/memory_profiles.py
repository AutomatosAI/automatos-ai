"""Memory-profile eval: does a domain profile keep more of what its domain needs? (Wave 4, §5.6)

Three sample exchanges, one per seeded domain profile (``customer``, ``incident``, ``case``).
Each names the facts the next conversation would need (``needed``, each a set of words that must
appear together in one distilled fact, with alternatives) and what it should leave out
(``unwanted``: a ruled-out hypothesis, general background). Each exchange is distilled with the
``general`` prompt (today's) and with its domain profile, ``--runs`` times each, through the real
prompt builder and parser (``SmartMemoryManager._build_distill_prompt`` /
``_parse_distilled_facts``, ``modules.memory.memory_profiles.with_profile``).

Live only for the model: ``--live --model <OpenRouter model id>`` calls OpenRouter directly
(``OPENROUTER_API_KEY``); no platform usage or cost row is written. Without ``--live`` it checks
its own fixtures (every profile exists, every prompt builds with the profile in it) and prints the
prompt sizes. The number is published, never a gate (exit 0), like ``evals.memory_recall``.

Run (from orchestrator/):
    python -m evals.memory_profiles
    python -m evals.memory_profiles --live --model <id> --runs 3 --json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

Words = Tuple[str, ...]
Need = Tuple[Words, ...]  # alternatives: any one set of words, all in one fact


@dataclass(frozen=True)
class Exchange:
    """One sample chat exchange, the profile for its domain, and what distilling it should keep."""

    profile: str
    user: str
    assistant: str
    needed: Dict[str, Need]
    unwanted: Dict[str, Need]


EXCHANGES: Tuple[Exchange, ...] = (
    Exchange(
        profile="customer",
        user=("Hi, it's Priya from Northwind's finance team. Our March invoice was charged twice, can you refund "
              "the duplicate? Also we'd like to move to quarterly billing from April if that's possible. And please "
              "email me rather than calling, I'm rarely at my desk."),
        assistant=("Sorry about that, Priya. I've raised a refund for the duplicate March invoice (£480); it will "
                   "reach your card by 20 November. Northwind is on the Team plan, billed monthly in GBP; I've asked "
                   "billing to switch you to quarterly from 1 April and they'll confirm by Friday. I've noted that "
                   "you prefer email."),
        needed={
            "refund promised by 20 November": (("refund", "20 november"),),
            "quarterly billing from April, pending": (("quarterly", "april"),),
            "billing confirms by Friday": (("friday",),),
            "on the Team plan": (("team plan",),),
            "Priya prefers email": (("prefer", "email"), ("email", "rather than"), ("email", "not phone")),
            "Priya is Northwind's billing contact": (("priya", "finance"), ("priya", "billing")),
        },
        unwanted={"the apology": (("sorry",), ("apolog",))},
    ),
    Exchange(
        profile="incident",
        user=("Checkout's down again: 502s from the eu-west load balancer since about 14:05 UTC. I restarted the "
              "payment workers at 14:15 and it made no difference. We think it's the 2026.11.3 release."),
        assistant=("Rolled back 2026.11.3 at 14:38 UTC and checkout recovered at 14:40. The release dropped the "
                   "database connection pool to 5, which starved the payment workers. Let's freeze deploys until "
                   "the pool fix ships, and a postmortem is owed in the incident channel by 12 November. It wasn't "
                   "DNS: I checked that first."),
        needed={
            "502s from 14:05 UTC": (("14:05",),),
            "restarting the workers made no difference": (("restart", "no difference"), ("restart", "did not"),
                                                           ("restart", "didn't"), ("14:15",)),
            "rollback of 2026.11.3 restored checkout": (("roll", "2026.11.3"), ("roll", "14:38"), ("14:40",)),
            "cause: the pool dropped to 5": (("pool", "5"),),
            "deploy freeze until the fix": (("freeze",),),
            "postmortem owed by 12 November": (("postmortem", "12 november"),),
        },
        unwanted={"the ruled-out DNS check": (("dns",),)},
    ),
    Exchange(
        profile="case",
        user=("Following up on Hartwell v Copperfield. The lease started on 1 March 2024 for five years. "
              "Copperfield's solicitors sent the dilapidations schedule on 2 November claiming £38,000, and the "
              "deadline for our counter-schedule is 28 November. My client doesn't want to contest the service "
              "charge: they'd rather keep the relationship with the landlord."),
        assistant=("Noted. Generally, a dilapidations claim is capped at the fall in value of the landlord's interest "
                   "under section 18 of the Landlord and Tenant Act 1927. For this case, I'll draft the "
                   "counter-schedule against the £38,000 claim for your review by 21 November, ahead of the "
                   "28 November deadline."),
        needed={
            "the lease: 1 March 2024, five years": (("2024", "five"), ("2024", "5 year"), ("2024", "5-year")),
            "the £38,000 claim": (("38,000",), ("38000",), ("38k",)),
            "the schedule came on 2 November": (("2 november",),),
            "counter-schedule due 28 November": (("28 november",),),
            "client won't contest the service charge": (("service charge",),),
            "draft for review by 21 November": (("21 november",),),
        },
        unwanted={"general legal background (section 18)": (("section 18",), ("1927",))},
    ),
)


def _held(facts: Sequence[str], need: Need) -> bool:
    """Some fact holds every word of one of ``need``'s alternatives."""
    lowered = [f.lower() for f in facts]
    return any(all(w in fact for w in words) for words in need for fact in lowered)


def score(exchange: Exchange, facts: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """What one distillation kept of what was needed, and what it kept that it should not have."""
    texts = [str(f.get("fact", "")) for f in facts]
    kept = [name for name, need in exchange.needed.items() if _held(texts, need)]
    unwanted = [name for name, need in exchange.unwanted.items() if _held(texts, need)]
    return {"kept": kept, "kept_share": len(kept) / len(exchange.needed), "unwanted": unwanted, "facts": len(texts)}


def profile_named(slug: str) -> Any:
    """The seeded profile ``slug`` as distillation reads it."""
    from core.seeds.seed_memory_profiles import SEEDED_PROFILES
    from modules.memory.memory_profiles import Profile

    p = next(p for p in SEEDED_PROFILES if p["slug"] == slug)
    return Profile(slug=p["slug"], name=p["name"], focus=p["focus"], keep=tuple(p["keep"]),
                   skip=tuple(p["skip"]), examples=tuple(p["examples"]))


def prompts(exchange: Exchange) -> Dict[str, str]:
    """The distillation prompt for ``exchange``, as ``general`` and with its domain profile."""
    from consumers.chatbot.smart_memory import SmartMemoryManager
    from modules.memory.memory_profiles import with_profile

    general = SmartMemoryManager._build_distill_prompt(exchange.user, exchange.assistant)
    return {"general": general, exchange.profile: with_profile(general, profile_named(exchange.profile))}


def check_fixtures() -> Dict[str, Any]:
    """Offline: every exchange's profile is seeded and its prompt carries it."""
    from modules.memory.memory_profiles import PROFILE_HEADING

    sizes = {}
    for exchange in EXCHANGES:
        built = prompts(exchange)
        if PROFILE_HEADING not in built[exchange.profile] or PROFILE_HEADING in built["general"]:
            raise SystemExit(f"the {exchange.profile} prompt does not carry its profile")
        sizes[exchange.profile] = {arm: len(text) for arm, text in built.items()}
    return {"prompt_chars": sizes}


async def _distil(client: Any, model: str, prompt: str) -> List[Dict[str, Any]]:
    from consumers.chatbot.smart_memory import SmartMemoryManager

    response = await client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}])
    return SmartMemoryManager._parse_distilled_facts(response.choices[0].message.content or "") or []


async def measure_live(model: str, runs: int) -> Dict[str, Any]:
    """Each exchange distilled ``runs`` times per arm; the mean share of needed facts kept,
    how often an unwanted one was kept, and the facts per distillation."""
    from openai import AsyncOpenAI

    from config import config

    if not config.OPENROUTER_API_KEY:
        raise SystemExit("--live needs OPENROUTER_API_KEY")
    client = AsyncOpenAI(api_key=config.OPENROUTER_API_KEY, base_url=config.OPENROUTER_BASE_URL)
    results: Dict[str, Any] = {}
    for exchange in EXCHANGES:
        arms = {}
        for arm, prompt in prompts(exchange).items():
            scored = [score(exchange, await _distil(client, model, prompt)) for _ in range(runs)]
            arms["profile" if arm != "general" else "general"] = _summary(exchange, scored)
        results[exchange.profile] = arms
    return results


def _summary(exchange: Exchange, scored: List[Dict[str, Any]]) -> Dict[str, Any]:
    missed = sorted({name for s in scored for name in exchange.needed if name not in s["kept"]})
    return {
        "kept_share": round(statistics.mean(s["kept_share"] for s in scored), 2),
        "unwanted_rate": round(statistics.mean(bool(s["unwanted"]) for s in scored), 2),
        "facts": round(statistics.mean(s["facts"] for s in scored), 1),
        "sometimes_missed": missed,
    }


def render(result: Dict[str, Any]) -> str:
    if "live" not in result:
        return "\n".join(f"{domain:<9} prompt chars: {sizes}" for domain, sizes in result["prompt_chars"].items())
    lines = [f"{'domain':<9} {'arm':<8} {'needed kept':>11} {'unwanted':>9} {'facts':>6}  sometimes missed"]
    for domain, arms in result["live"].items():
        for arm, s in arms.items():
            lines.append(f"{domain:<9} {arm:<8} {s['kept_share']:>11.0%} {s['unwanted_rate']:>9.0%} {s['facts']:>6}  "
                         f"{', '.join(s['sometimes_missed']) or '-'}")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--live", action="store_true", help="distil on OpenRouter and score both arms")
    parser.add_argument("--model", help="the OpenRouter model id for --live")
    parser.add_argument("--runs", type=int, default=3, help="distillations per exchange and arm for --live")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)
    if args.live and not args.model:
        parser.error("--live needs --model")
    result = check_fixtures()
    if args.live:
        result["live"] = asyncio.run(measure_live(args.model, args.runs))
        result["model"], result["runs"] = args.model, args.runs
    sys.stdout.write((json.dumps(result, indent=2) if args.json else render(result)) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
