# To: Andy — On Automatos and FUCK IT MODE

**From:** Gerard
**Re:** Why your missions feel "good but not great" and how we're about to fix that

---

## TL;DR

You've been driving Automatos in **Toyota Corolla mode** when there's a Ferrari in the garage you didn't know about. Tonight I'm wiring up a tier selector on missions. Pick **FUCK IT MODE**, pay the bill, see what Automatos actually does when nobody's holding it back. For you specifically — Shopify pilot, first real customer — I want you running it on the deliverables that matter so you can tell me whether the ceiling is high enough.

---

## Why your missions have been a bit meh

Every mission you've ever run has been on the same set of cost-safe defaults, hardcoded across three files:

- **Model:** `google/gemini-2.5-flash` — fast, cheap, decent. Not deep. Definitely not Opus.
- **Output cap:** 2,000 tokens per LLM call. ~1,500 words. A 19-page report needs ~10× that.
- **Per-agent context window:** 4,000 tokens. Claude.ai gets 200,000. 50× difference.
- **Tool iterations:** 10 max per task. Researcher gives up after 10 searches — can't actually go deep.
- **Decomposition:** Mandatory. Every mission gets fragmented across 4–6 specialist agents.

The defaults exist for a real reason: **multi-tenant cost safety**. A runaway Opus mission can burn $10 in a single agent-loop. We can't ship that as the global default. The cost is that the platform plays it safe — at the expense of being underwhelming on deliverables that need *more*.

---

## The thing I undersold you on: we already have a shared brain

Here's the part Claude.ai literally cannot do, and that you may not have noticed your missions are doing already:

**Field Memory** (PRD-108, live since late March). Every multi-agent mission gets its own **semantic vector field** — a Qdrant-backed shared memory space, one per mission. Agents inject findings into it (`platform_field_inject`) and query it (`platform_field_query`) by meaning, not by message-passing. The researcher writes "EU AI Act Article 6 requires X". The strategist later queries "regulatory constraints on rollout" — and gets the researcher's finding back, ranked by semantic relevance.

It's not the telephone game. It's not "the writer agent only sees what the orchestrator paraphrased forward". It's a shared brain — every agent reads from and writes to the same memory:

- **Patterns decay** over time (half-life ~7h) unless they get accessed
- **Frequently-used findings** get reinforced (Hebbian-style — bonus per access, capped at 2× original strength)
- **Mission-scoped** — each mission has its own isolated field, destroyed when the mission ends
- **Cross-agent** by default — a writer can pull what a researcher learned 30 minutes earlier, not as a summary but as the original semantic chunk

This is the architectural piece that makes "multi-agent" *not* mean "fragmentation". It's what Automatos has that Claude.ai doesn't — multiple specialist brains coordinating around a shared semantic field, instead of one mega-context that gets cut off at 200K.

**Honest disclaimer:** the formal benchmark we're running on this (target: 43% → 86% context coverage on synthesis tasks) hasn't been executed yet. The mechanism is wired and live; the empirical gate is still open. You're going to be one of the people whose mission outputs prove or disprove it, which is part of why I want you on FUCK IT MODE.

---

## The new tier model (shipping next)

Four modes. One dropdown on mission creation. Workspace default settable in Settings → Orchestrator.

| Tier | Vibe | Model | Output | Context | Tool calls | Field memory | Cost |
|---|---|---|---|---|---|---|---|
| **Economy** | Corner shop | Gemini Flash | 2K | 4K | 10 | Default decay | $0.05–$0.30 |
| **Balanced** | Default | Sonnet 4.6 | 4K | 16K | 20 | Default decay | $0.30–$2 |
| **Premium** | Get it right | Sonnet 4.6 | 8K | 50K | 30 | Slower decay, larger queries | $2–$8 |
| **FUCK IT MODE** 🚀 | To the moon | **Opus 4.6** | **16K** | **200K** | **50** | **Long half-life, top-k=50** | $8–$30+ |

FUCK IT MODE has two flavours, picked at mission creation:

- **🚀 Squad** — multi-agent + field memory + every agent on Opus. The Automatos-native answer: 4–6 specialist brains coordinating through the semantic field, each one operating at Claude.ai-Pro-class settings. The thing Claude.ai literally cannot replicate.
- **🧠 Solo** — single agent, no decomposition, Opus, 200K context, 50 tool iterations. Direct one-to-one match for a Claude.ai Pro session, with all your tools and integrations attached. Use this for one-shot deliverables (reports, websites, deep research memos).

Pick Solo when one mega-brain is the shape of the problem (a 20-page strategy doc). Pick Squad when the work genuinely benefits from specialists collaborating (researcher → analyst → strategist → writer, all sharing the field).

---

## What FUCK IT MODE actually unlocks for you

You're a power user. You know the platform. What you haven't seen is what it does when nothing's gated:

1. **Reports become deliverables, not summaries.** A "research my top 3 Shopify competitors and produce a strategy memo" mission goes from a 4-page bullet-point thing to an actual 15–20 page document with cited sources, comparison tables, and a real recommendation section. 16K output × Opus reasoning × 200K context = different category of output.

2. **Deep research goes deep.** 50 tool iterations means the researcher can actually search → read → cross-reference → search again — instead of giving up after 10 and writing whatever it half-found.

3. **Squad mode does what no solo Claude session can.** Researcher pulls 200 sources. Analyst groups them by theme. Strategist scores each theme against your goals. Writer synthesises. All sharing the same semantic field — every agent sees every previous agent's primary findings, not paraphrases. Opus on every link in the chain. This is the configuration you can't get out of any consumer AI tool.

4. **Solo mode matches Claude.ai Pro.** Same model, same context, same depth — but inside *your* workspace, with *your* tools (Shopify admin API, Klaviyo, GitHub, Drive, Slack), reading *your* documents, writing to *your* board, leaving deliverables that persist.

---

## What Automatos still does that Claude.ai *can't* — even on FUCK IT MODE

This is the part that matters most for the Shopify pilot. FUCK IT MODE makes Automatos competitive on Claude.ai's home turf. But the moat is everything Claude.ai *literally cannot do*:

- **Run while you sleep.** Playbooks + cron triggers. "Every Monday 6am, audit my Shopify store, write a report, raise board tasks." Claude.ai can't.
- **Coordinate across days.** Mission state persists. A research mission can pause for human input on Tuesday and resume Wednesday with full memory.
- **Tool integration.** Composio (3,000+ tools), GitHub, Slack, Shopify admin, Google Drive, Klaviyo. The agent can *act*, not just write.
- **Knowledge graph + RAG.** Every doc you upload, every report it generates, every conversation — searchable, surfaced, used in future missions. Claude.ai forgets the moment you close the tab.
- **Multi-agent + shared field memory** (the bit I just explained). When the work genuinely benefits from specialists, you get specialists who actually share a brain.
- **Reports, deliverables, board tasks, missions, agents** — persistent objects that outlive any session. They live in *your* workspace, not someone else's chat history.

So the right mental model:
> **Claude.ai = the world's best contractor.**
> **Automatos = the company you build to run your business 24/7, hire that contractor when needed, and remember everything.**

FUCK IT MODE is the "hire that contractor — and put them in a room with three more contractors who all share a brain" button.

---

## What I'd like you to do once it ships

1. **Set workspace default to Balanced.** Honest middle for everyday work.
2. **Use FUCK IT MODE on three things this week:**
   - One **Solo** run: competitor analysis / strategy memo / 3–6 month plan. Claude.ai-class, single deliverable.
   - One **Squad** run: deep multi-stage research that benefits from specialists (e.g. "audit my whole Shopify funnel, recommend 10 changes, prioritise by ROI, draft the implementation plan for the top 3"). Squad shines when there are clear hand-offs *and* shared knowledge.
   - One **end-to-end build**: Shopify section copy + page structure + email sequence. Squad, FUCK IT MODE, leave it running.
3. **Tell me where the ceiling is.** Two things I need from you specifically:
   - Where Solo still feels weaker than a Claude.ai Pro session (what's the gap?)
   - Whether Squad mode actually feels qualitatively different from Solo — i.e. is the field memory pulling its weight, or is it just expensive Solo-with-extra-steps?
4. **Watch the bill.** Per-mission cost cap defaults to $50, overridable. You won't accidentally moon-shot a $300 mission. But you'll see real numbers — that's the point of the tier system.

---

## Status

- Tier system shipping this week. ~1 day of plumbing across 4 files. No DB migration.
- Default workspace tier on ship will be **Balanced**. Your workspace gets **FUCK IT MODE pre-enabled** as default — first-pilot privilege.
- Field memory is live now — your existing missions are already using it; you just haven't been able to see it work because Flash + 4K context doesn't have enough headroom for the field to matter. On Opus + 200K it will.
- The destructive smoke-test that's been wiping my Irish CTO every night at 02:00 UTC is sorted (separate story, ask me sometime, it's funny).

---

Talk Monday. Tell me what to test first.

— G
