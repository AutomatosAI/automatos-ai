# Customer nights — the plan (PRD-247, revision after night 1)

**Gerard, 2026-09-18:** "Now we are on the same page and running REAL testing on the system… a plan to run all packs, gather and record, fix bugs and build as much data as we can… until we are comfortable a user can use 100% of the system via Auto or manually in the UI, via API calls, MCP — just a beautiful smooth system."

The engine is the customer night: a headless Claude Code session plays the owner of a made-up company in the operator's own local workspace, with the real tools, for hours (`scripts/ralph/overnight-customer.sh`, brief in `scripts/ralph/customer-night/`). Night 1 ran on 2026-09-18 from 18:59: five agents hired, 60 tickets by iteration 2, a website built by a Claude session, documents uploaded, a mission, Auto delegating, questions answered — and a findings list a customer would write. This plan turns one night into a programme.

## The loop (every night, every morning)

```
evening   preflight → launch the night on its theme → runner watch
night     persona works the product; sessions do the tickets; DB records everything
morning   read MORNING-REPORT + DIARY + OBSERVER → run the ledger → grade a sample
          → triage each finding: harness | product bug | product gap | design decision
          → tickets / PRs (small, CI-gated) → merge into the test branch
next      re-run the same theme when its fixes land (regression night) → compare
```

Three documents every night writes: `MORNING-REPORT.md` (the customer's verdict), `DIARY.md` (every action, how it felt), `OBSERVER.md` (what the supervising session saw that the persona could not — UI, code, DB — with file:line). One document the morning writes: the **ledger** (numbers, from the DB). One document that rolls across nights: `docs/testing/FINDINGS-LEDGER.md` (id · night · finding · basis · status: open / fixed in PR · verified by night N).

## What is recorded, and by what

| record | source | already there? |
|---|---|---|
| every model call — tokens, cost, latency, model, provider, request_type, agent, execution | `llm_usage` | yes (night 1: 852 rows) |
| every tool call an agent made, success, time | `tool_execution_logs` | yes |
| every ticket — status timeline, attempts, result, review feedback, runtime_ref (denials, asks) | `board_tasks` | yes |
| every report and deliverable, with grade columns | `agent_reports`, `deliverables` | yes; grades empty until the morning |
| every question/approval, what was asked, answered, expired | `approval_grants` | yes |
| Auto's conversations, turn by turn | `chats`, `messages` | yes |
| playbook runs, missions and their sub-tasks | `recipe_executions`, `orchestration_tasks` | yes |
| Auto's routing choices and the semantic tool graph | `routing_decisions`, `tool_routing_edges`, `tool_routing_affinities`, `tool_routing_intent_clusters`, `intent_classification_cache` | yes |
| the Harness's sweeps and prescriptions | `harness_prescriptions` | yes |
| Jev's shadow comparisons (PRD-248) | the seam's shadow rows | once the PoC branch is in the test checkout |
| the customer's own account, grades and friction | the night folder | yes |
| **the night's numbers in one place, comparable across nights** | **the ledger** (`tests.sim.customer ledger`) → `~/.automatos-sim/campaign.sqlite` | **to build** |

The ledger, per night (window = `night.status` STARTED→FINISHED, UTC; `llm_usage.created_at` is naive UTC): tickets by status and by runtime (cli Claude / cli Codex / API model), ticket durations p50/p95 per runtime, cost by request_type and by agent, Auto turns with tokens per turn and tools used, tool calls per agent and failure rate, asks raised / answered / expired, deliverables and reports counted and graded, retrieval hits, routing decisions, the decision lane (`request_type='decision'`: count, cost, latency p50/p95 by provider) with the `complexity_assessor` count beside it, **the tool-graph columns** (edges / affinities / intent clusters, first-try selection rate) and **the field-memory columns** (facts written, promoted, recalled, recalled by another agent, wrong recalls) — every night, so the two paper series accrue without anyone remembering to run them. Model-drift checks whitelist `request_type in ('decision','embedding')`. This is the data for the November analytics.

## Grading

- The persona grades everything it reads, 1–5, in the diary — the customer's opinion.
- The morning runs the independent judge over every report and deliverable of the night (`tests.sim.customer judge`, batch mode to build; a model from a different family than the one that wrote the work).
- Gerard hand-grades five a week; the judge's grades are trusted once they track his.
- Grades are written back to `agent_reports.grade` / `deliverables` so the product's own analytics carry them.

## The nights

Each theme is a brief file under `scripts/ralph/customer-night/nights/<theme>.md` that replaces "the evening you have in mind" in the prompt (to build: the `{{AGENDA}}` placeholder; the persona and rules stay). One theme per night; a theme re-runs after its fixes land.

| # | theme | the customer's evening | measured | needs |
|---|---|---|---|---|
| 1 | **General** (done 18 Sep) | hire, delegate, review, weird asks | baseline for everything | — |
| 2 | **Regression of night 1** | the same evening after the first fixes | the delta: same findings gone? new ones? |
| 2b | **Noise floor** (nobody has this number) | the same evening, the same brief, **the same build as the night before it — nothing changed at all** | run-to-run variance of every ledger column: tickets done, cost, tokens per turn, tool calls, grades. Until this exists, no night-over-night delta can be called a change rather than scatter. Cheapest useful night in the programme; run it once early, then after any big platform shift | a night where the operator changes nothing — hardest part is the discipline, not the code | fixes for: agents can't read the knowledge base; raw-command holds; stats tiles; reports panel; Answer button |
| 3 | **Knowledge** | upload 10–20 real-shaped documents (CSV, MD, PDF), ask Auto and agents questions only the documents answer, contradict a document and ask again, update one and ask again | retrieval hit rate, wrong answers, ingestion cost per document (night 1: $3.81 of graph extraction nobody asked for), does the knowledge graph build and does it help | `GET /api/knowledge/graph` reads; Harbourline corpus in `~/.automatos-sim/` |
| 4 | **Playbooks** | install playbooks from the marketplace, ask Auto to make one, run them, schedule one, break one on purpose | run success, step timing, what a failed step tells the owner | there are 0 playbooks in the workspace today |
| 5 | **Missions + shared field memory** | three missions of increasing size; explicit staffing instructions ("the words to the writer"); the second mission needs facts the first one learned | plan quality, assignment obeys the owner, approval flow, synthesis, deliverables per mission; **shared field memory (PRD-166, Qdrant `field_memory`, on since 18 Sep 21:06): facts written per mission, promoted to durable, recalled by a *different* agent in the next mission, recall hit rate, wrong recalls** — the follow-up numbers for Gerard's field-memory paper | night 1 found assignment ignored the instruction; memory was off until 21:06; **the PRD-245 bridge must first give sessions the mission tools and field-memory read+write (night 1: sessions wrote 0 field-memory points on a 9-task mission while API agents wrote 3) — Gerard: "missions are my most powerful tool"** |
| 6 | **Onboarding** | a fresh throwaway workspace; a new company onboarded through Auto; package proposed and accepted; first day of work | time to first value, questions asked, what the package installed, what the owner had to figure out alone | throwaway workspace (`tests.sim.workspace`); Composio pre-connected by hand if tools are in scope |
| 7 | **Command Centre, governance, harness** | the owner as manager: review modes human/llm, reject with feedback, SLAs, blocked handling, watchlist, digests, governance tab, what the Harness says about the night | every tile against the board's truth (night 1: WORKING/QUEUE/ATTENTION wrong), review round-trips, Harness sweeps present and honest | the tile fixes from night 1 |
| 8 | **Auto's brain — baseline** | 60–100 varied asks to Auto across every capability, in one evening | which tools it chose, wasted calls, tokens per turn (night 1: ~37k), latency, wrong routes; **the semantic tool-selection graph before/after each night (`tool_routing_edges`, `tool_routing_affinities`, `tool_routing_intent_clusters` counts and deltas; `routing_decisions` first-try rate; `intent_classification_cache` hits) — is the graph building, is it improving Auto's picks; the follow-up numbers for Gerard's tool-selection-graph paper**; the offline 77-query eval as the fixed yardstick | `python -m scripts.eval.tool_routing.run_eval` beside the live night |
| 9a | **Auto's brain — Jev, shadow** | the same evening word for word, `classifier_mode=shadow` + `tool_rerank_mode=shadow` | first assertion: chat rows, tokens and cost per turn within noise of night 8 (shadow must be inert; the decision lane's own `request_type='decision'` rows are excluded from chat cost); then agreement per field / tier / confidence band, rerank overlap and `nothing_fits` rate, decision latency p50/p95 and cost; `scripts.eval.decision_shadow.score --since --until --json` | PRD-248 branch merged with the dials OFF at the refresh after night 1's fixes, so nights 3–8 prove "off changes nothing" |
| 9b / 9c | **Jev, live — one hook at a time** | 9b `classifier_mode=live` only; 9c `tool_rerank_mode=live` only (one night with both if the rotation cannot afford two — the confound is then accepted) | vs night 8: first-try selection rate (`routing_decisions`), wrong-tool rate (`tool_execution_logs`), `__tool_gap__` rows, tool calls / input tokens / cost / latency per chat turn, `complexity_assessor` count (should fall toward zero in 9b), the decision lane's count/latency/cost, judge and persona grades, outcome rate; plus the two offline evals each night | the same brief as 8 and 9a |
| 9d | **any typed judge?** (optional) | `provider=llm` (the adapter on `system_llm`) in shadow | the same agreement log — is it Jev, or would any typed judge do | — |
| 10 | **Templates** | deliverables that must honour a template (Template Studio, layouts, data_table): reports, briefs, a slide-shaped one | does the output follow the template, where it breaks | PRD-242/243 templates seeded |
| 11 | **Hybrid runtimes** | the night-1 evening with a mixed team: Claude sessions, Codex sessions, API agents on OpenAI, DeepSeek, Kimi via OpenRouter — identical briefs to each; **plus one mission staffed across runtimes (Claude, Fable, Codex, Grok) sharing one field memory** | quality (judge), cost, time per runtime on the same brief; the "some agents perform better" table | the models present in the registry; budget |
| 12 | **Load** | 30–60 tickets released at once; many sessions | dispatcher fairness, session concurrency the Mac tolerates, SSE, Redis errors, p95 claim latency | evening with nothing else running |
| 13 | **Collective** | a week compressed into a night on the premium models — the scorecard night | usability / cost / quality / usefulness as one customer; the demo numbers | everything above green enough |
| 14 | **Socials (PRD-251)** — added 25 Sep (Gerard), placeholder | *written once HIGGS's build is done (W1 video engine → W2 Socials tab → W3 publishing)* | *written with the brief* | the PRD-251 waves built and in the local build; `socials.enabled` switched on for the test workspace; HIGGS's summary of what a customer can do; the brief and the testing personas are written THEN, not before |

Then the cycle repeats from 2 with whatever is still red. Nights 8→9 and 11 are the comparison pairs: same brief, one variable changed, the ledger and the judge do the comparing.

## What this is for — WebSummit Lisbon, 9 November 2026

Gerard has a stand. Between now and then the programme has to *prove* the difference — not another OpenClaw or Hermes: widgets, the maths, research, the Academy, knowledge graphs, missions with shared field memory, a tool-selection graph that learns — with numbers a stranger can check. So every night is also evidence:

- **Two paper follow-ups, measured every night from night 2:** the tool-selection graph (night 8's columns, tracked nightly, not only on night 8) and mission shared field memory (night 5's columns, tracked nightly). Each gets a dated before/after series in `campaign.sqlite` and a blog post when the series says something.
- **Three comparisons worth a post each:** Jev vs the embedding router (nights 8/9), session agents vs API agents on identical briefs (night 1 already: 11 of 11 5/5 pieces were sessions), and the hybrid-runtime table (night 11).
- **The collective night (13) is the demo:** one customer's week, one scorecard, the numbers on the slide.
- Cadence: ~50 nights available; one theme a night, a regression night after each round of fixes, the collective night at least three times (early, mid, final) so the slide shows a trend, not a point.

## Rules that hold every night

- **The running stack hot-reloads the checkout. A commit during a night is a deploy.** The backend runs
  `uvicorn --reload` over a bind mount of `automatos-ai/orchestrator`, so a commit (or any edit) in
  Gerard's checkout goes live inside seconds, under whatever sessions are mid-turn. Two consequences,
  learned 2026-09-18/19: a fix lands without a restart (night 2 ran on the fixed build with no
  redeploy), and **editing that checkout while a night runs corrupts the night** — the same class of
  damage as restarting the host mid-run (night 1's "180 ms cancellation sweep"). During a night:
  work in a worktree, never commit to the checkout, never restart backend/host/frontend until
  `night.status` shows `FINISHED`.
- **One night, one variable, wherever the night is meant to measure something.** A night that changes
  the model *and* the runtime *and* twenty fixes answers "did the findings recur?" well and "what did
  each fix buy?" not at all. Themed nights may carry a fix wave; comparison nights (8/9, 11, 2b) may
  not.

- Runtime **subscription session agents** for the team until the hybrid night (Gerard, 18 Sep). API agents only where the theme needs them.
- The persona's guardrails: everything tagged `sim-night-<date>`; nothing untagged touched; drafts only on connected apps; no code, git, Docker, database; a spend stop per night.
- **Spend stop: $40 a night** (Gerard: "worth a few hundred euros" over the programme) — the one line in the brief; the ledger reports actual.
- Nothing is purged after a night by default: the rows are the data. Agents and tickets that clutter the board get archived by tag when the night's review is done, never before.
- Product fixes go through PRs and CI, never edited live. The one exception on night 1 (the `--unlisted-bash allow` host switch, 19:42) is recorded in OBSERVER.md and is its own PR (#781).
- Every finding is labelled trace-backed or inferred; a finding without a file:line, an id, or a diary timestamp is not filed.

## To build before night 2 (small, in this order)

1. `nights/<theme>.md` + the `{{AGENDA}}` placeholder — one file per theme, the general one first.
2. `tests.sim.customer ledger --since --until` — the night's numbers from the DB into `campaign.sqlite` and a markdown table appended to the morning report.
3. Judge in batch over a night's reports and deliverables, grades written back to the product's grade columns.
4. `docs/testing/FINDINGS-LEDGER.md` seeded from night 1's OBSERVER.md and MORNING-REPORT.md.
5. The PRD-245 bridge at mission parity: `record_memory` (field + durable), the `## Field memory` block in the session ticket prompt, the mission tools sessions lack — prerequisite for night 5 and the mixed-runtime mission in night 11.
6. The night-1 fixes as PRs: agents' knowledge access; hold cards with intent + full command + `comm` allowlisted; Command Centre tiles; Agent Reports panel; the Answer button; the SLA-as-calendar entry.

## Decisions for Gerard

1. Spend stop per night — $40 proposed.
2. Auto's model for the programme — stay on gpt-5.4 (true cost, ~37k tokens a turn) or a cheaper tier for the volume nights and gpt-5.4 for the collective night.
3. Night 6 onboarding: which tools to pre-connect by hand in the throwaway workspace (OAuth cannot be scripted).
4. ~~When PRD-248's branch joins the test checkout~~ — agreed with the PRD-248 session 18 Sep: merged with the dials OFF at the first refresh after night 1's fixes (Fri 19 Sep), once #779's CI is green and its seed is shown to write only the `decision_engine` rows; night 9 is then a settings flip. The persona never opens Settings → System.
5. Which four hybrid runtimes for night 11.
