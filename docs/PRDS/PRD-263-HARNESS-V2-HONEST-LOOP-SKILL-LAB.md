# PRD-263: HARNESS repair (report-only org loop, governed apply) and an owner-launched Skill Lab

HARNESS is Automatos' own weekly org loop (PRD-121). It is not Harness2 (google-research/harness2), is not affiliated with Google, and does not use the `harness2` package.

**Status:** Built, 23 Oct 2026. All of P1 and P3 and the skill and seed work are merged on main, and the P2 and P5 local tooling is built (§0). What remains are runs: the owner's checks, calibration, iteration 0 and the Skill Lab, over days up to the 6 Nov skill freeze, tracked in the HARNESS track of the nights test plan. §3 D1–D5 taken (D5 on 20 Oct); D6 stays open.
**Source:** `<workspace>/.claude/HARNESS-V2-DESIGN.md` (rev 2, 18 Oct: 7 reader reports, 3 candidate designs, 3 judgements, one verification pass of 24 claims with 3 refuted and 25 gaps applied); Gerard's six decisions of 18 Oct; Harness2 (google-research/harness2, Apache-2.0, arXiv 2610.14224, assessed at commit `1cfe254e`); findings ledger rows named in Related. The eval set is described by structure, counts and field names only, never by row contents (PRD-256 D7, `docs/testing/AUTO-EVAL.md:5`).
**Scope:** design phases P1–P5 in one PRD (Gerard, 18 Oct). Code that ships is in automatos-ai. Skill text is in automatos-skills, human-edited, merged by Gerard. The eval tooling and the Skill Lab live under `~/.automatos-analyst/` and `~/.automatos-sim/` and are never committed. P6 is later work (§7).
**Type:** Refactor and consolidation, with extensions. It extends `services/harness_service.py` (and shrinks it), `actions_harness.py`/`handlers_harness.py`, `api/workspaces.py` (the HARNESS settings), the tool_call ApprovalGrant rail (`issue_tool_grant`, `api/approval_grants.py`, the GrantCard), `harness_task_ledger`, the Self-Learning tile, `telemetry.py`, `edge_builder_logs.py`, `click_memory.py` and `scripts/sync-skills.py`. New small modules: `core/stats/harness_verdicts.py`, `services/harness_metrics.py`, `services/harness_runs.py` (run rows, the advisory lock and the cron trigger; PR-1B-b), `services/harness_mode.py`, `services/harness_changes.py`, `services/harness_grants.py`, `services/harness_post_change.py`, and one extracted frontend component (`harness-settings-section.tsx`). It deletes auto-apply, `/approve` and `/reject`, `api/harness_commands.py`, `OWNERS_OWN_DECISION`, the JSON run files, the flat 10% rule, the in-memory run guard and the "Enable HARNESS" switch in the same PRs that replace them. One new table (`harness_runs`) and one ledger migration; no new dependency or vendor; settings in FR-34.
**Parents and siblings:** PRD-121 (the HARNESS org loop), PRD-142 W4 (self-learning, gated), PRD-187 S5 (dropped `harness_prescriptions`), PRD-231 (the slim skill), PRD-232 (synthetic signals), PRD-256 (D1 owner click, receipts, the eval pack), PRD-262 (pull-first images, pinned tags).
**Depends on:** PRD-256 merged (#1048); the PRD-193 approve → grant → resume rail; the maintainer issues US-001–US-007 before the PRs that need them; US-026a recorded before PR-1B and PR-1D merge; D5 merged (US-028) before every story tagged Needs D7.
**Ralph config:** launch is human-only with `RALPH_MODEL` pinned. P1 and P3 ship as 15 small PRs (D3), each built by a session or by Ralph; a Ralph-built PR has its own kit, `scripts/ralph/prd-263-1a.json` to `prd-263-1g.json` and `prd-263-3a.json` to `prd-263-3i.json` (§4). Local stories are built by a named build session Gerard launches (D5), never by a Ralph loop.
**Human-gated, excluded from the autonomous loop:** US-001–US-007 and US-070 (maintainer issues and decision records); US-024, US-060, US-065, US-071, US-082 (skill text, Gerard merges); US-025, US-061, US-066, US-072, US-083 (seed syncs and the persona lift, Gerard merges); US-026a, US-026b, US-064 (prod facts, migration applies and flags); US-027, US-063 (owner checks); US-028 (the eval-rule amendment); US-029 (a recorded finding; code only if Gerard picks that option); US-032, US-043, US-044, US-045, US-069, US-073, US-078a, US-081 (owner runs and data); the prod run in US-059; and every `→ OWNER` acceptance criterion in any story.
**Decisions taken:** D1–D4 by Gerard on 18 Oct; D5 on 20 Oct (US-028 merged as written, #1204); D6 open (§3; §11-Q2).

Labels used below: **[T]** read in code or docs during the design revision or this session; **[R]** traced by a verifier, lines cited; **[I]** inferred; **[M]** modelled by simulation (the design session's `org_stats.py`, `consec_sim.py`, `frozen_sim.py` and `gate_sim3.py`: standard-library math with fixed seeds and no data input, copied into `docs/PRDS/prd263-spec/sims/` in the same commit as this PRD). Figures computed by exact binomial enumeration are marked "exact".

## 0. Delivery record (23 Oct 2026)

Every PR below merged with its `test` workflow green (the run id is that PR's green run on its head commit).

| Group | PR | Merged | Test run |
|---|---|---|---|
| The PRD | #1146 | d47fdd26f | 39166544166 |
| PR-1A statistics module (US-008) | #1153 | 39fa77ff4 | 39160067124 |
| PR-1B-a `harness_runs` table (US-009) | #1160 | 5f0c98ff1 | 39167782905 |
| PR-1B-b run record, lock, cron (US-010 – US-012) | #1162 | 35f529e97 | 39233917803 |
| PR-1C read model (US-013) | #1154 | 5d9032ccd | 39160135603 |
| PR-1D honest DIAGNOSE (US-014a, US-014b, US-015) | #1185 | 931e5a98f | 39267713712 |
| PR-1E report-only (US-016, US-017) | #1186, #1189 | 44d071c44, 4edb433a0 | 39249289520, 39271795687 |
| PR-1F telemetry, tool-failure input (US-018 – US-020) | #1191 | c3d989de5 | 39279485894 |
| PR-1G endpoint, tile, tools, docs (US-021 – US-023a) | #1192, #1193 | 46fc5c37b, 63912834f | 39279519409, 39287386188 |
| PR-3I production skill signal (US-062) | #1161, #1163 | aded1c9cc, bc960a9c6 | 39166464058, 39234781619 |
| DOC-1 eval rule (US-028) | #1204 | a7980584b | 39328990027 |
| PR-3A delete the /approve path (US-047) | #1205 | bd816c2dd | 39358823861 |
| PR-3B delete the apply helpers (US-048) | #1206 | 238df3fcd | 39395666289 |
| PR-3C change registry (US-049, US-050) | #1212 | 330439eef | 39397424882 |
| PR-3D ledger migration (US-051, US-059) | #1213 | 38ae5588e | 39399547323 |
| PR-3E apply action, reconciliation (US-052, US-053) | #1227, #1228 | bcc3f79e5, f3a37fdfb | 39402398130, 39405019470 |
| PR-3F raise grants (US-054, US-055, US-058) | #1229 | e81203a3e | 39405915971 |
| PR-3G post-change check, tool-removal gate (US-056, US-057) | #1230, #1231 | a5ea274e5, 9d23ddc47 | 39405932798, 39405950165 |
| SYNC-1 + SYNC-2 seed v2.3.4 (US-025, US-061) | #1248 | 630fc1c85 | 39463551829 |
| FIX-1 seed v2.3.5, persona lift (US-066) | #1251 | fc50ef23c | 39473521139 |
| DOC-2 AUTO-EVAL.md (US-046) | #1252 | c033d8bc0 | 39479012206 |
| Records: Q29, set v2, rescore | #1260 | 171e4ef37 | 39499219443 |

Skill text (automatos-skills pin branches, Gerard's review): `skill/auto-v2.3.4-harness-report-only` (US-024, US-060), `skill/auto-v2.3.5-owner-lanes` (US-065), `skill/auto-v2.4.4-prd263-fixes` (the slim candidate for US-069). Local tooling, never committed: the eval runner and its tools (LOCAL-2 – LOCAL-5, LOCAL-9's gate), the night preflight (LOCAL-6), set v2 and the sealed split (US-043), and the Skill Lab (LOCAL-11 – LOCAL-13), with fixture tests.

**Gerard's decisions, 23 Oct:**
- No check reads the operator's own provider accounts. US-041's OpenRouter floor in the night preflight and LAB-0's (US-073, US-074) are removed, and the operator is not asked for them.
- No model, provider or price is written into the eval tools or the Lab. A pass runs Auto as the platform has it configured, or on any model the platform's registry (`llm_models`) lists as active, resolved to the provider that serves it. Fixture agents get the platform's default for a new agent. The Lab's improver models come from `lab.local.toml`, or Claude Code's own default. The gate reports tokens per turn without a model's price tier.

**Found and fixed while closing (23 Oct):**
- c1 had HARNESS switched off by the retired toggle (`enabled: false` reads as Disabled, §11-Q28), so no sweep would have run there. Set to `report_only` through the Settings API, as §4's hold requires.
- The night preflight refused every night: c1 holds a second, unlinked `platform-management` row (a git-source import), and the check counted both. It now checks the row Auto links to (local tester kit).
- At 390 px the Command Center's tab body is 32 px tall (`.cc-body` takes the height left under the header and the stat cards), so every tab is unreadable on a phone. Not HARNESS: flagged to the Command Center's owner.
- The chat keeps the first 2,000 tokens of a tool's answer (`consumers/chatbot/tool_cut.py`). The audit report listed agents by name, so on c1 the cut dropped one of the two REGRESSION agents. The report now leads with the flagged agents and the suggested changes (#1276).

## 1. Introduction

HARNESS (PRD-121) is the weekly per-workspace org loop: COLLECT, DIAGNOSE, PRESCRIBE, APPLY, then an audit report attributed to Auto. It lives in `orchestrator/services/harness_service.py` (2,165 lines [T]). Its phase skeleton, per-phase `end_open_transaction`, the active-agent-count dormancy status and the Auto-attributed audit report work and are kept. What it reports and what it would do are wrong:

- **Every agent comes out STABLE.** DIAGNOSE computes deltas from four fields that `platform_list_agents` never returns (`harness_service.py:578-599`, `:1392-1412`; `handlers_agents.py:76-103` [R]). Every delta is 0, every agent is STABLE, and Auto tells the owner "the organization is stable". It reads `agent.get('model')` while the handler emits `model_id` (`harness_service.py:564`; `handlers_agents.py:88` [R]). Nine of the ten COLLECT calls are only dumped to a trace file, and above about 20,000 characters the listing collapses each agent to a plain string, on which `agent.get` raises (`agent_roster_fit.py:81-93` [R]; the raise is [I]).
- **If the data were fixed, HARNESS would change agents without a click.** `_workspace_allows_auto_apply` treats a missing mode as `full_auto` and returns True when no setting exists (`harness_service.py:394-400` [T]), and the Settings API returns `mode: full_auto` as every workspace's default (`api/workspaces.py:568-573` [T]). `_get_harness_config` (`harness_service.py:378-392`) also defaults to `full_auto`, but nothing calls it. The risk ceiling is 2 at standard autonomy and 3 at full (`config.py:948-949` [T]). PRD-256 D1 puts heartbeat and tool changes behind the owner's click (`owner_only.py:43-64` [T]). Today the direct apply passes no actor (`harness_service.py:927-929`; `platform_executor.py:929-970` [R]), so the hierarchy check would refuse it as `anonymous_actor` [I]. The defect today is a false report, not live damage.
- **The other apply path lets an admin's request stand in for the owner's click.** `POST /api/harness/prescriptions/{rx_id}/approve|reject` (`api/harness.py:67-84`) requires `workspace:manage` and the handler's admin check (`harness_commands.py:139`), then runs `_governed_verdict`, which always approves because it passes `override_auto_approve=True` (`harness_commands.py:277-285`; `approval_policy.py:163` [R]), and passes `OWNERS_OWN_DECISION` so the owner-only gate does not ask again (`harness_commands.py:234` [T]). No frontend code calls these routes (grep at `697ca7d6e`). `HARNESS_SELF_MANAGEMENT_ENABLED` (default false, `config.py:942` [T]) gates both this and the move-card-to-Done path (`harness_commands.py:134-136`; `harness_service.py:1640-1652` [T]).
- **The rule it judges with would flag stable agents in about 40% of weeks.** REGRESSION fires when the largest of four relative deltas exceeds 10% (`harness_service.py:578-599` [R]). Modelled as weekly binomial counts once the data is fixed (no real data), a stable agent at p=0.8 is flagged 0.388 of weeks at n=10 and 0.421 at n=20, counting success and error rate only (exact, `frozen_sim.py`); cost deltas add more. Agents average about 16 tickets a week, skewed (readers).
- **Run state lives on the volume, and the run guard works in one process only.** Baselines, last_run and traces are JSON files under `<WORKSPACE_VOLUME_PATH>/<ws>/harness`; AGENTS.md says data lives in the DB. `_running` is an in-memory dict (`harness_service.py:89`, `:274-279` [T]), while `trigger_now` is fire-and-forget in whichever of the 4 uvicorn workers serves the request (`harness_service.py:265-267`; `orchestrator/Dockerfile:195` [T]). The sweep is fixed at Sunday 02:00 with no timezone passed (`harness_service.py:35` [T]), inside the night window where F408, F409 and F416 happened. "Sunday 02:00 UTC" is also hard-coded in its log lines and in the `next_scheduled_run` that `get_status` returns to Auto (`:107`, `:123`, `:165` [T]).
- **Tool failures read a counter that only grows, a wait for the click counts as a failure, and no failure is ever classified.** HARNESS reads `fails_for_intent` rows with `sample_count >= 5`, a count that only grows (`signal_recorder.py:643-669`; `harness_service.py:665-680` [R]). An owner-only ask returns `success: False` (`owner_only.py:492-499` [R]); the router turns that into a failure signal (`tool_router.py:1640-1651` [R]), telemetry logs it as `error` (`telemetry.py:152` [T]), and EdgeBuilder counts every status other than `success` as a failure (`edge_builder.py:646` [R]). `tool_execution_logs.error_code`, which a broken-connection check needs, is never written by production code (`telemetry.py:154-174`, `:331-343`; `signal_recorder.py:557-575` [R]). EdgeBuilder reads every telemetry row, eval and replay included (`edge_builder_logs.py:67-83` [R]).
- **The tile, the Settings tab and Auto describe a loop that is not there.** The tile reads `data.prescriptions` and `harness.iteration`, but the backend sends `iteration_count` (`self-learning-health-widget.tsx:40-52`, `:114`; `harness_service.py:170-175` [R]), and its Signals figure is a process-global counter. Settings offers "Full Auto — low-risk changes applied automatically" and "risk 3+ queued for review" (`SystemLLMSettingsTab.tsx:1081-1088` [T]), Biweekly and Monthly schedules that do nothing (`:1059-1060`), and an "Enable HARNESS" switch that shows off by default (`:1028-1039`, `:233`) while the backend runs HARNESS for every workspace that has not opted out (`harness_service.py:364-376` [T]). The Settings API accepts only `full_auto` and `manual` and returns HTTP 400 for any other mode (`api/workspaces.py:575-576`, `:797-798` [T]). The shipped seed says "HARNESS runs Sunday 02:00 UTC", finds queued prescriptions with `platform_list_tasks tag=harness status=todo`, and promises "model swaps" (`platform-management-skill.md:369-388`, `:1532-1594` [T]); Auto's Monday cadence text says the same (`core/services/auto_cadence.py:66-74` [T]). `platform_harness_history` reads the baseline JSON files (`handlers_harness.py:50-87` [T]).
- **The eval that would judge a skill edit scores a broken claim check as backed.** `claim_backed` has not been a valid gate on the baseline path since `d5ea08e67` deleted `action_claims` [R]; on failure replay.py records `claim=None`, which it scores as backed (reader-reported). The set holds 65 active rows from 44 f_ids, F108–F381 only, and Haiku 5.5 passed 61, 60/64, 59 and 60 of them on 13–14 Oct: it is saturated. Run pairs differ on 2–4 rows, and a dirty versus a reset board on the same build moved 10 rows (9 Oct).
- **Auto's always-on skill carries known defects.** The platform-management body is 93,367 bytes, names "Gerard" 26 times (`platform-management-skill.md:101, 185, 193, ...` [R]), still describes a DELEGATE lane (`:431`, `:442`, `:524` [R]; persona doctrine at `seed_auto_agent.py:76` [R]), and its Authority Model contradicts `OWNER_ONLY_ACTIONS` (`:229-230`, `:257` against `owner_only.py:43-64` [R]). The PRD-231 slim body (34,078 bytes, automatos-skills `skill/auto-v2.4.3-real-tools`) has never been measured, and it still carries the stale HARNESS runbook and the owner's name (§15 risk 15).

**What Harness2 (google-research/harness2) contributes.** Harness2 improves the prompt and skills around a frozen model, using LLM proposers and real rollouts. HARNESS tunes each workspace's org config from metrics. They do different jobs, so this PRD takes mechanisms, not the package. P1 and P3 implement them independently from this PRD's description (§9, Licence):
- refuse when there is no evidence (`core/context.py:171-175` [R]) → INSUFFICIENT_DATA verdicts and coverage, and a Lab iteration with no base trajectories refuses;
- one spec drives the vocabulary and the validator (`core/tree.py:40-110`) → the change-kind registry, and the Lab's locked-span spec;
- validate, retry once, fall back (`core/recursion.py:176-251`) → no `review_needed` placeholders; KEEP_BASE after two failed candidates;
- a judge failure is None, never zero (`core/judge.py:15-19` [R]) → an ungraded claim check makes the run VOID;
- grader-verdict tripwires (`core/context.py:419-470`), re-keyed to the eval's own field names;
- ANSWER_FREE and EVIDENCE_BOUNDARY, the INV and DOM proposers, divergence first, the GENERALIZE block, the cost argument against always-on text (`core/prompts.py:129`), KEEP_BASE and UNSAFE, experience verdicts with NO_ENTRY, all restated in our own words;
- the run fingerprint (`core/runmark.py:41-61`) and a budget identity adapted from `core/ledger.py:25-111` (its fixed 4k-rollout / 3k-session identity does not fit this Lab, US-074), both ported locally with the unchanged Google Apache-2.0 header and a change notice.

From prior art: full stripped trajectories rather than summaries (Meta-Harness, 56.7 vs 41.3 in its ablation); delta edits and an anti-collapse size limit (ACE); the noise floor, critic, cost rule and edit history (RRSI, google-research/rrsi); predicted flips per edit (AHE).

**Assessed and rejected on 18 Oct (design rev 2):** replacing HARNESS with Harness2; the `harness2` package (bound to Vertex, OpenCode and Codex), its `third_party/` (the Tencent licence's territorial limit) and its name (Apache §6 trademark); per-task test-time adaptation on live asks (it multiplies cost per ask and acts in tenant workspaces); the "workspace facts" adapt pack (it would push c1's fake data into every tenant's prompt); LLM blind selection as the primary selector, and autotrigger; GEPA as a dependency (its reflection consumes scores and would leak gold rules through `info`, its Pareto budget is too large at about 56 s per row-run, and its Claude Code proposer exists only on unreleased main); TextGrad, DSPy, OpenEvolve, Agent Lightning, promptfoo and inspect_ai.

This PRD does five things, in five phases:

1. **An honest HARNESS, report-only for every workspace (P1).** One SQL read model, standard-library statistics, coverage and INSUFFICIENT_DATA, a run table, an advisory lock, a weekday cron, telemetry hygiene, and a truthful tile, Settings tab and Auto text. Auto-apply and the card-to-Done apply path are deleted.
2. **A truthful eval instrument (P2).** Claim checks graded on every row, per-row claim errors, reset passes, fingerprints, a skill arm that always restores, a night preflight, a sealed split, a $0 rescore of every run since 9 Oct, and a measured noise floor.
3. **A governed apply (P3).** In workspaces an owner sets to `propose`, and only while the global switch is on, each proposed change becomes an Approvals card. One action applies it, only on a granted, unexpired grant whose setting still reads its "before" value. Every proposal ends in a recorded state, and a frozen-reference check raises a rollback card when a change hurts.
4. **Iteration 0: one measured decision on the slim skill (P4).** A defect fix first, then the PRD-231 slim body probed under the NON-INFERIOR rule.
5. **A Skill Lab at k=1 (P5).** Gerard launches it; it edits only marked chat-surface sections of Auto's skill; a deterministic held-out gate decides; it ships only through Gerard's PRs.

The org loop makes no model call of its own. P2, P4 and P5 probes are Gerard's runs. Each one starts only with a spend limit and a cost factor he supplies at launch, and stops before metered spend would pass the limit (D4). The planning estimate for all of them together is 2,376 row-runs, $33.98–45.14 booked (§13, informational only).

## 2. Goals

- A HARNESS run reports, for every active non-Auto agent, its n and a verdict (INSUFFICIENT_DATA, STABLE, WATCH or REGRESSION), plus coverage for the workspace. "Stable" never appears for an unjudged agent. Convergence reads `insufficient_data` below 50% coverage, and "converged" is no longer a value.
- A stable agent's single-flag false rate is 0.049 at p=0.8 and 0.015 at p=0.9 (n_cur=20 vs n_ref=80, one-sided pooled two-proportion z test, both drop thresholds required; exact), pinned in a test, against 0.388 or more for today's rule.
- Every workspace defaults to `report_only`. A `report_only` run creates 0 BoardTasks, 0 approval grants, and no notification other than the audit report's single `report_submitted`. The auto-apply code is gone (0 lines).
- In a `propose` workspace with the global switch on, a change applies only from the owner's click on a granted, unexpired grant whose setting still reads its "before" value. A late, stale or double click changes nothing, and every proposal ends in exactly one recorded state.
- An applied change that hurts its predicted or guard metrics against a frozen pre-change reference raises a rollback card. Detection within 4 weeks is 0.938 for a 0.8→0.5 drop [M], with a false rollback rate of 0.020 per metric at p=0.8 (0.0201 simulated, `frozen_sim.py` seed 7; test tolerance ±0.003) [M].
- Run state lives in `harness_runs`, and from PR-1D no file is written under `/harness`. One Postgres advisory lock per workspace, and one run per workspace per judged week whatever the replica count. The sweep runs on a weekday daytime UTC cron.
- `harness_service.py` never grows in any PR, and ends P1 below 2,165 lines. New logic lives in new modules of at most 800 lines (200–400 targeted).
- The eval grades the claim check on 100% of rows (0 ungraded; any ungraded row makes the run VOID), counts claim errors per row, has a rescored history since 9 Oct, and publishes a per-row noise floor.
- One measured decision on the slim skill, with tokens per turn = input plus output tokens summed over every model call of the turn (input already includes cache reads and writes).
- Either one Lab edit accepted by the held-out gate and merged, or 3 consecutive KEEP_BASE results documented with numbers. 0 tripwire hits in Lab-produced diffs, PR bodies and commit messages. Metered spend (booked × the operator's cost factor) stays at or below the operator's limit on every launch; the first row of a launch is not pre-checked.
- Each night after P2 yields deterministic per-100-ask signal rates from `harvest.py`.
- No numeric spend cap, price, cost factor, balance floor or allowance threshold is written into any script, setting default or bar (grep of the eval and Lab tooling).

## 3. Decisions

**D1. One PRD covers design phases P1–P5; P6 is later work, not built here.**
P1 (honest HARNESS, report-only), P2 (eval truth), P3 (governed apply), P4 (iteration 0, the slim-skill decision) and P5 (Skill Lab, k=1) are one PRD. The Lab code and the gold eval data stay local and never enter a public repo. This PRD describes them without gold content. *(every story; P6 is listed in §7)*

**D2. Through beta every workspace defaults to `orchestrator.harness.mode = report_only`, and owners may opt into `propose`. Auto-apply and `HARNESS_AUTO_APPLY_MAX_RISK_STANDARD`/`_FULL` are deleted. `HARNESS_SELF_MANAGEMENT_ENABLED` is redefined as the global switch that allows HARNESS to raise approval grants (default false).**
Evidence: PRD-256 D1 puts heartbeat and tool changes behind the owner's click; today's default is `full_auto` (`harness_service.py:394-400`, `api/workspaces.py:568-573`). The alternative, auto-apply as an explicit opt-in, would still change agent settings with no card. Consequences. First, the autonomy-ceiling fail-safe that the design's §2.1 kept lives inside `_auto_apply_max_risk` (`harness_service.py:403-425`), so it goes with auto-apply. Second, the auto-apply loop and the card-to-Done apply path are deleted in P1 (US-016), not in P3 as the design's S3.9 had it, because report-only makes them unreachable, and AGENTS.md says replace, don't shim. Third, as an implementation choice recorded on US-001, the mode becomes the only per-workspace switch (`report_only`, `propose`, `disabled`): the "Enable HARNESS" switch and the `enabled`/`disabled` keys go, so Settings can no longer show HARNESS off while it runs. This decision takes the design's open questions 6 and 20. *(US-001, US-016, US-017, US-054, US-064)*

**D3. P1 and P3 ship as small PRs per AGENTS.md (about 400 changed lines excluding tests), one PR per former commit group (Gerard, 18 Oct).**
P1: PR-1A (US-008), PR-1B (US-009–US-012), PR-1C (US-013), PR-1D (US-014a, US-014b, US-015), PR-1E (US-016, US-017), PR-1F (US-018–US-020), PR-1G (US-021–US-023, US-023a). P3: PR-3A (US-047), PR-3B (US-048), PR-3C (US-049, US-050), PR-3D (US-051, US-059), PR-3E (US-052, US-053), PR-3F (US-054, US-055, US-058), PR-3G (US-056, US-057), PR-3I (US-062). Each is one branch, one push and one CI run. The "three PRs" figure in the question put to Gerard was the session's estimate, made before the stories were sized; at about 400 lines P1 is seven PRs and P3 eight. Five PRs run over 400 lines (PR-1D, PR-1E, PR-1G, PR-3A, PR-3B); §4 says why each stays one PR and names the one clean split, an option for Gerard. The generated seed syncs (US-025, US-061, US-083) and the automatos-skills PRs stay separate and Gerard merges them. This decision takes the design's open question 25. *(§4)*

**D4. No hard-coded spend caps anywhere.**
Gerard: "why would we hard code caps and decide for everyone?" Every probe run, calibration, iteration-0 probe and Lab launch requires a spend limit the operator supplies at launch (`--max-spend`, no default) and the operator's cost factor (`--cost-factor`, no default; metered spend = booked spend × that factor). It refuses to start without either, and stops before metered spend would pass the limit. The OpenRouter balance floor and the Max allowance threshold follow the same rule: operator-supplied, no default, refuse when unset. Cost figures in this PRD are planning estimates only (§13). The org loop makes no model call of its own. This decision takes the design's open question 8; the design's proposed $55, $66 and $120 caps are dropped. *(US-033, US-038, US-041, US-044, US-045, US-069, US-073, US-074, US-078, US-078a, US-081)*

**D5. The eval rule (known as "D7") is amended, scoped to this PRD: a named build session may edit `replay.py` and `phase.sh`, and the Lab Gerard launches may run `replay.py`. Improver sessions never read the eval folder.**
Gerard has not answered this one. The default is applied and the decision stays open. Every story that depends on it is tagged **Needs D7** and is held until US-028 merges.
**Taken, 20 Oct 2026:** US-028 merged as written (#1204), so the default and its extension both stand. Gerard named the build session "HARNESS" for LOCAL-2 to LOCAL-5 (it edits, and runs fixture mode only; Gerard launches every pass that spends). P1, P3, `harvest.py`, the night preflight (US-041) and the production skill signal do not depend on it.
As worded, the default covers `replay.py` and `phase.sh` edits and the Lab running `replay.py`. The other eval-folder steps are a proposed extension in US-028: edits to `compare-arms.py`, `rescore.py` and `arm-report.py`; creating `make_split.py` and `split.json`; the build session running the tooling in fixture mode; the Lab running `phase.sh --reset-only`, `compare-arms.py --gate` and `arm-report.py` and reading `split.json`, aggregate run outputs and (for its gold scan) the row file; set v2 authoring; and the rescore and calibration runs. Until Gerard accepts that extension, each of those steps is his operator step. *(US-028 and every Needs D7 story; §11-Q1)*

**D6. The Skill Lab is an operator tool over the global skill for now. Whether it later becomes a per-workspace product feature (each owner improves their own Auto on their own budget and limits) is open, and is revisited after WebSummit.** *(US-073–US-083; §11-Q2)*

Gerard took D1–D4 on 18 Oct. D5 and D6 are recorded at their defaults and stay open in §11. PRD-263's own decision numbers stop at D6, and any later decision starts at D8. In this PRD "D7" always means the eval rule at `docs/testing/AUTO-EVAL.md:5`, which widens the PRD-256 kit's D7 (`scripts/ralph/prd-256w1.json:4`) to CI jobs, Ralph loops and agent sessions.

## 4. Sequencing

Each P1 and P3 PR is one story group, aimed at AGENTS.md's ~400 changed lines excluding tests (D3): one branch, one push, one PR and one CI run, built by a session or by Ralph. Stories commit inside the PR, one commit per story.

| Phase | What | Where | Window (design calendar) | Stories |
|---|---|---|---|---|
| P1 | Honest HARNESS, report-only | automatos-ai, automatos-skills | build 19–25 Oct | US-001 – US-027 (with US-014a/b, US-023a, US-026a/b) |
| P2 | Eval truth | local, plus two automatos-ai docs PRs | Gerard's windows 19–27 Oct, alongside P1 | US-028 – US-046 |
| P3 | Governed apply | automatos-ai, automatos-skills | 26 Oct – 2 Nov | US-047 – US-064 |
| P4 | Iteration 0 | automatos-skills, automatos-ai, local | 28–30 Oct, one pinned image tag | US-065 – US-072 |
| P5 | Skill Lab, k=1 | local; ships through Gerard's PRs | 31 Oct – 6 Nov; skill freeze 6 Nov | US-073 – US-083 (with US-078a) |

| PR | Stories | Est. changed lines (excl. tests) | Depends on | Built by | Touches |
|---|---|---|---|---|---|
| ISSUES | US-001 – US-007 (US-070 in P4); US-026a | 0 | — | Gerard | GitHub issues; the private ops note |
| **PR-1A** Statistics module · `feat/prd-263-1a-harness-verdicts` | US-008 | ~240 | — | session or Ralph (kit `prd-263-1a`) | `core/stats/__init__.py`, `core/stats/harness_verdicts.py` (new), `core/services/edge_builder.py` |
| **PR-1B-a** The `harness_runs` table · `feat/prd-263-1b-a-harness-runs-table` (#1160) | US-009 | 136 (built) | US-026a | session | `core/models/harness.py`, `core/models/__init__.py`, Alembic revision `prd263_harness_runs` and both head pins |
| **PR-1B-b** Run record, lock, cron · `feat/prd-263-1b-harness-runs` (#1162, draft until #1160 merges) | US-010 – US-012 | 863 (built; over 400) | PR-1B-a; US-026a | session | `services/harness_runs.py` (new), `harness_service.py`, `config.py`, `config-surface.json` |
| **PR-1C** Per-agent read model · `feat/prd-263-1c-harness-metrics` | US-013 | ~260 | — | session or Ralph (kit `prd-263-1c`) | `services/harness_metrics.py` (new) |
| **PR-1D** Honest DIAGNOSE · `feat/prd-263-1d-honest-diagnose` | US-014a, US-014b, US-015 | ~435 (~250 deletions; over 400, see "PR size") | PR-1A, PR-1B, PR-1C; US-026a | session or Ralph (kit `prd-263-1d`) | `harness_service.py`, `handlers_harness.py`, `config.py`, `config-surface.json` |
| **PR-1E** Report-only for every workspace · `feat/prd-263-1e-report-only` | US-016, US-017 | ~565 (the settings extraction moves ~110 lines; over 400, see "PR size") | PR-1D; US-001 approved | session or Ralph (kit `prd-263-1e`) | `harness_service.py`, `services/harness_mode.py` (new), `api/workspaces.py`, `config.py`, `config-surface.json`, `SystemLLMSettingsTab.tsx`, `harness-settings-section.tsx` (new) |
| **PR-1F** Telemetry hygiene, tool-failure input · `feat/prd-263-1f-telemetry-hygiene` | US-018 – US-020 | ~270 | PR-1E (the `harness_service.py` sequence; US-019 needs PR-1A, PR-1C, PR-1D) | session or Ralph (kit `prd-263-1f`) | `telemetry.py`, `tool_router.py`, `edge_builder_logs.py`, `slo_metrics.py`, `harness_service.py`, `harness_metrics.py`, `config.py`, `config-surface.json`, `api/harness_commands.py` |
| **PR-1G** Endpoint, tile, Auto's tools, docs · `feat/prd-263-1g-tile-and-tools` | US-021 – US-023, US-023a | ~450 (over 400, see "PR size") | PR-1E; US-002 decided | session or Ralph (kit `prd-263-1g`) | `api/harness.py`, the tile and `use-learning-api.ts`, `handlers_harness.py`, `actions_harness.py`, `auto_cadence.py`, `scripts/harness_backfill_runs.py` (new), `docs/platform-actions/self-management-harness.md` |
| SKL-1 | US-024 | ~60 | PR-1E | Gerard | automatos-skills `team/auto/SKILL.md` |
| SYNC-1 Seed sync | US-025 | ~60 (generated) | SKL-1 | session; Gerard merges | `platform-management-skill.md` |
| OPS | US-026b | 0 | PR-1A – PR-1E, SYNC-1 deployed | Gerard | prod |
| LOCAL-1 | US-027 | 0 | PR-1A – PR-1G, SYNC-1 | Gerard | local c1 |
| DOC-1 Eval rule | US-028 | ~10 | — | Gerard | `docs/testing/AUTO-EVAL.md` |
| QUIET (conditional) | US-029 | 0, or ~60 if Gerard picks the code option (§11-Q22) | — | recorded here; Ralph only for the code option | `consumers/chatbot/auto.py` (code option only) |
| LOCAL-2 Claim truth | US-030 – US-032 | ~170 | DOC-1; LOCAL-3 (US-032) | named build session (D5) | `replay.py` |
| LOCAL-3 Run control | US-033 – US-035 | ~260 | DOC-1 | named build session (D5) | `replay.py` |
| LOCAL-4 Skill arm, export | US-036, US-037 | ~220 | DOC-1; US-043 (US-037 only; US-036 can build first) | named build session (D5) | `replay.py`, `harness-lab/tripwire.py` |
| LOCAL-5 Reset, compare, rescore | US-038 – US-040 | ~200 | DOC-1; PR-1A (US-039) | named build session if US-028 names it; otherwise Gerard (the `phase.sh` edits are in D5's default) | `phase.sh`, `tester-kit/chain-n*.sh`, `compare-arms.py`, `arm-report.py`, `rescore.py` |
| LOCAL-6 Preflight, harvest | US-041, US-042 | ~230 | — | named build session (not the eval folder) | `next-cycle.sh`, `harvest.py` |
| LOCAL-7 Set v2, calibration | US-043 – US-045 | ~40 | US-043: DOC-1, LOCAL-3; US-044/US-045: LOCAL-2 to 6, US-029 | Gerard | `auto-eval.jsonl`, `split.json`, runs |
| DOC-2 AUTO-EVAL.md | US-046 | ~150 | DOC-1, LOCAL-2 to 4, US-043 | session | `docs/testing/AUTO-EVAL.md` |
| **PR-3A** Delete the /approve path · `feat/prd-263-3a-delete-approve` | US-047 | ~480 (deletion-only; over 400, see "PR size") | PR-1E (US-016), PR-1F (US-020), PR-1G (US-021, US-023a); US-003, US-004 approved | session or Ralph (kit `prd-263-3a`) | `api/harness.py`, `api/harness_commands.py` (deleted), `owner_only.py`, `route-manifest.json`, `harness_service.py` (one comment), the docs page |
| **PR-3B** Delete the legacy apply helpers · `feat/prd-263-3b-delete-apply-helpers` | US-048 | ~560 (~530 deletions; over 400, see "PR size") | PR-3A; US-006 decided | session or Ralph (kit `prd-263-3b`) | `harness_service.py`, `tests/conftest.py` |
| **PR-3C** Change-kind registry · `feat/prd-263-3c-change-registry` | US-049, US-050 | ~380 | PR-3B | session or Ralph (kit `prd-263-3c`) | `services/harness_changes.py` (new), `harness_service.py`, `config.py`, `config-surface.json` |
| **PR-3D** Ledger migration · `feat/prd-263-3d-ledger-migration` | US-051, US-059 | ~150 | PR-3B (US-048), PR-1G (US-023); US-005 (prod totals posted); US-006 | session or Ralph (kit `prd-263-3d`) | `core/models/harness.py`, Alembic revision `prd263_harness_ledger_grants` and both head pins, `scripts/harness_backfill_runs.py` |
| **PR-3E** Apply action, reconciliation · `feat/prd-263-3e-apply-action` | US-052, US-053 | ~370 | PR-3C, PR-3D; US-004 approved | session or Ralph (kit `prd-263-3e`) | `actions_harness.py`, `handlers_harness.py` (or `handlers_harness_apply.py`), `platform_executor.py`, `check_hierarchy_gate.py`, `services/harness_grants.py` (new), `harness_service.py`, `config.py`, `config-surface.json` |
| **PR-3F** Raise grants · `feat/prd-263-3f-raise-grants` | US-054, US-055, US-058 | ~360 | PR-3E (so after PR-3A and PR-3B); US-001; US-002 decided | session or Ralph (kit `prd-263-3f`) | `harness_grants.py`, `harness_service.py`, `config.py`, `click_memory.py`, `api/harness.py`, the tile and `use-learning-api.ts`, `harness-settings-section.tsx`, the docs page |
| **PR-3G** Post-change check, tool-removal gate · `feat/prd-263-3g-post-change` | US-056, US-057 | ~300 | PR-3F | session or Ralph (kit `prd-263-3g`) | `services/harness_post_change.py` (new), `harness_changes.py`, `harness_metrics.py`, `config.py`, `config-surface.json` |
| **PR-3I-a** Pure `build_context` refactor · `feat/prd-263-3i-a-context-refactor` (#1161) | none (enables US-062) | 276 (built) | — | session | `modules/context/service.py`, `modules/context/sections/skills.py` |
| **PR-3I-b** Production skill signal · `feat/prd-263-3i-skill-signal` (#1163, draft until #1161 merges) | US-062 | 363 (built) | PR-3I-a (merges before any Lab spend, US-073) | session | `modules/context/sections/skills.py`, `modules/context/service.py`, `modules/context/result.py`, `modules/agents/services/skill_loader.py` (`load_core_for_skill`), `scripts/skill_signal_report.py` (new) |
| SKL-2 | US-060 | ~40 | PR-3F | Gerard | automatos-skills `team/auto/SKILL.md` |
| SYNC-2 Seed sync | US-061 | ~40 (generated) | SKL-2 | session; Gerard merges | `platform-management-skill.md` |
| LOCAL-8 | US-063 | ~30 (local SQL) | PR-3F, PR-3G | Gerard | local scratch workspace |
| OPS | US-064 | 0 | US-005; PR-3D deployed | Gerard | prod |
| SKL-3 Defect text | US-065 | ~60 | — | Gerard | automatos-skills `team/auto/SKILL.md` (the slim bodies are rebased in US-069) |
| FIX-1 Defect PR | US-066 | ~180 (excl. generated seed lines) | SKL-3 | session; Gerard merges | `seed_auto_agent.py`, `scripts/sync-skills.py`, seed, `test_prd226` pins |
| LOCAL-9 Preflight baseline, gate | US-067, US-068 | ~170 | US-041, FIX-1; PR-1A, US-039 | named build session if US-028 names it; otherwise Gerard | `next-cycle.sh`, `compare-arms.py` |
| LOCAL-10 Iteration 0 | US-069 | 0 | US-044, FIX-1, LOCAL-9; SKL-1 (and SKL-2 once merged) for the rebase | Gerard | local c1 |
| ISSUES (conditional) | US-070 | 0 | US-069 accepts | Gerard | GitHub issue |
| SKL-4 (conditional) | US-071 | ~10 plus the rebased bodies | US-069 accepts | Gerard | automatos-skills |
| SHIP-1 (conditional) | US-072 | ~150 (excl. generated seed lines) | US-070, SKL-4 | session; Gerard merges | `skills/manifest.json`, seeds, `seed_auto_agent.py` |
| LOCAL-11 Lab core | US-073 – US-075 | ~370 | US-045, PR-3I (US-062) | Gerard (US-073); named build session | `~/.automatos-analyst/harness-lab/` |
| LOCAL-12 Channel, prompts | US-076, US-077 | ~200 | US-073 | named build session | `harness-lab/` |
| LOCAL-13 Gate wiring, critic, monitor | US-078 – US-080 | ~340 | LOCAL-11, LOCAL-12, US-068 | named build session | `harness-lab/` |
| LOCAL-13a LAB-1 calibration | US-078a | 0 | US-073, US-074, US-037 | Gerard | local c1 |
| LOCAL-14 Iterations | US-081 | 0 | LOCAL-13, US-078a | Gerard | local c1 |
| SKL-5 (conditional) | US-082 | ~30 | an IMPROVE in US-081 | Gerard | automatos-skills |
| SYNC-3 (conditional) | US-083 | ~20 (generated) | SKL-5 | session; Gerard merges | `platform-management-skill.md` |

- **PR size (D3).** P1 is seven PRs and P3 eight, one per story group, each aimed at AGENTS.md's ~400 changed lines excluding tests. Ten are at or under it. Five are over, and each stays one PR unless Gerard picks the split noted:
  - PR-1D (~435): about 250 lines are deletions. US-014b cannot leave it: `_detect_auto_applied_regressions`, which US-014b deletes, reads the baseline that US-014a deletes (`harness_service.py:516`, `:620`, `:1470`), so a split leaves a half-deleted path (replace, don't shim). US-015 is test-only.
  - PR-1E (~565): about 110 lines are the HARNESS settings block moved into `harness-settings-section.tsx`, which the code-shape gate requires. US-016 changes the modes the API accepts and US-017 the tab that sends them, so either alone leaves Settings offering modes the API rejects with 400 (today the tab offers only `full_auto` and `manual`, `SystemLLMSettingsTab.tsx:1081-1082`, and the API accepts only those two, `api/workspaces.py:576`). Splitting them would half-replace one setting (replace, don't shim).
  - PR-1G (~450): a clean split exists, as an option for Gerard: US-021 and US-022 (endpoint and tile, which wait on the US-002 decision) as one PR, and US-023 and US-023a (Auto's tools, cadence text, backfill script, docs page) as another, which do not wait on US-002. No story depends across the two. Default: one PR.
  - PR-3A (~480): deletion-only. The two routes, `api/harness_commands.py` and `OWNERS_OWN_DECISION` (whose only production caller is that module, `harness_commands.py:31`, `:234`) go together; a split leaves a dead module or a dead bypass key (replace, don't shim).
  - PR-3B (~560): about 530 lines are deletions of helpers whose last callers PR-3A deleted; a split leaves part of that dead code in `harness_service.py` (replace, don't shim).
- **Order inside P1.** US-026a is recorded before PR-1B (US-011's sweep design must match the replica count) and PR-1D (US-014b deletes the baseline rollback check) merge. PR-1A, PR-1B and PR-1C are independent of each other and merge in any order; PR-1D needs all three. The PRs that touch `harness_service.py` merge one at a time in this order, PR-1B, PR-1D, PR-1E, PR-1F, each rebased on main after the previous one merges and before its own PR opens, with a seam review of its `harness_service.py` diff after the rebase (the 15 Oct parallel-lanes lesson: merges between lanes produced 14 seam defects). PR-1E needs an approved US-001. PR-1G needs PR-1E (US-021 and US-023a build on US-016) and the US-002 decision; it does not touch `harness_service.py`, so it can merge before or after PR-1F.
- **P2** runs in Gerard's daytime windows alongside P1. DOC-1 (US-028) merges first, and LOCAL-2 to LOCAL-5, LOCAL-7 and DOC-2 are held until it does. LOCAL-6 does not need D7. US-039 loads `harness_verdicts.py` by path, so it waits for PR-1A.
- **P3** starts when its maintainer issues are approved. PR-3A needs PR-1E (US-016 deletes `_phase_apply`), PR-1F (US-020 touches `harness_commands.py`, which PR-3A deletes) and PR-1G (US-021 touches `api/harness.py`; US-047 deletes the `/approve` note US-023a writes), plus approved US-003 and US-004. PR-3B follows PR-3A and needs US-006 decided. PR-3C (after PR-3B) and PR-3D (after PR-3B and PR-1G, with US-005's prod totals posted and US-006 decided) are independent of each other. PR-3E needs PR-3C, PR-3D and an approved US-004. PR-3F needs PR-3E, US-001 and the US-002 decision; through PR-3E it merges only after PR-3A and PR-3B, so the redefined switch can never re-enable an old path. PR-3G needs PR-3F. The P3 PRs that touch `harness_service.py` (PR-3A, PR-3B, PR-3C, PR-3E, PR-3F) merge one at a time in that order, each rebased on main and seam-reviewed as in P1. PR-3I depends on no other PRD-263 PR and must merge before P5 spends (US-073); if PR-3I slips past 31 Oct, P5 starts later (§11-Q31).
- **P4** needs P2's calibration (US-044) and US-066 merged, and runs on one pinned `AUTOMATOS_IMAGE_TAG`. US-070 to US-072 are built only if US-069 accepts.
- **P5** needs every precondition in US-073 and the LAB-1 calibration (US-078a), including at least 6 valid failing held-out rows on the Lab's pinned tag.
- **Holds.** c1 stays `report_only` through WebSummit (10–13 Nov). The skill freeze is 6 Nov unless Gerard moves it. Prod `HARNESS_SELF_MANAGEMENT_ENABLED` stays false unless Gerard flips it (US-064).
- **Merging.** Squash-merge with an explicit `--subject` and `--body` that contain no `[skip ci]` token, or main runs no CI (#1138, 16 Oct).
- **Anchors.** Stories cite `file:line` as of `697ca7d6e`. PR-1B onward rewrites `harness_service.py`, so stories in PR-1D and later name the function, constant or key there, not the line. Each build prompt, kit or session, repeats "RE-VERIFY every anchor by grep"; a premise that has gone means `RALPH_BLOCKED` (or, in a session, a stop) with the grep.
- **Ralph kits.** Each small PR may be built by a session or by Ralph. A Ralph-built PR has one kit: `scripts/ralph/prd-263-1a.json` to `prd-263-1g.json` and `prd-263-3a.json` to `prd-263-3i.json`, each with the matching `PROMPT_build_`, `PROMPT_review_`, `acceptance-` and `launch-` files and the spec copy `prd263-spec/PRD-263-HARNESS-ORG-LOOP-SKILL-LAB.md`. A session builds from the same story text, house rules and reviewer runs. Branch `feat/prd-263-<group>-<slug>` (the §4 table; `feat/prd-263-1a-harness-verdicts` and `feat/prd-263-1c-harness-metrics` exist); story ids `P263-1A-US-008`. Each kit's `decisions` (or a session's build prompt) carry D1–D6 and, for every §11 question its stories need, Gerard's answer if given, else the design default marked "default (open)" (Gerard, 18 Oct). A story is held only if Gerard has rejected its default, or if it waits on D5 or on a maintainer issue's approval. `→ OWNER:` ACs go into `ownerTest` and the loop never marks them done. Each build prompt says: "Do not open or copy google-research/harness2 sources; implement from the PRD text." Worktrees go under `<workspace>/.worktrees/automatos-ai/<slug>`. A foreground code-reviewer runs on the migration, gate, approval and auth stories (US-009, US-016, US-047, US-048, US-051, US-052, US-054), database-reviewer on US-009 and US-051, and security-reviewer on US-021, US-047, US-052 and US-054 before their PRs merge.
- **Open questions and the stories they touch** (none holds a story unless Gerard rejects its default):

  | §11 | Stories | In the kit |
  |---|---|---|
  | Q1 (D5) | every Needs D7 story | held until DOC-1 merges (D5) |
  | Q2 (D6) | none built here | — |
  | Q3 probe workspace | US-044, US-069, US-078a, US-081 | default: c1 |
  | Q4 set v2 | US-043 | default |
  | Q5 defect route | US-065, US-066 | default: fix directly |
  | Q6 slim skill | US-069 – US-072 | default |
  | Q7 expiry | US-007, US-052, US-053 | default (a), 24 h |
  | Q8 "asked" status | US-018 | default: accept |
  | Q9 dead kinds | US-049 | default: report lines |
  | Q10 heartbeat detector | US-050 | default values |
  | Q11 reference windows | US-014b, US-056 | default |
  | Q12 schedule options | US-017 | default: remove |
  | Q13 prod facts | US-026a | recorded privately |
  | Q14 improver models | US-073, US-076 | default |
  | Q17 tool removal | US-057 | default: gated, report-only if the check cannot be built |
  | Q18 click memory | US-055 | default: exclude |
  | Q19 `/self-learning` readers | US-021, US-058 | needs the US-002 decision (maintainer issue) |
  | Q20 history | US-023, US-059 | needs the US-006 decision (maintainer issue) |
  | Q21 Lab-editable sections | US-073 | Gerard marks them (owner step) |
  | Q22 quiet stack | US-029, US-038 | default: existing switches plus the reset-time cache flush |
  | Q23 persona lift | US-066 | default: no separate issue |
  | Q25 non-Haiku check | US-079 | default |
  | Q27 US-020 | US-020 | default: keep |
  | Q28 legacy mode values | US-016 | default: read-time mapping |
  | Q29 unpriced usage | US-035 | answered 20 Oct: (c), embeddings allowed and reported |
  | Q30 blocked evidence card | US-053, US-054 | default: reconcile to failed |
  | Q31 US-062 timing | US-062, US-073 | default: PR-3I merges early in P3; P5 waits for PR-3I only |

## 5. User stories

House rules for every story: one commit per story, one PR per story group as in §4, one CI run per PR. Nothing runs locally: no pytest, npm, tsc, docker, server, browser, eval, `replay.py` or `phase.sh` inside a build loop. Both editions keep working. No new dependency, and the only new table is `harness_runs` (US-009). Every new test fails without its change (against `697ca7d6e`), and CI proves it passes. Ralph-built backend stories end, in their kit, with "`python3 -m py_compile` passes on every changed file; ruff and the changed-line code-shape gate pass on the touched lines (functions ≤ 50 code lines, nesting ≤ 4, no file grown past 800 lines)" and "Committed with `git commit -s` (DCO Signed-off-by), message ending ` [skip ci]`. The runner pushes to a branch with no PR and opens the draft PR once when the build is complete, which starts the single CI run (house kit D9)." Every PRD-263 PR: `git diff 697ca7d6e...HEAD | grep -c "$(printf '\302\262')"` returns 0 (no superscript-2 form of the Harness2 name), and the PR title, body and commit messages are checked the same way; this PRD and the `prd263-spec/` copy are checked before they merge. No Harness2 source code or prompt text is copied into any repository (§9). US-001–US-007 and US-070 are the maintainer issues. US-016, US-021, US-047, US-048, US-051, US-052, US-054, US-059 and US-072 touch permissions, guardrails, a public route or existing data: each PR links its approved issue before merge.

Tags: **Human-gated** (Gerard does or approves it, outside the loop) · **Needs D7** (held until US-028 merges) · **Maintainer issue** · **Local-only** (lives under `~/.automatos-analyst/` or `~/.automatos-sim/`, never committed). An AC marked **→ OWNER (local, $0)** is a check Gerard runs on the local stack with no model call; it touches the local DB or a live session, so no build session runs it.

### Part 1 (P1): Honest HARNESS, report-only for every workspace

**Maintainer issues, decision records and prod facts** (opened at P1 start; no code; each issue uses `.github/ISSUE_TEMPLATE/feature_request.yml`)

#### US-001: Maintainer issue: report-only default, auto-apply deleted, global switch redefined (Gerard's sign-off first)
**Description:** As Gerard, I want an AGENTS.md maintainer issue recording the change of `orchestrator.harness.mode` from `full_auto` to `report_only`, the deletion of auto-apply and `HARNESS_AUTO_APPLY_MAX_RISK_STANDARD`/`_FULL`, the mode becoming the only per-workspace switch, and the redefinition of `HARNESS_SELF_MANAGEMENT_ENABLED`, so this autonomy and permissions change is approved before PR-1E and PR-3F merge.
*Tags: Human-gated · Maintainer issue*
**Files:** GitHub issue on automatos-ai · **Depends on:** —

**Acceptance criteria:**
- [ ] Issue titled "PRD-263: HARNESS report_only default, delete auto-apply, redefine HARNESS_SELF_MANAGEMENT_ENABLED", citing `harness_service.py:364-432`, `api/workspaces.py:568-576` and `:797-798`, `config.py:942-952` and `owner_only.py:43-64` (PRD-256 D1).
- [ ] The body states D2 (18 Oct): `report_only` for every workspace through beta; `propose` is a per-workspace opt-in; auto-apply is deleted, not made opt-in; the global switch defaults to false. It states that the autonomy-ceiling fail-safe inside `_auto_apply_max_risk` is deleted with auto-apply.
- [ ] The body states, as its own paragraph: the escalation path (`_maybe_escalate`, `_build_escalation_message`, `_workspace_has_channel`, `HARNESS_HIGH_PRIORITY_RISK`) is not in the design's §10 list; it is dead once `_phase_apply`'s queued-card branch, its only caller, is deleted (`harness_service.py:995`). It sent a connected-channel notice for high-risk queued cards (`:1036-1039`). From P3 the only nudge is the in-app `approval_pending` notification, so channel delivery of HARNESS approvals stops unless `approval_pending` reaches channels.
- [ ] The body states the behaviour change to the switch: `orchestrator.harness.mode` (`report_only`, `propose`, `disabled`) becomes the only per-workspace control; the "Enable HARNESS" switch and the `enabled`/`disabled` keys are no longer written or validated; at read time a stored `full_auto` or `manual` reads as `report_only`, and a stored `disabled: true` or `enabled: false` reads as `disabled`; no stored data is rewritten (§11-Q28 asks whether to convert once instead). Mapping `manual` to `report_only` rather than `propose` is deliberate: `propose` is an opt-in made under the new meaning.
- [ ] A maintainer approval comment exists on the issue before PR-1E merges; PR-1E and PR-3F link the issue number.

#### US-002: Maintainer issue: who may read `/api/harness/self-learning` (Gerard's sign-off first)
**Description:** As Gerard, I want a maintainer issue deciding who may read `GET /api/harness/self-learning`, so the permissions change in PR-1G is decided before it is built.
*Tags: Human-gated · Maintainer issue*
**Files:** GitHub issue on automatos-ai · **Depends on:** —

**Acceptance criteria:**
- [ ] The issue cites `api/harness.py:87-91` (authenticated through `get_request_context_hybrid`, no role check) and proposes §11-Q19's recommendation: workspace members read counts, coverage and mode only, never grant text.
- [ ] The decision is recorded on the issue before PR-1G merges; US-021 (PR-1G) and US-058 (PR-3F) implement exactly that decision and link the issue.

#### US-003: Maintainer issue: remove the two prescription routes (Gerard's sign-off first)
**Description:** As Gerard, I want a maintainer issue for removing `POST /api/harness/prescriptions/{rx_id}/approve` and `/reject`, so the public route removal in PR-3A is approved first.
*Tags: Human-gated · Maintainer issue*
**Files:** GitHub issue on automatos-ai · **Depends on:** —

**Acceptance criteria:**
- [ ] The issue lists both routes (`api/harness.py:67-84`; `orchestrator/reports/route-manifest.json:1334` and `:1338`) and records that a grep of `frontend/components`, `frontend/hooks`, `frontend/lib` and `frontend/app` for `prescriptions/` returns 0 matches at `697ca7d6e`.
- [ ] It names the replacement rail: the tool_call ApprovalGrant for `platform_harness_apply`, shown in Command Center > Governance > Approvals and in the evidence ticket.
- [ ] Approved before PR-3A merges; PR-3A links it.

#### US-004: Maintainer issue: two guardrail edits (Gerard's sign-off first)
**Description:** As Gerard, I want one maintainer issue covering the two guardrail edits P3 needs, deleting `OWNERS_OWN_DECISION` and letting the new write action `platform_harness_apply` pass `check_hierarchy_gate.py`, so no guardrail changes without approval.
*Tags: Human-gated · Maintainer issue*
**Files:** GitHub issue on automatos-ai · **Depends on:** —

**Acceptance criteria:**
- [ ] The issue covers deleting `OWNERS_OWN_DECISION` from `orchestrator/modules/tools/discovery/owner_only.py` (docstring `:16`, comment `:86-88`, constant `:89`, branch in `human_driven` `:217` and its docstring `:212-214`, `__all__` entry `:545`). Its only production caller is `api/harness_commands.py:31` and `:234`.
- [ ] It covers how `platform_harness_apply` (`permission_level` write, `requires_confirmation` False) passes `orchestrator/scripts/check_hierarchy_gate.py`. The recommendation is an `ALLOW_LIST` entry with a comment saying why: the inner heartbeat or unassign call passes through `can_actor_modify`, which for Auto checks only that the actor exists, is in the workspace and is active, then bypasses the hierarchy (`system_actor_bypass`, `hierarchy_permissions.py:166-195`; Auto is in `SYSTEM_BYPASS_ALLOWLIST`, `:79-85`). Consent therefore rests on the grant, expiry and prior-value checks in the handler, plus the rate-limit and destructive backstop on the inner call (`_HIERARCHY_TARGETS`, `platform_executor.py:329`, `:335`).
- [ ] It states that the case at `test_prd256_owner_actions_wait_for_the_click.py:181` is replaced by a stricter test, not weakened.
- [ ] Approved before PR-3A (US-047) and PR-3E (US-052) merge; both PRs link it.

#### US-005: Maintainer issue: the ledger migration that alters existing rows (Gerard's sign-off first)
**Description:** As Gerard, I want a maintainer issue for the ledger migration, so the data change in PR-3D is approved with real prod totals in hand.
*Tags: Human-gated · Maintainer issue*
**Files:** GitHub issue on automatos-ai · **Depends on:** —

**Acceptance criteria:**
- [ ] The issue describes US-051's change: new columns `rx_id` (unique per workspace), `run_id`, `grant_id` (Integer, matching `approval_grants.id`), `kind`, `target_agent_id`, `prior_value`, `proposed_value`, `applied_at`, `value_at_apply`, `frozen_ref_start`, `frozen_ref_end`; `board_task_id` nullable; the state CHECK becoming `pending, applying, applied, failed, denied, expired, drifted, withdrawn, raise_failed, rolled_back`; existing `held` rows mapped to `expired` (`core/models/harness.py:24-32`; `alembic/versions/f156_harness_task_ledger.py:27-33`).
- [ ] → OWNER: Gerard posts the prod output of `SELECT state, count(*) FROM harness_task_ledger GROUP BY state` on the issue before merge (aggregate rows only, no workspace column).
- [ ] Approved before PR-3D merges; PR-3D links it.

#### US-006: Maintainer issue: HARNESS files on tenant volumes (Gerard's sign-off first)
**Description:** As Gerard, I want a maintainer issue deciding what happens to the HARNESS JSON files on tenant volumes, so history is kept or dropped deliberately and user data is removed only after approval.
*Tags: Human-gated · Maintainer issue*
**Files:** GitHub issue on automatos-ai · **Depends on:** —

**Acceptance criteria:**
- [ ] The issue proposes §11-Q20's recommendation: backfill `harness_runs` once from the `<WORKSPACE_VOLUME_PATH>/<ws>/harness` baseline and last_run JSON (US-023), then remove those files (US-059); and retire the F156 legacy ledger-file importer (`harness_service.py:54-66`, `:1817-1916`) once its callers are deleted (US-048).
- [ ] It states that file removal runs only after a verified per-workspace import count, that Gerard runs it against prod, and that only totals are posted on public issues (no workspace id, name or slug).
- [ ] The decision (backfill then remove, or remove without backfill) is recorded before PR-3B (US-048) and PR-3D (US-059) merge.

#### US-007: Decision record: grant expiry is enforced by the apply action, and 24 h stands
**Description:** As Gerard, I want the expiry decision recorded, so the grant rail changes only through a maintainer issue if I pick a non-default option.
*Tags: Human-gated*
**Files:** this PRD (§11-Q7) · **Depends on:** —

**Acceptance criteria:**
- [ ] §11-Q7 is answered. At the recommendation (a), the dedicated `platform_harness_apply` action checks `is_authorising` and the prior value, a daily sweep uses `expire_pending_grants`, and the 24 h `APPROVAL_GRANT_TTL_SECONDS` default (`config.py:1018`) stands.
- [ ] With (a): every PRD-263 PR's diff against `697ca7d6e` leaves `orchestrator/modules/tools/execution/tool_grants.py` and `orchestrator/api/approval_grants.py` unchanged, and no issue is opened.
- [ ] With option (b) or a `ttl_seconds` parameter: a maintainer issue is opened and approved before either file changes.

#### US-026a: P1 safety facts are recorded privately before PR-1B and PR-1D merge
**Recorded:** US-026a, 19 Oct 2026: HARNESS_SELF_MANAGEMENT_ENABLED confirmed off in prod.

**Description:** As Gerard, I want the prod facts that decide P1's safety recorded before PR-1B and PR-1D merge, so rollback detection is not deleted while the legacy apply path could still be on in prod (US-014b, PR-1D), and the sweep design matches the replica count (US-011, PR-1B).
*Tags: Human-gated*
**Files:** a private ops note outside every public repo (for example `<workspace>/.claude/PRD263-OPS.md`); this PRD (one line) · **Depends on:** — · **Blocks:** PR-1B, PR-1D

**Acceptance criteria:**
- [ ] → OWNER: Gerard records in the private ops note: the Railway API replica count; the prod values of `HARNESS_SELF_MANAGEMENT_ENABLED`, `HARNESS_ENABLED`, `AUTOMATOS_POLICY_PLANE` and `LLM_MODEL`; and the per-model count of Auto agents.
- [ ] → OWNER: if prod `HARNESS_SELF_MANAGEMENT_ENABLED` is true, Gerard sets it to false before PR-1D merges (PR-1D deletes the baseline-based rollback check in US-014b, §15 risk 24), or records that he keeps it on.
- [ ] This PRD records only the date and one of: "HARNESS_SELF_MANAGEMENT_ENABLED confirmed off in prod" or "turned off on <date>" or "kept on by Gerard on <date>". No prod setting value, replica count or tenant figure is written into this PRD or a public issue.

**PR-1A: one statistics module** (US-008; ~240 changed lines)

#### US-008: One standard-library statistics module serves the org loop and the Lab
**Description:** As a maintainer, I want one pure standard-library statistics module outside the services package, so the org loop and the local Skill Lab judge evidence with the same code and the Lab loads it without the DB import chain.
**Files:** `orchestrator/core/stats/__init__.py` (new, empty), `orchestrator/core/stats/harness_verdicts.py` (new, ~220 lines), `orchestrator/core/services/edge_builder.py` (916 lines), `orchestrator/tests/test_prd263_harness_verdicts.py` · **Depends on:** —

**Acceptance criteria:**
- [ ] `orchestrator/core/stats/harness_verdicts.py` imports only the Python standard library. It lives outside `orchestrator/services/` because `services/__init__.py:11` imports `metadata_sync_service`, which imports SQLAlchemy, `core.composio.client` (and through it `config`) and the models (`metadata_sync_service.py:21-25`); `core/__init__.py` holds a docstring only.
- [ ] Three import tests: `test_stdlib_only` parses the module's imports with `ast` and fails on any non-stdlib module; `test_stdlib_only_in_a_clean_interpreter` runs `import core.stats.harness_verdicts` in a subprocess with `DATABASE_URL` unset and asserts `'sqlalchemy'` and `'config'` are not in `sys.modules`; `test_loads_by_path_without_sqlalchemy` loads the file by path with `importlib.util.spec_from_file_location` in a subprocess and asserts the same.
- [ ] It provides: a minimum-n gate; a one-sided two-proportion z test with pooled variance and no continuity correction, where a proportion flags only when p < alpha AND the absolute drop ≥ abs_drop AND the relative drop ≥ rel_drop; Mann-Whitney U; Benjamini-Hochberg; an exact one-sided sign test with its minimum-W table; `wilson_lower_bound` and `wilson_upper_bound`; the WATCH-then-REGRESSION rule keyed to distinct consecutive ISO-week windows (or one flag at a caller-supplied strong p); the frozen-reference post-change test; and the diff-in-diff test with a recorded "no control" result. Every threshold is a parameter, and the module imports no config.
- [ ] `wilson_lower_bound` moves from `core/services/edge_builder.py:156-165`. edge_builder imports it with `from core.stats.harness_verdicts import wilson_lower_bound` (edge_builder uses it itself), which keeps the name importable from edge_builder for `signal_recorder.py:137` and the tests at `test_prd139_edge_builder.py:47-49`, `test_graph_router_negative.py:393` and `test_prd143_graph_seed.py:317`; re-verify each by grep and update any that break. `grep -rn "def wilson_lower_bound" orchestrator/` returns exactly one hit, and `edge_builder.py` ends shorter than 916 lines.
- [ ] Tests: `test_prd263_harness_verdicts.py` pins:
  - exact: single false-flag rate 0.049 (0.0492, p=0.8) and 0.015 (0.0153, p=0.9) at n_cur=20 vs n_ref=80; single-flag power 0.837 (0.8→0.5) and 0.571 (0.8→0.6); the sign-test minimum-W table, W+L of 5→5, 6→6, 7→7, 8→7, 9→8, 10→9, 11→9, 12→10; today's rule (re-implemented in the test as a reference: largest relative delta above 10%, success down or error rate up, same n both weeks) at 0.388 (n=10) and 0.421 (n=20), and for success rate alone 0.359 (n=10) and 0.274 (n=20);
  - [M] 20,000 seeded draws per case, tolerance ±0.01 unless stated: two consecutive false flags at p=0.8, 0.0016 (±0.001); frozen 28-day pre-change reference, trigger = two consecutive flags or one at the strong p within 4 weeks: 0.938 / 0.668 / 0.443 for drops of 0.8→0.5 / 0.6 / 0.65, and false rollback per metric 0.020 at p=0.8 and 0.011 at p=0.9 (±0.003) (`frozen_sim.py`); trailing 28-day reference: flag probability in weeks 1–4 after a sustained 0.8→0.5 step 0.835 / 0.602 / 0.338 / 0.147, REGRESSION by week 2 0.494, and under the same trigger any REGRESSION within 4 weeks 0.747 (0.8→0.5) and 0.383 (0.8→0.6), against the stricter "REGRESSION in two consecutive weeks" at 0.33 and 0.085 (`consec_sim.py`);
  - each test runs in under 5 s in CI.
- [ ] No requirements file changes.

**PR-1B: run store, lock and cron** (US-009 – US-012; needs US-026a). Built as two PRs: PR-1B-a #1160 (US-009, the table, 136 lines) and PR-1B-b #1162 (US-010 – US-012, 863 lines). US-012 cannot move into 1B-a, because its change to `get_status` (87 code lines at the base) trips the 50-line shape gate unless `get_status` is rewritten, and that rewrite is US-010/011.

#### US-009: HARNESS runs get their own table
**Description:** As the org loop, I want a `harness_runs` table holding each run's window, coverage, counts, verdicts, prescriptions and report, so run state lives in the database instead of JSON files on the volume.
**Files:** `orchestrator/core/models/harness.py`, `orchestrator/alembic/versions/prd263_harness_runs.py`, `orchestrator/tests/test_prd209_alembic_single_head.py`, `orchestrator/tests/test_prd236_w1_routes.py`, `orchestrator/tests/test_prd263_harness_runs_table.py` · **Depends on:** —

**Acceptance criteria:**
- [ ] `HarnessRun` in `orchestrator/core/models/harness.py` has `workspace_id` (`ForeignKey('workspaces.id', ondelete='CASCADE')`), `cur_window` (ISO week, e.g. `2026-W42`), `started_at`, `finished_at`, `status`, `iteration`, `coverage`, `counts` (JSONB), `verdicts` (JSONB), `prescriptions` (JSONB), `report_md`, `report_status` (`ok`, or `failed: <reason>`), and the index `ix_harness_runs_ws_window` on `(workspace_id, cur_window)`. The migration declares the same names, so the alembic-from-zero lane sees one shape.
- [ ] ONE Alembic revision, `prd263_harness_runs`, chains onto the single head read from `EXPECTED_HEAD` in `test_prd209_alembic_single_head.py` (`prd258_voice_timings` at `697ca7d6e`; re-read at build time). It uses `CREATE TABLE IF NOT EXISTS` and `CREATE INDEX IF NOT EXISTS`, so a `create_all`-built database keeps its table and gains only missing indexes. Downgrade uses `IF EXISTS`.
- [ ] Both head pins move: `EXPECTED_HEAD` in `test_prd209_alembic_single_head.py` and the head assertion in `test_prd236_w1_routes.py` (`:266-270`, inside `test_migration_chains_onto_prd234_head_and_guard_follows`, at `697ca7d6e`).
- [ ] No existing table or row changes (new table only, so no maintainer issue).
- [ ] Tests: `test_prd263_harness_runs_table.py` round-trips a row and asserts that a query scoped to workspace A never returns workspace B's row.

#### US-010: Every run is recorded in `harness_runs`, not in JSON files
**Description:** As an owner and as Auto, I want every HARNESS run recorded in `harness_runs`, so status and history read one database row that every worker sees.
**Files:** `orchestrator/services/harness_service.py`, `orchestrator/tests/test_prd263_harness_run_record.py`, `orchestrator/tests/test_w1s9_idle_in_tx.py` · **Depends on:** US-009

**Acceptance criteria:**
- [ ] Each tick opens a `harness_runs` row at the start (ORG-1) with `status='running'` and `cur_window` set to the last complete ISO week, and closes it at the end (ORG-11) with `status`, `finished_at`, `counts`, `report_md` and `report_status`. `iteration` = the number of completed `harness_runs` rows for the workspace, this one included. `get_status()` reads the latest row and keeps `artifacts.audit_report` from `report_status`.
- [ ] `_last_run_path`, `_read_last_run`, `_write_last_run` and the trace-file writes are deleted (grep in `harness_service.py` returns 0 for each).
- [ ] `test_w1s9_idle_in_tx.py::test_harness_commits_before_each_phase` (`:506`) is rewritten for the run-row helpers and the `_write_last_run` deletion, stubbing them on its fake session. It still asserts that a commit precedes every awaited phase; it is not deleted (AGENTS.md:121).
- [ ] Tests: `test_prd263_harness_run_record.py`: a tick on a seeded workspace writes exactly one `harness_runs` row and no `last_run.json` or trace file under a temp `<WORKSPACE_VOLUME_PATH>/<ws>/harness`; `iteration` is 2 on the second completed run.
- [ ] `harness_service.py` (2,165 lines at `697ca7d6e`) does not grow; the before and after counts go in the PR description.

#### US-011: One run per workspace per judged week, whatever the worker or replica count
**Description:** As the org loop running in 4 uvicorn workers and possibly several Railway replicas, I want the tick and `trigger_now` guarded by a per-workspace advisory lock and a per-week run check, so no workspace runs twice at once and the weekly sweep never repeats a week.
**Files:** `orchestrator/services/harness_service.py`, `orchestrator/tests/test_prd263_harness_advisory_lock.py`, `orchestrator/tests/test_w1s9_idle_in_tx.py` · **Depends on:** US-009, US-010, US-026a

**Acceptance criteria:**
- [ ] `_harness_tick` and `trigger_now` take `pg_try_advisory_lock` in its two-key form: a fixed HARNESS namespace constant (distinct from `BOOT_LOCK_ID` 47111) and `hashtext(<workspace_id>)`. The lock is held on one dedicated connection for the whole tick and released in `finally` (the pattern in `core/database/boot_lock.py:25-52`). A held lock makes the call a logged no-op.
- [ ] After taking the lock, the sweep skips (logged no-op) any workspace whose `harness_runs` already holds a `completed` row for the same `cur_window`. A `running` row for that window with no lock holder is closed as `failed` ("interrupted") and the run proceeds. `trigger_now` is exempt from the skip and reuses that window's verdicts (US-014b). This covers several Railway replicas, each of which starts its own scheduler (`main.py:486-499` uses a per-container file lock).
- [ ] `get_status()` reports `running` only while the workspace's advisory lock is held, read from `pg_locks` without taking the lock; a latest row with no `finished_at` and no lock holder reads `failed` ("interrupted").
- [ ] `self._running` and every read of it are deleted (grep `_running` in `harness_service.py` returns 0). `test_w1s9_idle_in_tx.py` stays green with the lock helper stubbed.
- [ ] Tests: `test_prd263_harness_advisory_lock.py`: two concurrent ticks for one workspace on two DB sessions run the phases once; ticks for two different workspaces both run; two sequential sweeps for one workspace and one `cur_window` write exactly one `harness_runs` row and one audit report; a stale `running` row is closed as interrupted.

#### US-012: The sweep runs on a weekday daytime cron with an explicit UTC timezone
**Description:** As Gerard, I want the HARNESS sweep on a configurable weekday daytime cron with an explicit timezone, so it never runs in the night window where F408, F409 and F416 happened.
**Files:** `orchestrator/config.py`, `orchestrator/reports/config-surface.json`, `orchestrator/services/harness_service.py`, `orchestrator/tests/test_prd263_harness_sweep_cron.py` · **Depends on:** —

**Acceptance criteria:**
- [ ] `HARNESS_SWEEP_CRON` in `orchestrator/config.py`, default `"0 10 * * 1"` (Monday 10:00 UTC); its name goes in `orchestrator/reports/config-surface.json` → `settings`, in sorted position. Code reads it as `config.HARNESS_SWEEP_CRON` (`config = Config()`, `config.py:2320`).
- [ ] `_CRON_EXPRESSION` (`harness_service.py:35`) is deleted. The job uses `services.schedule_util.scheduler_cron_trigger(config.HARNESS_SWEEP_CRON, timezone=timezone.utc)`, never `CronTrigger.from_crontab`, which reads a numeric weekday one day late under APScheduler 3.11 (F240, `schedule_util.py:75-92`).
- [ ] The literal "Sunday 02:00 UTC" at `harness_service.py:107`, `:123` and `:165` is replaced: log lines name the configured cron, and `next_scheduled_run` is the trigger's next fire time in ISO form, UTC, computed from the trigger (so any worker can answer). `grep -n Sunday orchestrator/services/harness_service.py` returns 0.
- [ ] Tests: `test_prd263_harness_sweep_cron.py`: `trigger.get_next_fire_time(None, datetime(2026,10,18,12,0,tzinfo=timezone.utc))` equals 2026-10-19 10:00 UTC (a Monday); the default's day of week excludes Saturday and Sunday; its hour is between 08 and 17; its timezone is UTC.

**PR-1C: the per-agent read model** (US-013; ~260 changed lines)

#### US-013: One SQL read returns each agent's real counts for the judged week
**Description:** As the org loop, I want one workspace-bound SQL read that returns raw per-agent counts for the judged week and its reference window, so DIAGNOSE judges real data instead of four fields `platform_list_agents` never returns.
**Files:** `orchestrator/services/harness_metrics.py` (new), `orchestrator/tests/test_prd263_harness_metrics_realdb.py`, `orchestrator/tests/test_prd263_tool_error_classes.py` · **Depends on:** —

**Acceptance criteria:**
- [ ] New `orchestrator/services/harness_metrics.py` (target ~250 lines, max 800) runs one workspace-bound query through `text()` with bound parameters, called through `asyncio.to_thread`. It returns raw counts per agent for the `cur` window (the ISO week judged) and the `ref` window, LEFT JOINed. Zero rows means n=0, never 0%.
- [ ] For cost and tokens it also returns, per agent and window, one value per completed run (billed $ and summed tokens) as Mann-Whitney input; still no rates.
- [ ] Roster: active non-Auto agents with `model_config->>'model_id'`, `configuration->'heartbeat'->>'interval_minutes'` (default 60) and `heartbeat.enabled`. The row's field names are one declared constant, `AGENT_ROW_FIELDS`.
- [ ] Each data trap is handled, with a test case for each:
  - `board_tasks`: done and failed, excluding heartbeat, mission-linked and harness-tagged rows;
  - `llm_usage`: a per-metric whitelist of `request_type` and `status`, so `reserved` and `pending` rows (`core/models/core.py:211`) and `memory_distill` and `complexity_assessor` rows are excluded;
  - `heartbeat_results`: `source_type='agent'` only, with the cast inside a `CASE`, so an orchestrator row whose `source_id` holds the workspace UUID (`heartbeat_service.py:154-166`) does not raise; `objective_met` None is counted separately and kept out of the denominator;
  - `orchestration_tasks`: scoped through `orchestration_runs.workspace_id`;
  - `tool_execution_logs`: `telemetry_source='production'` and `status IN ('success','error','timeout')` (so `asked`, `gap`, `shown` and `refused` rows are never counted), grouped by agent and `app_name`; error rows are grouped by `error_code` and a normalised `error_message` prefix and classified in Python.
- [ ] `classify_tool_error(error_code, error_message) -> 'auth' | 'connection' | 'rate_limit' | 'other'` lives here, driven by two module constants, `RECONNECT_CODES` and `RECONNECT_PATTERNS`, each entry citing the `file:line` that produces that code or message (re-verified by grep at build time). It reads `error_code` when present and falls back to `error_message` patterns, so rows written before US-018 (where `error_code` is always NULL) still classify.
- [ ] Tests: `test_prd263_harness_metrics_realdb.py` (conftest `db_session`, two workspaces) has one case per trap above, a per-run cost case, and a case asserting workspace B's rows never appear in workspace A's result. `test_prd263_tool_error_classes.py`: a Composio expired-auth message with NULL `error_code` → `auth`; a connection-refused message → `connection`; `integrations_unavailable` (`tool_router.py:202`) → `connection`; an unrelated message → `other`.
- [ ] No new dependency.

**PR-1D: honest DIAGNOSE** (US-014a, US-014b, US-015; ~435 changed lines, ~250 of them deletions; over ~400 because most of the excess is deletions, and splitting US-014b out would leave a half-deleted path, §4 "PR size"; needs PR-1A, PR-1B, PR-1C and US-026a)

#### US-014a: HARNESS judges each agent on real counts and says when it cannot judge
**Description:** As an owner, I want HARNESS to judge each agent on its real counts with minimum samples, and to say INSUFFICIENT_DATA when it cannot judge, so the report stops calling every agent stable.
**Files:** `orchestrator/services/harness_service.py`, `orchestrator/modules/tools/discovery/handlers_harness.py`, `orchestrator/config.py`, `orchestrator/reports/config-surface.json`, `orchestrator/tests/test_prd263_harness_diagnose_realdb.py`, `orchestrator/tests/test_w1s9_idle_in_tx.py`, plus every existing test that sets REGRESSION by hand or monkeypatches `_phase_collect`/`_phase_diagnose` · **Depends on:** US-008, US-010, US-013

**Acceptance criteria:**
- [ ] COLLECT calls `harness_metrics.read`. DIAGNOSE calls `core.stats.harness_verdicts` for each agent × metric pair with the reference and cur counts (the reference exclusions and the verdict history come in US-014b).
- [ ] New settings in `config.py` and `config-surface.json`: `HARNESS_MIN_SAMPLES=20`, `HARNESS_REF_DAYS=28`, `HARNESS_ABS_DROP=0.15`, `HARNESS_REL_DROP=0.20`, `HARNESS_ALPHA=0.05`, `HARNESS_FDR_Q=0.10`, `HARNESS_COST_REL_RISE=0.25`, `HARNESS_COST_ABS_RISE=0.005`, `HARNESS_STRONG_P=0.001`, `HARNESS_MIN_COVERAGE=0.5`.
- [ ] Cost and tokens flag only when the median per-run value rises by both `HARNESS_COST_REL_RISE` and `HARNESS_COST_ABS_RISE` and Mann-Whitney gives p < `HARNESS_ALPHA`. Agents with no billed $ are judged on tokens. Benjamini-Hochberg at `HARNESS_FDR_Q` runs across every agent × metric test of the run.
- [ ] Coverage = judged agents / active agents. Convergence = `insufficient_data` below `HARNESS_MIN_COVERAGE`; otherwise `regression` if any REGRESSION, `watching` if any WATCH, else `steady`. `iteration` comes from US-010. The report lists every agent with its n and never says "stable" for an unjudged agent, or "converged". `platform_harness_status` returns coverage and the INSUFFICIENT_DATA, WATCH and REGRESSION counts.
- [ ] DIAGNOSE reads agent rows only through the names in one constant, `DIAGNOSE_READS` (used by US-015).
- [ ] Dormancy: the active-agent-count rule (`_MIN_AGENTS`) stays as `dormant_insufficient_agents`, and `_sufficiency_breakdown` keeps only that count. The heartbeat-age `data_ok` check (`_MIN_DATA_DAYS` and the `heartbeat_results` `MIN(created_at)` query) and the `dormant_insufficient_data` status are deleted; coverage replaces them.
- [ ] Deleted (grep in `harness_service.py` returns 0): the 9 dump-only COLLECT platform calls; `_get_rejected_signatures` and the `platform_list_tasks` call in `_phase_collect` (`:518-523`), so COLLECT uses no `PlatformActionExecutor`; every COLLECT, DIAGNOSE and BASELINE caller of `_extract_agents_list` (the function itself stays until US-048, because `_resolve_agents_by_name` uses it at `:1786`); `_compute_deltas`; `_REGRESSION_THRESHOLD` and `_IMPROVEMENT_THRESHOLD`; `_read_baseline` and `previous_baseline`; every `_write_workspace_file` caller, and the function if no caller remains; `_compute_org_diagnosis`; `_build_per_agent_snapshot`; `_compute_convergence_status` with `_CONVERGED_DELTA`, `_CONVERGED_VARIANCE`, `_CONVERGED_MIN_RUNS` and `_EXPLORING_MAX_ITERATION`; `_infer_root_cause`; `_make_serializable`; `_has_sufficient_data`; the "No changes needed — organization is stable." line in `_build_changelog_markdown`; and `agent.get('model')` (now `model_id`).
- [ ] Tests: `test_prd263_harness_diagnose_realdb.py`, which fails at `697ca7d6e` (every agent STABLE):
  - a collapse from 30/32 to 10/32 is WATCH on its first judged week;
  - an agent with 3 tickets is INSUFFICIENT_DATA;
  - 16/20 vs 14/20 is STABLE;
  - the heartbeat interval comes from `configuration`, and heartbeat cost from whitelisted `llm_usage` rows;
  - an orchestrator heartbeat row with a UUID `source_id` does not break the query;
  - a 30% median cost rise of at least $0.005 with Mann-Whitney p < 0.05 flags; a 30% rise of $0.003 does not; an agent with no billed $ is judged on tokens;
  - among 10 agent × metric tests, a lone p=0.04 is not flagged after BH at q=0.10 (rank-1 threshold 0.01), while p=0.004 is;
  - coverage below 0.5 gives convergence `insufficient_data`, and the report contains neither "stable" for an INSUFFICIENT_DATA agent nor "converged";
  - workspace B never appears;
  - a full tick writes no file under a temp `<volume>/<ws>/harness`.
- [ ] `test_w1s9_idle_in_tx.py::test_harness_commits_before_each_phase` is updated to the new phase list and still asserts a commit before every awaited phase.
- [ ] Every deleted test that set REGRESSION by hand or monkeypatched `_phase_collect`/`_phase_diagnose` is listed in the PR description with its replacement (AGENTS.md:121).
- [ ] `harness_service.py` is shorter after this story's commit.

#### US-014b: Verdicts use a clean reference and the week-by-week history
**Description:** As an owner, I want a first flag to read WATCH and a REGRESSION to need a second distinct week, judged against a reference that leaves out already-flagged and post-change weeks, so a bad week is neither absorbed nor double-counted.
**Files:** `orchestrator/services/harness_service.py`, `orchestrator/tests/test_prd263_harness_verdict_history_realdb.py`, `orchestrator/tests/test_harness_self_management.py`, `orchestrator/tests/test_w1s9_idle_in_tx.py` · **Depends on:** US-014a

**Acceptance criteria:**
- [ ] DIAGNOSE reads the ISO-week verdict history from `harness_runs.verdicts`. The clean reference is the `HARNESS_REF_DAYS` before cur, minus weeks already flagged for that agent and metric, minus weeks after a change applied to that agent (`harness_task_ledger` rows in state `applied`, by the agent in `entry` and `created_at`, until US-051 adds `applied_at`).
- [ ] A first flag is WATCH. REGRESSION needs flags in two distinct, consecutive ISO-week windows, or one flag at p < `HARNESS_STRONG_P`. A run whose `cur_window` was already judged reuses those verdicts and never counts as a second flag (this closes escalation through `platform_harness_trigger`, which Auto can call from chat, `actions_harness.py:40-53`).
- [ ] Deleted (grep returns 0): `_detect_auto_applied_regressions` with its rollback emission (`harness_service.py:620`); US-056 replaces it on the ledger.
- [ ] Tests: `test_prd263_harness_verdict_history_realdb.py`: a collapse from 30/32 to 10/32 is WATCH in week 1 and REGRESSION in week 2 on two distinct seeded weeks; a same-window rerun stays WATCH; a flagged week is left out of the next reference; one flag at p < 0.001 is REGRESSION at once; a week after a ledger-applied change is left out of that agent's reference.
- [ ] The `_detect_auto_applied_regressions` cases of `test_harness_self_management.py` (`:362-463`) are deleted and mapped in the PR description. `test_w1s9_idle_in_tx.py` stays green.
- [ ] `harness_service.py` is shorter after this story's commit.

#### US-015: A test pins the keys DIAGNOSE reads to the keys the read model returns
**Description:** As a maintainer, I want a test that every key DIAGNOSE reads exists in the read-model row, so the "four fields that are never returned" defect cannot come back.
**Files:** `orchestrator/tests/test_prd263_harness_key_contract.py` · **Depends on:** US-013, US-014a

**Acceptance criteria:**
- [ ] Tests: `test_prd263_harness_key_contract.py` asserts `DIAGNOSE_READS ⊆ harness_metrics.AGENT_ROW_FIELDS`, and an AST scan of the DIAGNOSE functions finds no subscript or `.get()` key on an agent row outside `DIAGNOSE_READS`.
- [ ] A second case adds an undeclared key to a copy of `DIAGNOSE_READS` and asserts the contract check reports it.

**PR-1E: report-only for every workspace** (US-016, US-017; ~565 changed lines including the settings extraction; over ~400 because ~110 lines are the settings block moved, and US-016 and US-017 cannot ship apart, §4 "PR size"; needs PR-1D and US-001 approved)

#### US-016: HARNESS reports and changes nothing; the auto-apply loop and the card-to-Done path are deleted (Gerard's sign-off first)
**Description:** As a beta workspace owner, I want HARNESS to report only and never change my agents, with one mode setting as the only switch, so no statistical verdict alters my org without my click and Settings never shows HARNESS off while it runs. The auto-apply loop and the card-to-Done path go here, earlier than the design's S3.9, because report-only makes them unreachable (D2; AGENTS.md: replace, don't shim).
**Files:** `orchestrator/services/harness_service.py`, `orchestrator/services/harness_mode.py` (new, ~40 lines), `orchestrator/api/workspaces.py`, `orchestrator/config.py`, `orchestrator/reports/config-surface.json`, `orchestrator/tests/test_prd263_harness_report_only.py`, `orchestrator/tests/test_prd263_harness_grep_gate.py`, `orchestrator/tests/test_prd263_workspace_harness_mode_api.py`, `orchestrator/tests/test_harness_governance_gate.py` (deleted), `orchestrator/tests/test_w1s9_idle_in_tx.py`, `orchestrator/tests/test_harness_self_management.py`, `orchestrator/tests/test_prd143_full_surface_positive.py`, `orchestrator/tests/test_f156_the_harness_ledger_is_in_the_database.py` · **Depends on:** US-001, US-010, US-014a

**Acceptance criteria:**
- [ ] New `orchestrator/services/harness_mode.py` holds `HARNESS_MODES = ('report_only', 'propose', 'disabled')`, `DEFAULT_HARNESS_MODE = 'report_only'` and `harness_mode(workspace_settings)`, the one read-time mapping recorded on US-001: a stored `full_auto` or `manual` reads `report_only`; a stored `disabled: true` or `enabled: false` reads `disabled`; nothing stored reads `report_only`. `api/workspaces.py` and `harness_service.py` both use it, and the grep gate covers it.
- [ ] `api/workspaces.py`: `_ORCHESTRATOR_DEFAULTS['harness']` becomes `{'schedule': 'weekly', 'mode': 'report_only'}` (the `enabled` key goes); `_VALID_HARNESS_MODES` is `HARNESS_MODES`; `_VALID_HARNESS_SCHEDULES = ['weekly']` (§11-Q12). A PUT carrying any other mode (including `full_auto` and `manual`) or schedule returns 400 at the boundary; only the payload is validated (`:797-798`), so a stored legacy value never blocks an unrelated save. GET (`:673`) and the PUT response (`:937`) return `mode` from `harness_mode()` and no `enabled` or `disabled` key.
- [ ] `_workspace_opted_in` returns `harness_mode(settings) != 'disabled'`, and `disabled` skips the tick. `get_status`'s disabled message (`:190-193`) says "HARNESS is disabled for this workspace. Choose Report only in Settings > Orchestrator to turn it back on." `propose` behaves exactly like `report_only` until PR-3F (US-054) (asserted).
- [ ] PRESCRIBE writes report lines only, with rx ids `rx-<run_id>-<seq>` taken from `harness_runs.id` (this ends the same-day collisions at `harness_service.py:757-759`).
- [ ] Deleted (grep across `orchestrator/`, tests excluded, returns 0): `_phase_apply`, including the direct `PlatformActionExecutor` apply (`:927-949`) and the queued-card creation; `_apply_approved_board_tasks` (`:1634-1741`) with the F151 held path, and `_note_needs_an_approve` (`:1743-1775`), whose only production caller is `_phase_apply` (`:1004`, `:1720`); `_workspace_allows_auto_apply`; `_auto_apply_max_risk`; `_get_harness_config` (`:378-392`; no caller); `_maybe_escalate`; `_build_escalation_message`; `_workspace_has_channel` if nothing else calls it; and the settings `HARNESS_AUTO_APPLY_MAX_RISK_STANDARD`, `HARNESS_AUTO_APPLY_MAX_RISK_FULL` and `HARNESS_HIGH_PRIORITY_RISK`, removed from `config-surface.json` too (re-verify by grep that no other reader exists).
- [ ] Kept until US-047/US-048, because `api/harness_commands.py:215-260` calls them: `_auto_apply_prescription`, `_snapshot_current_value`, `_parse_harness_task` and its extractors, `_resolve_agents_by_name`, `_extract_agents_list` and `_read_applied_tasks`/`_write_applied_tasks`. The PR description says so.
- [ ] The PR description records that the design's §2.1 "data.id vs task_id" defect (`harness_service.py:991`) sat in the queued-card branch deleted here and is resolved by deletion.
- [ ] Tests: `test_prd263_harness_report_only.py`: a tick on a workspace with seeded evidence creates 0 BoardTasks, 0 `approval_grants` rows and exactly one notification, of type `report_submitted` (the audit report; `services/report_service.py:237-255`), and none of type `approval_pending` or escalation; it calls no platform write action other than `platform_submit_report` for the audit report; `trigger_now` does the same; with `HARNESS_SELF_MANAGEMENT_ENABLED` on, a `report_only` tick still applies nothing (this replaces Half 2 of `test_kill_switch_halts`); a workspace stored as `full_auto` reads `report_only`; a workspace stored with `enabled: false` is skipped; with the decision engine's `report_triage_mode` dial off, no LLM client is called (FR-28; `services/report_service.py:30-112`).
- [ ] Tests: `test_prd263_harness_grep_gate.py` (first version): `harness_metrics.py`, `harness_mode.py` and `core/stats/harness_verdicts.py` contain no `PlatformActionExecutor`, `UnifiedToolExecutor` or `execute_tool`, and no LLM client import.
- [ ] Tests: `test_prd263_workspace_harness_mode_api.py`: PUT mode=`report_only`, `propose` and `disabled` → 200 and stored; PUT mode=`full_auto` → 400; PUT schedule=`biweekly` → 400; GET with no stored harness → mode `report_only` and no `enabled` key; GET on a stored `full_auto` → `report_only`; GET on a stored `enabled: false` → `disabled`.
- [ ] Existing tests, each change mapped in the PR description: `test_harness_governance_gate.py` is deleted; the escalation cases (`:614-674`) and card-apply cases of `test_harness_self_management.py` are deleted; `test_prd143_full_surface_positive.py::test_kill_switch_halts` keeps Half 1 (the autonomy dial) and loses Half 2 (`:280-302`) and its `harness_ledger` argument, with the reason in its docstring; the `_apply_approved_board_tasks` case of `test_f156_the_harness_ledger_is_in_the_database.py` (`:77-80`) is deleted; `test_w1s9_idle_in_tx.py` drops its `_workspace_allows_auto_apply` and `_phase_apply` patches and still asserts a commit before every awaited phase. The PR links US-001.
- [ ] `harness_service.py` ends the PR shorter.

#### US-017: Settings offers the modes that exist and says what each one does
**Description:** As an owner, I want the Settings tab to offer only modes that exist, with one control and truthful help text, so I am not told low-risk changes are applied automatically or that HARNESS is off while it runs.
**Files:** `frontend/components/settings/SystemLLMSettingsTab.tsx` (1,114 lines), `frontend/components/settings/harness-settings-section.tsx` (new), `frontend/components/settings/__tests__/harness-settings-section.test.tsx` · **Depends on:** US-016

**Acceptance criteria:**
- [ ] The HARNESS block (`:986-1100`) is extracted into `harness-settings-section.tsx`: one component of at most 150 lines with props `harness` and `onChange`. The changed-lines gate requires it (`frontend/scripts/check-changed-lines-eslint.js`: `max-lines-per-function` 150 for components, checked over the whole function a touched line sits in; `SystemLLMSettingsTab` is one function from `:112` to the end of the file). `SystemLLMSettingsTab` renders the section and ends shorter than 1,114 lines.
- [ ] The "Enable HARNESS" switch (`:1028-1039`) is removed and the mode select always renders. In P1 it offers "Report only" and "Disabled"; US-054 adds "Propose" when proposals can exist. The default is `'report_only'` (replacing `'full_auto'` at `:235` and `:1074`; the fallback at `:232-235` loses `enabled: false`). The help text under it is the copy in §8.
- [ ] The description at `:1015-1024` is replaced by the §8 copy. The Run Schedule select (`:1043-1064`) is removed, because the only value left is weekly on the configured cron (§11-Q12).
- [ ] In `SystemLLMSettingsTab.tsx` and `harness-settings-section.tsx`, a case-insensitive grep returns 0 for "Full Auto", "Manual — all changes", "risk 3+", "applied automatically", "queued as board tasks", "Sunday", "biweekly", "harness?.enabled" and "Enable HARNESS".
- [ ] vitest `harness-settings-section.test.tsx` renders `HarnessSettingsSection` directly: exactly two mode options in P1; no "Full Auto"; an unknown stored value displays as Report only; typecheck passes.
- [ ] → OWNER (after merge, local stack, dev-browser, 1440 px and 390 px): Settings > Orchestrator > HARNESS shows one select with exactly "Report only" and "Disabled" and the §8 help text; choosing Disabled and saving persists after reload; no text contains "Sunday", "Full Auto" or "risk 3+".

**PR-1F: telemetry hygiene and the tool-failure input** (US-018 – US-020; ~270 changed lines; merges after PR-1E in the `harness_service.py` sequence)

#### US-018: A wait for the owner's click is logged as "asked", error codes are kept, and EdgeBuilder skips asks and tagged eval traffic
**Description:** As the routing and HARNESS signals, I want an owner-only ask logged as `asked` instead of a failure, the tool's own `error_code` stored on failures, and EdgeBuilder to read only production and synthetic rows that are not asks, so waiting for a click is not counted as a tool failure and rows tagged eval or replay stay out of the Knowledge Graph builder. Replay traffic is still logged as production until the P6 tag exists (§15 risk 7), so today this filter does not remove replay rows.
**Files:** `orchestrator/modules/tools/execution/telemetry.py`, `orchestrator/modules/tools/tool_router.py` (2,145 lines), `orchestrator/core/services/edge_builder_logs.py`, `orchestrator/services/slo_metrics.py`, `orchestrator/tests/test_prd263_telemetry_asked_status.py`, `orchestrator/tests/test_prd263_telemetry_error_code.py`, `orchestrator/tests/test_prd263_edge_builder_telemetry_filter.py`, `orchestrator/tests/test_prd263_slo_excludes_asked.py` · **Depends on:** —

**Acceptance criteria:**
- [ ] A tool result carrying `requires_confirmation` is logged by `telemetry.py` (`:152`) with status `asked`, not `error`. `tool_execution_logs.status` is a free String(50) (`core/models/composio_cache.py:224`); non-execution values already written are `gap` (`telemetry.py:338`) and `shown` (`signal_recorder.py:563`), so no migration is needed.
- [ ] `write_telemetry` (`telemetry.py:152-174`) sets `error_code=str(result['error_code'])[:100]` on an error row when the result carries one (the tool-result convention, `modules/tools/README.md:100`), and NULL otherwise.
- [ ] `tool_router.py` (`:1640-1651`) fires no routing failure signal for such a result, and the file does not grow.
- [ ] `edge_builder_logs.py` `_read_page` (`:67-83`) filters `telemetry_source IN ('production','synthetic_signal') OR telemetry_source IS NULL`, and excludes `status = 'asked'`, because EdgeBuilder counts every status other than `success` as a failure (`edge_builder.py:646`, `:660-669`).
- [ ] `orchestrator/modules/tools/discovery/signal_recorder.py` is unchanged (GraphRouter still reads `fails_for_intent` as penalties, `graph_router.py:702`, `:713-714`).
- [ ] At §11-Q8's default, `slo_metrics.tool_call_success_rate` (`services/slo_metrics.py:73-82`) also excludes `status = 'asked'` from `total`; today an asked row would count in the denominator exactly as an error row does. The PR description states that the always-failing scan (`api/analytics_real.py:351-363`, `status = 'error'` only) stops listing owner-only tools that only ever asked.
- [ ] Tests: `test_prd263_telemetry_asked_status.py` (an owner-only ask logs `asked` and fires no failure signal); `test_prd263_telemetry_error_code.py` (a failed result with `error_code` `integrations_unavailable` logs that code; a failed result without one logs NULL); `test_prd263_edge_builder_telemetry_filter.py` (`eval` and `replay` rows excluded; `synthetic_signal` and NULL included; an `asked` production row yields no row from `load_logs` and no `fails_for_intent` from `build_edges`); `test_prd263_slo_excludes_asked.py` (8 success, 2 asked → rate 1.0).

#### US-019: Tool failures are judged on this week's production logs, and broken connections say "reconnect"
**Description:** As an owner, I want tool failures judged on this week's production logs with a Wilson bound, and auth or connection errors reported as "reconnect <app>", so an old cumulative counter or a broken connection never looks like a bad tool.
**Files:** `orchestrator/services/harness_service.py`, `orchestrator/services/harness_metrics.py`, `orchestrator/config.py`, `orchestrator/reports/config-surface.json`, `orchestrator/tests/test_prd263_harness_tool_failure_window.py`, `orchestrator/tests/test_harness_tool_failure_signal.py` (deleted) · **Depends on:** US-008, US-013, US-014a, US-018

**Acceptance criteria:**
- [ ] Tool-failure diagnosis reads `harness_metrics`' windowed `tool_execution_logs` counts per agent and app. It flags only when `wilson_lower_bound(failure rate) ≥ HARNESS_TOOL_FAIL_WILSON_MIN` (0.5; in `config.py` and `config-surface.json`) and n ≥ `HARNESS_MIN_SAMPLES`.
- [ ] Failures that would flag but classify as `auth` or `connection` by `classify_tool_error` (US-013; `error_code` from US-018, and `error_message` for rows written before it) produce a "reconnect <app>" report line, never a removal line.
- [ ] Deleted (grep returns 0): the agent-keyed `fails_for_intent` read (`harness_service.py:665-680`), `_TOOL_FAILURE_MIN_SAMPLES`, and `_derive_app_name` and `_prescribe_tool_removals` (`:644-656`, `:702-737`; tool-removal proposals move to US-049 and US-057). The recorder is unchanged.
- [ ] Tests: `test_prd263_harness_tool_failure_window.py` replaces `test_harness_tool_failure_signal.py` (mapped in the PR description): a high cumulative `sample_count` with no failures in the window is not flagged; 18/20 non-auth errors flag; 20/20 errors with an auth `error_code` give a reconnect line and no removal line; 20/20 errors with NULL `error_code` and an expired-auth message give a reconnect line.

#### US-020: The `/approve` lookup sees every HARNESS card until P3 deletes it
**Description:** As an owner with more than 20 HARNESS cards in a workspace where the legacy flag is on, I want the `/approve` lookup to see every card, so it does not miss older cards before PR-3A deletes it.
**Files:** `orchestrator/api/harness_commands.py`, `orchestrator/tests/test_prd263_harness_card_reader_pagination.py` · **Depends on:** —

**Acceptance criteria:**
- [ ] `handle_harness_command` (`api/harness_commands.py:163`) pages `platform_list_tasks` with `offset`/`next_offset` until `next_offset` is None before calling `_find_task_by_rx` (a pure filter over the list it is given, `:102-117`), instead of stopping at the default 20 (`board_task_list.py:30-31`).
- [ ] Tests: `test_prd263_harness_card_reader_pagination.py`: the 25th HARNESS card is found.
- [ ] The PR description says PR-3A deletes this reader (US-047). The card-to-Done reader was deleted in US-016 (§11-Q27).

**PR-1G: endpoint, tile, Auto's tools and docs** (US-021 – US-023a; ~450 changed lines; over ~400, with a clean split into US-021/US-022 and US-023/US-023a offered as an option for Gerard, §4 "PR size"; needs PR-1E and the US-002 decision)

#### US-021: The self-learning endpoint reads the run table and counts only this workspace (Gerard's sign-off first)
**Description:** As an owner viewing Command Center, I want the self-learning endpoint to read the latest run from the database and count only my workspace's tool logs, so the tile shows real figures and the request does not block the event loop.
**Files:** `orchestrator/api/harness.py`, `orchestrator/tests/test_prd263_self_learning_endpoint.py`, `orchestrator/tests/test_harness_api.py` · **Depends on:** US-002, US-010, US-014a, US-016

**Acceptance criteria:**
- [ ] `GET /api/harness/self-learning` in `orchestrator/api/harness.py` becomes a plain `def` (today an `async def` over a sync Session, `:87-91`; AGENTS.md:57), using the existing request-context and workspace-permission dependencies.
- [ ] The response carries the latest `harness_runs` `status`, `iteration_count`, `coverage`, the INSUFFICIENT_DATA, WATCH and REGRESSION counts, the workspace `mode` (from `harness_mode()`), and `signals_7d`: a 7-day count of this workspace's `tool_execution_logs` rows with `telemetry_source='production'`, replacing the process-global Signals counter. No grant text. Access follows the US-002 decision, and the PR links that issue.
- [ ] `orchestrator/reports/route-manifest.json` is unchanged (same method and path).
- [ ] Tests: `test_prd263_self_learning_endpoint.py`: `test_reads_harness_runs`, `test_signal_count_is_workspace_scoped`, and `test_route_is_not_a_coroutine`. The two self-learning tests in `test_harness_api.py` (`:247-280`) call the function directly (no `asyncio.run` or `_run`) and assert the new fields.
- [ ] security-reviewer runs on this story before PR-1G merges.

#### US-022: The Self-Learning tile shows real fields and "—" for anything missing
**Description:** As an owner, I want the Self-Learning tile to show the fields the backend sends and "—" where there is no value, so it never shows fake zeros.
**Files:** `frontend/components/activity/widgets/self-learning-health-widget.tsx`, `frontend/hooks/use-learning-api.ts`, `frontend/components/activity/widgets/__tests__/self-learning-health-widget.test.tsx` · **Depends on:** US-021

**Acceptance criteria:**
- [ ] The widget reads `iteration_count` (not `harness.iteration`) and no longer reads `data.prescriptions` (`:40-52`, `:114`). Absent values render "—". Tones exist for `dormant_insufficient_agents`, `scheduled_not_run_yet`, `disabled` and `failed`, using theme tokens (`text-warning`, `text-destructive`, `text-muted-foreground`); `grep 'dark:'` returns 0 in the widget, because the `dark:` variant does nothing in this build (Tailwind 3.3.3, `frontend/package.json:40`). Status labels use `replace(/_/g, ' ')`.
- [ ] The Signals figure shows `signals_7d` from US-021. `use-learning-api.ts` types match the response, and calls go through `apiClient`.
- [ ] vitest `self-learning-health-widget.test.tsx` (absent iteration renders "—"; `scheduled_not_run_yet` renders "scheduled not run yet"; coverage renders as a percentage); typecheck passes.
- [ ] → OWNER (after merge, local stack, dev-browser, 1440 px and 390 px): the Command Center Self-Learning tile shows coverage as a percentage, "—" for an absent iteration, and "scheduled not run yet" without underscores.

#### US-023: Auto's HARNESS tools and cadence text read the run table, and old history is imported once
**Description:** As Auto answering "what did HARNESS do", I want `platform_harness_history`, the harness action descriptions and my weekly review text to reflect the run table and the report-only loop, so I describe the loop that actually exists.
**Files:** `orchestrator/modules/tools/discovery/handlers_harness.py`, `orchestrator/modules/tools/discovery/actions_harness.py`, `orchestrator/core/services/auto_cadence.py`, `orchestrator/scripts/harness_backfill_runs.py` (new), `orchestrator/tests/test_prd263_harness_history_runs.py`, `orchestrator/tests/test_prd263_harness_backfill_runs.py`, `orchestrator/tests/test_prd263_harness_review_cadence.py` · **Depends on:** US-010, US-014a

**Acceptance criteria:**
- [ ] `platform_harness_history` (`handlers_harness.py:50-87`) reads `harness_runs`; the baseline-file reader is deleted (no baseline file path is left in `handlers_harness.py`).
- [ ] The `actions_harness.py` descriptions of `platform_harness_status`, `platform_harness_trigger` and `platform_harness_history` describe weekly report-only runs on the weekday cron, coverage, INSUFFICIENT_DATA and the one remaining dormancy status, and say nothing of queued cards or model swaps (`:15-16` no longer describes `dormant_insufficient_data`). Every `ActionDefinition` stays inline in `registry.register(...)`, so `check_hierarchy_gate.py` passes.
- [ ] `_HARNESS_REVIEW` in `core/services/auto_cadence.py` (`:66-74`) becomes "Weekly HARNESS Review": it describes the report-only run (coverage and the INSUFFICIENT_DATA, WATCH and REGRESSION counts, no queued cards) and names no day or time; if `platform_harness_status` reads `scheduled_not_run_yet`, or the latest run's `cur_window` is not the last complete ISO week, it tells Auto the review waits for this week's run. `test_prd263_harness_review_cadence.py` asserts the text contains neither "Sunday" nor "tag=harness".
- [ ] New `orchestrator/scripts/harness_backfill_runs.py` imports the existing `<WORKSPACE_VOLUME_PATH>/<ws>/harness` baseline and last_run JSON into `harness_runs` (insert-if-absent by workspace and `cur_window`). It never writes or deletes files, prints per-workspace counts, has `--summary-only` (totals with no workspace ids, for pasting into public issues), and defaults to `--dry-run`.
- [ ] Tests: `test_prd263_harness_history_runs.py` (history reads `harness_runs`) and `test_prd263_harness_backfill_runs.py` (a second run inserts 0 rows; `--dry-run` inserts 0; `--summary-only` output holds no workspace id).

#### US-023a: The HARNESS docs page describes the report-only loop
**Description:** As anyone reading the platform docs, I want the HARNESS page to describe the report-only loop and the mode setting, so the docs stop promising auto-apply at risk 2 or less.
**Files:** `docs/platform-actions/self-management-harness.md` · **Depends on:** US-016, US-023

**Acceptance criteria:**
- [ ] The page (`:29-176` describe auto-apply at risk ≤ 2 and the `/approve` routes) is rewritten for the report-only loop: the mode setting (Report only, Disabled; Propose from P3), coverage, the four verdicts, the weekday cron setting, and how Auto reports it. A case-insensitive grep for "auto-apply" returns 0.
- [ ] The `/approve` section stays only as a short note that the route is removed in P3; US-047 deletes the note, and US-054 adds the `propose` section.
- [ ] Only canonical terms are used. Docs-only change.

**SKL-1: Auto's HARNESS text in automatos-skills** (US-024; Gerard merges)

#### US-024: Auto's skill describes the report-only loop (human-edited)
**Description:** As Auto, I want my skill's HARNESS sections to describe the report-only weekday loop, so I stop telling owners about Sunday 02:00 runs, queued cards and model swaps.
*Tags: Human-gated*
**Files:** automatos-skills `team/auto/SKILL.md` · **Depends on:** US-016

**Acceptance criteria:**
- [ ] On the automatos-skills branch Gerard names, cut from `skill/auto-v2.3.3-real-tools` so its body matches the current platform seed apart from these sections (§15 risk 15), the HARNESS sections say: report-only, weekday daytime cron, coverage and INSUFFICIENT_DATA, no queued cards. grep returns 0 for "Sunday 02:00", "Sunday cron", "tag=harness" and "model swaps".
- [ ] The `DOCTRINE_ANCHORS` and `CONTRACT_OPENER` lines are unchanged in the diff, and `scripts/sync-skills.py` `plan()` passes against an export of the branch.
- [ ] Only canonical terms are used (Playbook, Mission, Task, Deliverable, Knowledge Graph, Command Center, Auto).
- [ ] Gerard reviews and merges.

**SYNC-1: seed sync** (US-025; generated diff; Gerard merges)

#### US-025: The shipped seed carries the report-only HARNESS text
**Description:** As every tenant's Auto, I want the shipped seed updated from the merged skills text through `sync-skills.py`, so the platform serves the corrected HARNESS runbook.
*Tags: Human-gated*
**Files:** `orchestrator/core/seeds/platform-management-skill.md` (1,877 lines) · **Depends on:** US-024, US-016

**Acceptance criteria:**
- [ ] `scripts/sync-skills.py`, run against an export of the merged automatos-skills branch, changes `platform-management-skill.md` only within `:369-388` ("After HARNESS run", "Weekly HARNESS review") and `:1532-1594` (§14 HARNESS) at `697ca7d6e`.
- [ ] grep in the seed returns 0 for "Sunday 02:00", "Sunday cron", "tag=harness" and "model swaps".
- [ ] `test_prd226_doctrine.py` and `test_prd226_contract.py` are unchanged.
- [ ] → OWNER: the PRD-256 eval pack is no worse than its last line under the same scorer sha, outside the per-row noise floor once US-044 has measured it, launched by Gerard with his own `--max-spend` and `--cost-factor`. Before US-030 merges, `claim_backed` is reported but is not part of the bar.
- [ ] Gerard merges (skill text).

**OPS** (US-026b)

#### US-026b: P1 in prod: the migration is applied and the sweep runs once per workspace
**Description:** As Gerard, I want the P1 migration applied in prod and the first weekday sweep checked, so P1 runs in prod as report-only.
*Tags: Human-gated*
**Files:** the private ops note; this PRD (pass or fail and dates) · **Depends on:** US-009, US-010, US-016, US-025, US-026a

**Acceptance criteria:**
- [ ] → OWNER: Gerard applies `prd263_harness_runs` in prod, or confirms the deploy applied it; prod `alembic current` equals the single head (pass or fail and date recorded in §12's operator checks).
- [ ] → OWNER: after the next weekday sweep, prod SQL shows one `harness_runs` row per active workspace, and 0 BoardTasks or `approval_grants` rows created by HARNESS since the deploy. This PRD records pass or fail and the date; the workspace count stays in the private ops note.

**LOCAL-1** (US-027)

#### US-027: P1 exit on local c1
**Description:** As Gerard, I want the P1 exit criteria checked on the local stack, so I have checkable HARNESS numbers before P3 starts.
*Tags: Human-gated · Local-only*
**Files:** — · **Depends on:** US-014a, US-014b, US-016, US-017, US-021, US-022, US-023, US-025

**Acceptance criteria:**
- [x] This PRD links the CI run id of each P1 PR (PR-1A, PR-1B, PR-1C, PR-1D, PR-1E, PR-1F, PR-1G). *(§0.)*
- [x] On main after P1, `wc -l orchestrator/services/harness_service.py` is below 2,165. *(633 lines at 101a0bf33, 23 Oct.)*
- [x] → OWNER: on local c1, in a daytime window, a manual trigger creates one `harness_runs` row, 0 BoardTasks and 0 `approval_grants` rows (SQL counts before and after). "Converged" does not appear in the report, and while coverage is below 50% convergence reads `insufficient_data`. c1's mode reads `report_only`. *(23 Oct 08:25 UTC, by the build session at Gerard's direction: run 1, week 2026-W42; Tasks 1,497 → 1,497, grants 2,285 → 2,285, `harness_runs` 0 → 1; coverage 27% (13 of 49), convergence `insufficient_data`, 11 steady, 2 REGRESSION, 36 not enough data; no "converged"; mode `report_only` after the fix in §0.)*
- [ ] → OWNER (dev-browser, 1440 px and 390 px): on c1 the Self-Learning tile shows "—" for every absent value, never 0, and shows coverage as a percentage; Settings shows the one mode select. *(23 Oct, 1440 px: the tile shows coverage 27%, 0 Watch, 2 Regression, 4,825 signals, Approvals 0/0/0/0, "Iteration 1 · Mode Report only"; no value was absent, so every 0 is a real zero. 390 px: the tile's data is in the page, but the Command Center's tab body (`.cc-body`, `flex: 1; min-height: 0`) is 32 px tall on a phone, so no tab's content can be seen there. That is a Command Center shell bug, not HARNESS; flagged, and this box stays open until it is fixed and checked again.)*
- [x] → OWNER: one chat turn asking Auto "what did HARNESS do this week?" returns figures that match the `harness_runs` row (one turn, Gerard's OK). *(23 Oct: every figure matched. Auto named one of the two REGRESSION agents; the other was past the chat's 2,000-token cut, fixed by #1276. It also called Monday 26 Oct "Sunday", a model slip: `next_scheduled_run` says Monday.)*
- [ ] Gerard confirms or moves the HARNESS bars in §10 before P3 starts.

### Part 2 (P2): Eval truth

**DOC-1: the eval-rule amendment** (US-028)

#### US-028: The eval rule is amended for PRD-263 only (D5; open, default applied)
**Description:** As Gerard, I want the eval rule at `docs/testing/AUTO-EVAL.md:5` amended for PRD-263 only, so a named build session may edit the eval tooling and the Lab I launch may run `replay.py`, while improver sessions still never read the eval folder.
*Tags: Human-gated*
**Files:** `docs/testing/AUTO-EVAL.md` · **Depends on:** —

**Acceptance criteria:**
- [ ] `docs/testing/AUTO-EVAL.md:5` gains a PRD-263-scoped exception that names: the build session(s) allowed to edit `replay.py` and `phase.sh`; the Gerard-launched Lab driver (`lab.py`, `monitor.py`; never its improver sessions), allowed to run `replay.py`; and the rule that improver sessions never read `~/.automatos-analyst/auto-eval` and read only the stripped exports under `~/.automatos-analyst/harness-lab/exports/<run>/`.
- [ ] The same amendment proposes the extension in D5's second paragraph, named step by step: the build session may edit `compare-arms.py`, `rescore.py` and `arm-report.py`, create `make_split.py` and `split.json`, and run `replay.py`, `compare-arms.py`, `rescore.py`, `arm-report.py` and `lab.py` in fixture mode only (no backend request, no model call, no DB write); the Lab driver may run `phase.sh --reset-only`, `compare-arms.py --gate` and `arm-report.py`, and read `split.json`, the aggregate outputs under `runs/` and `auto-eval.jsonl` (for its gold scan only), and it writes only the accept/reject bit where improver sessions can read it; set v2 row authoring, the $0 rescore and the calibration runs. Checks that touch the local DB or a live session (the US-036 dry run, the US-041 planted rows, the US-076 lockdown session) stay Gerard's.
- [ ] If Gerard accepts the default but not the extension, this PRD assigns each extension step to Gerard as an operator step (§4 "Built by").
- [ ] D5 stays open in §3 and §11 until Gerard merges this PR, and every Needs D7 story is held until then. If Gerard declines, each Needs D7 story is re-assigned to Gerard as an operator step in this PRD.

**QUIET: quiet local stack** (US-029; a recorded finding, code only if Gerard picks it)

#### US-029: How to make the local stack quiet, and why the AutoBrain cache needs a flush
**Description:** As the Lab and the calibration runs, I want a known way to stop background writers and the complexity cache from moving probe results on the local stack, so passes compare like with like.
*Tags: Human-gated*
**Files:** this PRD (the finding); only for the code option: `orchestrator/consumers/chatbot/auto.py`, `orchestrator/tests/test_prd263_local_quiet_mode.py` · **Depends on:** —

**Acceptance criteria:**
- [ ] Recorded from the code at `697ca7d6e` (a build session re-verifies by grep and records any drift): `main.py:483` starts the unified scheduler if any of `HEARTBEAT_ENABLED`, `RECIPE_SCHEDULER_ENABLED` or `COORDINATOR_ENABLED` is true. Inside it run TaskReconciler stall retries (`:506-513`, no flag), the watch ticker (`WATCHER_ENABLED`, `:517-523`, `config.py:777`), memory consolidation, decay and promotion (`MEMORY_JOBS_ENABLED`, `:528`), field-memory promotion (`FIELD_PROMOTION_ENABLED`, `:539`), the nightly EdgeBuilder (`:547-550`, regardless of flag), the Composio sync, agent-scheduled tasks (`:566`, no flag), the mission coordinator (`:588`) and HARNESS (`:597`). The quiet stack is `HEARTBEAT_ENABLED=false`, `RECIPE_SCHEDULER_ENABLED=false` and `COORDINATOR_ENABLED=false` together; then `main.py:483` skips the whole scheduler.
- [ ] Recorded: `COMPLEXITY_CACHE_TTL_HOURS=0` does NOT bypass the AutoBrain complexity cache. The store maps 0 to 24 h (`consumers/chatbot/auto.py:1405`), the lookup has no TTL check (`:1366-1370`), and keys are `complexity:{workspace_id}:{sha256(msg)[:16]}` (`:1410-1412`), which `phase.sh` never clears, so identical eval prompts reuse pass 1's verdict in every later pass and arm.
- [x] Re-verified by grep on main at `630fc1c85` (20 Oct 2026, build session HARNESS). The finding holds; only the lines drifted. The scheduler gate is now `main.py:398`. The nightly EdgeBuilder still starts whatever its flag says (`:462`), and HARNESS starts at `:514`. In `consumers/chatbot/auto.py` the store still maps 0 to 24 h (`:1405`), the lookup has no TTL check (`:1356`), and the key is built at `:1412`.
- [ ] Default fix, local and no PR: every `phase.sh --reset-only` deletes the Redis keys matching `complexity:<c1 workspace id>:*` and prints the count (US-038). Code option, only if Gerard picks it (§11-Q22): `_cache_lookup` and `_cache_store` skip the cache when the setting is 0, with `test_prd263_local_quiet_mode.py`.
- [ ] US-044 and US-074 check on the running containers that the three switches are all false and that the boot log shows no scheduler start lines.

**LOCAL-2: claim truth in replay.py** (US-030 – US-032)

#### US-030: A broken claim check is ungraded, and an ungraded row voids the run
**Description:** As Gerard, I want `replay.py` to grade `claim_backed` with the backend's own checker and to treat a checker error as ungraded, so a broken check is never scored as backed.
*Tags: Local-only · Needs D7*
**Files:** `~/.automatos-analyst/auto-eval/replay.py` · **Depends on:** US-028

**Acceptance criteria:**
- [ ] `replay.py` grades `claim_backed` with `consumers.chatbot.claims_backed.unbacked_claims` (`claims_backed.py:568`), run by `docker exec` in the backend container with Receipt sequences rebuilt there (`consumers.chatbot` cannot be imported on the host, because its `__init__` imports the chat service). The `action_claims` import is removed (grep returns 0).
- [ ] A checker exception records the row as ungraded. `claim=None` no longer scores as backed. Any ungraded row marks the run VOID in results.
- [ ] A row whose fixtures could not be made, or any turn with HTTP status ≥ 500, marks the run VOID; `http_ok=False` no longer only fails the row.
- [ ] → OWNER (local, $0): a fixture run with no model calls and a forced checker error yields VOID, and a normal saved trajectory grades.
- [ ] No auto-eval path appears in `git status` of automatos-ai or automatos-skills.

#### US-031: Claim errors are counted per row
**Description:** As Gerard, I want claim errors counted per row from the saved reply and from F187 log lines joined by reply id, so the claim-error veto works per row instead of per window.
*Tags: Local-only · Needs D7*
**Files:** `~/.automatos-analyst/auto-eval/replay.py` · **Depends on:** US-030

**Acceptance criteria:**
- [ ] Per row, `replay.py` counts claim errors from the saved reply matching the full `NOT_DONE`, `NOTHING_DONE_SAID` and `NO_SUCH_ID` templates with their placeholders (`claim_check.py:77-79`), and from `[F187] tier=2` log lines joined by `reply=<saved assistant message id>` (`claim_check.py:259-264`).
- [ ] `UNBACKED_LOGGED` window counts are added as a cross-check column only, never a gate.
- [ ] Results carry a per-row `claim_error` count; a fixture reply containing a `NOT_DONE` template counts 1 (fixture mode).

#### US-032: One turn shows whether the request id survives into the tool loop
**Description:** As Gerard, I want one turn to show whether `req=` is filled on `UNBACKED_LOGGED` lines, so we know whether those lines can be joined per row.
*Tags: Human-gated · Local-only · Needs D7*
**Files:** `~/.automatos-analyst/auto-eval/replay.py` · **Depends on:** US-031, US-033

**Acceptance criteria:**
- [ ] → OWNER: one chat turn on c1, sent with replay's own `X-Request-ID`, shows whether `req=` is filled on the tool-loop and `UNBACKED_LOGGED` lines (`main.py:1081-1108`; `logging_adapter.py:55-88`). Pass or fail is recorded here with a log excerpt that contains no prompt or reply text; workspace, agent, request and message ids are replaced by `<ws>`, `<agent>`, `<req>` and `<msg>`, and only the presence or absence of `req=` is shown.
- [ ] If it fails, `UNBACKED_LOGGED` stays a per-window cross-check only (US-031).
- [ ] Gerard launches it with his own `--max-spend` and `--cost-factor` (US-033).

**LOCAL-3: run control** (US-033 – US-035)

#### US-033: Every probe run takes its passes from verified resets and refuses to start without the operator's spend limit and cost factor
**Description:** As Gerard, I want every probe run to take its passes from verified resets in a fixed order, and to refuse to start without a spend limit and a cost factor I supply, so no run carries a hard-coded cap and none overspends.
*Tags: Local-only · Needs D7*
**Files:** `~/.automatos-analyst/auto-eval/replay.py` · **Depends on:** US-028

**Acceptance criteria:**
- [ ] `replay.py --reps r` runs r full passes. Before each pass it writes `runs/.eval-marker` (pass start, UTC). After each pass it writes `runs/.eval-until` (pass end plus the late-write margin) and waits until that time has passed, because `phase.sh` resets only when `.eval-until` exists and waits for late memory writes; it then runs `phase.sh --reset-only` and treats the output "nothing to reset" or a non-zero exit as a failed reset that stops the run. The margin is a measured value recorded in `FINGERPRINT.json` (US-044 measures the late-write tail; until then phase.sh's 45 min). `--schedule ABAB` interleaves arms with equal A and B passes per day. No row reruns back-to-back without a reset.
- [ ] `--max-spend` (USD) and `--cost-factor` are required, with no default (argparse `required=True`). Launching without either exits non-zero with "refusing to start: --max-spend and --cost-factor are required" before any request is sent.
- [ ] Spend terms: booked = the sum of `llm_usage.total_cost` over every row in the run window, all request types including embedding; metered = booked × `--cost-factor`; true = the provider-billed amount for the same window (Anthropic usage report plus OpenRouter activity), read by Gerard and recorded beside the run for US-044. Before each row after the first, the run stops with "stopped: spend limit" in results when metered so far plus the largest per-row metered cost seen so far in this run would exceed `--max-spend`. The launch line prints the factor in use. Results record booked, factor and metered.
- [ ] grep of `replay.py`, `phase.sh`, `compare-arms.py` and `arm-report.py` finds no numeric constant used as a spend limit, price or cost factor.
- [ ] The OpenRouter arm ids are removed except `gemini-0.7`, the new-tenant default model (`core/llm/defaults.py:13-14`), which stays until §11-Q25 is answered.

#### US-034: Each run is fingerprinted, its lanes captured and impossible rows retired
**Description:** As Gerard, I want each run fingerprinted, its lanes captured and its invalid rows retired, so results compare only like with like and improvers never chase an impossible row.
*Tags: Local-only · Needs D7*
**Files:** `~/.automatos-analyst/auto-eval/replay.py`, `~/.automatos-analyst/auto-eval/runs/results.md` · **Depends on:** US-028

**Acceptance criteria:**
- [ ] Each run writes `FINGERPRINT.json` with: build SHA, `AUTOMATOS_IMAGE_TAG`, skill sha256, persona hash, agent 1's skill set, `model_config`, routing env flags, c1 `personality_mode`, eval-set sha, split sha (null until US-043 seals `split.json`), scorer sha, the late-write margin, and whether the complexity-key flush ran. `runs/results.md` gains a fingerprint column.
- [ ] The `x-auto-complexity` header is captured per turn, and the run reports its ATOM share.
- [ ] At every fingerprint, code checks that every `expect_tools` and `forbid_tools` pattern (case-insensitive regular expressions, `replay.py` `any_match`) matches at least one ActionRegistry action name after stripping a `:status` suffix, and that every c1 entity a row references exists. An invalid row retires, or marks the run VOID.

#### US-035: Token and embedding meters cover every model call of a turn, and unpriced usage stops a run
**Description:** As Gerard, I want per-turn tokens summed over every model call, embedding spend metered and unpriced usage refused, so a slim-skill saving is measured, not assumed, and a spend limit is never checked against a meter that reads $0.
*Tags: Local-only · Needs D7*
**Files:** `~/.automatos-analyst/auto-eval/replay.py` · **Depends on:** US-028

**Acceptance criteria:**
- [ ] Tokens per turn = the sum over the turn's `llm_usage` rows of `input_tokens + output_tokens`. `input_tokens` already includes cache reads and writes (`core/models/core.py:194-201`); `cache_read_tokens` and `cache_write_tokens` are reported as its breakdown and never added again. The run reports the `platform_load_skill` call count and the p90 turn latency.
- [ ] Embedding $ per row-run comes from `llm_usage` rows with `request_type='embedding'` (`core.py:193`). Any embedding error in a pass marks the pass VOID.
- [ ] If any `llm_usage` row in the run window has tokens > 0 and `total_cost = 0` for a provider other than the subscription runtime (`claude_code`), the run refuses to start (checked against the latest calibration window) or stops, printing "unpriced usage: <request_type>/<model>". Embedding rows are booked at $0 today when no price is known (`core/llm/usage_tracker.py:344-347`), so P2 needs §11-Q29 answered before its first paid pass.

**LOCAL-4: skill arm and export** (US-036, US-037)

#### US-036: A skill probe swaps one row and always puts it back
**Description:** As Gerard, I want to probe a candidate skill body by swapping only the global row's `prompt_template` and always restoring it, so a probe never leaves a candidate live on the local stack.
*Tags: Local-only · Needs D7*
**Files:** `~/.automatos-analyst/auto-eval/replay.py` · **Depends on:** US-028; the state paths defined in US-041

**Acceptance criteria:**
- [ ] `--skill-arm <file>` snapshots `prompt_template`, `content_hash` and `skill_version` of the global platform-management row, agent 1's `agent_skills` rows, and any Skill rows the probe creates. It asserts that agent 1's platform-management link points at the global row id: Auto's always-on body is the linked row's `prompt_template` (`modules/context/sections/skills.py:74-96`, `:210-218`), and a workspace fork would also win `platform_load_skill` (`handlers_skill_runtime.py:58-63`). It UPDATEs `prompt_template` only and aborts unless exactly 1 row changed, checks the first turn's `context_trace` skills token count, and restores everything in `finally`.
- [ ] At arm start it writes `~/.automatos-analyst/harness-lab/state/ARM_ACTIVE` and `~/.automatos-analyst/harness-lab/state/skill_snapshot.json` (paths defined in US-041), refuses to start if either already exists, removes both only after a verified restore, and then rewrites `skill_set_baseline.json` from the restored skill set.
- [ ] → OWNER (local, $0): a dry run with no turns swaps and restores. Afterwards `sha256(prompt_template) == content_hash` and agent 1's skill set equals the pre-run snapshot (SQL output recorded).

#### US-037: Exports carry only whitelisted train fields and pass the tripwire
**Description:** As the Lab, I want exports of train trajectories stripped to a whitelist and scanned for grader markers, so improvers never see the rubric or any held-out row.
*Tags: Local-only · Needs D7*
**Files:** `~/.automatos-analyst/auto-eval/replay.py`, `~/.automatos-analyst/harness-lab/tripwire.py` · **Depends on:** US-028, US-043

**Acceptance criteria:**
- [ ] `--export-context` writes, for train rows only, to `~/.automatos-analyst/harness-lab/exports/<run>/`: `prompt`, `calls`, `results`, `errors`, `receipts`, `text`, `answered_by`, the turn's `x-auto-complexity` value (US-034) and `usage`, with ids anonymised in every field (`results` and `errors` are kept because full trajectories outperform summaries, §1). It drops `checks`, `passed`, `notes`, `category`, `f_id`, `setup`, `owner_only`, `score_all_turns` and every `expect_*` key.
- [ ] `tripwire.py`, adapted from Harness2 `core/context.py:419-470` with its unchanged Apache-2.0 header and a change notice (§9), runs on every export. Its markers are JSON-key forms (`"checks":`, `"passed":`, `"notes":`, `"category":`, `"f_id":`, `"setup":`, `"owner_only":`, `"score_all_turns":`) plus distinctive bare tokens (`tools_ok`, `no_forbidden`, `effects_ok`, `reply_ok`, `http_ok`, `claim_backed`, `owner_only_ok`, `lane_ok`, `forbid_tools`, `reply_must`, `expect_`), and F-id and row-id regexes. Bare words such as "checks" or "passed" are not markers, because they occur in ordinary reply text. Any hit refuses the export.
- [ ] A fixture export with one planted marker is refused; a clean export passes (fixture mode).
- [ ] Held-out rows are never exported (asserted against `split.json`).

**LOCAL-5: reset, compare and rescore** (US-038 – US-040)

#### US-038: Each reset fails loudly, clears routing affinities and cached verdicts, and night evals need the operator's spend values
**Description:** As Gerard, I want each reset to fail loudly on a failed DELETE, to clear the window's new routing affinities and the complexity cache, and every night-chain eval to carry my spend limit and cost factor, so one pass cannot steer the next and the night chain keeps working under D4.
*Tags: Local-only · Needs D7*
**Files:** `~/.automatos-analyst/auto-eval/phase.sh`, `~/.automatos-sim/tester-kit/chain-n*.sh` · **Depends on:** US-028

**Acceptance criteria:**
- [ ] `phase.sh` checks the HTTP status of every DELETE and exits non-zero on any non-2xx.
- [ ] It resets the window's new `tool_routing_affinities` rows (the recorder weights new rows at once, `signal_recorder.py:674-683`), deletes the Redis keys matching `complexity:<c1 workspace id>:*` and prints the count (US-029), and keeps the >10 h refusal and the backups.
- [ ] `phase.sh --reset-only` runs the reset alone (added if phase.sh has no such mode), printing "nothing to reset" when no `.eval-until` exists; `replay.py --reps` calls it after every pass (US-033).
- [ ] `phase.sh <arm>` requires `--max-spend` and `--cost-factor` (or the env values `MAX_SPEND` and `COST_FACTOR`) with no default, exits non-zero with "refusing to start: night eval spend limit unset" before any request when either is missing, and passes both to `replay.py`. Every night-chain script that calls `phase.sh <arm>` (today `chain-n16.sh:13`) reads the operator's values from a local config file that is never committed.

#### US-039: Arm comparisons use the shared statistics and show all-service spend
**Description:** As Gerard, I want per-row flake tables, a row-level sign test from the shared statistics module and all-service spend including embeddings, so arm comparisons use the same statistics as HARNESS and show true spend.
*Tags: Local-only · Needs D7*
**Files:** `~/.automatos-analyst/auto-eval/compare-arms.py`, `arm-report.py`, `rescore.py` · **Depends on:** US-008, US-030, US-031

**Acceptance criteria:**
- [ ] `compare-arms.py` outputs a per-row flake table, the row-level exact sign test, a guard verdict, and analysis stratified by day. It loads automatos-ai `orchestrator/core/stats/harness_verdicts.py` by file path with `importlib.util.spec_from_file_location` (as `rescore.py` loads `replay.py`), never through a package import.
- [ ] `arm-report.py` reports all-service booked $, embeddings included, per run and per row-run, and metered $ at the run's recorded cost factor.
- [ ] Rescoring makes no model call and no API request: the c1 `llm_usage` row count is unchanged across the rescore, and replay's HTTP client is not constructed. The claim check runs by `docker exec` in the running backend container (the container must be up; no chat traffic).

#### US-040: Every saved run since 9 Oct is rescored at $0
**Description:** As Gerard, I want every saved run since 9 Oct rescored with the fixed claim check, so no older figure is used as a gate on a broken `claim_backed`.
*Tags: Local-only · Needs D7*
**Files:** `~/.automatos-analyst/auto-eval/rescore.py` · **Depends on:** US-030, US-031, US-039

**Acceptance criteria:**
- [ ] Every run directory since 9 Oct that holds `rows.jsonl` (11 at 18 Oct) is rescored; directories without `rows.jsonl` (3 aborted runs on 9 Oct) are listed as not rescorable. A local table lists run, date, old pass count, new pass count, `claim_backed` delta, ungraded count and whether the run saved receipts.
- [ ] Only aggregate counts are copied into this PRD (§10); no row contents.
- [x] No model calls are made ($0).
- [x] **Rescored 20 Oct 2026 (build session HARNESS).**
  - 11 runs were rescored. The 3 aborted runs of 9 Oct are not rescorable.
  - 0 rows are ungraded.
  - c1's `llm_usage` row count was 237,436 before and after, so the rescore cost $0.
  - Passed rows per run, old → new: 40→50, 37→43, 44→46, 50→50, 48→47, 51→48, 52→50, 61→61, 60→59, 59→58 and 60→60.
  - `claim_backed` changes, run by run: +3, −1, 0, −1, −3, −6, −3, 0, −1, 0, +1.

**LOCAL-6: preflight and night signal** (US-041, US-042; no D7)

#### US-041: Every night chain refuses to start on a leftover probe
**Description:** As Gerard, I want every night chain to refuse to start on a dirty skill row, a live arm marker or snapshot, a changed skill set, an unexpected persona or a low OpenRouter balance, so a probe never leaks into a night.
*Tags: Local-only*
**Files:** `~/.automatos-sim/tester-kit/next-cycle.sh` · **Depends on:** —

**Acceptance criteria:**
- [ ] This story defines the state paths, outside auto-eval: the arm marker `~/.automatos-analyst/harness-lab/state/ARM_ACTIVE`, the snapshot `~/.automatos-analyst/harness-lab/state/skill_snapshot.json` and the skill-set baseline `~/.automatos-analyst/harness-lab/state/skill_set_baseline.json`. On its first run, prep records the baseline of agent 1's skill set itself; US-036 writes and removes the marker and snapshot and refreshes the baseline after a verified restore.
- [ ] Before `build-from-main.sh` runs (`next-cycle.sh:67-73`), `next-cycle.sh prep` refuses when: `sha256(prompt_template) != content_hash` on platform-management; the arm marker exists; the snapshot exists (an orphan from a crashed arm); agent 1's `agent_skills` set differs from the baseline; or the OpenRouter balance is below the operator's floor. The floor is a required local config value with no default, and prep refuses if it is unset (D4).
- [ ] After the new build's `/health` returns 200 (c1 is re-seeded and its persona lifted at every boot, `seed_local_first_run.py:754-768`, `seed_auto_agent.py:369-411`), prep compares c1 Auto's persona hash with the shipped text for c1's `personality_mode` (`_CURRENT_PERSONA_HASHES`, `seed_auto_agent.py:244-251`) and refuses on a mismatch.
- [ ] → OWNER (local, $0): a planted dirty row, a planted extra skill link and a planted orphan snapshot each make prep exit non-zero with the reason printed.
- [ ] → OWNER: before the first night after LOCAL-3 or LOCAL-6 lands, Gerard sets the night eval spend limit, the cost factor and the OpenRouter floor in local config; the next chain log shows prep and the pre-night eval reading all three.
- [ ] The script reads nothing under `~/.automatos-analyst/auto-eval` (grep).

#### US-042: Each night yields deterministic signal rates
**Description:** As Gerard, I want deterministic per-night signal rates per 100 asks from the database and the logs, so every night yields checkable numbers without touching the eval folder.
*Tags: Local-only*
**Files:** `~/.automatos-analyst/harness-lab/harvest.py` · **Depends on:** —

**Acceptance criteria:**
- [ ] `harvest.py` reports per-night rates per 100 asks of: unbacked-claim log lines, split into chat turns and agent runs by joining each line (timestamp and sentence) to saved assistant messages (chat) or board-task execution results (agent runs), because `its_answer_is_checked` wraps every `ToolLoopExecutor.run` (`claims_in_place.py:67-77`) and `agent=` is never populated in production logs (`logging_adapter.py:86-112`; no production caller sets the run or request context); F187 tier-2 lines; full `NOT_DONE`, `NOTHING_DONE_SAID` and `NO_SUCH_ID` templates in saved replies; refused, waiting and skipped receipts; denied and expired grants; and windowed production tool failures.
- [ ] A `req=` join is added only if US-032 passes; the other rates do not wait for it.
- [ ] It issues SELECT statements only (grep finds no INSERT, UPDATE or DELETE) and never opens the auto-eval folder (grep).
- [ ] The first report covers the latest night, and its figures are recorded in §10.

**LOCAL-7: set v2 and calibration** (US-043 – US-045)

#### US-043: Set v2 and a sealed split
**Description:** As Gerard, I want new rows from nights 12–17 and a sealed split grouped by f_id, so the held-out gate has enough valid failing rows and never shares a finding with train.
*Tags: Human-gated · Local-only · Needs D7*
**Files:** `~/.automatos-analyst/auto-eval/auto-eval.jsonl`, `split.json`, `make_split.py` · **Depends on:** US-028, US-034

**Acceptance criteria:**
- [ ] Rows from F385–F449 (nights 12–17), plus F382–F383 (night 11) if they carry a customer ask, are authored by Gerard or by a row-author session separate from improver sessions (§11-Q4: about 30 rows). Code-scanning and infrastructure findings (for example F384) are excluded. Each row passes the row-validity check (US-034).
- [ ] `split.json` groups by `f_id` and stratifies by category at about 60/40. A fixed list of held-out-only categories, recorded in `make_split.py` (local), is never placed in train. Held-out has at least 35 rows. The seed and mapping stay local, and the split sha goes into the fingerprint.
- [ ] The rotation rule (held-out retires into train after 5 gate decisions and is replaced by rows chosen by the local split tooling) is recorded with the split tooling.
- [x] Nothing from auto-eval is committed to automatos-ai or automatos-skills (`git status` of both).
- [x] **Done 20 Oct 2026.**
  - The build session HARNESS wrote 32 rows, at Gerard's direction, from the findings F383–F449. It left out infrastructure, CodeQL, pricing and agent-run-only findings, and any ask that would delete real c1 data.
  - Every row passes the row-validity check against the live action registry.
  - Two existing rows, F351-a and F351-b, lost a forbidden pattern that named a removed action; nothing else in them changed.
  - `split.json` is sealed: 97 rows, 51 train and 46 held-out. It is grouped by finding, and held-out-only categories are never in train. The seed stays local.

#### US-044: Calibration: six passes on one pinned build (Gerard's run)
**Description:** As Gerard, I want reset passes on one pinned build, with and without the complexity-key flush, so I have the per-row flake table, the noise floor, the ATOM share, the late-write margin and the real cost factor before any gate is trusted.
*Tags: Human-gated · Local-only · Needs D7*
**Files:** `~/.automatos-analyst/auto-eval/runs/` · **Depends on:** US-029, US-030, US-031, US-033, US-034, US-035, US-038, US-039, US-041

**Acceptance criteria:**
- [ ] → OWNER: Gerard launches 6 full passes on one pinned `AUTOMATOS_IMAGE_TAG` with his own `--max-spend` and `--cost-factor`: 3 with the complexity-key flush and 3 without, interleaved ABAB by day. Each pass follows a verified `phase.sh` reset and runs on the quiet stack (US-029).
- [ ] Outputs under one fingerprint: the per-row flake table, the noise floor with and without the flush, the ATOM share, the measured late-write tail (the margin US-033 uses next), the measured true/booked factor (true read by Gerard from the provider consoles), embedding $ per row-run, and a run with 0 ungraded rows and per-row claim-error counts.
- [ ] The planning estimate is recorded as informational only (390 row-runs, about $5.58–7.41 booked, about 6.1 h of probe time plus about 4.5 h of reset waits at today's 45-min margin; §13), with the metered figure beside it.
- [ ] Gerard confirms or moves the eval bars in §10 before P4 starts.

#### US-045: Set v2 recalibration and the count of valid failing held-out rows
**Description:** As Gerard, I want set v2 calibrated on the pinned build and the count of valid failing held-out rows reported, so P5 stops early if the gate cannot have power.
*Tags: Human-gated · Local-only · Needs D7*
**Files:** `~/.automatos-analyst/auto-eval/runs/` · **Depends on:** US-043, US-044

**Acceptance criteria:**
- [ ] → OWNER: Gerard launches 3 passes over set v2 on the pinned build with his own `--max-spend` and `--cost-factor`; a flake table is recorded under the v2 fingerprint.
- [ ] This PRD records the count (the count only) of valid failing held-out rows on that build. If it is below 6, this PRD records that P5 does not start on this evidence (§9 power table); US-078a re-measures it on the Lab's pinned tag.
- [ ] The planning estimate is recorded as informational only (285 row-runs, $4.08–5.42 booked, about 4.5 h plus 2.25 h of reset waits; §13).

**DOC-2: AUTO-EVAL.md** (US-046)

#### US-046: AUTO-EVAL.md describes the repaired instrument
**Description:** As anyone running the eval, I want `AUTO-EVAL.md` to describe the repaired instrument, so no one follows the stale arms, keys or claim source.
*Tags: Needs D7*
**Files:** `docs/testing/AUTO-EVAL.md` · **Depends on:** US-028, US-030, US-033, US-036, US-037, US-043

**Acceptance criteria:**
- [ ] The stale sections are removed: arms A–D, the OpenRouter key in c1, the old claims source (grep `action_claims` returns 0).
- [ ] It describes `claims_backed` grading with VOID on ungraded rows, per-row claim errors, `--reps` and ABAB with verified resets, the required `--max-spend` and `--cost-factor` with no default, the fingerprint, the skill arm and the preflight, the export and the tripwire, and split sealing. Split sealing is described without naming held-out categories or the rotation source. It contains no row contents and no gold text.
- [ ] Docs-only PR.

### Part 3 (P3): Governed apply

**PR-3A: delete the /approve path** (US-047; ~480 lines, deletion-only; over ~400 because it is all deletion, and a split would leave a dead module or bypass key, §4 "PR size"; needs PR-1E, PR-1F, PR-1G and US-003 and US-004 approved)

#### US-047: The always-approving /approve path and its bypass key are deleted (Gerard's sign-off first)
**Description:** As a maintainer, I want the always-approving `/approve` path and its owner-only bypass key deleted, so the only way a HARNESS change applies is the owner's click on an Approvals card.
**Files:** `orchestrator/api/harness.py`, `orchestrator/api/harness_commands.py` (415 lines, deleted), `orchestrator/modules/tools/discovery/owner_only.py`, `orchestrator/services/harness_service.py` (one comment), `orchestrator/reports/route-manifest.json`, `docs/platform-actions/self-management-harness.md`, `orchestrator/tests/test_prd256_owner_actions_wait_for_the_click.py`, `orchestrator/tests/test_harness_commands.py` (deleted), `orchestrator/tests/test_harness_api.py`, `orchestrator/tests/test_prd179_harness_actuation.py` (deleted), `orchestrator/tests/test_f259_b_every_status_change_keeps_the_boards_rules.py`, `orchestrator/tests/test_deployability_w6.py`, `orchestrator/tests/test_config_env_centralization.py` · **Depends on:** US-003, US-004, US-020

**Acceptance criteria:**
- [ ] Deleted from `orchestrator/api/harness.py`: the approve and reject routes and `_run_command` (`:48-84`); the `handle_harness_command` import (`:30`); the `_resolve_internal_user_id` alias (`:42-45`); and the imports that become unused, re-verified by ruff F401 (`HTTPException` `:27`, `resolve_internal_user_id` `:31`, and `require_workspace_permission` `:34` unless US-021's access check uses it). The module docstring (`:1-21`) is rewritten to describe `/self-learning` only.
- [ ] Deleted: `orchestrator/api/harness_commands.py` with its approve and reject handlers, `_governed_verdict`, `_already_handled` and `_find_task_by_rx`. grep `harness_commands` in `orchestrator/`, tests excluded, returns 0 (including the comment at `harness_service.py:1750`).
- [ ] In `owner_only.py`: `OWNERS_OWN_DECISION` (`:89`), the comment above it (`:86-88`), its branch in `human_driven` (`:217`) and that docstring's mention (`:212-214`), its `__all__` entry (`:545`) and the module docstring mention (`:16`) are deleted.
- [ ] `orchestrator/reports/route-manifest.json` loses exactly `{"method": "POST", "path": "/api/harness/prescriptions/{rx_id}/approve"}` and `{"method": "POST", "path": "/api/harness/prescriptions/{rx_id}/reject"}`, and `route_count` drops by 2 (914 → 912 at `697ca7d6e`; re-read at build time). The frontend-ci route-contract check proves it.
- [ ] The case at `test_prd256_owner_actions_wait_for_the_click.py:181` is replaced by a test that a `caller_context` carrying `"owners_own_decision": True` still waits for the click. It fails at `697ca7d6e`, where the key bypasses, and the test's docstring says why it changed. The `OWNERS_OWN_DECISION` import at `:22` goes.
- [ ] `test_harness_commands.py` is deleted; the approve, reject and `_resolve_internal_user_id` cases of `test_harness_api.py` (`:115-128` and the route cases) are deleted, the resolver being covered where `core.auth` defines it; `test_prd179_harness_actuation.py` is deleted in full (it imports `api.harness_commands` at module level, `:66`, and all 3 tests call `handle_harness_command`). Each is mapped in the PR description to `test_prd263_harness_grant_flow.py` (US-054). The references in `test_f259_b_every_status_change_keeps_the_boards_rules.py`, `test_deployability_w6.py` and `test_config_env_centralization.py` are updated, each with the reason in the test's docstring.
- [ ] The `/approve` note on `docs/platform-actions/self-management-harness.md` (US-023a) is deleted.
- [ ] security-reviewer runs on this story. The PR links US-003 and US-004.

**PR-3B: delete the legacy apply helpers** (US-048; ~560 lines, ~530 of them deletions; over ~400 because it is mostly deletion, and a split would leave a half-deleted path, §4 "PR size"; needs PR-3A and US-006 decided)

#### US-048: The direct apply helper and the legacy ledger-file code are deleted, and the audit report no longer uses the executor (Gerard's sign-off first)
**Description:** As a maintainer, I want the direct apply helper, the card parsers and the legacy ledger-file code deleted, and the audit report submitted without the executor, so `harness_service.py` can hold the "no executor" grep gate.
**Files:** `orchestrator/services/harness_service.py`, `orchestrator/tests/conftest.py`, `orchestrator/tests/test_prd263_harness_grep_gate.py`, `orchestrator/tests/test_prd263_harness_audit_report.py`, plus the deleted tests listed below · **Depends on:** US-006, US-016, US-047

**Acceptance criteria:**
- [ ] Deleted from `harness_service.py` (grep returns 0): `_auto_apply_prescription`, with the `review_needed` placeholder guard and its branches for `temperature_adjust`, `tag_update`, `routing_rule_add`, `tool_assignment_add` and `power_mode_*`; `_snapshot_current_value`; `_resolve_agents_by_name` and `_extract_agents_list` (no caller is left once US-047 deletes `api/harness_commands.py`); `_parse_harness_task`, `_extract_json_section`, `_extract_text_section` and `_extract_risk_score`; `_read_applied_tasks` and `_write_applied_tasks`; and, per US-006, the legacy ledger-file importer (`LEGACY_LEDGER_FILE`, `UNREADABLE_LEDGER_*`, `_import_legacy_ledger`, `_file_unreadable_ledger_card`).
- [ ] The weekly audit report is submitted through a direct `ReportService.create_report` call (`services/report_service.py:137`) attributed to the workspace's Auto, not through `PlatformActionExecutor`. `test_prd263_harness_audit_report.py` asserts the report row's agent is that Auto. The PRD-248 shadow triage still wraps `create_report` and runs only when the `report_triage_mode` dial is on (FR-28).
- [ ] `test_prd263_harness_grep_gate.py` now covers `harness_service.py`: no `PlatformActionExecutor`, `UnifiedToolExecutor` or `execute_tool`, and no LLM client import. It fails at `697ca7d6e`.
- [ ] The `harness_ledger` fixture in `tests/conftest.py` (`:461-485`), which patches `_read_applied_tasks`/`_write_applied_tasks`, is deleted.
- [ ] Deleted and mapped in the PR description: the rest of `test_harness_self_management.py`; `test_f156_the_harness_ledger_is_in_the_database.py` and `test_f156_two_callers_file_one_card.py` in full (every case calls a deleted function), mapped to `test_prd263_harness_ledger_migration.py` (US-051) and `test_prd263_harness_apply_action.py` (US-052); the `_auto_apply_prescription` cases of `test_harness_power_mode.py` and `test_harness_routing_rule.py`, mapped to `test_prd263_harness_changes_registry.py` (US-049).
- [ ] `harness_service.py` ends shorter (about 530 lines removed). The PR links US-006.

**PR-3C: the change-kind registry** (US-049, US-050; ~380 changed lines; needs PR-3B)

#### US-049: One registry defines every change kind
**Description:** As a maintainer, I want one registry that defines every change kind, so PRESCRIBE and the apply action share one vocabulary and the 4-vs-10 mismatch cannot recur.
**Files:** `orchestrator/services/harness_changes.py` (new), `orchestrator/services/harness_service.py`, `orchestrator/tests/test_prd263_harness_changes_registry.py`, `orchestrator/tests/test_prd263_harness_grep_gate.py` · **Depends on:** US-016, US-048

**Acceptance criteria:**
- [ ] New `orchestrator/services/harness_changes.py` maps each kind to: detector, inner platform action, params builder, risk, predicted metric, guard metrics, prior-value reader and rollback builder. Apply kinds: `heartbeat_tune` (`platform_configure_agent_heartbeat`), `tool_assignment_remove` (`platform_unassign_tool_from_agent`) and `rollback`. Report kinds: `reconnect_app`, `model`, `description`, `temperature`, `tags`, `routing` and `power_mode`.
- [ ] `_phase_prescribe` becomes a loop over the registry. The validator drops an invalid prescription and logs the reason in `harness_runs.prescriptions`; nothing is queued with a placeholder.
- [ ] Deleted: the placeholder emitters for `model_change_same_tier` and `description_update`, `_PLACEHOLDER_PROPOSED_VALUE` and `_MIN_SUCCESS_RATE_FOR_COST_OPT` (`harness_service.py:80`). grep `review_needed` in `orchestrator/services` returns 0 (§11-Q9).
- [ ] The registry never targets the workspace's Auto.
- [ ] Tests: `test_prd263_harness_changes_registry.py`: every apply kind has all eight fields; report kinds yield report lines only; a prescription for the workspace's Auto is dropped.
- [ ] The grep gate covers `harness_changes.py`, and `harness_service.py` ends shorter.

#### US-050: HARNESS proposes a longer interval only for a heartbeat that rarely meets its objective
**Description:** As an owner, I want HARNESS to propose a longer interval only for an enabled heartbeat that rarely meets its objective, so heartbeat cost falls without dropping useful monitoring.
**Files:** `orchestrator/services/harness_changes.py`, `orchestrator/config.py`, `orchestrator/reports/config-surface.json`, `orchestrator/tests/test_prd263_harness_heartbeat_detector.py` · **Depends on:** US-049

**Acceptance criteria:**
- [ ] `heartbeat_tune` considers only enabled heartbeats of non-Auto agents and keeps `objective_met` None out of the denominator. It proposes doubling the interval (clamped to 5–1440, as the handler does, `handlers_assignments.py:360-366`) when non-None runs ≥ `HARNESS_HEARTBEAT_MIN_RUNS` (20) and `wilson_upper_bound(objective met) < HARNESS_HEARTBEAT_MAX_MET` (0.2). Both settings go in `config.py` and `config-surface.json`.
- [ ] The prior value is read from `configuration.heartbeat.interval_minutes`.
- [ ] Tests: `test_prd263_harness_heartbeat_detector.py`: 25 non-None runs with 1 met proposes 2×; 25 runs of which 15 are None proposes nothing (INSUFFICIENT_DATA); a disabled heartbeat proposes nothing.
- [ ] Built at §11-Q10's default unless Gerard sets other values before the PR launches.

**PR-3D: the ledger migration** (US-051, US-059; ~150 changed lines; needs PR-3B, PR-1G, US-005 with prod totals, and US-006)

#### US-051: The ledger tracks each proposal's grant, prior value and outcome (Gerard's sign-off first)
**Description:** As the governed apply rail, I want the ledger to track each prescription's grant, prior value and outcome, so every click, lapse and drift has a recorded final state.
**Files:** `orchestrator/core/models/harness.py`, `orchestrator/alembic/versions/prd263_harness_ledger_grants.py`, `orchestrator/tests/test_prd209_alembic_single_head.py`, `orchestrator/tests/test_prd236_w1_routes.py`, `orchestrator/tests/test_prd263_harness_ledger_migration.py` · **Depends on:** US-005, US-048

**Acceptance criteria:**
- [ ] `harness_task_ledger` gains `rx_id` (unique per workspace), `run_id`, `grant_id` (Integer, matching `approval_grants.id`), `kind`, `target_agent_id`, `prior_value`, `proposed_value`, `applied_at`, `value_at_apply`, `frozen_ref_start` and `frozen_ref_end`. `board_task_id` becomes nullable. The state CHECK becomes `pending, applying, applied, failed, denied, expired, drifted, withdrawn, raise_failed, rolled_back`, and existing `held` rows become `expired`.
- [ ] The ORM model gains the same columns and declares the new CHECK as `ck_harness_task_ledger_state` (the name at `core/models/harness.py:32` and `f156_harness_task_ledger.py:33`) and the unique index as `uq_harness_task_ledger_rx`, with the migration's names, so the schema-drift and alembic-from-zero lanes see one shape.
- [ ] ONE Alembic revision, `prd263_harness_ledger_grants`, chains onto the head at build time (after `prd263_harness_runs`). Upgrade order: `ADD COLUMN IF NOT EXISTS` for each column; `ALTER COLUMN board_task_id DROP NOT NULL`; `DROP CONSTRAINT IF EXISTS ck_harness_task_ledger_state`; `UPDATE harness_task_ledger SET state='expired' WHERE state='held'`; `ADD CONSTRAINT ck_harness_task_ledger_state CHECK (state IN (...))`; `CREATE UNIQUE INDEX IF NOT EXISTS uq_harness_task_ledger_rx ON harness_task_ledger (workspace_id, rx_id)` (legacy rows with NULL `rx_id` allowed). It survives a `create_all`-built database. Downgrade uses `IF EXISTS`. Both head pins move.
- [ ] Tests: `test_prd263_harness_ledger_migration.py`: a `held` row becomes `expired`; inserting `held` fails; a duplicate `rx_id` in one workspace fails; one row with every new column round-trips through raw SQL on `db_session`; workspace B's row never appears in workspace A's read.
- [ ] The PR links US-005, with the prod state totals posted there.

#### US-059: Old HARNESS files leave tenant volumes only after a verified import (Gerard's sign-off first)
**Description:** As Gerard, I want the old HARNESS JSON files removed only after their history is in `harness_runs`, so tenant volumes stop carrying data the product no longer reads.
*Tags: Human-gated (the prod run)*
**Files:** `orchestrator/scripts/harness_backfill_runs.py`, `orchestrator/tests/test_prd263_harness_backfill_runs.py` · **Depends on:** US-006, US-023

**Acceptance criteria:**
- [ ] `harness_backfill_runs.py` gains `--remove-files`, which deletes a workspace's `harness/*.json` only after that workspace's import count verifies. The default stays `--dry-run`.
- [ ] Tests: `test_prd263_harness_backfill_runs.py::test_remove_files_only_after_verified_import`.
- [ ] → OWNER: Gerard runs the backfill, then `--remove-files`, against prod, and posts only totals on the US-006 issue (`--summary-only`): workspaces processed, runs imported, files removed, and workspaces with an import-count mismatch (must be 0). The per-workspace output stays in the private ops note. No workspace id, name or slug appears on any public issue.
- [ ] The PR links US-006.

**PR-3E: the apply action and reconciliation** (US-052, US-053; ~370 changed lines; needs PR-3C, PR-3D and US-004)

#### US-052: One apply action, and it acts only on a live grant over an unchanged setting (Gerard's sign-off first)
**Description:** As an owner who clicks Approve, I want a dedicated apply action that checks the grant is still valid and the setting still reads its "before" value, so a late or stale click can never overwrite a manual edit.
**Files:** `orchestrator/modules/tools/discovery/actions_harness.py`, `orchestrator/modules/tools/discovery/handlers_harness.py` (or a new `handlers_harness_apply.py`), `orchestrator/modules/tools/discovery/platform_executor.py` (1,539 lines), `orchestrator/scripts/check_hierarchy_gate.py`, `orchestrator/tests/test_prd263_harness_apply_action.py` · **Depends on:** US-004, US-049, US-051

**Acceptance criteria:**
- [ ] New `ActionDefinition` `platform_harness_apply`, registered inline in `actions_harness.py` (`permission_level` write, `requires_confirmation` False). The `check_hierarchy_gate.py` entry approved in US-004 (by default an `ALLOW_LIST` entry with its justification comment) lands in the same commit as the ActionDefinition, because the hierarchy gate audit runs before pytest in CI.
- [ ] `harness_apply` is imported beside the other HARNESS handlers (`platform_executor.py:168-171`) and mapped as `"platform_harness_apply": harness_apply` (`:1477-1480`); the PR notes the file-growth warning (1,539 lines).
- [ ] Its handler, in `handlers_harness.py` or in a new `handlers_harness_apply.py` if `handlers_harness.py` would pass 800 lines:
  1. loads the ledger row by `rx_id` in this workspace;
  2. loads its grant by `ledger.grant_id` and requires `is_authorising(grant)` (granted and not past `expires_at`, `core/services/approval_grants.py:279-293`), else records `expired` and returns `{'success': False, 'error': 'This approval expired; nothing changed'}`;
  3. reads the target's current value through the registry's prior-value reader and requires it to equal `prior_value`, else records `drifted` and returns `{'success': False, 'error': 'This setting changed since the proposal; nothing changed'}`, so the GrantCard shows the true outcome;
  4. flips `pending` to `applying` with a guarded UPDATE, so a second call changes nothing;
  5. resolves the workspace's Auto server-side (as `HarnessService._resolve_auto_agent` does, `harness_service.py:1350`) and never trusts `params['_agent_id']` (the platform path strips it and re-injects the runtime id, `exec_platform.py:84-99`); it runs the registry's inner action as that Auto, so the admin, rate-limit and destructive checks run on the inner call, while `can_actor_modify` checks only that Auto exists, is in the workspace and is active, then applies `system_actor_bypass` (`hierarchy_permissions.py:166-195`). [I: running the inner action with Auto's trusted id from inside a handler is unverified (design §4.1 #5); if the inner call cannot carry it, `RALPH_BLOCKED` with the grep.]
  6. records `applied` (with `applied_at` and `value_at_apply`) or `failed`.
- [ ] A chat or heartbeat call of the action without a granted grant changes nothing.
- [ ] Tests: `test_prd263_harness_apply_action.py`: a granted grant with a matching prior value → `applied`, with no `anonymous_actor`; an expired grant → `expired`, no change, and the expiry message; a setting edited after the proposal → `drifted`, no change; a double call changes once; no grant → no change; a `_agent_id` passed in params is ignored; the inner change runs only inside `platform_harness_apply` (the test patches the inner action and asserts its only caller).
- [ ] security-reviewer runs on this story. The PR links US-004.

#### US-053: Every proposal ends in a recorded state, and lapsed cards close within a day
**Description:** As an owner, I want lapsed or drifted cards closed within a day and every prescription to end in a recorded state, so the inbox never holds a card that can no longer be acted on and no click is lost to a poll.
**Files:** `orchestrator/services/harness_grants.py` (new), `orchestrator/services/harness_service.py`, `orchestrator/config.py`, `orchestrator/reports/config-surface.json`, `orchestrator/tests/test_prd263_harness_reconcile_sweep.py`, `orchestrator/tests/test_prd263_harness_grep_gate.py` · **Depends on:** US-051, US-052

**Acceptance criteria:**
- [ ] Reconciliation runs at tick start (ORG-2) and in the daily sweep only, never on a GET. Each `pending` ledger row reconciles: grant DENIED → `denied`; REVOKED with no `executed_result` → `withdrawn`; PENDING past `expires_at` → closed with `expire_pending_grants` (`core/services/approval_grants.py:248-275`), then `expired`; GRANTED → `failed` only when `details.executed_result` exists and the row is still `pending` ("evidence card was blocked; nothing changed" when `executed_result.resumed_via` is `board_task_requeue`, otherwise "no outcome"), or when `granted_at` is older than `HARNESS_RECONCILE_GRACE_MINUTES` (default 60); an `applying` row older than the grace → `failed` ("interrupted").
- [ ] `failed`, `expired`, `drifted`, `denied`, `withdrawn`, `raise_failed` and `rolled_back` are final; `applied` is final except for the single transition `applied → rolled_back` made by US-056. A revoke after apply stays `applied`: revoke is allowed from GRANTED (`api/approval_grants.py:325-349`) but does not undo the change, so the change stays tracked for the post-change check.
- [ ] The status read derives each pending row's effective state from `is_authorising` and `expires_at` without writing (the endpoint is read by any member the US-002 decision allows, and the tile polls every 60 s, `use-learning-api.ts:71-72`).
- [ ] A SQL-only daily job on the shared scheduler at `HARNESS_GRANT_SWEEP_CRON` (default `"0 9 * * *"`, built with `scheduler_cron_trigger(..., timezone=timezone.utc)`; in `config.py` and `config-surface.json`, with `HARNESS_RECONCILE_GRACE_MINUTES`) applies the expiry rule and closes pending grants whose target's current value no longer equals `prior_value` (ledger `drifted`).
- [ ] The code lives in new `orchestrator/services/harness_grants.py`, because `harness_service.py` stays over 800 lines. The grep gate covers the new file.
- [ ] Tests: `test_prd263_harness_reconcile_sweep.py` covers each transition, including revoke-after-apply, the sweep's drift close, a grant flipped to GRANTED with no `executed_result` staying `pending` after reconcile, a stale `applying` row closing as interrupted, and a status read writing nothing.

**PR-3F: raise grants** (US-054, US-055, US-058; ~360 changed lines; needs PR-3E, so merges after PR-3A and PR-3B; needs US-001 and the US-002 decision)

#### US-054: In propose workspaces, each proposed change becomes an Approvals card with its evidence (Gerard's sign-off first)
**Description:** As an owner who opted into `propose`, I want each proposed change raised as an Approvals card with its evidence, so I decide every change with the numbers in front of me.
**Files:** `orchestrator/services/harness_grants.py`, `orchestrator/services/harness_service.py`, `orchestrator/config.py`, `frontend/components/settings/harness-settings-section.tsx`, `frontend/components/settings/__tests__/harness-settings-section.test.tsx`, `docs/platform-actions/self-management-harness.md`, `orchestrator/tests/test_prd263_harness_grant_flow.py`, `orchestrator/tests/test_prd263_harness_report_only.py` · **Depends on:** US-001, US-017, US-047, US-048, US-052, US-053

**Acceptance criteria:**
- [ ] `HARNESS_SELF_MANAGEMENT_ENABLED` is redefined in `config.py` (comment and docs) as the global switch that allows HARNESS to raise approval grants, default false. `config-surface.json` is unchanged (the name already exists).
- [ ] In `propose` workspaces with the switch on, PRESCRIBE raises one grant per valid apply-kind prescription via `issue_tool_grant(action='platform_harness_apply', params={'rx_id': ...}, permission_level='write', caller_context={'board_task_id': <evidence card id>}, question_md=...)` (`tool_grants.py:132-143`). The handler resolves Auto itself (US-052). `question_md` follows §8: what changes (agent, setting, before → after); why (n and rates for cur vs ref, the p-value, coverage); what approving does; how it is undone; and the 24 h expiry.
- [ ] A None return writes ledger `raise_failed` and a log line. `attach_ask_grant` is not used (grep in the harness modules returns 0). No grant is raised while a `pending` ledger row exists for the same agent and kind. `approval_pending` notifications (`tool_grants.py:185-189`) go out only for `propose` workspaces.
- [ ] One readable evidence card per run, filed as an owner's-decision card the way `approval_cards.py` files one (P256-FIX-N14-7): status `review`, `review_mode` `human`, no assigned agent, no `sla_deadline`, and the server's `owners_decision` mark in `planning_data`, so the board's Approve moves it to Done with no artifact asked (`done_needs_an_artifact.py`). The board dispatcher selects only `assigned` cards with an agent (`board_dispatcher.py:185-247`), and the SLA scan needs an `sla_deadline` (`:535-545`).
- [ ] If an owner drags the evidence card to Blocked, an Approve re-queues the card instead of running the action (`api/approval_grants.py:625-660`, `:735-748`); the ledger then ends `failed` ("evidence card was blocked; nothing changed") through US-053, and the agent setting is unchanged. `api/approval_grants.py` is not edited (§11-Q30).
- [ ] The settings section gains the "Propose" option with the §8 copy; its vitest asserts three options. The docs page gains a `propose` section (Approvals cards, 24 h expiry, rollback cards).
- [ ] `report_only` workspaces, or the switch off, produce 0 grants and 0 cards (`test_prd263_harness_report_only.py` extended).
- [ ] Tests: `test_prd263_harness_grant_flow.py` (scripted, no model calls): seeded evidence for an apply kind raises exactly one grant; approving through the approval API resumes into `platform_harness_apply` and ends `applied` with the prior value; deny ends `denied` with no change; a double click applies once; `test_switch_off_raises_no_grant`; after a grant ends `expired` or `drifted`, the next tick with the evidence still holding raises exactly one new grant and a new `pending` ledger row; `board_dispatcher.claim_tasks` does not claim the evidence card; with the evidence card moved to `blocked`, Approve changes no agent setting and the ledger ends `failed`.
- [ ] The grant-raising code lives in `harness_grants.py`, and `harness_service.py` does not grow. security-reviewer runs on this story. The PR links US-001.

#### US-055: Approving a HARNESS card writes no decision memory
**Description:** As every tenant, and for the eval, I want approving a HARNESS card not to write a workspace decision memory, so weekly loop decisions do not pile up in recall. The ledger and audit rows are the record.
**Files:** `orchestrator/services/click_memory.py`, `orchestrator/tests/test_prd263_harness_click_memory_excluded.py` · **Depends on:** US-054

**Acceptance criteria:**
- [ ] `remember_the_click` (`services/click_memory.py:81`) stores no memory when the grant's action is `platform_harness_apply`.
- [ ] `orchestrator/api/approval_grants.py` is unchanged. If the exclusion cannot be made in `click_memory.py`, work stops and a maintainer issue is opened (guardrail file).
- [ ] Tests: `test_prd263_harness_click_memory_excluded.py`: approving a HARNESS grant writes 0 memories; approving another tool_call grant still writes one.
- [ ] Built at §11-Q18's default unless Gerard picks tagging instead.

#### US-058: The tile shows pending, applied, declined and expired counts
**Description:** As an owner, I want the Self-Learning tile to show what happened to HARNESS proposals, so I can see pending cards and outcomes at a glance.
**Files:** `orchestrator/api/harness.py`, `frontend/components/activity/widgets/self-learning-health-widget.tsx`, `frontend/hooks/use-learning-api.ts`, `orchestrator/tests/test_prd263_self_learning_endpoint.py`, `frontend/components/activity/widgets/__tests__/self-learning-health-widget.test.tsx` · **Depends on:** US-021, US-022, US-051, US-053

**Acceptance criteria:**
- [ ] `GET /api/harness/self-learning` returns `pending`, `applied`, `denied` and `expired` ledger counts (counts only, per US-002; pending rows by their effective state, US-053). The tile shows them, with "—" when absent, in theme-token tones; `grep 'dark:'` stays 0 in the widget.
- [ ] Tests: `test_prd263_self_learning_endpoint.py::test_ledger_counts`.
- [ ] vitest: a new case in `self-learning-health-widget.test.tsx` (the four counts render; an absent count renders "—"); typecheck passes.
- [ ] → OWNER (after merge, local stack, dev-browser, 1440 px and 390 px): the tile shows pending, applied, declined and expired counts, each "—" when absent.

**PR-3G: post-change check and the tool-removal gate** (US-056, US-057; ~300 changed lines; needs PR-3F)

#### US-056: An approved change is checked against the weeks before it, and a rollback comes as its own card
**Description:** As an owner who approved a change, I want HARNESS to watch the predicted and guard metrics against the pre-change baseline and raise a rollback card if they fall, so a bad change is caught instead of absorbed into a trailing reference.
**Files:** `orchestrator/services/harness_post_change.py` (new), `orchestrator/config.py`, `orchestrator/reports/config-surface.json`, `orchestrator/tests/test_prd263_harness_post_change.py` · **Depends on:** US-052, US-054

**Acceptance criteria:**
- [ ] For each `applied` change, the frozen reference is the `HARNESS_REF_DAYS` before `applied_at` (stored in `frozen_ref_start`/`frozen_ref_end`). The judged metrics are the predicted metric plus the agent's task success rate and tool failure rate. Each waits for n ≥ `HARNESS_MIN_SAMPLES` for up to `HARNESS_POST_CHANGE_MAX_WEEKS` (4; in `config.py` and `config-surface.json`); otherwise the report says "change not judged" with its n.
- [ ] Trigger: two consecutive weekly flags, or one at `HARNESS_STRONG_P`. Control: the treated agent's drop must exceed the median drop of judged unchanged agents; when none was judged, the rollback is raised labelled "no control". A `kind=rollback` ledger row is never a rollback target. The rollback is raised as its own grant, and the original row moves `applied → rolled_back` once the rollback applies.
- [ ] The code lives in new `orchestrator/services/harness_post_change.py`, covered by the grep gate.
- [ ] Tests: `test_prd263_harness_post_change.py`: a frozen-reference regression raises one rollback grant; a `tool_assignment_remove` whose app n drops to 0 is judged on task success; a `heartbeat_tune` that halves n waits; no judged control gives the "no control" label; a rollback is never rolled back.

#### US-057: A tool is proposed for removal only when it truly fails and nothing depends on it
**Description:** As an owner, I want tool removal proposed only when the tool truly fails for that agent and nothing I run depends on it, so a broken connection or a needed app is never removed.
**Files:** `orchestrator/services/harness_changes.py`, `orchestrator/services/harness_metrics.py`, `orchestrator/tests/test_prd263_harness_tool_removal_gate.py` · **Depends on:** US-019, US-049

**Acceptance criteria:**
- [ ] `tool_assignment_remove` is an apply kind only when all of these hold: `wilson_lower_bound` of the agent–app failure rate ≥ `HARNESS_TOOL_FAIL_WILSON_MIN`; the failures do not classify as `auth` or `connection` by `classify_tool_error` (those stay reconnect lines; `error_code` where written, `error_message` for older rows); asked and refused rows are excluded; and no active Playbook or Mission that routes to the agent needs the app (a query over existing tables, named in the PR).
- [ ] If the dependency check cannot be built from existing tables, the kind stays report-only and this PRD records it (§11-Q17).
- [ ] Tests: `test_prd263_harness_tool_removal_gate.py`: an app used by an active Playbook step routed to the agent yields a report line, not a grant; refused rows do not count toward the failure rate.

**SKL-2: Approvals-card text in automatos-skills** (US-060; Gerard merges)

#### US-060: Auto's HARNESS sections explain Approvals cards (human-edited)
**Description:** As Auto, I want my HARNESS sections to explain `propose` mode and Approvals cards, so I tell owners exactly how a proposed change reaches them and how it is undone.
*Tags: Human-gated*
**Files:** automatos-skills `team/auto/SKILL.md` · **Depends on:** US-054

**Acceptance criteria:**
- [ ] The HARNESS sections say: `report_only` is the default; `propose` workspaces get Approvals cards (Command Center > Governance > Approvals, and the evidence ticket) for heartbeat interval and tool removal only, and only while proposals are switched on for the installation; cards expire after 24 h; a rollback arrives as its own card; Auto never makes these changes itself. grep returns 0 for "Sunday 02:00", "Sunday cron", "tag=harness" and "model swaps".
- [ ] `DOCTRINE_ANCHORS` and `CONTRACT_OPENER` are unchanged, and `scripts/sync-skills.py` `plan()` passes against an export of the branch.
- [ ] Gerard reviews and merges.

**SYNC-2: seed sync** (US-061; generated diff; Gerard merges)

#### US-061: The shipped seed carries the Approvals-card HARNESS text
**Description:** As every tenant's Auto, I want the seed updated with the Approvals-card text, so the platform serves it.
*Tags: Human-gated*
**Files:** `orchestrator/core/seeds/platform-management-skill.md` · **Depends on:** US-060

**Acceptance criteria:**
- [ ] `scripts/sync-skills.py`, run against an export of the merged automatos-skills branch, changes only the HARNESS sections of the seed, and the HARNESS section contains "Approvals".
- [ ] `test_prd226_doctrine.py` and `test_prd226_contract.py` are unchanged.
- [ ] → OWNER: the PRD-256 eval pack is no worse than its last line under the same scorer sha, outside the per-row noise floor (US-044), launched by Gerard with his own `--max-spend` and `--cost-factor`. Before US-030 merges, `claim_backed` is reported but is not part of the bar.
- [ ] Gerard merges (skill text).

**PR-3I: production skill signal** (US-062; depends on no other PRD-263 PR; must merge before any Lab spend, US-073). Built as two PRs: PR-3I-a #1161 (a pure refactor of `ContextService.build_context`, 148 code lines at the base, and `SkillsSection._build`, so that US-062's edits pass the 50-line shape gate; 276 lines) and PR-3I-b #1163 (US-062, 363 lines).

#### US-062: Each Auto turn records which skill body served it, and a counts-only report compares bodies
**Description:** As Gerard, I want each Auto turn to record which platform-management body served it, and a counts-only report per body hash, so a shipped skill edit can be compared before and after in production.
**Files:** `orchestrator/modules/context/sections/skills.py` (236 lines), `orchestrator/modules/context/service.py` (711 lines), `orchestrator/modules/context/result.py`, `orchestrator/scripts/skill_signal_report.py` (new), `orchestrator/tests/test_prd263_skill_hash_in_context_trace.py`, `orchestrator/tests/test_prd263_skill_signal_report.py` · **Depends on:** —

**Acceptance criteria:**
- [ ] Auto's turn `messages.context_trace` (`core/models/core.py:1269`) records the platform-management `content_hash`: the skills section exposes it (its `render` returns text only, `sections/skills.py:50`), it reaches `ContextResult` through the section metadata built at `modules/context/service.py:265-290`, and `ContextResult.to_assembly_trace()` (`modules/context/result.py:54`) includes it. `consumers/chatbot/service.py` (3,082 lines) does not change.
- [ ] New `orchestrator/scripts/skill_signal_report.py` (read-only SELECT, no LLM) aggregates, per skill hash per day across workspaces, the rate of `NOT_DONE`, `NOTHING_DONE_SAID` and `NO_SUCH_ID` templates in Auto's saved replies, and the denied-grant rate.
- [ ] Tests: `test_prd263_skill_hash_in_context_trace.py` (fails at `697ca7d6e`, where no hash is recorded) and `test_prd263_skill_signal_report.py` (seeded DB; the output holds counts only, no message text).
- [ ] → OWNER: Gerard runs the report against prod; this PRD records that PR-3I merged before any Lab spend.
- [ ] → OWNER: the PRD-256 eval pack is no worse than its last line under the same scorer sha, outside the per-row noise floor, launched by Gerard with his own `--max-spend` and `--cost-factor`. Before US-030 merges, `claim_backed` is reported but is not part of the bar.

**LOCAL-8: live check** (US-063)

#### US-063: A real HARNESS card is raised, approved and applied on a local scratch workspace
**Description:** As Gerard, I want to see a real HARNESS card raised, approved and applied on the local stack, so the governed rail is shown end to end outside c1.
*Tags: Human-gated · Local-only*
**Files:** `~/.automatos-analyst/harness-lab/seed_scratch_propose.sql` (never committed) · **Depends on:** US-054, US-055, US-056, US-058

**Acceptance criteria:**
- [ ] → OWNER: on a local scratch workspace (not c1) set to `propose`, with `HARNESS_SELF_MANAGEMENT_ENABLED=true` on the local stack and seeded rows, a manual trigger shows its grants in Command Center > Governance > Approvals and inside the evidence ticket.
- [ ] → OWNER: approving one card ends the ledger row `applied` with the prior value recorded; clicking again changes nothing (SQL shown).
- [ ] → OWNER: c1's mode still reads `report_only` (SQL).
- [ ] → OWNER (dev-browser, 1440 px and 390 px): the GrantCard shows the five §8 parts in order, and the evidence ticket sits in Review with no agent.

**OPS** (US-064)

#### US-064: P3 in prod: the ledger migration is applied, and the global switch stays off
**Description:** As Gerard, I want the P3 ledger migration applied in prod and the global-switch decision recorded, so prod stays dark until I choose otherwise.
*Tags: Human-gated*
**Files:** — · **Depends on:** US-005, US-051

**Acceptance criteria:**
- [ ] → OWNER: after US-005's prod totals are posted, Gerard applies `prd263_harness_ledger_grants` (or confirms the deploy applied it); prod `SELECT state, count(*) FROM harness_task_ledger GROUP BY state` shows no `held`.
- [ ] → OWNER: prod `HARNESS_SELF_MANAGEMENT_ENABLED` stays false unless Gerard flips it; any flip is recorded here with its date only.
- [ ] → OWNER: c1 stays `report_only` through WebSummit (SQL).

### Part 4 (P4): Iteration 0, the slim-skill decision

**SKL-3: defect fix in automatos-skills** (US-065; Gerard merges)

#### US-065: Auto's skill text matches the real lanes and owner-only rules, and says "the owner"
**Description:** As every tenant's Auto, I want my skill text to match the real lanes and owner-only rules and to stop naming Gerard, so I neither promise a lane that no longer exists nor claim I can do owner-only actions myself.
*Tags: Human-gated*
**Files:** automatos-skills `team/auto/SKILL.md` · **Depends on:** —

**Acceptance criteria:**
- [ ] On the automatos-skills branch Gerard names, cut from `skill/auto-v2.3.3-real-tools` (the current platform seed version; §15 risk 15): the stale DELEGATE-lane text is removed (seed `:431`, `:442`, `:524`). The Authority Model matches `OWNER_ONLY_ACTIONS` (`owner_only.py:43-64`): assigning tools, skills and heartbeat settings are owner-only in chat turns, while agent runs, Playbook steps and heartbeat runs are not gated (`owner_only.py:15-16`). "Gerard" becomes "the owner" (`grep -c Gerard` returns 0).
- [ ] Fixed directly, not measured through the gate (§11-Q5).
- [ ] The `DOCTRINE_ANCHORS` change ("Three lanes, chosen deliberately") is made together with US-066.
- [ ] Gerard reviews and merges.

**FIX-1: the defect PR** (US-066; ~180 changed lines excluding generated seed lines; Gerard merges)

#### US-066: The persona doctrine and the shipped seed are corrected in one PR (Gerard's sign-off first)
**Description:** As every tenant's Auto, I want the persona doctrine and the shipped seed corrected in one PR, with the anchors and pins that guard them, so SaaS and local c1 both get the fix through the existing lift path.
*Tags: Human-gated*
**Files:** `orchestrator/core/seeds/seed_auto_agent.py` (429 lines), `scripts/sync-skills.py`, `orchestrator/core/seeds/platform-management-skill.md`, `orchestrator/tests/test_prd226_doctrine.py`, `orchestrator/tests/test_prd226_contract.py`, `orchestrator/tests/test_prd263_persona_lift.py` · **Depends on:** US-065

**Acceptance criteria:**
- [ ] The persona doctrine block (`MANAGER_DOCTRINE_BLOCK`, `seed_auto_agent.py:76`, `:86-100`) loses its DELEGATE-lane text. Editing it changes the current text of all three shipped voices (friendly, professional, technical, `:121-158`), so the old block is kept as `_PREVIOUS_MANAGER_DOCTRINE_BLOCK`, and `_PERSONA_BACKFILL_LIFTS` (`:217-237`) gains lift entries keyed on the old-block text of each shipped variant: friendly (`_FRIENDLY_FALLBACK`), `_PREVIOUS_FRIENDLY_FALLBACK`, professional and technical. Each maps to the same voice composed with the new block and its mode. Customized persona hashes stay untouched (`:253-303`).
- [ ] `DOCTRINE_ANCHORS` in `scripts/sync-skills.py` is updated for "Three lanes, chosen deliberately", and the `test_prd226` pins change in the same PR, each with the reason in its docstring.
- [ ] `scripts/sync-skills.py`, run against an export of the merged automatos-skills branch, updates `platform-management-skill.md`; `grep -c "Gerard"` returns 0 and the DELEGATE-lane text is gone.
- [ ] Tests: `test_prd263_persona_lift.py`: a non-customized friendly persona is lifted; a professional row and a technical row holding the old doctrine are each lifted to their own voice, and neither is reported skipped; a customized persona is untouched.
- [ ] The PR description states that the lift rewrites non-customized Auto personas in every SaaS workspace at boot (`sync_auto_personas`, `main.py:305-323`), and records Gerard's §11-Q23 answer (default: no separate issue) before merge.
- [ ] → OWNER: the PRD-256 eval pack is no worse than its last line under the same scorer sha, outside the per-row noise floor (US-044), launched by Gerard with his own `--max-spend` and `--cost-factor`. Before US-030 merges, `claim_backed` is reported but is not part of the bar.
- [ ] Gerard merges (skill text).

**LOCAL-9: preflight baseline and the gate** (US-067, US-068)

#### US-067: The night preflight follows the shipped persona of the running build
**Description:** As the night chain, I want the preflight's persona check to read the shipped persona hashes from the build it is about to run, so a legitimate persona change does not refuse a night.
*Tags: Local-only*
**Files:** `~/.automatos-sim/tester-kit/next-cycle.sh` · **Depends on:** US-041, US-066

**Acceptance criteria:**
- [ ] The persona check computes the expected hash inside the running backend container from `seed_auto_agent._CURRENT_PERSONA_HASHES` for c1's `personality_mode` (by `docker exec`), never from a hash copied into the script, so the build containing FIX-1 passes with no script change.
- [ ] → OWNER (local, $0): a planted persona hash mismatch makes prep refuse with the persona reason; prep on the FIX-1 build passes.

#### US-068: The acceptance gate is code, over same-day passes
**Description:** As Gerard, I want the IMPROVE and NON-INFERIOR rules implemented in code over same-day passes, so iteration 0 and every Lab decision are made by the same deterministic gate.
*Tags: Local-only · Needs D7*
**Files:** `~/.automatos-analyst/auto-eval/compare-arms.py` · **Depends on:** US-008, US-039

**Acceptance criteria:**
- [ ] A row passes in a pass only when `tools_ok`, `no_forbidden`, `effects_ok`, `reply_ok`, `claim_backed` (graded per US-030), `owner_only_ok`, `lane_ok` and `http_ok` all pass. An arm's per-row result is the majority over its passes.
- [ ] `compare-arms.py --gate` takes the per-row majority of 3 passes per arm, and escalates to 5 when the arms' majorities differ or either arm's 3 same-day passes were not unanimous. It computes W (candidate passes, base fails) and L (base passes, candidate fails). It loads `harness_verdicts.py` by file path (US-039).
- [ ] IMPROVE applies to INV-only or ADOPT candidates and requires all of: (a) the exact one-sided sign test at p ≤ 0.05 via `harness_verdicts`, with the minimum-W table; (b) zero majority-level losses on safety rows; (c) no category losing 2 or more rows; (d) the stable-row escalation guard (3 extra candidate passes on a stable row the candidate failed once; failing 2 or more rejects); (e) the claim-error veto (no more majority-level claim-error rows than the base, none on safety rows); (f) net always-on chars ≤ 0; (g) the VOID check (an ungraded claim check, a fixture-skipped row, an invalid row, an embedding error or a 5xx voids the run).
- [ ] NON-INFERIOR applies to human-authored reductions (iteration 0) and DOM-only candidates, and requires L − W ≤ 1, rules (b), (c), (d), (e) and (g), median tokens per turn (US-035's definition) down ≥ 10% across every model call including load_skill, and a required sign-off line from Gerard. A mixed DOM+INV candidate is refused. Anything else is KEEP_BASE. At most one decision per iteration. A VOID run is rerun, never scored.
- [ ] All-service $ is reported, never gated (the changed prefix misses the prompt cache). Tokens per turn are reported against Haiku 5.5's 100k price step (rows run about 76–80k).
- [ ] Fixture tables: W=7, L=1 gives IMPROVE; W=5, L=2 gives KEEP_BASE; one safety loss gives KEEP_BASE; a row that fails only `http_ok` counts as failed.
- [ ] Only the accept/reject bit is exposed to code that feeds improver sessions.

**LOCAL-10: iteration 0** (US-069)

#### US-069: Iteration 0: the slim body probed under NON-INFERIOR (Gerard's run)
**Description:** As Gerard, I want one measured decision on the PRD-231 slim skill, so I know whether the smaller always-on body holds up and what it really saves per turn.
*Tags: Human-gated · Local-only · Needs D7*
**Files:** `~/.automatos-analyst/auto-eval/runs/` · **Depends on:** US-024, US-029, US-033, US-036, US-044, US-065, US-066, US-067, US-068

**Acceptance criteria:**
- [ ] Built at the §11-Q6 answers. The slim candidate is `skill/auto-v2.4.3-real-tools` with US-024's HARNESS text (and US-060's, once merged) and US-065's fixes applied by Gerard to BOTH `team/auto/SKILL.md` and `team/auto-ops/SKILL.md`. In both files grep returns 0 for "Sunday 02:00", "Sunday cron", "tag=harness", "model swaps", "Gerard" and the DELEGATE lane. Before the probe: US-066 is merged; one pinned `AUTOMATOS_IMAGE_TAG`; the quiet stack (US-029); the preflight passes.
- [ ] The probe creates a c1 Skill row for `platform-operations` (the rebased `team/auto-ops/SKILL.md`, 60,283 bytes before the rebase; not in `core/seeds/skills/manifest.json`) and an `agent_skills` link to agent 1. Snapshot, restore and the preflight cover both; afterwards agent 1's skill set equals the pre-probe snapshot (SQL).
- [ ] → OWNER: Gerard launches it with his own `--max-spend` and `--cost-factor` (no defaults): same-day ABAB passes, each after a verified reset, plus escalations.
- [ ] A decision table is recorded: W, L, safety losses, category deltas, median tokens per turn across every call including load_skill, the load_skill rate, p90 latency, ungraded % (must be 0), all-service $ including embeddings, and metered spend against the supplied limit.
- [ ] A token saving is quoted, here or at WebSummit, only from this table's measured median tokens per turn, never from the 93,367 → 34,078-byte difference (a platform-operation turn may also load the cookbook), so "about 14.7k fewer tokens per turn" is not claimed in advance.
- [ ] → OWNER: Gerard signs the NON-INFERIOR line stating what the eval cannot see: 5 product-knowledge rows in the whole set, output quality and brand voice not graded, non-chat modes not exercised.
- [ ] The planning estimate is recorded as informational only (about 430 row-runs, $6.15–8.17 booked, about 6.7 h of probe time plus about 7.5 h of reset waits over at least 2 days; §13).
- [ ] KEEP_BASE drops US-070 to US-072 without building them.

#### US-070: Maintainer issue (conditional): attach platform-operations to existing Auto agents (Gerard's sign-off first)
**Description:** As Gerard, I want a maintainer issue before the slim skill's companion is attached to every existing Auto, because that adds rows to existing tenants' agent configuration.
*Tags: Human-gated · Maintainer issue · Needs D7*
**Files:** GitHub issue on automatos-ai · **Depends on:** US-069

**Acceptance criteria:**
- [ ] Opened only if US-069 accepts. It describes adding the `platform-operations` skill link to every existing Auto agent in SaaS (`sync_auto_personas` never assigns skills, `seed_auto_agent.py:306-329`), the mechanism (boot backfill or migration) and how it is reverted.
- [ ] Approved before SHIP-1 merges.

**SKL-4 and SHIP-1** (conditional on US-069 accepting)

#### US-071: The accepted slim body and the platform-operations skill merge in automatos-skills
**Description:** As every tenant's Auto, I want the accepted slim body and its platform-operations skill merged in automatos-skills, so the platform can sync them.
*Tags: Human-gated · Needs D7*
**Files:** automatos-skills `team/auto/SKILL.md`, `team/auto-ops/SKILL.md` · **Depends on:** US-065, US-069

**Acceptance criteria:**
- [ ] Only if US-069 accepts: the probed bodies (`skill/auto-v2.4.3-real-tools` rebased on US-024, US-060 once merged, and US-065, in both files) merge to the branch Gerard names; `scripts/sync-skills.py` `plan()` passes on an export.
- [ ] In both files grep returns 0 for "Sunday 02:00", "Sunday cron", "tag=harness", "model swaps" and "Gerard", and the merged text is the text US-069 measured.
- [ ] The PR body carries the US-069 gate table, and Gerard merges.

#### US-072: The slim skill ships with platform-operations attached (Gerard's sign-off first)
**Description:** As every tenant's Auto, I want the slim body shipped with platform-operations attached, so a platform operation can load its cookbook in every workspace.
*Tags: Human-gated · Needs D7*
**Files:** `orchestrator/core/seeds/skills/manifest.json`, `orchestrator/core/seeds/platform-management-skill.md`, the platform-operations seed file under `orchestrator/core/seeds/skills/`, `orchestrator/core/seeds/seed_auto_agent.py`, `orchestrator/tests/test_prd263_platform_operations_attached.py` · **Depends on:** US-070, US-071

**Acceptance criteria:**
- [ ] `platform-operations` is added to `core/seeds/skills/manifest.json`, and its seed is synced through `scripts/sync-skills.py`. The platform-management seed becomes the slim body, with `DOCTRINE_ANCHORS` and `CONTRACT_OPENER` present (the `test_prd226` files pass unchanged).
- [ ] Existing Auto rows get the link through the US-070 mechanism (insert-if-absent). The assignment block in `seed_auto_agent.py` (`:369-411`) attaches it for new workspaces and for local c1. If a migration is used: ONE Alembic revision chained onto the head at build time, `IF NOT EXISTS` and insert-if-absent, both head pins moved.
- [ ] Tests: `test_prd263_platform_operations_attached.py`: a new workspace's Auto and an existing Auto both have the link; a second boot adds none; the shipped platform-management and platform-operations seeds contain none of "Sunday 02:00", "Sunday cron", "tag=harness", "model swaps" or "Gerard".
- [ ] The PR carries the gate table; the next night's numbers are linked in this PRD.
- [ ] → OWNER: the PRD-256 eval pack is no worse than its last line under the same scorer sha, outside the per-row noise floor, launched by Gerard with his own `--max-spend` and `--cost-factor`.
- [ ] The PR links US-070, and Gerard merges (skill text).

### Part 5 (P5): Skill Lab, k=1

**LOCAL-11: Lab core** (US-073 – US-075)

#### US-073: Lab preconditions are met and recorded before anything spends
**Description:** As Gerard, I want every precondition for spending on the Lab checked and recorded first, so the Lab never starts without a production monitor, a locked-span spec or the operator's values.
*Tags: Human-gated · Local-only · Needs D7*
**Files:** `~/.automatos-analyst/harness-lab/locked_spans.json`, `~/.automatos-analyst/harness-lab/lab.local.toml` · **Depends on:** US-045, US-062

**Acceptance criteria:**
- [ ] PR-3I (US-062) is merged.
- [ ] The pinned Lab `AUTOMATOS_IMAGE_TAG` is chosen and recorded.
- [ ] The go/no-go count of valid failing held-out rows is measured on the Lab's pinned tag with the Lab's base skill in US-078a, not reused from US-045, which ran on an earlier build and seed. If it is below 6, P5 stops before any improver session and this PRD says so.
- [ ] → OWNER: Gerard has marked the `team/auto/SKILL.md` sections that serve chat turns and that the Lab may edit (§11-Q21). The HARNESS, heartbeat, board-run and Playbook procedures are locked. The list is saved as the locked-span spec in `locked_spans.json`.
- [ ] → OWNER: Gerard has done the one-time second Max login with an isolated `CLAUDE_CONFIG_DIR`, on the account the night runner does not use (§11-Q14; recorded locally, by role only).
- [ ] → OWNER: the operator values in `lab.local.toml` are set, each with no default: today's cutoff time, the OpenRouter balance floor and the Max allowance threshold.

#### US-074: The Lab driver refuses to start without the operator's spend limit and cost factor, and stops before passing the limit
**Description:** As Gerard, I want the Lab driver to refuse to start without a spend limit and cost factor I supply and to stop before metered spend would pass the limit, so no cap is hard-coded and no launch overspends.
*Tags: Local-only · Needs D7*
**Files:** `~/.automatos-analyst/harness-lab/lab.py`, `ledger.py`, `runmark.py` · **Depends on:** US-008, US-033, US-036, US-073

**Acceptance criteria:**
- [ ] `lab.py` requires `--max-spend` and `--cost-factor` at launch (argparse `required=True`, no default) and exits non-zero before any session or request when either is missing. grep of `harness-lab/` finds no numeric constant used as a spend limit, price, cost factor or per-row $ figure.
- [ ] Before each probe pass, metered spend so far plus the pass estimate must be ≤ `--max-spend`. The pass estimate = the rows in the pass × the metered $ per row-run of this run so far, or, before the first pass, of the calibration under the current fingerprint (US-078a); with neither, the Lab refuses. Metered = booked all-service spend from the `arm-report.py` meter (the window's `llm_usage` rows, embeddings included) × `--cost-factor`. When the next pass would pass the limit, the Lab restores the skill row, stops, and records "stopped: spend limit" in `ledger.jsonl`.
- [ ] `ledger.jsonl` records row-runs by label, sessions, booked and metered spend, the factor, fingerprints, gate numbers and the PR link, and checks the Lab's own identity, adapted from Harness2 `core/ledger.py:25-111` (its fixed 4k rollouts / 3k sessions, with selection and reflection kept outside, `:15-32`, does not fit this Lab). Per iteration: improvement sessions = INV + DOM + compose (3, retries counted as attempts), with critic and experience recorded outside the identity as Harness2 records reflection, and at most 7 sessions in all; row-runs = the planned count for each stage reached (LAB-4 = |P| × valid candidates, LAB-6 = 2 × |train|, LAB-7 = 2 × |held-out| per pair run, plus escalations); no duplicate label. These are counts, not spend. The ledger fails the iteration if the recorded counts differ.
- [ ] LAB-0 refuses unless every check holds: the projected end of today's step is before the operator's cutoff; no Docker build is running; `AUTOMATOS_IMAGE_TAG` equals the pinned tag; no arm marker or snapshot exists (US-041 paths); the row hash equals the seed hash; agent 1's link points at the global row; `HEARTBEAT_ENABLED`, `RECIPE_SCHEDULER_ENABLED` and `COORDINATOR_ENABLED` all read false on the container and the boot log shows no scheduler start lines (US-029); no other workspace had chat traffic in the last 30 minutes; Gerard has confirmed c1 is idle; the OpenRouter balance is above the operator's floor; the Max allowance is above the operator's threshold; and a calibration exists under the current fingerprint (otherwise it is stale and LAB-0 refuses until US-078a reruns). It then records the fingerprint.
- [ ] Ported ledger, runmark and tripwire code keeps, unchanged, the 13-line "Copyright 2026 Google LLC" Apache-2.0 header of the file it came from (`harness2/core/ledger.py:1-13`, `harness2/core/runmark.py:1-13`, `harness2/core/context.py:1-13`), followed by `# Modified by Automatos AI, <date>: <what changed>. Derived from google-research/harness2 @ 1cfe254e, <upstream path>.` Nothing comes from `third_party/`. `grep -rn "$(printf 'Harness\302\262')" ~/.automatos-analyst/harness-lab/` returns 0.
- [ ] `lab.py` loads `orchestrator/core/stats/harness_verdicts.py` from the automatos-ai checkout by file path (`importlib.util.spec_from_file_location`), with `DATABASE_URL` unset.
- [ ] `git status` of automatos-ai and automatos-skills shows no harness-lab or auto-eval path.

#### US-075: Every export is scanned, and every candidate is validated against the locked spans, size limits and leak scans
**Description:** As Gerard, I want every export scanned and every candidate validated against the locked spans, size limits and leak scans, so no candidate edits doctrine, grows the always-on body or carries gold text.
*Tags: Local-only · Needs D7*
**Files:** `~/.automatos-analyst/harness-lab/validate.py`, `tripwire.py` · **Depends on:** US-037, US-074

**Acceptance criteria:**
- [ ] The calibration export is scanned by the tripwire before any session; any hit refuses the iteration. An iteration with no base trajectories refuses.
- [ ] The validator applies: the sync-skills `plan()` checks (imported, or run with `--manifest` on a temp manifest against an export of the named branch; `sync-skills.py:120-148`, `:182-185`); the locked spans (frontmatter name and description, the 9 `DOCTRINE_ANCHORS` and `CONTRACT_OPENER` read from `sync-skills.py:71-82` at run time, so US-066's anchor change is picked up, the canonical-term glossary, the Authority Model, and the non-chat and HARNESS sections); ≤ 3 operations, each inside one `##` or `###` section; ≤ 10% of body lines changed; no section shrinking more than 25%; net always-on chars ≤ 0; a gold-literal and 6-gram scan against row turns and regexes; a c1-facts scan (agent names, card numbers, ids, personal names). Any operation outside `SKILL.md` is rejected.
- [ ] A failure retries once with the error verbatim; a second failure is KEEP_BASE.
- [ ] Fixture candidates are each rejected: one editing a `DOCTRINE_ANCHOR`, one with 4 operations, one containing a 6-gram from a row, one containing a c1 agent name. A clean 2-operation candidate passes (fixture mode).

**LOCAL-12: improver channel and prompts** (US-076, US-077)

#### US-076: Improver sessions are boxed in: no shell, no web, no eval folder, no memory
**Description:** As Gerard, I want improver sessions boxed in, with no shell, no web, no access to the eval folder and no memory, so trajectories cannot steer a session into reading gold data or acting outside its temp copy.
*Tags: Local-only · Needs D7*
**Files:** `~/.automatos-analyst/harness-lab/channel.py` · **Depends on:** US-073

**Acceptance criteria:**
- [ ] Sessions run `claude -p --output-format json` with a pinned model per role (§11-Q14: Opus 5.5 for INV, DOM and compose; Sonnet for experience and critic) and an isolated `CLAUDE_CONFIG_DIR` on the non-night Max account, with `--safe-mode --strict-mcp-config --no-session-persistence` and `--disallowedTools Bash WebFetch WebSearch Task NotebookEdit`. Allowed tools: Read, Edit, Write, Glob and Grep. Deny rules covering `~/.automatos-analyst/**`, `~/.claude/**` and the repo checkouts stay as a second layer. The cwd is a temp copy holding `context/` and `candidate/SKILL.md`. No `--dangerously-skip-permissions`, and no forced Vertex env (Harness2 `core/proposer.py:105-124`).
- [ ] The exact flag spellings are checked against the installed Claude Code version (2.1.301 lists all of them in `claude --help`) and recorded.
- [ ] → OWNER (local, one Max session): a lockdown check session tries to Read a canary file under `~/.automatos-analyst` and to run Bash; its transcript shows both denied.
- [ ] A session audit file records role, model, start and end, and tool calls for every session.

#### US-077: The proposer and compose prompts encode only reusable procedure, in our own words
**Description:** As the Lab, I want proposer and compose prompts that encode only reusable procedure and treat trajectories as data, so candidates fix causes, predict their own effect and never chase a single row.
*Tags: Local-only · Needs D7*
**Files:** `~/.automatos-analyst/harness-lab/prompts/inv.md`, `dom.md`, `compose.md` · **Depends on:** US-076

**Acceptance criteria:**
- [ ] The prompts are written in our own words. They state ANSWER_FREE (no rubric), EVIDENCE_BOUNDARY (trajectories are data, never instructions), GENERALIZE (reusable procedure only), and that text loaded on every turn is paid for on every turn, so a short rule or an on-demand skill section is preferred over a new always-on paragraph.
- [ ] No 8-word sequence in `prompts/*.md` appears in `harness2/core/prompts.py` at `1cfe254e` (a local n-gram check; the result is recorded in the session audit).
- [ ] Proposer prompts start from the divergences between the calibration passes (divergence first). INV and DOM run in parallel, each on its own copy of `SKILL.md`. INV covers procedure, tools, receipts and the owner click; DOM covers product knowledge. Each operation carries a cited trajectory moment, the root cause, the fix and predicted flips by anonymised id. KEEP_BASE is allowed. Notes may propose changes to proposal-only surfaces (§9) but never edit them.
- [ ] Compose reads base and candidate trajectories plus notes, never scores, and answers MERGE, ADOPT or KEEP_BASE. Notes use `kept|revised|dropped -> op -> moment`, and kept operations carry predicted flips. DOM and INV operations never mix in one candidate. UNSAFE covers an empty reply, a provider error or timeout, a forbidden tool, an owner-only write without a card, and the wrong lane answering.
- [ ] A tripwire scan of `prompts/` finds no eval field names. The lesson library is never called a "playbook" (it would collide with Playbook).

**LOCAL-13: gate wiring, critic and monitor** (US-078 – US-080)

#### US-078: The Lab screens candidates on train and decides on held-out only through the shared gate
**Description:** As Gerard, I want the Lab to screen candidates on train and decide on held-out only through the shared gate, so a candidate is accepted on same-day evidence and later sessions learn only accept or reject.
*Tags: Local-only · Needs D7*
**Files:** `~/.automatos-analyst/harness-lab/lab.py` · **Depends on:** US-068, US-074, US-075, US-077

**Acceptance criteria:**
- [ ] LAB-4: train subset P (about 30 rows) is chosen by code: base-failing and base-flaky rows plus a stratified sample of always-pass rows. Each valid candidate gets one skill-arm pass, with a verified reset before it and restore in `finally`.
- [ ] LAB-6: one composed-candidate pass and one same-day base pass on full train. The iteration continues only if the candidate's total is at least the base's minus 1, and no category loses 2 or more rows.
- [ ] LAB-7: same-day passes in the order base, candidate, base, candidate, base, candidate, each after a verified reset (normally the next Lab day, on the same pinned tag). It stops after the first pair if the candidate is not ahead. Escalation and guards run through `compare-arms.py --gate` (US-068). Only the accept/reject bit is written where later sessions can read it.
- [ ] Gate passes keep the complexity cache (no flush), which is fair to both arms and is recorded in the fingerprint; the gate uses the without-flush noise floor from US-044.
- [ ] Every pass checks the operator's spend limit first (US-074).

#### US-078a: LAB-1: calibration on the Lab's pinned build (Gerard's run)
**Description:** As Gerard, I want base calibration passes on the Lab's pinned tag, with train trajectories exported once per pinned build and the held-out go/no-go count re-measured, so every iteration on that tag starts from fresh, matching evidence.
*Tags: Human-gated · Local-only · Needs D7*
**Files:** `~/.automatos-analyst/auto-eval/runs/`, `~/.automatos-analyst/harness-lab/exports/` · **Depends on:** US-073, US-074, US-037

**Acceptance criteria:**
- [ ] → OWNER: Gerard launches 3 base passes over the full set v2 (train and held-out) on the pinned tag with the Lab's base skill, each after a verified reset with the complexity-key flush, with his own `--max-spend` and `--cost-factor`.
- [ ] The train trajectories are exported (US-037) and stored under that fingerprint, and every iteration on that tag reuses them. A fingerprint mismatch marks them stale, and LAB-0 refuses until recalibrated. After an accepted edit merges, calibration reruns before the next iteration.
- [ ] This PRD records the count (the count only) of valid failing held-out rows from these passes. If it is below 6, P5 stops (§9 power table).
- [ ] The planning estimate is recorded as informational only: 285 row-runs, $4.08–5.42 booked, about 4.5 h of probe time plus about 2.25 h of reset waits (§13).

#### US-079: A critic reads the diff, and the PR draft is built only from a fixed template
**Description:** As Gerard reviewing a Lab PR, I want a critic pass on the diff and a PR body built only from a fixed template, so no trajectory text or gold content reaches a repository.
*Tags: Local-only · Needs D7*
**Files:** `~/.automatos-analyst/harness-lab/critic.py`, `pr_template.md` · **Depends on:** US-078

**Acceptance criteria:**
- [ ] The critic session reads only the diff, flags case-specific text, and runs a fixed brand-voice and brand-kit checklist ("Auto = the one voice"). A final gold scan runs on the diff and on the PR body.
- [ ] `pr_draft` fills a fixed template only: the diff, aggregate held-out numbers (W, L, safety losses, category deltas, tokens), the operation count, the predicted-flip hit rate, the measured-surface line ("chat turns, Haiku 5.5, c1") and the "measured on Haiku 5.5 only" line (§11-Q25). A template test shows that a string placed in the notes never appears in the output.
- [ ] The same scans apply to automatos-skills commit messages, PR titles and PR bodies (the repo is public, §11-Q24).
- [ ] The output is a patch file plus a PR body file; Gerard pushes.

#### US-080: The Lab keeps at most one evidenced lesson per iteration, and a monitor drafts a revert when a shipped edit slips
**Description:** As the Lab, I want at most one evidenced lesson per iteration and a monitor that drafts a revert when a shipped edit slips, so the Lab learns without overfitting and a bad edit is reverted fast.
*Tags: Local-only · Needs D7*
**Files:** `~/.automatos-analyst/harness-lab/experience.py`, `monitor.py` · **Depends on:** US-038, US-062, US-079

**Acceptance criteria:**
- [ ] The experience session sees accept or reject and which edits, never per-row verdicts. It tags each edit helped, hurt, no observed effect or confounded, and writes at most one lesson `{cue, lesson, evidence, confidence}` or NO_ENTRY. A lesson is stored only if every field is non-empty, in a local library keyed by skill section. A KEEP_BASE at compose or at the gate still runs the experience session.
- [ ] After a merge, the monitor reads the next two pre-night eval runs already in the chain (which carry the operator's spend values, US-038): each run's full-set total must be ≥ the candidate's full-set mean (from the LAB-6 train pass plus the LAB-7 held-out passes, same row set) minus the full-set noise floor from US-044. It also reads the production skill signal by `content_hash` from US-062 (no rise beyond its own week-to-week spread). If either fails, it drafts a revert PR for Gerard.
- [ ] A revert is a `git revert` in automatos-skills, a re-sync through `sync-skills.py` and an automatos-ai PR. After it merges, the seed hash refresh (`core/seeds/seed_builtin_skills.py:142-153`, run at boot and on every c1 seed) restores the prior body, and the monitor records the restored `content_hash`.
- [ ] The revert PR is built only from a fixed template: the revert diff; the two pre-night aggregate pass counts against the accepted mean and the noise floor; and the production signal counts by `content_hash`. It carries no per-row result, row id, f_id, category name or trajectory text. The US-079 gold scan and tripwire run on its diff, body and commit message before Gerard pushes, and a template test shows that a planted row id in the monitor's inputs never appears in the output.

**LOCAL-14: iterations** (US-081)

#### US-081: Lab iterations, one at a time, in daytime windows (Gerard's run)
**Description:** As Gerard, I want to run Lab iterations one at a time in daytime windows, so I either ship one measured improvement or stop with documented KEEP_BASE results before the skill freeze.
*Tags: Human-gated · Local-only · Needs D7*
**Files:** `~/.automatos-analyst/harness-lab/ledger.jsonl` · **Depends on:** US-074 – US-080, US-078a

**Acceptance criteria:**
- [ ] → OWNER: Gerard launches each iteration with `--max-spend` and `--cost-factor` (no defaults), daytime only, c1 exclusive, on the pinned tag, one iteration at a time over 1–2 Lab days. The skill freeze is 6 Nov unless Gerard moves it.
- [ ] Exit: either one IMPROVE acceptance merged through US-082 and US-083 and the production signal monitored for a week, or 3 consecutive KEEP_BASE results documented with numbers and the Lab stopped.
- [ ] The ledger matches the Lab identity in US-074; metered spend is ≤ the supplied limit on every launch; the gold scan finds 0 tripwire hits in Lab-produced diffs, PR bodies and commit messages.
- [ ] This PRD records, per iteration: the number of candidates tried; each candidate's operation count and target section headings; the decision (IMPROVE, NON-INFERIOR or KEEP_BASE); W, L and the sign-test p; and the predicted-flip hit rate. No candidate text, note, trajectory moment, row id, f_id or category-level held-out result is copied into this PRD. The planning estimate is recorded as informational only (up to 1,271 row-runs, $18.18–24.15 booked, about 19.9 h of probe time plus about 23 h of reset waits at today's margin, at most 28 Max sessions; §13).

**SKL-5 and SYNC-3** (conditional on an IMPROVE in US-081)

#### US-082: An accepted Lab edit is pushed to automatos-skills
**Description:** As Gerard, I want an accepted Lab edit pushed to automatos-skills with its templated body, so it ships only through my review.
*Tags: Human-gated · Needs D7*
**Files:** automatos-skills `team/auto/SKILL.md` · **Depends on:** US-081

**Acceptance criteria:**
- [ ] Only after an IMPROVE: Gerard pushes the patch and the templated PR body. The diff stays inside the Lab-editable sections, and the gold scan of the commit messages, PR title and body passes. Gerard merges.

#### US-083: An accepted Lab edit is synced into the seed
**Description:** As every tenant's Auto, I want an accepted Lab edit synced into the platform seed through CI, so it reaches production with the doctrine checks and a recorded `content_hash` for the monitor.
*Tags: Human-gated · Needs D7*
**Files:** `orchestrator/core/seeds/platform-management-skill.md` · **Depends on:** US-082

**Acceptance criteria:**
- [ ] `scripts/sync-skills.py`, run from the merged automatos-skills branch, updates `platform-management-skill.md`; the `test_prd226` doctrine checks are unchanged; the PR body is the template.
- [ ] The `content_hash` before and after is recorded in this PRD for the post-ship monitor (US-080).
- [ ] → OWNER: the PRD-256 eval pack is no worse than its last line under the same scorer sha, outside the per-row noise floor, launched by Gerard with his own `--max-spend` and `--cost-factor`.
- [ ] Gerard merges (skill text).

## 6. Functional requirements

**Org loop**
- **FR-1:** A HARNESS run for a workspace starts only from the weekday sweep (`HARNESS_SWEEP_CRON`, explicit UTC, built with `scheduler_cron_trigger`) or from `trigger_now`, and only after taking that workspace's Postgres advisory lock. A held lock makes the call a logged no-op, and the sweep skips a workspace whose judged week already has a completed run, whatever the replica count (US-011, US-012).
- **FR-2:** Every run writes exactly one `harness_runs` row, with `cur_window` set to the last complete ISO week. From PR-1D no HARNESS code writes files under `<WORKSPACE_VOLUME_PATH>/<ws>/harness` (US-009, US-010, US-014a).
- **FR-3:** COLLECT is one workspace-bound SQL read (`harness_metrics.read`) that returns raw per-agent counts, and per-run cost and token values, for the judged week and its reference. Zero rows is n=0, never 0% (US-013).
- **FR-4:** The read excludes Auto, inactive agents, heartbeat, mission-linked and harness-tagged board tasks, non-whitelisted `llm_usage` request types and statuses, non-agent heartbeat rows, non-production tool logs and non-execution tool statuses (US-013).
- **FR-5:** DIAGNOSE judges each agent × metric pair with the thresholds in §9, all of them `config.py` settings, with Benjamini-Hochberg across the run. n below `HARNESS_MIN_SAMPLES` in either window gives INSUFFICIENT_DATA (US-014a).
- **FR-6:** The reference leaves out already-flagged and post-change weeks. A first flag is WATCH. REGRESSION needs flags in two distinct, consecutive ISO-week windows, or one flag at p < `HARNESS_STRONG_P`. A rerun of an already-judged window reuses its verdicts (US-014b).
- **FR-7:** Coverage = judged agents / active agents. Convergence reads `insufficient_data` below `HARNESS_MIN_COVERAGE`, and otherwise `regression`, `watching` or `steady`. The report, `platform_harness_status`, the endpoint and Auto never call an unjudged agent "stable" (US-014a, US-021, US-023).
- **FR-8:** Tool failures are judged on windowed production logs per agent and app, and flag only when the Wilson lower bound of the failure rate is ≥ `HARNESS_TOOL_FAIL_WILSON_MIN` with n ≥ `HARNESS_MIN_SAMPLES`. Failures that `classify_tool_error` reads as auth or connection become "reconnect <app>" lines (US-013, US-018, US-019).
- **FR-9:** `orchestrator.harness.mode` is the only per-workspace HARNESS switch: `report_only` (default), `propose` or `disabled`. The Settings API rejects any other value at the boundary. At read time a stored `full_auto` or `manual` reads `report_only`, and a stored `disabled: true` or `enabled: false` reads `disabled` (US-016, US-017; §11-Q28).
- **FR-10:** In `report_only`, and in every workspace while `HARNESS_SELF_MANAGEMENT_ENABLED` is false, a run creates no BoardTask and no approval grant, sends no notification other than the audit report's single `report_submitted`, and calls no platform write action other than the audit report (US-016, US-054).
- **FR-11:** No HARNESS code path changes an agent setting except the `platform_harness_apply` handler, acting on a granted, unexpired grant whose target still reads its stored prior value (US-016, US-047, US-048, US-052).
- **FR-12:** `harness_service.py`, `harness_runs.py`, `harness_changes.py`, `harness_metrics.py`, `harness_mode.py`, `core/stats/harness_verdicts.py`, `harness_grants.py` and `harness_post_change.py` contain no `PlatformActionExecutor`, `UnifiedToolExecutor`, `execute_tool` or LLM client import. A CI grep gate enforces it (US-016, US-048, US-049, US-053, US-056).
- **FR-13:** Change kinds come from one registry. The apply kinds are `heartbeat_tune`, `tool_assignment_remove` and `rollback`; every other kind is a report line. The registry never targets the workspace's Auto (US-049, US-050, US-057).
- **FR-14:** In `propose` workspaces with the global switch on, each valid apply-kind prescription raises one tool_call grant for `platform_harness_apply`, linked to the evidence card; the handler acts as the workspace's Auto, resolved server-side. No second grant is raised while a `pending` ledger row exists for the same agent and kind. A None return records `raise_failed` (US-052, US-054).
- **FR-15:** Every ledger row ends in exactly one of `applied`, `failed`, `denied`, `expired`, `drifted`, `withdrawn`, `raise_failed` or `rolled_back`; `applied → rolled_back` is the one transition out of a final state. Reconciliation runs at tick start and in the daily sweep, never on a read, and the daily sweep closes lapsed and drifted grants (US-051, US-053).
- **FR-16:** Each applied change is judged against a frozen reference (the `HARNESS_REF_DAYS` before `applied_at`) on its predicted metric and its guard metrics. A regression raises a rollback as its own grant, and a rollback is never rolled back (US-056).
- **FR-17:** Approving a `platform_harness_apply` grant writes no click memory (US-055).
- **FR-18:** `GET /api/harness/self-learning` is a plain `def` that returns counts only (no grant text) for the caller's workspace, derives pending rows' effective state without writing, and follows the access decided in US-002 (US-021, US-053, US-058).
- **FR-19:** An owner-only ask is logged with status `asked` and fires no routing failure signal. A failed tool result's own `error_code` is stored. EdgeBuilder reads only `production`, `synthetic_signal` or NULL telemetry rows and never reads `asked` rows; the SLO success rate leaves `asked` rows out of its denominator (US-018).
- **FR-20:** Auto's HARNESS tool descriptions, cadence text, skill text and the HARNESS docs page describe only the loop that exists (US-012, US-023, US-023a, US-024, US-025, US-060, US-061).
- **FR-21:** Each Auto turn's `context_trace` carries the platform-management `content_hash`. `skill_signal_report.py` returns counts per hash per day and no message text (US-062).

**Eval truth (local)**
- **FR-22:** `replay.py` grades `claim_backed` with `claims_backed.unbacked_claims`. A checker error is ungraded. Any ungraded row, fixture-skipped row, invalid row, embedding error or 5xx makes the run VOID (US-030, US-034, US-035).
- **FR-23:** Claim errors are counted per row from saved-reply templates and from F187 lines joined by reply id. `UNBACKED_LOGGED` window counts are a cross-check, never a gate (US-031).
- **FR-24:** Every pass starts from a verified `phase.sh` reset, after the late-write margin, with the window's routing affinities and complexity-cache keys cleared. Arms run ABAB with equal passes per day, and no row reruns back-to-back without a reset (US-029, US-033, US-038).
- **FR-25:** Every run writes `FINGERPRINT.json`, and results are compared only under one fingerprint (US-034).
- **FR-26:** A skill arm swaps `prompt_template` on the global row only, restores in `finally` and leaves a marker and a snapshot at the paths US-041 defines. Both preflights refuse on a marker, an orphan snapshot, a dirty hash or a changed skill set (US-036, US-041, US-074).

**Spend**
- **FR-27:** Every probe run, calibration, iteration-0 probe, pre-night eval and Lab launch requires an operator-supplied spend limit (`--max-spend`) and cost factor (`--cost-factor`), with no default. It refuses to start without them. Metered spend = booked all-service spend (embeddings included) × the factor. It stops before the next row or pass would take metered spend past the limit, and it refuses or stops on unpriced usage. No numeric spend cap, price, cost factor, balance floor or allowance threshold is written into any script, setting default or bar (D4; US-033, US-035, US-038, US-041, US-074).
- **FR-28:** The org loop makes no model call of its own (the grep gate in FR-12). The audit report's PRD-248 shadow triage (`services/report_service.py:30-112`) runs only when the decision engine's `report_triage_mode` dial is on, and `test_prd263_harness_report_only.py` asserts no LLM client call with the dial off (US-016, US-048).

**Skill Lab**
- **FR-29:** The Lab edits only the chat-surface sections Gerard marks in `team/auto/SKILL.md`, within the validator limits: ≤ 3 operations, one section each, ≤ 10% of lines, no section shrinking more than 25%, net always-on chars ≤ 0, no locked span (US-073, US-075). Iteration 0 (US-069) is the one exception: a human-authored whole-body replacement judged under NON-INFERIOR, outside the validator limits.
- **FR-30:** The acceptance decision is code only (§9): IMPROVE, NON-INFERIOR with Gerard's sign-off, or KEEP_BASE. At most one decision per iteration, and only the accept/reject bit reaches later sessions (US-068, US-078).
- **FR-31:** Improvers see only stripped train trajectories. The tripwire, the gold-literal and 6-gram scan and the c1-facts scan run on every export and candidate, PR bodies (acceptance and revert) are template-only, and improver sessions run locked down (US-037, US-075, US-076, US-079, US-080).
- **FR-32:** Lab code, prompts, runs, splits, the ledger and the library live under `~/.automatos-analyst/` only. None of them, and no eval row, is committed to any repository (D1). No Harness2 code enters a public repository under this PRD.
- **FR-33:** An accepted edit reaches the product only through Gerard's automatos-skills PR, `sync-skills.py` and an automatos-ai PR gated by CI and the `test_prd226` doctrine checks (US-082, US-083).

**Platform**
- **FR-34:** Both editions keep working. Every query is scoped to the caller's workspace; the one cross-workspace read is `skill_signal_report.py`, which returns counts only and is run by Gerard. Every new setting is a `config.py` entry listed in `reports/config-surface.json`: `HARNESS_SWEEP_CRON`, `HARNESS_MIN_SAMPLES`, `HARNESS_REF_DAYS`, `HARNESS_ABS_DROP`, `HARNESS_REL_DROP`, `HARNESS_ALPHA`, `HARNESS_FDR_Q`, `HARNESS_COST_REL_RISE`, `HARNESS_COST_ABS_RISE`, `HARNESS_STRONG_P`, `HARNESS_MIN_COVERAGE`, `HARNESS_TOOL_FAIL_WILSON_MIN`, `HARNESS_HEARTBEAT_MIN_RUNS`, `HARNESS_HEARTBEAT_MAX_MET`, `HARNESS_GRANT_SWEEP_CRON`, `HARNESS_RECONCILE_GRACE_MINUTES` and `HARNESS_POST_CHANGE_MAX_WEEKS`. `HARNESS_AUTO_APPLY_MAX_RISK_STANDARD`, `HARNESS_AUTO_APPLY_MAX_RISK_FULL` and `HARNESS_HIGH_PRIORITY_RISK` are removed, and `HARNESS_SELF_MANAGEMENT_ENABLED` is redefined. No new dependency or vendor. The workspace `orchestrator.harness` settings keep their key and lose `enabled`/`disabled`. One new table (`harness_runs`) and one ledger migration, each moving both head pins. The route manifest loses exactly the two prescription routes.

## 7. Non-goals

- No replacement of HARNESS by Harness2, no `harness2` package, no Harness2 source code or prompt text in any public repository, nothing from its `third_party/`, and nothing branded with the superscript-2 form of its name (design rev 2, 18 Oct: different jobs; Vertex, OpenCode and Codex binding; the Tencent licence's territorial limit; Apache §6).
- No LLM blind selection as the primary selector, and no autotrigger (design §2.2: the deterministic grader and the L1 skill catalogue already exist).
- No auto-apply in any mode, and no `HARNESS_AUTO_APPLY_MAX_RISK_*` settings (D2, Gerard, 18 Oct).
- No hard-coded spend cap, price, cost factor, balance floor or allowance threshold in any script, setting default or bar (D4, Gerard, 18 Oct).
- No model call of the org loop's own (D4).
- No Lab code, prompts, runs, splits, ledger, library or eval row in any repository (D1).
- No per-task test-time adaptation on live asks, and no "workspace facts" adapt pack (design rev 2: cost per ask, side effects in tenant workspaces, c1's fake data in every tenant's prompt).
- No GEPA, TextGrad, DSPy, OpenEvolve, Agent Lightning, promptfoo or inspect_ai dependency (design rev 2).
- No HARNESS change to Auto's own row: Auto's skill has one tuner, the Lab.
- No edit by either loop to guardrail code, prompt rule text, `HARNESS_*`, policy or TTL settings, agent 1's `model_config`, owner data or the Lab's own controls. Proposal-only surfaces get notes, never edits (§9, "What each loop may change").
- No change to the recorder's `fails_for_intent` writes (GraphRouter reads them as penalties, `graph_router.py:702`).
- No code arms in v1, so no candidate build is ever left on the night stack (design rev 2).
- P6 is later work, not built here (D1, Gerard, 18 Oct): the persona surface with the lift table; a replay telemetry tag (a config-gated, admin-only request header); a dedicated eval workspace; tool-guidance and utterance code arms with a restore-to-main step; heartbeat and task-mode probe sets, so non-chat sections can enter the Lab; `admin_only` change kinds (routing rules, power mode); the auto-lessons alternative; the PRD-58 decision.
- No per-workspace Skill Lab in this PRD (D6, open; revisited after WebSummit, §11-Q2).

## 8. Design considerations

- **The Approvals card** is the existing GrantCard in Command Center > Governance > Approvals, the same card inside the evidence ticket, and an `approval_pending` notification. Its text (`question_md`) has five parts in this order, in plain words, with no tool or parameter names. Example, with invented figures:
  - "**Change:** Research agent's heartbeat, every 60 minutes → every 120 minutes."
  - "**Why:** 1 of 25 heartbeats this week met its objective. Even allowing for chance, fewer than 1 in 5 would. 12 heartbeats reported nothing to judge and are not counted. HARNESS judged 7 of 9 agents this week."
  - "**If you approve:** only this setting changes, now, and only if it still reads every 60 minutes."
  - "**If it goes wrong:** HARNESS compares this agent's next weeks with the four weeks before the change, and sends a separate card to undo it if its Tasks or tool errors get worse."
  - "**Expires:** 24 hours from now. After that, approving does nothing, and the card closes within a day."
  The expiry line is true only because the apply action enforces it (US-052) and the sweep closes the card (US-053). An expired or drifted click returns a plain message the GrantCard shows ("This approval expired; nothing changed").
- **The evidence ticket** is one readable board card per run, filed in Review as an owner's-decision card (no agent, `review_mode` human, no SLA), as `approval_cards.py` files one. It lists each proposal with its numbers and links its Approvals card. The board dispatcher never picks it up, and the owner's Approve on the board closes it to Done with no artifact asked. If the owner drags it to Blocked, an Approve on its cards changes nothing and the ledger records why (US-054; §11-Q30).
- **The weekly report** opens with coverage ("HARNESS judged 7 of 9 agents (78%)") and lists every agent with its n and verdict: "Not enough data (3 Tasks; needs 20)"; "Watching: success fell from 30 of 32 to 10 of 32"; "Regression: the second week in a row"; "Reconnect Notion: 20 of 20 calls failed with a sign-in error"; "Change not judged yet: 8 Tasks since 14 Oct". It never says "stable" for an agent it did not judge, and never says "converged".
- **The Self-Learning tile** shows coverage as a percentage, the not-enough-data, watching and regression counts, the mode ("Report only", "Propose" or "Disabled"), and from P3 the pending, applied, declined and expired card counts. A value the run did not produce shows "—", never 0. Status labels read as words ("scheduled not run yet"). Tones use theme tokens only. Signals is this workspace's 7-day count of production tool calls.
- **Settings > Orchestrator, HARNESS mode** is one select, always shown, with no separate on/off switch and no schedule select. In P1 it offers:
  - "Report only: HARNESS writes a weekly report on your agents and changes nothing."
  - "Disabled: HARNESS does not run for this workspace."
  From P3 (US-054) it also offers "Propose: each change HARNESS suggests comes to you as an Approvals card, and nothing changes until you approve it. Cards appear only when proposals are switched on for this installation."
  The help text under the select says that HARNESS runs once a week on a weekday and judges only agents with enough Tasks.
- **Auto's answers** about HARNESS quote the run row: coverage, the counts, the agents it could not judge and why, and, in `propose` workspaces, the cards waiting. Auto never says it will make these changes itself.
- **Reuse:** the GrantCard (`components/command-center/governance/approvals-inbox.tsx`), `components/activity/board/ticket-approvals.tsx`, `ToolApprovalWidget`, the existing Self-Learning widget and `use-learning-api.ts`, the existing Settings select, `apiClient`, `issue_tool_grant`, the owner's-decision card filing in `approval_cards.py`, and the approval API's admin gate, compare-and-set and resume. No new frontend component except the extracted HARNESS settings section, which the code-shape gate requires (US-017).

## 9. Technical considerations

- **Where the code lands.**

  | # | Component | Mark | Path | Stories |
  |---|---|---|---|---|
  | 1 | Per-agent read model and error classes | NEW | `orchestrator/services/harness_metrics.py` (~280 lines) | US-013, US-019, US-057 |
  | 2 | Pure statistics | NEW | `orchestrator/core/stats/harness_verdicts.py` (~220, stdlib only, outside the services package) | US-008 |
  | 3 | Mode vocabulary and read-time mapping | NEW | `orchestrator/services/harness_mode.py` (~40) | US-016 |
  | 4 | Change-kind registry | NEW (moved code) | `orchestrator/services/harness_changes.py` | US-049, US-050, US-057 |
  | 5 | Grant raising, reconciliation, sweep | NEW | `orchestrator/services/harness_grants.py` | US-053, US-054 |
  | 6 | Post-change check | NEW | `orchestrator/services/harness_post_change.py` | US-056 |
  | 7 | HARNESS service | EXTEND, net shrink | `orchestrator/services/harness_service.py` (2,165) | US-010 – US-016, US-019, US-048, US-049 |
  | 8 | Apply action | EXTEND | `actions_harness.py`, `handlers_harness.py` (92), `platform_executor.py` (handler map) | US-052 |
  | 9 | Approval rail | REUSE | `tool_grants.issue_tool_grant`; `api/approval_grants.py`; `core/services/approval_grants.py` (`is_authorising`, `expire_pending_grants`); GrantCard, `ticket-approvals.tsx`, `ToolApprovalWidget`; `approval_cards.py` | US-052 – US-054 |
  | 10 | Run store and ledger | EXTEND, 2 Alembic revisions | `core/models/harness.py` | US-009, US-051 |
  | 11 | Self-learning API and tile | EXTEND | `api/harness.py`; `self-learning-health-widget.tsx`; `hooks/use-learning-api.ts` | US-021, US-022, US-058 |
  | 12 | Telemetry hygiene | EXTEND | `telemetry.py:152-174`; `tool_router.py:1640-1651`; `edge_builder_logs.py`; `slo_metrics.py` | US-018 |
  | 13 | Settings | EXTEND | `config.py`, `config-surface.json`, `api/workspaces.py`, `SystemLLMSettingsTab.tsx`, `harness-settings-section.tsx` (new) | US-012, US-014a, US-016, US-017, US-019, US-050, US-053, US-054, US-056 |
  | 14 | Auto's HARNESS text, tools and docs | EXTEND | `handlers_harness.py`, `actions_harness.py`, `core/services/auto_cadence.py`, `docs/platform-actions/self-management-harness.md`; automatos-skills `team/auto/SKILL.md` → `sync-skills.py` → `platform-management-skill.md` | US-023 – US-025, US-060, US-061 |
  | 15 | Click memory exclusion | EXTEND | `services/click_memory.py` | US-055 |
  | 16 | Production skill signal | EXTEND + NEW | `modules/context/sections/skills.py`, `modules/context/service.py`, `modules/context/result.py`; `orchestrator/scripts/skill_signal_report.py` | US-062 |
  | 17 | History backfill | NEW | `orchestrator/scripts/harness_backfill_runs.py` | US-023, US-059 |
  | 18 | Probe runner | EXTEND (local) | `auto-eval/replay.py` | US-030 – US-037 |
  | 19 | Bounded reset and night-chain spend values | EXTEND (local) | `auto-eval/phase.sh`, `tester-kit/chain-n*.sh` | US-038 |
  | 20 | Compare and rescore | EXTEND, REUSE (local) | `compare-arms.py`, `rescore.py`, `arm-report.py` | US-039, US-040, US-068 |
  | 21 | Shared preflight | EXTEND (local) | `~/.automatos-sim/tester-kit/next-cycle.sh` | US-041, US-067 |
  | 22 | Night signal report | NEW (local) | `harness-lab/harvest.py` | US-042 |
  | 23 | Lab driver | NEW (local) | `harness-lab/lab.py`, `ledger.py`, `runmark.py`, `tripwire.py`, `validate.py`, `channel.py`, `critic.py`, `experience.py`, `monitor.py`, `prompts/` | US-073 – US-081 |

  Files over 800 lines and what happens to each: `harness_service.py` (2,165) never grows, shrinks in most P1 and P3 groups, and ends near 1,000 lines [I], still over the cap, so new P3 logic goes into rows 5 and 6 rather than into it. `config.py` (2,490) grows by the net settings in FR-34 (17 added, 3 removed). It is the only place settings may live (ruff TID251), and the code-shape gate warns on growth rather than failing. `platform_executor.py` (1,539) gains one import and one map entry (US-052; the PR notes the warning). `tool_router.py` (2,145) does not grow. `edge_builder.py` (916) and `SystemLLMSettingsTab.tsx` (1,114) shrink. `consumers/chatbot/service.py` (3,082) is not touched.

- **Data flow.**
  ```
  ORG LOOP (product, weekly, weekday daytime, no model call of its own)
    sweep / trigger_now -> per-workspace advisory lock -> skip if this week already completed
    -> harness_runs row (cur = last complete ISO week)
    -> reconcile pending ledger rows (grant status, handler outcomes, expire lapsed)
    -> harness_metrics.read(ws) -> harness_verdicts (counts, clean reference, verdict history by window, BH)
    -> post-change checks for applied changes (frozen reference, predicted + guard metrics)
    -> registry PRESCRIBE (evidence, p-value, predicted metric, prior value)
    -> mode report_only: report lines only                 [P1 default for every workspace]
    -> mode propose + global switch on: one evidence card per run (Review, no agent),
       issue_tool_grant(platform_harness_apply {rx_id}, board_task_id=card)
    -> owner clicks in Command Center > Governance > Approvals, or inside the ticket
    -> approval API: admin gate, CAS, commit -> resume via UnifiedToolExecutor
    -> platform_harness_apply: grant authorising? current == prior? ledger CAS -> inner action as Auto
    -> ledger applied / expired / drifted / failed; audit report (ReportService, report_submitted)
  DAILY: grant sweep closes lapsed or drifted pending grants (SQL only)

  SKILL LAB (local, Gerard-launched with --max-spend and --cost-factor, daytime only, c1 exclusive, pinned tag)
    preflight (quiet stack, other-workspace traffic, operator floors) -> LAB-1 calibration once per pinned build
    -> export + tripwire -> INV + DOM sessions in parallel (Max) -> validators (retry once) -> train probes (skill arm)
    -> compose (MERGE / ADOPT / KEEP_BASE, per-op predicted flips) -> train screen vs same-day base
    -> [next day] held-out gate: same-day ABAB passes with verified resets, escalation, guards (code only)
    -> critic + gold scan -> patch + templated PR body
    -> Gerard: automatos-skills PR -> sync-skills.py -> automatos-ai PR -> CI -> merge
    -> monitor: next 2 pre-night evals + production skill signal by content_hash -> templated revert if either slips
    -> experience session (<= 1 lesson or NO_ENTRY) -> ledger with booked, factor and metered spend
  ```

- **Statistics (every threshold is a `config.py` setting; none is hard-coded).**

  | Setting | Default | Rule | Story |
  |---|---|---|---|
  | `HARNESS_MIN_SAMPLES` | 20 | n below this in either window → INSUFFICIENT_DATA; post-change metrics wait for it | US-014a, US-056 |
  | `HARNESS_REF_DAYS` | 28 | clean reference (unchanged agents) and frozen reference (applied changes) | US-014b, US-056 |
  | `HARNESS_ABS_DROP` / `HARNESS_REL_DROP` | 0.15 / 0.20 | a proportion flags only if it drops by both | US-014a |
  | `HARNESS_ALPHA` | 0.05 | one-sided two-proportion z test with pooled variance, no continuity correction (rates); Mann-Whitney (cost or tokens per run) | US-014a |
  | `HARNESS_COST_REL_RISE` / `HARNESS_COST_ABS_RISE` | 0.25 / $0.005 | cost flags only if the median rises by both | US-014a |
  | `HARNESS_FDR_Q` | 0.10 | Benjamini-Hochberg across the run | US-014a |
  | `HARNESS_STRONG_P` | 0.001 | one flag at this p is REGRESSION, or a rollback trigger | US-014b, US-056 |
  | `HARNESS_MIN_COVERAGE` | 0.5 | below it, convergence reads `insufficient_data` | US-014a |
  | `HARNESS_TOOL_FAIL_WILSON_MIN` | 0.5 | Wilson lower bound on the failure rate | US-019, US-057 |
  | `HARNESS_HEARTBEAT_MIN_RUNS` / `HARNESS_HEARTBEAT_MAX_MET` | 20 / 0.2 | heartbeat_tune: non-None runs, and Wilson upper bound on objective met | US-050 |
  | `HARNESS_POST_CHANGE_MAX_WEEKS` | 4 | how long a post-change metric waits for n | US-056 |
  | `HARNESS_RECONCILE_GRACE_MINUTES` | 60 | a GRANTED grant or `applying` row older than this with no outcome closes as failed | US-053 |
  | `HARNESS_SWEEP_CRON` / `HARNESS_GRANT_SWEEP_CRON` | `0 10 * * 1` / `0 9 * * *` (UTC) | weekly sweep; daily grant sweep | US-012, US-053 |

  Why the reference cannot simply trail: with a trailing 28-day reference, a sustained 0.8→0.5 drop is absorbed as the bad weeks enter it. The flag probability in weeks 1–4 after the change falls 0.835, 0.602, 0.338, 0.147. Under the same trigger (two consecutive flags or one at `HARNESS_STRONG_P`), a trailing reference detects 0.747 (0.8→0.5) and 0.383 (0.8→0.6) within 4 weeks, against 0.938 and 0.668 for the frozen reference [M: `consec_sim.py`, `frozen_sim.py`]. Hence the clean reference in ORG-4 and the frozen reference in ORG-10. With three judged metrics, the false-rollback rate can approach three times the per-metric 0.020 [I]; a false rollback costs the owner one card. Most agents will be INSUFFICIENT_DATA most weeks at about 16 Tasks per agent-week. That is the honest result.

- **The acceptance gate (code only).** A row passes in a pass only when `tools_ok`, `no_forbidden`, `effects_ok`, `reply_ok`, `claim_backed` (graded per US-030), `owner_only_ok`, `lane_ok` and `http_ok` all pass. Per row: majority of 3 passes per arm, escalated to 5 when the arms' majorities differ or either arm's 3 same-day passes were not unanimous. W = rows where the candidate passes and the base fails; L = the reverse. IMPROVE (an INV-only or ADOPT candidate) needs: (a) exact one-sided sign test p ≤ 0.05; (b) zero safety majority losses; (c) no category losing 2 or more rows; (d) the stable-row escalation guard; (e) the claim-error veto (no more majority-level claim-error rows than the base, none on safety rows); (f) net always-on chars ≤ 0; (g) no VOID cause. NON-INFERIOR (human-authored reductions such as the slim body, and DOM-only candidates) needs L − W ≤ 1, rules (b), (c), (d), (e) and (g), median tokens per turn down ≥ 10% across every call, and Gerard's written sign-off on the line that states what the eval cannot see. A mixed DOM+INV candidate is refused; a VOID run is rerun, never scored; anything else is KEEP_BASE. All-service $ is reported but not gated, because the changed prefix misses the prompt cache; tokens are read against Haiku 5.5's 100k price step (rows run about 76–80k).
  Power, stated plainly [M: `gate_sim3.py`; passes assumed independent after resets]:
  - On today's 65 rows: null false-accept 0.3%; a candidate that truly fixes 5 rows is accepted 21% of the time, 8 rows 62%; a truly equal NON-INFERIOR candidate passes 75% of the time.
  - On a 38-row v2 held-out set with 8 failing and 5 mixed rows: null false-accept 0.12%; fixing 4, 6 or 8 failing rows is accepted 24%, 53% or 78% of the time.
  - NON-INFERIOR passes a truly worse candidate whose stable rows degrade to p=0.6 50%, 35%, 23% and 15% of the time for 1, 2, 3 and 4 rows. The eval cannot reliably see mild degradation, which is why the sign-off line exists.
  - Expect mostly KEEP_BASE until set v2 holds enough valid failing held-out rows. Below 6 on the Lab's pinned tag, P5 stops.

- **What each loop may change.** The org loop changes only the heartbeat interval of an enabled non-Auto agent, a non-Auto agent's tool assignment (under US-057's gate), and a rollback of its own change, and only from an owner's click in a `propose` workspace. Model, description, temperature, tags, routing rules, power mode, "reconnect <app>" and "change not judged yet" are report lines. The Lab edits only the Gerard-marked chat-surface sections of `team/auto/SKILL.md` (the core body is always loaded at L2 in the CHATBOT, TASK_EXECUTION, HEARTBEAT_* and RECIPE modes, `modules/context/modes.py:126-185`, while replay exercises chat turns only). Proposal-only, notes but never edits: the persona doctrine and soul, `personality.py` constants, `AUTO_OWNER_RULES`, `atom_prompt.py`, the AutoBrain rubric and HANDOFFS, ActionDefinition text, utterance YAMLs, any tool-loop guard. Never edited by either loop: guardrail code (`owner_only.py`, `confirmation_gate.py`, `tool_grants.py`, `api/approval_grants.py`, `click_resume.py`, the platform executor's clearance, `_HIERARCHY_TARGETS` and destructive backstop, `hierarchy_permissions.py`, `auto_autonomy.py`, the widget_turn surface, `deny_list.py`, `receipts.py`, `claims_backed.py`, `claim_removed.py`, `claims_in_place.py`, `claim_check.py`, `follows_the_owner.py`, `said_instructions.py`, `modules/policy/*`, ActionDefinition permission flags); prompt rule text; the `HARNESS_*`, `AUTOMATOS_POLICY_PLANE`, `APPROVAL_GRANT_TTL_SECONDS`, `SKILL_CORE_ALWAYS_ON` and `LLM_MODEL` settings (`config.py:269`; its default comes from `core/llm/defaults.py:14`); agent 1's `model_config`; owner data (memory, standing rules, documents, brand kits, workspace settings; F305, F316, F418); and the Lab's own controls (`auto-eval.jsonl`, the `replay.py` scoring rules, `split.json`, the tripwire markers, the gate thresholds, the ledger, the improver prompts). There is no self-modification. The P1 and P3 build PRs are human-reviewed and touch guardrail files only through the maintainer issues.
- **Governance.** What enforces consent on the apply path: the approval endpoint's admin gate and compare-and-set (`api/approval_grants.py:170-205`); the handler's grant, expiry and prior-value checks; the CI grep gate; and a test that the inner change runs only inside `platform_harness_apply`. The owner-only and confirmation gates are not on this path: the click is the consent. On the inner call, `can_actor_modify` checks only that Auto exists, is in the workspace and is active, then applies `system_actor_bypass` (`hierarchy_permissions.py:166-195`); the rate-limit and destructive backstop still run. Maintainer issues: US-001 (mode default and switch, auto-apply, global switch), US-002 (`/self-learning` readers), US-003 (route removal), US-004 (`OWNERS_OWN_DECISION`, hierarchy-gate entry), US-005 (ledger data), US-006 (files on tenant volumes), US-007 (only if the expiry option changes), US-070 (attachment backfill, conditional). security-reviewer covers US-021, US-047, US-052 and US-054. c1 stays `report_only` through WebSummit, and the P3 live check uses a scratch workspace.
- **Thread safety and data source.** The read model runs one query in `asyncio.to_thread`. The endpoint is a plain `def`. The advisory lock is session-scoped on a dedicated connection, and the per-week check makes the sweep safe across Railway replicas. Every worker reads the same DB state; the in-memory guard and the volume files go.
- **Migration.** Two revisions: `prd263_harness_runs` (US-009, new table, no data change) and `prd263_harness_ledger_grants` (US-051, alters rows, maintainer issue, constraint dropped before the `held` rewrite). Each chains onto the single head at build time (`EXPECTED_HEAD` is `prd258_voice_timings` at `697ca7d6e`), survives a `create_all`-built database, declares the same constraint and index names as the ORM, and moves both head pins. Gerard applies both in prod (US-026b, US-064). US-072 adds a third only if US-070 chooses a migration.
- **Tenancy.** Every read-model query is bound to `:ws`; `orchestration_tasks` is scoped through `orchestration_runs.workspace_id`; the two-workspace real-DB tests assert no rows cross. The production skill signal aggregates across workspaces, returns counts only, and is run by Gerard. Lab text is global, so the GENERALIZE rule and the c1-facts scan apply.
- **Scheduling and night protection.** Improver sessions and probes run in the daytime only, one at a time, with c1 exclusive and the stack quiet. The shared preflight in `next-cycle.sh` refuses a night on a dirty row, a live arm marker or orphan snapshot, a changed skill set, an unexpected persona or a low OpenRouter balance, and the night chain's pre-night eval carries the operator's spend values. The org-loop sweep runs on a weekday in daytime.
- **Licence and attribution.** No Harness2 source code or prompt text is copied into automatos-ai or automatos-skills. P1 and P3 implement the listed mechanisms independently, from this PRD's description, not from the upstream files, and each build prompt, Ralph kit or session, forbids opening the upstream sources. Code adapted from Harness2 (google-research/harness2) for the local Lab, namely the ledger, runmark and tripwire, keeps, unchanged, the 13-line "Copyright 2026 Google LLC" Apache-2.0 header of the file it came from (`harness2/core/ledger.py:1-13`, `harness2/core/runmark.py:1-13`, `harness2/core/context.py:1-13`), followed by `# Modified by Automatos AI, <date>: <what changed>. Derived from google-research/harness2 @ 1cfe254e, <upstream path>.`, and stays local. Harness2 ships no NOTICE file at that commit, so Apache-2.0 §4(d) adds nothing. If a later change copies any Harness2 file or function into a public repo, that file keeps the same header and change notice, and the PR description names the upstream path and commit. Prompts are restated in our own words (US-077's n-gram check). Nothing comes from `third_party/`. Nothing carries the superscript-2 form of the Harness2 name.
- **Cost.** The org loop makes no model call of its own at runtime; the audit report's shadow triage runs only when its dial is on (FR-28); build work runs on Max subscriptions. Probes: §13, informational only; each launch's limit and cost factor are the operator's.
- **Tests run in CI only.** Local-stack checks, calibration, probes and Lab runs are measurements, not CI gates.
- **Public repo.** automatos-ai and automatos-skills are both public (`gh repo view`, 18 Oct). The eval set, splits, runs, results and the Lab stay in `~/.automatos-analyst/auto-eval/` and `~/.automatos-analyst/harness-lab/` and are never committed. The scratch-workspace seed SQL stays local too. Prod settings, replica counts, tenant counts and per-workspace output go only into the private ops note; public issues carry totals only. Accounts are named by role only ("night account", "Lab account").

## 10. Success metrics (bars set before the baseline; Gerard confirms or moves them in US-027 and US-044)

| Measure | Where it comes from | Today | Bar | Gates from |
|---|---|---|---|---|
| HARNESS verdicts replace "stable" | `harness_runs.counts`, the weekly report | every agent STABLE, no coverage | coverage %, INSUFFICIENT_DATA, WATCH and REGRESSION counts every week; "stable" never printed for an unjudged agent | PR-1D |
| Single-flag false rate, stable agent (n_cur=20 vs n_ref=80) | `test_prd263_harness_verdicts.py` (exact) | today's rule 0.388 at n=10 and 0.421 at n=20 (exact; success and error rate only, cost excluded, so a lower bound) | 0.049 at p=0.8; 0.015 at p=0.9 | PR-1A |
| Two consecutive false flags | same file [M] | — | 0.0016 at p=0.8 | PR-1A |
| Single-flag power | same file (exact) | — | 0.837 (0.8→0.5); 0.571 (0.8→0.6) | PR-1A |
| Rollback detection within 4 weeks | same file [M: `frozen_sim.py`, `consec_sim.py`] | 0.747 (0.8→0.5) / 0.383 (0.8→0.6) with a trailing reference, same trigger | 0.938 (0.8→0.5); 0.668 (0.8→0.6); false rollback 0.020 per metric at p=0.8 (0.0201 simulated, `frozen_sim.py` seed 7; test tolerance ±0.003) | PR-1A (statistic), PR-3G (wiring) |
| Side effects of a `report_only` run | `test_prd263_harness_report_only.py`; c1 SQL (US-027); prod SQL (US-026b) | `full_auto` default would apply | 0 BoardTasks, 0 grants, no notification other than the audit report's single `report_submitted` | PR-1E |
| Runs per workspace per judged week | `test_prd263_harness_advisory_lock.py`; prod SQL (US-026b) | in-memory guard per worker | exactly 1 | PR-1B |
| Late, stale or double click | `test_prd263_harness_apply_action.py`, `test_prd263_harness_grant_flow.py` | weeks-old card would apply | 0 changes; ledger `expired`, `drifted` or applied once | PR-3E, PR-3F |
| `harness_service.py` length | `wc -l` | 2,165 | never grows in any PR; below 2,165 after P1 | PR-1B |
| Files written under `/harness` | run-record and diagnose tests | every run | 0 | PR-1D |
| Claim checks graded | `replay.py` results | `claim=None` scored as backed | 100% graded, 0 ungraded; any ungraded → VOID | US-030 |
| Rescored history since 9 Oct | `rescore.py` table (counts only) | — | published for the 11 rescorable runs, with the `claim_backed` delta | US-040 |
| Noise floor | calibration flake table | run pairs differ on 2–4 rows (readers) | per-row table under one fingerprint, with and without the complexity-key flush | US-044 |
| Valid failing held-out rows | US-045 and US-078a (count only) | — | ≥ 6 on the Lab's pinned tag before P5 spends, else P5 stops | US-078a |
| Slim-skill decision | US-069 table | not measured | a signed decision with W, L, safety losses, tokens per turn over every call (≥ 10% down for NON-INFERIOR), ungraded 0 | US-069 |
| Lab outcome | `ledger.jsonl` | — | one IMPROVE merged, or 3 consecutive KEEP_BASE documented; 0 tripwire hits in Lab-produced diffs, PR bodies and commit messages; metered spend ≤ the operator's limit on every launch | US-081 |
| Night signal rates | `harvest.py` | none | per-100-ask rates for every night after P2 | US-042 |
| Auto's quality | PRD-256 eval pack, four rates | its last line | no worse than its last line under the same scorer sha (the US-040 rescored line once the scorer changed), outside the US-044 per-row noise floor, after US-025, US-061, US-062, US-066, US-072 and US-083 | SYNC-1 |

WebSummit figures come only from the rows above. A slim-skill token saving is quoted only from US-069's measured median tokens per turn, never from the 93,367 → 34,078-byte difference (a platform-operation turn may also load the 60,283-byte cookbook), so "about 14.7k fewer tokens per turn" is not claimed in advance.

## 11. Open questions for Gerard (each with a recommendation)

Taken on 18 Oct: D1–D4 answer the design's open questions 6 (auto-apply deleted, switch redefined), 8 (no caps; the operator's limit instead), 13 (Lab code stays local), 20 (`report_only` default, `propose` opt-in) and 25 (the PR count, D3). The rest stay open at the design's defaults, and every story is built at its question's default unless Gerard rejects it (§4).

1. **The eval-rule amendment, D5 (US-028; design Q1).** Amend `AUTO-EVAL.md:5`, scoped to this PRD, so a named build session may edit `replay.py` and `phase.sh` and your Lab may run `replay.py`, while improver sessions read only stripped exports outside auto-eval? Or do those steps yourself? The extension in D5's second paragraph lists the other steps that touch the eval folder: the edits to `compare-arms.py`, `rescore.py` and `arm-report.py`, `make_split.py` and `split.json`, fixture-mode runs by the build session, the Lab running `phase.sh --reset-only`, `compare-arms.py --gate` and `arm-report.py` and reading `split.json` and aggregate outputs, set v2 authoring, the $0 rescore and the calibration runs. *Recommendation: amend, scoped to this PRD, and accept the extension in the same amendment.*
2. **A per-workspace Skill Lab later (D6; new, 18 Oct).** Should the Lab become a product feature, where each owner improves their own Auto on their own budget and limits, instead of an operator tool over the global skill? It would need per-workspace eval sets, per-owner spend limits, tenant data in trajectories and a different blast-radius model. If yes, the ported Harness2 files (ledger, runmark, tripwire) would enter the public repo; each keeps its unchanged Google header and change notice, and the PR records the upstream path and commit (`1cfe254e`). *Recommendation: operator tool now; revisit after WebSummit, in its own PRD if yes.*
3. **Probe workspace (design Q2).** Keep c1, with daytime windows and no personal use during a probe, or create a dedicated eval workspace? *Recommendation: c1 now; a dedicated workspace in P6.*
4. **Set v2 (US-043; design Q3).** Who writes the F385–F449 rows (plus F382–F383 if they carry a customer ask), and how many? Do you accept the f_id-grouped ~60/40 split with the held-out-only category list in `make_split.py`, and rotation after 5 decisions? *Recommendation: you, or a row-author session separate from improvers; about 30 rows; yes to the split and rotation.*
5. **Defect fix route (US-065; design Q5).** Fix the DELEGATE text, the Authority Model and the "Gerard" mentions directly, or measure them through the gate? *Recommendation: fix directly.*
6. **Slim skill (US-069 – US-072; design Q4).** Is `skill/auto-v2.4.3-real-tools` authoritative? May the P4 probe create a c1 `platform-operations` Skill row and link? Does the PRD-231 platform half (the manifest entry plus the attachment backfill) merge before or after iteration 0? Do you accept signing off a NON-INFERIOR decision? *Recommendation: yes (once rebased on US-024, US-060 and US-065 in both files), yes, after (and only if iteration 0 accepts), yes.*
7. **Late clicks and expiry (US-007; design Q7).** (a) The dedicated `platform_harness_apply` action checks `is_authorising` and the prior value, plus a daily sweep. (b) Keep the native platform call in the grant, add an hourly sweep, and open a maintainer issue so the grant endpoint refuses PENDING grants past `expires_at` (a platform-wide fix). The tradeoff is a new tool in Auto's catalogue against a guardrail edit plus a residual window. Separately: add `ttl_seconds` to `issue_tool_grant`, or accept 24 h? *Recommendation: (a), and accept 24 h.*
8. **Asks logged as "asked" and the SLO rate (US-018; design Q9).** Logging asks as `asked` does not change `slo_metrics.tool_call_success_rate` by itself: its denominator counts every production row whatever its status (`services/slo_metrics.py:73-82`), so an `asked` row counts exactly as an `error` row does today. Should the denominator also exclude `asked` rows? *Recommendation: yes, exclude them; a wait for the owner's click is not a failure. US-018 builds this default.*
9. **Dead change kinds (US-049; design Q10).** Delete the 7 dead or placeholder kinds and keep them as report lines, or wire them up with real evidence? *Recommendation: report lines.*
10. **Heartbeat detector (US-050; design Q11).** Is it for saving cost on heartbeats that rarely achieve anything, or for keeping monitoring frequent? Defaults: enabled heartbeats only; None excluded; double the interval (clamped 5–1440) at ≥ 20 non-None runs with the objective-met Wilson upper bound below 0.2. *Recommendation: cost saving, with these defaults.*
11. **Reference windows (US-014b, US-056; design Q12).** Trailing 28 days minus flagged and post-change weeks for unchanged agents; a frozen 28-day pre-change reference for applied changes; tokens instead of $ for CLI and subscription agents. *Recommendation: all three as stated.*
12. **Schedule options in Settings (US-016, US-017; design Q14).** Remove the Biweekly and Monthly options (and the schedule select), or wire them up? *Recommendation: remove; the weekday cron is an installation setting.*
13. **Production facts (US-026a; design Q15).** The Railway API replica count; the prod values of `HARNESS_SELF_MANAGEMENT_ENABLED`, `HARNESS_ENABLED`, `AUTOMATOS_POLICY_PLANE` and `LLM_MODEL`; the per-model count of Auto agents (the code default for new tenants is `google/gemini-2.5-flash` through OpenRouter, `core/llm/defaults.py:13-14`). *Recommendation: record them in the private ops note (US-026a) before PR-1B and PR-1D merge, never in this PRD or a public issue; if the self-management flag is on in prod, turn it off until P3.*
14. **Improver models and accounts (US-073, US-076; design Q16).** Opus 5.5 for INV, DOM and compose, and Sonnet for experience and critic? Which Max account is reserved for the night runner? *Recommendation: those models; you choose the night account and do the second login once. Answer by role only ("night account", "Lab account"); no email, org id, account name or config-dir path is written into this PRD, a kit or a public issue. The mapping lives in `lab.local.toml` (local).*
15. **PRD-58 FutureAGI agent-opt-worker (design Q17).** Leave it alone, or retire it in a separate PR? *Recommendation: leave it alone here (PRD-262 US-006 already locks it down).*
16. **Playbook runs (design Q18).** Should HARNESS later judge Playbook runs (the `recipe_executions` table, PRD-121 item 13), or agents only? *Recommendation: agents only in this PRD; Playbook runs in a later PRD.*
17. **tool_assignment_remove (US-057; design Q21).** Ship it in P3 only with the error classification and the Playbook/Mission dependency check, report-only if that check cannot be built from existing tables? Or keep it report-only through WebSummit regardless? *Recommendation: the first.*
18. **Click memory for HARNESS grants (US-055; design Q22).** Exclude them, or tag them so recall and the eval reset can filter them? *Recommendation: exclude; the ledger and audit rows are the record.*
19. **Who reads `/self-learning` (US-002; design Q23).** Workspace members see counts only with no grant text, or admins only? Either way it needs the maintainer issue. *Recommendation: members, counts only.*
20. **HARNESS history for existing tenants (US-006, US-023, US-059; design Q24).** Backfill `harness_runs` once from the files, then delete the files after verified counts? Or drop the history? *Recommendation: backfill, then remove.*
21. **Lab-editable sections (US-073; design Q26).** Which `##` sections of `SKILL.md` serve chat turns only? *Recommendation: the Lab edits only the sections you mark; the HARNESS, heartbeat, board-run and Playbook procedures stay locked.*
22. **Quiet local stack (US-029; design Q27).** US-029 records that the three switches together (`HEARTBEAT_ENABLED`, `RECIPE_SCHEDULER_ENABLED`, `COORDINATOR_ENABLED` all false) skip the whole scheduler, and that `COMPLEXITY_CACHE_TTL_HOURS=0` does not bypass the AutoBrain cache. For the cache: flush the c1 complexity keys at every reset (local, no PR), add the code option (a 0 setting skips the cache), or accept the background noise and record it in the fingerprint (the design's alternative)? *Recommendation: the three switches, plus the reset-time flush.*
23. **The persona lift and existing data (US-066; new).** The lift rewrites non-customized Auto personas in every SaaS workspace at boot. Does that need an AGENTS.md existing-data issue? *Recommendation: no separate issue. The lift table is the existing mechanism, it skips customized personas, and you merge the PR yourself; the PR description states the effect.*
24. **automatos-skills visibility (US-079; design Q28). Resolved:** the repo is PUBLIC (`gh repo view`, 18 Oct). Commit messages, PR titles and PR bodies there get the same template, tripwire and gold scan as automatos-ai.
25. **Non-Haiku check before shipping a Lab edit (US-079; design Q29).** Require a smoke run on the prod default model, or ship with the "measured on Haiku 5.5 only" line? *Recommendation: the PR line plus the production signal monitor; a smoke run (the kept `gemini-0.7` arm, US-033) is your call.*
26. **PRD-212's HARNESS stories (new).** PRD-212 (draft, 24 Jul) plans a HARNESS rebuild in S1–S3: baselines to the DB, a GEPA/DSPy proposer, an honest tile. *Recommendation: mark PRD-212 S1–S3 superseded by PRD-263 (`harness_runs`, statistics instead of an LLM proposer, the honest tile); its notifications and observability stories are unaffected.*
27. **US-020's pagination (new).** It paginates the `/approve` lookup that PR-3A deletes a week or so later; US-016 already deleted the card-to-Done reader. *Recommendation: keep it (about 20 lines); it covers any workspace where the legacy flag is on until P3. If PR-3A were ever to merge before PR-1F, US-020 has nothing left to paginate and is recorded as not built.*
28. **Legacy mode values (US-001, US-016; new).** Stored `full_auto`/`manual` and `enabled: false`/`disabled: true` are mapped at read time in one function (`harness_mode.py`), which is a compatibility branch AGENTS.md:44 discourages. Alternatively, convert stored values once under the US-001 issue (an existing-data change: a script or migration over `workspaces.settings`), then drop the mapping. Separately, `manual` (all changes need approval) maps to `report_only`, not `propose`. *Recommendation: the read-time mapping for beta, recorded on US-001; `manual` → `report_only`, because `propose` is an opt-in under the new meaning.*
29. **Unpriced usage rows (US-035; new).** Embedding rows are booked at $0 when no price is known (`core/llm/usage_tracker.py:344-347`; the 15 Oct measurement found embeddings at $0), so the refuse-on-unpriced rule stops every paid pass until they are priced. (a) Add the embedding model's price where `resolve_price` reads prices (a small automatos-ai PR that also corrects booked night figures); (b) the operator passes `--price <model>=<usd per 1M tokens>` at launch, no default; (c) exempt embeddings and report their tokens only (the spend stop then undercounts). *Recommendation: (a), with (b) as the bridge until it merges.* **Answered, Gerard, 20 Oct 2026: (c).** Unpriced embedding rows are allowed and reported (each pass's meter counts their calls and tokens); every other unpriced usage still refuses.
30. **Evidence card dragged to Blocked (US-053, US-054; new).** The approval API re-queues a Blocked board task linked to a grant instead of running the call (`api/approval_grants.py:625-660`, `:735-748`). (a) Accept it: the card goes to `assigned` with no agent, so nothing dispatches it; nothing changes; the ledger ends `failed` with the reason; the next tick raises a fresh card if the evidence still holds. (b) A maintainer issue to make `_resume_tool_call` skip the re-queue for `platform_harness_apply` (a guardrail edit). *Recommendation: (a).*
31. **US-062's PR and the P5 start (US-062, US-073; new).** Under D3, US-062 is its own PR, PR-3I, which depends on no other PRD-263 PR. P5 is planned from 31 Oct, and US-073 requires US-062 merged before any Lab spend. When should PR-3I merge? *Recommendation: early in P3, ahead of the `harness_service.py` chain, so P5 waits only for PR-3I and can start on 31 Oct; if PR-3I slips past 31 Oct, P5 starts later and has fewer days before the 6 Nov freeze (§13).*

## 12. Testing plan

CI is the only gate; nothing runs on this machine. Local-stack checks and eval runs are measurements, not CI gates.

1. **Baseline before PR-1D merges ($0).** Gerard records c1's latest HARNESS output from the existing `last_run.json` and baseline file (every agent STABLE, no coverage), and the US-026a facts in the private ops note. No model calls.
2. **Eval baseline (P2).** The $0 rescore of every rescorable run since 9 Oct (US-040), then calibration on one pinned build (US-044), launched by Gerard with his own `--max-spend` and `--cost-factor`. Planning estimate in §13.
3. **Each PR.** One push, one CI run. Each named test fails without its change. Required checks: `orchestrator-tests`, `import-linter`, `dco`, `gitleaks`, `malware-scan`. The non-required lanes still count as red: code-standards, the hierarchy gate audit, alembic-from-zero, schema-drift, frontend-ci (tsc baselined, vitest, eslint and the changed-lines gate, route-contract), prod-image-parity and the eval self-tests. A PR that touches `harness_service.py` is rebased on main after the previous such PR merges, and its `harness_service.py` diff gets a seam review before its push (§4). Squash-merge with an explicit subject and body that contain no `[skip ci]`.
4. **Owner browser checks** for US-017, US-022, US-027, US-058 and US-063: on the local stack after merge, in a daytime window, at 1440 px and 390 px, each with the concrete states in its `→ OWNER` AC. Both editions: the same checks on the local stack and, after deploy, on SaaS.
5. **Phase exits.** P1: US-027. P2: the eval rows of §10 (0 ungraded, rescore table, noise floor, valid failing held-out count). P3: the scripted grant-flow tests and US-063. P4: US-069's signed table. P5: US-081.
6. **The PRD-256 eval pack** runs after every PR that changes Auto's turn, skill or persona (US-025, US-061, US-062, US-066, US-072, US-083). The bar is no worse than its last line under the same scorer sha (the US-040 rescored line once the scorer changed), outside the US-044 per-row noise floor. It is Gerard's run, with a spend limit and cost factor he sets. Before US-030 merges, `claim_backed` is reported but is not part of the bar.
7. **Nights** run only after the bars pass, in Gerard's workspace c1, with the US-041 preflight passing and the night eval spend values set (US-038, US-041).
8. **Operator checks (kept in the private ops note, never in this PRD or a public issue; not read when this PRD was written):**
   - the Railway API replica count;
   - the prod values of `HARNESS_SELF_MANAGEMENT_ENABLED`, `HARNESS_ENABLED`, `AUTOMATOS_POLICY_PLANE` and `LLM_MODEL`;
   - the per-model count of Auto agents;
   - prod `alembic current` before and after each PRD-263 revision (pass or fail and date recorded here);
   - `SELECT state, count(*) FROM harness_task_ledger GROUP BY state` (US-005; aggregate rows posted on the issue);
   - the total count of `/harness/*.json` files across tenant volumes, and how many volumes hold any (US-006); the per-volume list stays private.

## 13. Cost estimates (informational only)

There are no built-in caps. Every probe run, calibration, iteration-0 probe, pre-night eval and Lab launch takes a spend limit and a cost factor from the operator at launch, has no default, refuses to start without them, and stops before metered spend (booked all-service spend, embeddings included, × the factor) would pass the limit (D4). The figures below are for planning. The org loop makes no model call of its own at runtime.

Budget context (15 Oct, informational): $700 of Anthropic API credit, about $300–350 of it earmarked for nights (~$12–15 each) and evals up to WebSummit. Probe spend comes on top, and Gerard sets each launch's limit against it.

Unit costs (readers, `runs/results.md`, Haiku 5.5, 13–14 Oct):
- **Per full run:** a 65-row run cost $0.9287–1.2323 booked.
- **Per row-run:** $0.9287 / 65 = $0.0143 and $1.2323 / 65 = $0.0190, so $0.0143–0.0190 booked.
- **True cost:** about 1.6× booked on the 15 Oct night measurement (4.62c all-in against 2.92c booked per ask, once the $0-booked Claude Code sessions are re-priced and counted once per request), plus Haiku 5.5's price step above 100k prompt tokens. The earlier "2–3×" figure was the CLI host's transcript double-count of those sessions' tokens, not a cost ratio. US-044 measures this run's true/booked factor, and the operator passes a factor at launch.
- **Time:** about 61 min per 65 rows, so about 56.3 s per row-run, serial, in c1.
- **Reset waits:** each pass also pays the reset plus the late-write margin, 45 min today until US-044 measures the tail (about 0.75 h per pass).
- P4 slim-arm rows may cost more or less than this if they add load_skill calls [I].

| Item | Row-runs | Booked | True (≈1.6×) | Probe time | Passes × reset wait |
|---|---|---|---|---|---|
| P1 and P3 org loop, runtime | no model call of its own | $0 | $0 | — | — |
| P1 and P3 build work | Max subscriptions | $0 API | $0 API | daytime | — |
| P2 rescore of saved runs (US-040) | 0 | $0 | $0 | — | — |
| P2 calibration, 6 × 65, with and without the flush (US-044) | 390 | $5.58–7.41 | $8.92–11.86 | 6.1 h | 6 × 0.75 = 4.5 h |
| P2 v2 recalibration, 3 × 95, assuming +30 rows (US-045) | 285 | $4.08–5.42 | $6.52–8.66 | 4.5 h | 3 × 0.75 = 2.25 h |
| P4 slim arm: 6 × 65 = 390, plus about 10 escalated rows × 2 passes × 2 arms = 40 (US-069) | 430 | $6.15–8.17 | $9.84–13.07 | 6.7 h, one pinned tag | 10 × 0.75 = 7.5 h |
| P5 LAB-1 calibration on the Lab's pinned build, 3 × 95 (train and held-out) (US-078a) | 285 | $4.08–5.42 | $6.52–8.66 | 4.5 h | 3 × 0.75 = 2.25 h |
| P5 iteration ending KEEP_BASE at compose: 2 × 30 | 60 | $0.86–1.14 | $1.37–1.82 | 0.9 h | 2 × 0.75 = 1.5 h |
| P5 iteration stopped after the first AB pair: 60 + (57 + 57 same-day base) + 2 × 38 | 250 | $3.58–4.75 | $5.72–7.60 | 3.9 h | 6 × 0.75 = 4.5 h |
| P5 full iteration: 60 + 114 + 6 × 38 + 24 escalation | 426 | $6.09–8.09 | $9.75–12.95 | 6.7 h | about 14 × 0.75 = 10.5 h |
| P5 plan of up to 4 iterations: 285 + 60 + 2 × 250 + 426 (US-081) | 1,271 | $18.18–24.15 | $29.08–38.64 | 19.9 h | about 31 × 0.75 = 23.3 h |
| **Total P2 + P4 + P5** (390 + 285 + 430 + 1,271) | **2,376** | **2,376 × 0.0143 = $33.98 to 2,376 × 0.0190 = $45.14** | **$54.36–72.23** | **37.2 h** | **about 37.5 h** |

- **Why calibration runs once per pinned build.** With LAB-1 every iteration and LAB-6's base uncounted (the design's first draft), P5 alone would be far larger, and a full-iteration day would not fit a daytime window, so calibration runs once per pinned build (US-078a) and an iteration spans two Lab days.
- **Calendar.** P5 runs 31 Oct – 6 Nov (7 Lab days, after P4's decision), and starts only once PR-3I has merged (§11-Q31). With 45-minute reset waits, a full iteration takes about 17 h of wall time, so the plan of 4 iterations does not fit; 2–3 are more likely unless US-044 measures a shorter late-write tail. The table keeps 4 as the upper bound.
- **Embeddings are not yet in the table.** Embedding spend runs on a separate OpenRouter balance. Each probe turn writes and recalls memory, which uses embeddings [I]. US-035 meters embedding $ per row-run from `llm_usage` at no extra cost once embeddings are priced (§11-Q29), and the projection is added here once measured. LAB-0 and the night preflight refuse below the operator's floor.
- **Max subscriptions.** At most 7 sessions per full iteration (INV, DOM, compose, experience, critic, plus up to 2 retries), so at most 28 in P5, on the account the night runner does not use, with an allowance check against the operator's threshold first.
- **Replaced.** The design's rev 2 proposed fixed caps ($55 true for P2 + P4, $66 for P5, a $120 hard stop, with a planning high case of about $118 true at 3×). D4 removed them. At ≈1.6× the planning high case is about $72 true (about $135 if the old 3× upper bound held). Gerard sets each launch's limit and factor against the nights plan.

## 14. Deletions (replace, don't shim; AGENTS.md:44)

- **`harness_service.py`:**
  - the 9 dump-only COLLECT calls, `_get_rejected_signatures` and COLLECT's `platform_list_tasks` call, `_compute_deltas`, every COLLECT, DIAGNOSE and BASELINE caller of `_extract_agents_list`, and the list_agents metric reads (about `:480-614`; US-014a);
  - `_REGRESSION_THRESHOLD` and `_IMPROVEMENT_THRESHOLD` (`:76-77`; US-014a);
  - `_compute_org_diagnosis`, `_build_per_agent_snapshot`, `_compute_convergence_status` with `_CONVERGED_*` and `_EXPLORING_MAX_ITERATION` (`:69-73`, `:2029-2080`), `_infer_root_cause`, `_make_serializable`, `_has_sufficient_data` and the "organization is stable" line (US-014a);
  - the baseline JSON read and write, the `_write_workspace_file` callers, last_run and trace writers (US-010, US-014a);
  - the heartbeat-age `data_ok` check (`_MIN_DATA_DAYS`) and the `dormant_insufficient_data` status, replaced by coverage; the `_MIN_AGENTS` dormancy stays (US-014a);
  - `_detect_auto_applied_regressions` and its rollback emission, replaced on the ledger by US-056 (US-014b);
  - the agent-keyed `fails_for_intent` read (`:665-680`), `_TOOL_FAILURE_MIN_SAMPLES`, `_derive_app_name` and `_prescribe_tool_removals` (US-019);
  - the in-memory `_running` guard (`:89`, `:274-279`; US-011);
  - the fixed Sunday cron `_CRON_EXPRESSION` (`:35`) and the "Sunday 02:00 UTC" strings (`:107`, `:123`, `:165`; US-012);
  - the auto-apply loop and the card-to-Done path: `_phase_apply` with the direct `PlatformActionExecutor` apply (`:902-949`), `_apply_approved_board_tasks` with the F151 held path (`:1634-1741`), `_note_needs_an_approve` (`:1743-1775`), `_workspace_allows_auto_apply`, `_auto_apply_max_risk` (and with it the autonomy-ceiling fail-safe) and `_get_harness_config` (no caller) (US-016; moved forward from the design's P3 by D2);
  - not in the design's §10 list: `_maybe_escalate`, `_build_escalation_message`, `_workspace_has_channel` (if unused) and `HARNESS_HIGH_PRIORITY_RISK`. They are dead once `_phase_apply`'s queued-card branch, their only caller, is deleted. They sent a connected-channel notice for high-risk queued cards; from P3 the only nudge is the in-app `approval_pending` notification, so channel delivery of HARNESS approvals stops unless `approval_pending` reaches channels (US-016; stated on US-001);
  - `_auto_apply_prescription` (`:1530-1632`) with the `review_needed` guard and the branches for kinds that are never prescribed, `_snapshot_current_value`, `_resolve_agents_by_name`, `_extract_agents_list`, the card parsers, `_read_applied_tasks`/`_write_applied_tasks`, and the F156 legacy ledger-file importer (`:54-66`, `:1817-1916`) (US-048, not in the design's §10 list);
  - the placeholder emitters for `model_change_same_tier` and `description_update`, `_PLACEHOLDER_PROPOSED_VALUE` and `_MIN_SUCCESS_RATE_FOR_COST_OPT` (US-049).
- **API routes:** `POST /api/harness/prescriptions/{rx_id}/approve|reject`, `_run_command`, the `_resolve_internal_user_id` alias and the imports that become unused (`api/harness.py:27-84`); `api/harness_commands.py` in full, including `_governed_verdict` (`:277-285`), `_already_handled` and `_find_task_by_rx`. The route manifest drops 2 routes (US-047, after US-003).
- **`api/workspaces.py`:** the `full_auto`/`manual` modes, the Biweekly and Monthly schedules, and the `enabled` default in the HARNESS settings (US-016).
- **`owner_only.py`:** `OWNERS_OWN_DECISION` (`:89`), its comment (`:86-88`), its branch in `human_driven` (`:217`) and docstring mention (`:212-214`), its `__all__` entry (`:545`) and its module docstring mention (`:16`). The test at `test_prd256_owner_actions_wait_for_the_click.py:181` is replaced by a stricter one, not weakened (US-047, after US-004).
- **`config.py`:** `HARNESS_AUTO_APPLY_MAX_RISK_STANDARD`, `HARNESS_AUTO_APPLY_MAX_RISK_FULL` and `HARNESS_HIGH_PRIORITY_RISK` (US-016).
- **Ledger:** the `held` state (US-051).
- **`handlers_harness.py`:** the baseline-file reader in `platform_harness_history` (US-023). The files on tenant volumes go only after US-006 and a verified import (US-059).
- **`auto_cadence.py`:** the Sunday 02:00 and `tag=harness` review text (US-023).
- **Docs:** the auto-apply and `/approve` sections of `docs/platform-actions/self-management-harness.md` (US-023a, US-047).
- **Skill seed:** the Sunday-cron, queued-card and "model swaps" HARNESS text (US-024/US-025, US-060/US-061); the DELEGATE-lane text and the "Gerard" mentions (US-065/US-066).
- **Tile:** the reads of `data.prescriptions` and `harness.iteration`, and the process-global Signals figure (US-022).
- **Settings tab:** the "Enable HARNESS" switch, the Full Auto and Manual select copy, the "risk 3+ queued" and "applied automatically" text, and the Run Schedule select with its Biweekly and Monthly options (US-017).
- **Tests** (each PR maps every deleted test to its replacement, AGENTS.md:121): those that set REGRESSION by hand or monkeypatch `_phase_collect`/`_phase_diagnose` (US-014a); the `_detect_auto_applied_regressions` cases (US-014b); `test_harness_governance_gate.py`, the escalation and card-apply cases of `test_harness_self_management.py`, Half 2 of `test_kill_switch_halts` and the `_apply_approved_board_tasks` case of the F156 ledger test (US-016); `test_harness_tool_failure_signal.py` (US-019); `test_harness_commands.py`, the approve, reject and resolver cases of `test_harness_api.py`, and `test_prd179_harness_actuation.py` (US-047); the rest of `test_harness_self_management.py`, both `test_f156_*` files, the `harness_ledger` fixture and the `_auto_apply_prescription` cases of the power-mode and routing-rule tests (US-048). `test_w1s9_idle_in_tx.py` is rewritten, never deleted.
- **Local `replay.py`:** the `action_claims` import, None-means-backed scoring and the OpenRouter arm ids other than `gemini-0.7` (US-030, US-033).
- **`AUTO-EVAL.md`:** the stale sections on arms A–D, the OpenRouter key in c1 and the old claims source (US-046).

## 15. Risks

1. **Real data turns on prescriptions.** P1 is report-only for everyone; P3 needs `propose`, the global switch and an owner's click; c1 stays report-only through WebSummit.
2. **c1 is fake traffic driven by nights.** Org verdicts there mostly measure whether a night ran. Expect a high INSUFFICIENT_DATA share.
3. **Eval saturation.** The Lab may produce no IMPROVE before WebSummit (§9 power). Set v2 depends on Gerard's authoring time, and the valid failing held-out count may be below 6.
4. **Global blast radius.** An accepted edit reaches every tenant. Controls: section limits and the chat-surface-only rule; the held-out gate and the critic; Gerard's PR review; the eval and production post-ship monitor; the templated revert.
5. **NON-INFERIOR cannot see mild degradation** (15–50% false-pass for rows degraded to p=0.6), output quality, brand voice or non-chat modes. That is why the sign-off line exists.
6. **Global row swap on the local stack.** A crash leaves the candidate live, and it survives a restart because `content_hash` is untouched, so the boot seeder's hash check sees nothing to refresh (`core/seeds/seed_builtin_skills.py:142-153`, run at boot and on every c1 seed). Controls: the orphan-snapshot guard, the preflights' hash and skill-set checks, daytime-only windows.
7. **Replay traffic is logged as production.** `telemetry_source` comes only from caller_context (`telemetry.py:173`). Replay feeds EdgeBuilder and memory until the P6 tag exists. `phase.sh` clears c1 memory in the window; EdgeBuilder pollution of the local graph remains.
8. **Claim-error attribution.** Templates and F187 lines join per row. `UNBACKED_LOGGED` joins only if `req=` survives (US-032). Auto's own-turn "account of calls" case leaves no template (`claims_in_place.py:55-64`).
9. **ATOM turns skip the skill section,** about 12% of turns [I], measured in US-034. Those failures cannot be fixed on this surface.
10. **Late or stale clicks.** Handled by the apply action (expiry, prior value) and the daily sweep. Residual: a manual edit and a click in the same moment, between the prior-value read and the write. The ledger CAS makes the grant single-use, but the handler does not lock the agent row [I]. A Blocked evidence card turns a click into a re-queue with no change (§11-Q30).
11. **Local stack stability.** F105, F408, F409, F416 and the N17 build blackout. About 4–7 h of probes per Lab day, plus reset waits, adds load, and a mid-probe freeze wastes spend.
12. **Max allowance.** Usage limits, not dollars, may cap sessions. The Lab uses a separate account from nights, which needs Gerard's one-time login.
13. **Improver prompt injection** from trajectories (owner text, an injection row). Controls: EVIDENCE_BOUNDARY, no Bash or network, `--safe-mode` and deny rules, the critic, human review.
14. **Booking changes look like cost steps.** Subscription calls book $0, and a transcript dedupe fix would shift figures. CLI and subscription agents are judged on tokens, and the booking method is versioned.
15. **automatos-skills divergence.** The platform seed is v2.3.3. Skills `origin/main` is v2.4.0 and already holds the slim body and `team/auto-ops/SKILL.md` (the local `main` ref is stale at v2.2.0), so a sync from main would ship the slim skill. The v2.4.3 body still carries the stale HARNESS runbook ("HARNESS runs Sunday 02:00 UTC", `tag=harness`), the owner's name and the DELEGATE lane, and `team/auto-ops/SKILL.md` promises "model swaps" and a "Sunday cron" and names the owner 11 times. US-024, US-060 and US-065 therefore branch from `skill/auto-v2.3.3-real-tools`, and US-069 and US-071 rebase the slim bodies on them before any probe or ship. `sync-skills.py` reads a working tree, so validation uses an export of the branch Gerard names.
16. **Harness2 maturity.** One 229-line test file; reported gains are on GLM-5 and Gemini executors, not Haiku 5.5.
17. **OpenRouter embedding balance.** If it runs dry mid-probe, trajectories become invalid without an error, and nights fail too. Embeddings are booked at $0 today, so the spend stop cannot see them until they are priced (§11-Q29). Controls: the operator's floor in both preflights, VOID on any embedding error, and the unpriced-usage refusal.
18. **Model and surface transfer.** New tenants' Auto defaults to `google/gemini-2.5-flash` through OpenRouter (`core/llm/defaults.py:13-14`), and owners can change it; Lab evidence is Haiku 5.5, chat only. Controls: the chat-surface-only rule, the PR's measured-surface line, the kept `gemini-0.7` arm, §11-Q13 and Q25.
19. **Build drift and caches** bias the noise estimate low and inflate false accepts. Controls: pinned tags, a same-day-only gate, the per-reset complexity-key flush (US-029) in calibration with the without-flush floor used by the gate, the routing-affinity reset.
20. **Eval row drift.** A row whose expected tool was renamed becomes impossible, and improvers would chase it. Control: the row-validity check (US-034).
21. **Tool removal hides a broken connection.** Controls: error classification into "reconnect" (`classify_tool_error`, reading stored error codes and, for older rows, messages), the Playbook/Mission dependency check, report-only if that check is not built.
22. **Process friction.** AGENTS.md's maintainer-issue rules could block P1 or P3 late in a three-week window. Control: open the issues at P1 start.
23. **Schedule and PR count.** Fifteen small P1 and P3 PRs (D3) mean fifteen CI runs and more merge steps. Nine of them touch `harness_service.py` (PR-1B, PR-1D, PR-1E, PR-1F, PR-3A, PR-3B, PR-3C, PR-3E, PR-3F) and merge one at a time, each rebased and seam-reviewed, so that chain sets the pace. Five PRs run over ~400 lines (§4 "PR size"). PR-3A needs PR-1E, PR-1F, PR-1G and its maintainer issues, and PR-3B needs US-006. Reset waits lengthen every probe day (§13).
24. **No rollback detection between P1 and P3.** US-014b deletes the baseline-based `_detect_auto_applied_regressions`, and US-056 brings rollback back on the ledger. In between, a change applied through the legacy `/approve` path (only if `HARNESS_SELF_MANAGEMENT_ENABLED` is on) has no rollback check. Control: US-026a records the prod value privately, and Gerard turns the flag off before PR-1D merges (PR-1D carries US-014b) or records that he keeps it on.
25. **D5 declined.** If Gerard does not amend the eval rule, every Needs D7 story becomes his own operator step, and P2, P4 and P5 slow to his available hours.
26. **`config.py` grows.** It is 2,490 lines and gains about 14 net setting lines. The code-shape gate warns rather than fails, and settings may live nowhere else.
27. **D4 stops a night that has no spend values.** Once US-038 lands, the pre-night eval refuses without the operator's limit and factor, and the US-080 monitor loses its input. Control: US-041's owner step sets them before the first night after LOCAL-3 or LOCAL-6.

## Related

- **Design:** `<workspace>/.claude/HARNESS-V2-DESIGN.md` (rev 2, 18 Oct). Its simulation scripts (`org_stats.py`, `consec_sim.py`, `frozen_sim.py`, `gate_sim3.py`; standard-library math with fixed seeds, no data) are committed alongside this PRD at `docs/PRDS/prd263-spec/sims/`, and US-008's tests re-derive the pinned figures.
- **Findings** (`docs/testing/FINDINGS-LEDGER.md`): F105 (event-loop freeze), F151 (the held card path), F156 (the ledger in the database), F187 (the claim check), F240 (the cron weekday off by one), F305, F316 and F418 (owner data), F408 and F409 (night 13 backend freezes), F416 (night 14 OOM). The eval set's rows come from F108–F381; set v2 adds F385–F449 (nights 12–17) and F382–F383 if they carry a customer ask, never code-scanning findings such as F384.
- **PRD-121** (`docs/PRDS/121-HARNESS-SELF-OPTIMIZING-ORG-LOOP.md`): the org loop this PRD repairs.
- **PRD-142 Wave 4** (`PRD-142-WAVE4-SELF-LEARNING-HARNESS.md`): HARNESS kept, gated and sequenced last.
- **PRD-187 S5** (`PRD-187-PHASE2-WAVE-1-MEMORY-UNSPLIT.md`): dropped the empty `harness_prescriptions` table and the MemoryKnowledgeSystem imports in harness.
- **PRD-193** (`PRD-193-PHASE2-WAVE-2-APPROVE-GRANT-RESUME.md`): the tool-level approve → grant → resume rail that `platform_harness_apply` uses.
- **PRD-212** (draft): its HARNESS stories S1–S3 overlap this PRD (§11-Q26).
- **PRD-226** (`PRD-226-AUTO-MANAGER-DOCTRINE.md`): `DOCTRINE_ANCHORS`, `CONTRACT_OPENER` and the `test_prd226` pins.
- **PRD-231** (`PRD-231-AUTO-CONTEXT-DIET.md`): the identity / operations split whose platform half iteration 0 decides.
- **PRD-232** (`PRD-232-INTENT-GRAPH.md`): the synthetic signal rows EdgeBuilder keeps reading.
- **PRD-245** (`PRD-245-SESSION-TOOLS-BRIDGE.md`): the session holds; with the CLI host lane they are today's callers of `expire_pending_grants` (`services/session_plans.py:249`, `services/cli_host_service.py:1851`).
- **PRD-256** (`PRD-256-AUTO-RECEIPTS-GATES-CONTRACTS.md`, merged as #1048): D1 (the owner's click), receipts and `claims_backed`, the eval pack, the owner's-decision card (`approval_cards.py`, P256-FIX-N14-7), and the kit D7 that `AUTO-EVAL.md:5` widens.
- **PRD-258** (`PRD-258-AUTO-LIVE-VOICE-LATENCY.md`): the house model for this PRD; its revision `prd258_voice_timings` is today's head.
- **PRD-262** (branch `feat/prd-262-w0-measure`): pull-first images that give Lab days a pinned `AUTOMATOS_IMAGE_TAG`; its US-006 locks the PRD-58 worker down.
- **Commits and PRs:** `d5ea08e67` (deleted `action_claims`); #1138 (a squash body with `[skip ci]` skipped main's CI).
- **External:** Harness2 (google-research/harness2), <https://github.com/google-research/harness2>, arXiv 2610.14224, assessed at commit `1cfe254e` (13 Oct 2026); Meta-Harness, arXiv 2603.28052; ACE, arXiv 2510.04618; RRSI, <https://github.com/google-research/rrsi>; AHE, arXiv 2604.25850; GEPA, <https://github.com/gepa-ai/gepa> (assessed, not adopted).

