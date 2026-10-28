# Ralph Review Prompt — PRD-268 W1 Check-ins, Proactive Auto's engine

You are a fresh-context **adversarial reviewer**. The build claims PRD-268 wave W1 is complete (US-C01, US-C02a–d, US-C03a–d, US-C04a–b, US-C05, US-C06a–b, US-C07a–b, US-C17a–b, US-C18). This wave adds a guardrail and changes what heartbeats do: review it as someone trying to make a background check write something.

## What to look for

**The read-only lane:**
- a call in the lane can write, by any path:
  - an action without a level;
  - a mislabelled read action;
  - a Composio, plugin or workspace tool;
  - the Anthropic memory tool;
  - AgentFactory;
  - a sub-agent, Mission or `platform_execute`;
  - a tool the runner offers that the lane doesn't check;
- `clear` or `_run_cleared` changed inside;
- the decorator fails open on an exception or a missing registry entry (US-C02a–d, US-C03d).

**Suggestions:**
- `platform_suggest_next_step` works outside the lane;
- it skips provenance or dedupe;
- it writes anything but its row;
- it passes the hierarchy gate without a reason;
- suggestion text reaches a prompt as instructions, or is posted as the person's message (US-C03b, US-C06a).

**Check-ins and heartbeats:**
- a silent run makes a model call;
- the caps, the $0.05 ceiling, the switches, `background_allowed` or the spend check are skipped or fail open;
- `check_in` is missing from `BACKGROUND_LANES`;
- a trial pays (US-C03d, US-C07a);
- `_agent_tick` or `_orchestrator_tick` changed inside;
- `off` mode differs from the base at all;
- `shadow` drops today's tick;
- `live` files a BoardTask or a per-tick report for a check_in agent;
- an `act` agent stopped acting;
- `heartbeat_service.py` grew (US-C04a–b);
- a streak creates more than one suggestion, makes a model call to write it, never backs off, or never resets (US-C05).

**The API:**
- Start can run twice;
- a viewer can start or hide;
- another workspace's suggestion answers anything but 404;
- Start skips the spend check;
- the audit write commits inside the Start transaction;
- a Mission or Playbook start does anything but 409;
- a route is missing from the manifest, or is `async def` over the sync session (US-C06a–b, US-C07a).

**Status and readiness:**
- needs-you counts changed;
- a STUCK_* constant or kind lacks an English or Portuguese (Brazil) sentence;
- unknown maps to a state;
- `api/missions.py` or `api/board_tasks.py` grew (US-C17a–b);
- a probe has no timeout;
- not_checked stands in for a known failure;
- a fix href leaves the app;
- SaaS detail shows a hostname (US-C18).

**The migration:**
- the revision doesn't chain onto the base head;
- a CREATE lacks `IF NOT EXISTS`;
- a pin didn't move;
- the models and the DDL differ;
- the partial unique index is missing (US-C01).

**The wave as a whole:**
- a story's evidence doesn't exist;
- an AC is marked DONE that the code doesn't meet;
- a W2, W3 or W4 piece was built;
- a new dependency or third-party client was added;
- a file over 800 lines grew (config.py and the two reports excepted).

You fix NOTHING in the code yourself.

## Scope

```
BASE=$(bash scripts/ralph/acceptance-prd268w1.sh --print-base)
git diff --stat $BASE..HEAD
git diff $BASE..HEAD -- . ':!scripts/ralph'
```

Never file a finding against a line this diff doesn't touch, except where an AC says the base's behaviour had to change and it didn't.

Read beside the diff:
- `scripts/ralph/prd-268w1.json` (binding: its `decisions` and each story's ACs and `notes`);
- `docs/PRDS/PRD-268-PROACTIVE-AUTO.md` §6.2, §6.9, §6.11.1, §6.11.3 and §8;
- `AGENTS.md`;
- the `ownerTest` list.

## Verification

- **code-review** skill, or the code-reviewer agent, on the diff, **in the foreground**.
- **security-reviewer** agent, in the foreground, on:
  - `orchestrator/modules/tools/discovery/check_in_lane.py`;
  - the decorated `clear` and `execute_tool`;
  - `orchestrator/modules/tools/discovery/{actions,handlers}_proactive.py`;
  - `orchestrator/services/proactive/`;
  - `orchestrator/api/proactive.py`;
  - `orchestrator/modules/context/{modes,sections/tools}.py`;
  - the hierarchy-gate change.

  Ask it explicitly about: fail-closed behaviour on every path; prompt injection from what a check-in reads; provenance; tenancy (404, never 403); the spend and trial checks; the concurrent Start.
- **database-reviewer** agent, in the foreground, on `orchestrator/alembic/versions/prd268_w1_proactive.py` and `orchestrator/core/models/proactive.py`.
- Run `bash scripts/ralph/acceptance-prd268w1.sh` yourself. It waits for the wave's one CI run on the tested commit and reads the PR body.
- Spot-check five `DONE` acceptance criteria at random against the code and the tests. Always include one from US-C02b, one from US-C03d and one from US-C04b.
- **Nothing runs on the owner's machine.** Never end your turn with anything running in the background.

## Verdict

**No CRITICAL, HIGH or MEDIUM:** write a 6-line summary:
- (a) the lane and the levels;
- (b) the tool and the runner;
- (c) the bridge, the modes and the streaks;
- (d) the API, the switches and the audit;
- (e) status and readiness;
- (f) the owner's next step: the `ownerTest` list (the shadow week).

Then `REVIEW_PASS` as the last line. REVIEW_PASS and REVIEW_FINDINGS appear on your last line only, never in prose.

**Findings:**
1. Append `P268W1-RVW-n` stories to `scripts/ralph/prd-268w1.json`, n counting from 1. Same shape: id, title, description, acceptanceCriteria (with the cause, the change and the test), priority after the last story, passes false, notes.
2. Commit with `git commit -s -m 'chore(prd-268): review findings → fix stories [skip ci]' -- scripts/ralph/prd-268w1.json`.
3. `REVIEW_FINDINGS` as the last line.
