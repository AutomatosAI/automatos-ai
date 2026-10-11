# Ralph Review Prompt — PRD-258 Auto Live, Wave 1 (measure the voice turn)

You are a fresh-context **adversarial reviewer**. The build claims PRD-258 Wave 1 is complete. Find where:
- a timing is invented, mis-measured or measured from the wrong point (the 'One moment.' ack counted as Auto's first frame; `total_ms` semantics changed; a superseded turn writes a made-up number);
- a live turn can write zero or two `voice_turns` rows, or telemetry can fail a turn;
- the transcript or reply text is stored (the lengths-only rule);
- the migration is not the single new head, does not survive a `create_all`-built database, or a head pin was not moved;
- the events webhook can return non-2xx because of the get-call fallback, or verifies HMAC after doing work, or overwrites a stored latency with None;
- a re-arm can point the agent at a wrong or empty host, leaks the key in a log or error, or the agent name ignores the edition;
- the runner writes a file, ships data, uses a new dependency, or its frames don't match what the WS handler reads;
- `voice_retell.py` grew, a shim re-exports a moved helper, or anything changed latency or Auto's behaviour;
- a story's evidence does not exist.

You fix NOTHING in the code yourself.

## Scope

```
BASE=$(bash scripts/ralph/acceptance-prd258w1.sh --print-base)
git diff --stat $BASE..HEAD
git diff $BASE..HEAD
```

Read, beside the diff:
- `scripts/ralph/prd-258w1.json` (binding, with its `decisions` and each story's `notes`: the loop's choices for the PRD's ambiguities are the contract, not findings);
- `docs/PRDS/PRD-258-AUTO-LIVE-VOICE-LATENCY.md` (US-001..US-004, FR-1..FR-3);
- the `ownerTest` list in the JSON (the owner's checks; the loop must not claim them).

## Hunt list: every item is a confirmed-risk class

1. **Timings (US-001, FR-1).** First frame = the first content frame sent after `response_required` (the ack counts); first Auto frame excludes the ack; `tools_before_speech` counts only tool-call lines before Auto's first text, by the prefixes `StreamingHandler` really emits; `prep_ms` comes from this turn's own assistant message's `context_trace`, by the key the builder writes. A wrong anchor or an invented value = HIGH.
2. **One row per turn.** Trace every path: normal end, barge-in cancel, socket death mid-reply (the turn keeps draining), a binding that fails closed, a stream error. Zero or two rows on any path = HIGH. A raise out of telemetry = CRITICAL.
3. **Migration (D2).** One revision; `down_revision` = the base's head; `IF NOT EXISTS` / `IF EXISTS`; both pins moved; the ORM matches. Read the CI `Alembic from-zero` and `Schema-drift check` jobs' logs for it. A second revision or a broken chain = CRITICAL.
4. **Vendor latency (US-002, FR-2).** HMAC before anything; the subset keeps only the named keys and numbers; idempotent; the get-call fallback runs at most once per event, uses the existing timeout, logs without the key, and never changes the 2xx. Any path to a non-2xx from the fallback = HIGH.
5. **Arming (US-003, FR-3).** One `PUBLIC_API_HOST`; the endpoints helper is the single source of both URLs; re-arm PATCHes tuning + endpoints once; a PATCH failure stays non-fatal; the key never appears in a log line, exception message or response; the local name only on `AUTH_EDITION == 'local'`; the tunnel script commits no secret or domain and `bash -n` passes. A logged key = CRITICAL.
6. **Runner (US-004).** Frames match the WS handler (feed them through `parse_llm_request`); pure parts tested; no file writes; the `.gitignore` covers `results.md` and `asks*.json`; refuses clearly when Auto Live is off; no new dependency.
7. **Scope and conventions.** No latency change; no edit to `consumers/chatbot/service.py`; `voice_retell.py` ≤ its base length and no re-export shim; no `os.getenv` outside `config.py`; no hardcoded values; every commit DCO-signed; story commits carry ` [skip ci]`; no `node_modules`; functions ≤ 50 code lines and nesting ≤ 4 on touched code; no route-manifest change.
8. **Claims.** Nothing in the diff, the commit messages or the DONE marks may claim a run of the simulator, a tunnel, a Retell call, a browser check or local tests. A claimed run = CRITICAL.

## Verification

- Run the **code-review** skill (or the code-reviewer agent) on the diff **in the foreground**, and the database-reviewer agent on the migration. Any CRITICAL or HIGH either reports is a finding.
- Run `bash scripts/ralph/acceptance-prd258w1.sh` yourself (the gate reads the wave's one CI run on the draft PR). A green build with a red gate is a finding.
- Spot-check three `DONE` acceptance criteria at random against the code and the tests. Evidence that does not exist = CRITICAL.
- **Nothing runs on the owner's machine:** no server, docker, browser, pytest, database, tunnel or runner. CI is the evidence. Never end your turn with anything running in the background.

## Verdict

- **No CRITICAL/HIGH/MEDIUM:** a 5-line summary:
  - (a) the timing trace: `response_required` → first frame / first Auto frame → the one `voice_turns` row, and the test that proves the ack is not counted;
  - (b) the migration: revision id, its `down_revision`, both pins;
  - (c) the vendor-latency path and its fallback test;
  - (d) the re-arm PATCH body and the key-never-logged check;
  - (e) the owner's next step: the `ownerTest` list in `scripts/ralph/prd-258w1.json` (tunnel, arm, the baseline).

  The final line of your reply is the pass sentinel, alone.
- **Findings:**
  1. Append `P258-RVW-n` stories to `scripts/ralph/prd-258w1.json` (n continues from the highest existing `P258-RVW-` number), each with a title, `file:line` evidence, mechanical ACs, `priority` after the last story, `passes: false`.
  2. Commit with `git commit -s -m 'chore(prd-258): Wave 1 review findings → fix stories [skip ci]'`.

  The final line of your reply is the findings sentinel, alone.
- **Sentinels:** the pass sentinel is `REVIEW_PASS`, the findings sentinel is `REVIEW_FINDINGS`. They are read ONLY from the last line of your reply; never write either word anywhere else in the reply.
