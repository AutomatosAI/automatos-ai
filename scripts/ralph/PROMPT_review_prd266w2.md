# Ralph Review Prompt — PRD-266 W2 Push, the Automatos Go backend

You are a fresh-context **adversarial reviewer**. The build claims PRD-266 wave W2 is complete (US-010a, US-010b, US-011a, US-010c, US-011b, US-010d, US-010e, US-009, US-012a, US-012b, US-014a, US-013, US-014b, US-019a, US-019b, US-019c). This wave decides who is told what on their phone, sends data to a third party (Expo) and signs what the phone may act on: review it as an attacker and as a confused member would. Find where:
- the Expo client logs a token, a title or a body, sends to any host but `exp.host`, retries a 4xx, raises out of `send`/`receipts`, trusts an unvalidated answer, or another module in the wave speaks HTTP (US-010a);
- the revision does not chain onto W1's head, a CREATE lacks `IF NOT EXISTS`, a pin did not move, the model and the DDL describe different tables, or a digest post is not unique per (workspace, person, local date) (US-010b);
- a push carries a question's text or options when redaction is on, carries text for any non-question type, truncates an option, lacks the AI badge, or the signature's canonical form differs from K9 or from the vector file; the GDPR export still carries a token hash, the pairing code hash, the push key or the Expo token (US-011a);
- a local install pushes anything before the operator agreed, the consent route answers on SaaS, a member who is not owner/admin can change it, or discovery's `push.expo` disagrees with the stored answer (US-010c);
- a viewer's phone gets a needs-an-answer push, a removed member's phone gets anything, another workspace's phone gets anything, a revoked/web/push-off device is sent to, SaaS sends to a paired row or a local install to a clerk row, a person's push switch is ignored, `hand_over` does work when nothing is eligible, or the delivery runs inline on the event loop (US-011b);
- `dispatch` is still over 50 code lines, lost a behaviour (the credit-outage hold, auto-reporting, the in_app default, quiet hours, silent, external, channel), stopped calling a helper a test patches, lets a push row cost the bell row, pushes an event outside the five, pushes in quiet hours for normal severity, or one of the four dispatcher tests was edited (US-010d);
- the receipts job disables the wrong device, reads tickets too young, keeps tickets forever, registers with pickled args, or raises (US-010e);
- an Expo token moves between two different people's installs, a clerk row gets a token hash, push turns on without a token, or a signed-out phone keeps receiving pushes (US-009);
- `review_pending` fires twice for one entry, fires for review → review or a rolled-back move, misses a writer (the sweeper's raw UPDATE), fires for `_reopen_review` or a Mission's own card (`source_type 'orchestration'`, already announced by `mission_plan_ready`), emits SQL in `after_commit`, or the listener is registered twice (US-012a);
- `auto_message` fires for the person's own action (`question_answer`, a click result), for a non-Message, more than once per chat in ten minutes (the digest excepted), reaches anyone but the chat's owner, or `deliver_background_message`'s return value changed; the bell's `chat` case or its fixture is missing (US-012b);
- a `push` row is accepted for an event outside the five plus `daily_digest`, GET hides the defaults, or `bulk_update_preferences` was touched (US-014a);
- a person gets two digests on one local date (several phones, two ticks, a race), a digest posts outside its window or in the wrong timezone or on a DST day at the wrong hour, the switch off still posts, or a failed post burns the day (US-013);
- a non-owner/admin can set `push_redact_questions`, or `api/workspaces.py` was touched (US-014b);
- Studio asks the Expo question more than once, mints the code before the answer, shows the switch on SaaS, shows another person's push switch, or hides whose phone it is from an admin (US-019a/b); a QR encoder appears while K17 is not approved (US-019c);
- a story's evidence does not exist, an AC is marked DONE that the code does not meet, a W3 piece or US-020 was built, a new dependency or a second third-party client was added, a file over 800 lines grew (config.py's settings and the generated reports excepted), or W1's kit files changed.

You fix NOTHING in the code yourself.

## Scope

```
BASE=$(bash scripts/ralph/acceptance-prd266w2.sh --print-base)
git diff --stat $BASE..HEAD
git diff $BASE..HEAD -- . ':!scripts/ralph'
```

The base is W1's tip while this wave is stacked on it (W1's own diff is reviewed in W1's PR), else the `origin/main` merge-base. Never file a finding against a line this diff does not touch, except where an AC says the base's behaviour had to change and it did not. Read, beside the diff: `scripts/ralph/prd-266w2.json` (binding: its `decisions` and each story's ACs and `notes`), `docs/PRDS/PRD-266-AUTOMATOS-GO-BACKEND.md` (§3, §4.4, §4.5, §4.10, §8, §11), `AGENTS.md`, and the `ownerTest` list.

## Verification

- Run the **code-review** skill (or the code-reviewer agent) on the diff **in the foreground**, and the **security-reviewer** agent in the foreground on `orchestrator/services/push/`, `orchestrator/api/mobile_push.py`, `orchestrator/services/mobile/devices.py`, `orchestrator/services/mobile/discovery.py`, `orchestrator/services/gdpr_service.py`, `orchestrator/core/services/notification_dispatcher.py`, `orchestrator/services/notification_events/` and `orchestrator/api/notifications.py`. Ask the security reviewer explicitly about: who receives each push type (roles, membership at send time, tenancy), the signature and its canonical form, what text leaves the install (redaction, generic lines), the local consent gate, the Expo token handover, logging of tokens and keys, the GDPR export, and the outbound client (host, TLS, retries, validation).
- Run the **database-reviewer** agent in the foreground on `orchestrator/alembic/versions/prd266_w2_push.py`, `orchestrator/core/models/mobile_push.py` and the claim query in `orchestrator/services/mobile/digest.py`.
- Run `bash scripts/ralph/acceptance-prd266w2.sh` yourself (it waits for the wave's one CI run on the tested commit and reads the PR body and base).
- Spot-check five `DONE` acceptance criteria at random against the code and the tests; always include one from US-011b, one from US-010d and one from US-012a or US-013.
- No CI lane runs `next build` for this PR: read every new `'use client'` boundary for what a build would refuse (a hook in a server component, a server-only import in a client file).
- **Nothing runs on the owner's machine** (no pytest, alembic, npm, vitest, tsc, next, docker, server, browser). Never end your turn with anything running in the background.

## Verdict

- **No CRITICAL/HIGH/MEDIUM:** a 7-line summary: (a) the sender and the tables: the one client, validation, the revision and the pins; (b) payload and signature: the types, the badge, redaction, K9's canonical form and the vector, the GDPR export; (c) consent and fan-out: the local gate, recipients and roles at send time, eligibility, the best-effort hand-over; (d) the dispatcher: the rewrite's behaviour kept, push as external, the five events only, the row id; (e) receipts, registration and the two events: DeviceNotRegistered, the token handover, review_pending's entries, auto_message's origins and coalescing; (f) preferences, the digest and redaction: the defaults, the switch, once per local day across timezones and DST; (g) Studio and the owner's next step: the consent question and switch, push states, then the frontend image build (no CI `next build`) and the `ownerTest` list.

  Then `REVIEW_PASS` as the last line (the words REVIEW_PASS and REVIEW_FINDINGS appear on your last line only, never in prose).
- **Findings:** append `P266W2-RVW-n` stories to `scripts/ralph/prd-266w2.json` (n from 1; same shape: id, title, description, acceptanceCriteria with the cause, the change and the test, priority after the last story, passes false, files, notes), commit `git commit -s -m 'chore(prd-266): review findings → fix stories [skip ci]' -- scripts/ralph/prd-266w2.json`.

  Then `REVIEW_FINDINGS` as the last line.
