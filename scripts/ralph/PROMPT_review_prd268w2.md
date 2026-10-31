# Ralph Review Prompt — PRD-268 W2 Delivery, Proactive Auto reaches people

You are a fresh-context **adversarial reviewer**. The build claims PRD-268 wave W2 is complete (US-C19a–d, US-C10a–b, US-C08a–b, US-C09a–b, US-C20a–c, US-C21a–f, US-W01a–b, US-W02, US-W04, US-W05). This wave delivers suggestions and the brief to people, adds a push type, changes approval text and bumps the SDK. Review it as someone trying to make Auto post twice, push when it shouldn't, leak across workspaces, or let a theme reach a protected surface.

## What to look for

**The brief (US-C19c, C10a, C10b):**
- two briefs for one person and local date, from any mix of devices, tabs, the schedule and an open;
- a failed build that keeps the claim;
- a local_date outside one day of UTC accepted;
- a viewer's brief with questions or approvals;
- counts.needs_you differing from needs_you_counts;
- a fold at 6;
- a second model call;
- server copy missing a language;
- an open brief announced as auto_message;
- a web-only person who gets none;
- the cadence Daily Brief still reachable;
- a route that is async def over the sync session, or missing from the manifest.

**Message parts (US-C19b):**
- post_background_message's owner check changed (another member's chat targeted);
- a part type outside the allow-list saved;
- an oversized part saved;
- a call without parts no longer writing exactly one text part;
- the rewrite left over 50 code lines.

**get_brief (US-C19d):**
- it writes anything;
- it is not registered as read;
- platform_executor.py grew.

**Delivery and the live lane (US-C08a, C08b):**
- delivery in off or shadow;
- a suggestion posted as the person's message, or followed as instructions;
- an auto_message raised for it;
- more than one bell row per run;
- a bell row for a can_wait-only run or past the cap;
- suggestion_changed missing on create, start, hide or expire;
- a payload carrying suggestion text;
- another workspace's event forwarded.

**The push (US-C09a, C09b):**
- SUGGESTION_PUSH_ENABLED defaulting on;
- a push while it is off;
- a push to a device without the category;
- more than one push per run, or more than two a day;
- a push in quiet hours;
- a can_wait push;
- a new signed field, or the field order changed;
- the title on the lock screen without the workspace switch;
- registration letting a caller set another device's categories;
- discovery listing suggestion_push while off;
- the other three push types changed.

**Approvals (US-C20a, C20b, C20c):**
- Always allow in any form (route, button, confirm sheet, policy write, 'always_allow' in answers);
- new button words;
- a preview carrying raw params past the card's digest;
- a rule sentence that is machine text or missing a language;
- the view route open to a viewer, or 403 instead of 404 across workspaces;
- format_for_frontend changed inside;
- result_formatter.py grown;
- evaluate_board_task_approval still over 50 lines;
- api/board_tasks.py touched;
- blocked-reason.ts losing the grant id;
- keys 1/2/3 acting without focus, or from a document listener;
- the cards no longer protected surfaces;
- a preview rendered as HTML.

**SDK and defaults (US-C21a–f):**
- --running missing from a tone, or Classic not drawing working in info;
- globals.css grown beyond the token lines;
- lookToStudioTokens changing MOBILE_TOKEN_NAMES;
- a parity test loosened rather than extended;
- a status through --accent or by colour alone;
- data-ai-generated missing on Auto's words, or put on controls;
- BriefVoice auto-playing, outside its protected surface, or a theme able to speak;
- the calibration hooks making a request;
- the SDK version not 1.3.0 everywhere, or THEME_HOST_VERSION bumped;
- the d.ts not matching the runtime;
- surface.test.ts weakened;
- a new npm dependency (the axe AC must be OWNER-marked, K18).

**The web (US-W01a, W01b, W02, W04, W05):**
- message.tsx touched;
- a non-import change to chat.tsx;
- the pinned card outside the Auto thread, or shown while a Mission suggestion is unanswered;
- the thread card's state not surviving reload;
- useBoardEventStream still over 50 code lines, or its tests edited;
- ActivityPage, MissionsBody or BoardCardFace touched inside;
- the brief route called more than once per local date per tab, or called with no workspace, or navigating or playing sound on its own;
- a calendar row;
- credits on anything but the web card;
- Studio rules broken (orange, hex, glass, rounded-3xl, a second primary).

**The migration (US-C19a):**
- the revision doesn't chain onto W1's head;
- a step without IF EXISTS / IF NOT EXISTS;
- a third change beyond K6's two;
- the downgrade unsafe;
- a pin didn't move;
- the models and the DDL differ.

**The wave as a whole:**
- a story's evidence doesn't exist;
- an AC is marked DONE that the code doesn't meet;
- a W3 or W4 piece was built (K23);
- a `*268w1*` kit file changed;
- a file over 800 lines grew (the K3 exceptions aside);
- a K19 pin file lost a test.

You fix NOTHING in the code yourself.

## Scope

```
BASE=$(bash scripts/ralph/acceptance-prd268w2.sh --print-base)
git diff --stat $BASE..HEAD
git diff $BASE..HEAD -- . ':!scripts/ralph'
```

The base is W1's tip while W1 is unmerged (the wave is stacked): never file a finding against W1's own diff. Never file a finding against a line this diff doesn't touch, except where an AC says the base's behaviour had to change and it didn't.

Read beside the diff:
- `scripts/ralph/prd-268w2.json` (binding: its `decisions` and each story's ACs and `notes`);
- `docs/PRDS/PRD-268-PROACTIVE-AUTO.md` §6.4, §6.5, §6.11, §6.12 and §12;
- `AGENTS.md` and `frontend/DESIGN_SYSTEM.md` §22;
- the `ownerTest` list.

## Verification

- **code-review** skill, or the code-reviewer agent, on the diff, **in the foreground**.
- **security-reviewer** agent, in the foreground, on:
  - `orchestrator/services/chat_messenger.py` and `services/chat_parts.py`;
  - `orchestrator/services/brief/` and `orchestrator/api/proactive_brief.py`;
  - `orchestrator/modules/tools/discovery/{actions,handlers}_brief.py`;
  - `orchestrator/services/proactive/delivery.py` and the `suggestion_changed` lane in `services/board_events.py`;
  - `orchestrator/services/push/` (payload, fanout, suggestion_rules) and the device registration's push_categories;
  - `orchestrator/modules/policy/rule_sentences.py`, `modules/tools/formatting/approval_view.py`, `api/approval_views.py` and `services/board_approval.py`;
  - the approval widgets, `lib/chat/approvals.ts`, `components/theme-host/protected-surface.tsx`, `components/voice/brief-voice.tsx` and `lib/theme-sdk/proactive.ts`.

  Ask it explicitly about:
  - tenancy (404, never 403);
  - roles (viewers: D16);
  - the brief's one-per-day claim under concurrency;
  - what a lock screen can show;
  - the signature;
  - model-written text reaching a prompt or a person's message;
  - the preview's redaction;
  - the protected surfaces against theme CSS and code.
- **database-reviewer** agent, in the foreground, on `orchestrator/alembic/versions/prd268_w2_delivery.py`, the model changes, and the claim's queries in `services/brief/claim.py`.
- Run `bash scripts/ralph/acceptance-prd268w2.sh` yourself. It waits for the wave's one CI run and the browser lane on the tested commit, and reads the PR body and its base.
- Spot-check five marked acceptance criteria at random against the code and the tests. Always include one from US-C10a, one from US-C09b and one from US-C20c.
- **Nothing runs on the owner's machine.** Never end your turn with anything running in the background.

## Verdict

**No CRITICAL, HIGH or MEDIUM:** write a 7-line summary:
- (a) the brief: claim, route, strings;
- (b) message parts and get_brief;
- (c) delivery and the live lane;
- (d) the push, shipped off;
- (e) approvals: the view, the sentences, the cards;
- (f) SDK 1.3.0 and the web surfaces;
- (g) the owner's next step: the `ownerTest` list (images.yml first, then live for Gerard).

Then `REVIEW_PASS` as the last line. REVIEW_PASS and REVIEW_FINDINGS appear on your last line only, never in prose.

**Findings:**
1. Append `P268W2-RVW-n` stories to `scripts/ralph/prd-268w2.json`, n counting from 1. Same shape: id, title, description, acceptanceCriteria (with the cause, the change and the test), priority after the last story, passes false, notes.
2. Commit with `git commit -s -m 'chore(prd-268): W2 review findings → fix stories [skip ci]' -- scripts/ralph/prd-268w2.json`.
3. `REVIEW_FINDINGS` as the last line.
