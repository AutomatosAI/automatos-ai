# Ralph Review Prompt — PRD-266 W3 Store duties, the Automatos Go backend

You are a fresh-context **adversarial reviewer**. The build claims PRD-266 wave W3 is complete (US-016a, US-016b, US-016c, US-017a, US-017b, US-017c, US-018a, US-018b, US-018c, US-018d). This wave adds a user-input route, an outbound post to a credential URL, a list built from every key store, and three public frontend paths: review it as an attacker would. Find where:
- a report reaches a row the reporter cannot see (another workspace's message or deliverable, another member's private chat, a user message), answers 403 or leaks existence instead of 404, writes more than one audit row, or lets a caller with no person (API key, SDK key) file one; the route lacks the `members:read` gate or a viewer is refused; the rate limit runs after a write or edits `DEFAULT_LIMITS`; a duplicate notifies or posts again (US-016a);
- the owner's notice goes to anyone but the owner (a `user_id` NULL row every member sees), carries the excerpt or the note, or a dispatch failure loses the report; a line inside `NotificationDispatcher.dispatch` changed (US-016a);
- a local install can post to the moderation webhook (setting set or not); a non-https value posts; the posted text carries more than 200 characters of the reported content, or the reporter's name, email or id, or the workspace's name; the URL (or a driver error that can quote it) reaches any log; the post delays the response or runs for a duplicate; a new HTTP client or dependency appears; a test or doc holds a Slack-webhook-shaped URL (US-016b);
- the bell's drift guard lost an entry or a self-check, `content_report` routes nowhere, or the audit filter's prefix matches nothing the backend writes (US-016c);
- `ai_providers` decrypts, logs, returns or caches any key (the Retell key included), calls `resolve_provider_key` or `get_platform_workspace_key`, lists a byok_only provider on saas, misses the workspace's own key with its override on, lists one with its override off, double-lists a provider, or makes a query per provider (US-017a);
- the embeddings, reranker or session-runtime sources disagree with what the platform actually calls; Retell is listed when the platform switch is off, Retell is not armed, the workspace's Auto Live is off, or (saas) the plan lacks `families.voice` (enterprise excepted), or is NOT listed when all of them hold (local: no plan gate); the cap was made part of the rule; the rule copies the mint's logic instead of reading `modules/voice/live_settings.py` and `services/plan_tiers.py`; `api/voice_retell.py`, `modules/voice/` or `EmbeddingManager._load_provider` changed (US-017b);
- `GET /api/workspaces/current` 500s when the provider read fails, a pre-existing test of the route was edited, `api/workspaces.py` grew, the moved masking helper changed behaviour, or `get_current_workspace` passed 50 code lines (US-017c);
- a cancelled task appears in the schedule, a paused one lacks `paused: true`, the source makes more than one `execute()` or moved position, another source's items changed, or `services/activity_service.py` did not shrink (US-018a); the web still offers Pause on a paused task, shows it in Next up, or marks it by colour alone (US-018b);
- the AASA or assetlinks body is not exactly the kit's, is served with another content type or through a redirect, a malformed fingerprint passes, the middleware's public list opened more than the exact paths (a prefix like `/.well-known/(.*)` or `/go/(.*)`), `next.config.js` changed, or a custom URL scheme appears (US-018c);
- `/go/pair` makes any network call, stores anything, pairs anything, renders a query value as HTML or into an `href`, shows an unvalidated code or a non-http(s) server, lacks the shared frame, or edits `HOST_OWNED_ROUTES` (US-018d);
- a story's evidence does not exist, an AC is marked DONE that the code does not meet, a migration or head-pin edit appears, a W1 file (`services/mobile/`, `api/mobile_devices.py`, `api/well_known.py`, `core/auth/{pairing,bearer_registry,device}.py`, `core/models/mobile_devices.py`, `core/security/{rate_limiter,ip_bucket}.py`, `components/settings/phones-devices/`, `api/chat.py`, `consumers/chatbot/service.py`, `api/voice_retell.py`, `modules/voice/`) was created or edited, US-019 or a W2 piece was built, an `.env*` file was written, or a file over 800 lines grew (config.py's setting and the generated reports excepted).

You fix NOTHING in the code yourself.

## Scope

```
BASE=$(bash scripts/ralph/acceptance-prd266w3.sh --print-base)
git diff --stat $BASE..HEAD
git diff $BASE..HEAD -- . ':!scripts/ralph'
```

The base is the `origin/main` merge-base (the PRD is already on main through #1257). Never file a finding against a line this diff does not touch, except where an AC says the base's behaviour had to change and it did not. Read, beside the diff: `scripts/ralph/prd-266w3.json` (binding: its `decisions` K1–K20 and each story's ACs and `notes`), `docs/PRDS/PRD-266-AUTOMATOS-GO-BACKEND.md` (§3, §4.8–4.10, US-016 … US-018), `AGENTS.md`, and the `ownerTest` list. W1 (`feat/prd-266-w1-connect`) is a separate, parallel PR: the merge seams the kit names (K5) are expected, not findings.

## Verification

- Run the **code-review** skill (or the code-reviewer agent) on the diff **in the foreground**, and the **security-reviewer** agent in the foreground on `orchestrator/services/content_reports/`, `orchestrator/api/content_reports.py`, `orchestrator/services/ai_providers.py`, `orchestrator/services/activity_scheduled_task_items.py`, `frontend/middleware.ts`, `frontend/app/.well-known/`, `frontend/app/go/pair/` and `frontend/components/go-pair/`. Ask the security reviewer explicitly about: tenancy and existence leaks on the report targets, the members-only gate, the duplicate and rate-limit order, the owner-only notice, the moderation post (SaaS only, https only, the 200-character cap, no identity, the URL never logged), key presence versus key values in `ai_providers` (Retell included), the exact public paths in the middleware, and query values on `/go/pair`.
- Run `bash scripts/ralph/acceptance-prd266w3.sh` yourself (it waits for the wave's one CI run on the tested commit and reads the PR body).
- Spot-check five `DONE` acceptance criteria at random against the code and the tests; always include one from US-016a, one from US-016b and one from US-017b.
- No CI lane runs `next build` for this PR: read every new `'use client'` boundary, page and Next.js route handler for what a build would refuse (a hook in a server component, a server-only import in a client file, `useSearchParams` rules, a route handler under `app/.well-known/` exporting anything but HTTP methods and route config).
- **Nothing runs on the owner's machine** (no pytest, alembic, npm, vitest, tsc, next, docker, server, browser). Never end your turn with anything running in the background.

## Verdict

- **No CRITICAL/HIGH/MEDIUM:** a 7-line summary: (a) reports: the route, the gate, the target rules, the audit row, duplicates, the limiter; (b) the owner's notice and the moderation post: owner only, SaaS and https only, the 200-character excerpt, no identity, no URL in logs; (c) the web: the bell link, the audit filter; (d) ai_providers: the six sources, key presence only, Retell's voice rule, null on failure, `api/workspaces.py` not grown; (e) paused tasks: the feed and the web calendar; (f) universal links and `/go/pair`: exact public paths, the JSON bodies, no API call; (g) the owner's next step: the frontend image build (no CI `next build`), then the `ownerTest` list.

  Then `REVIEW_PASS` as the last line (the words REVIEW_PASS and REVIEW_FINDINGS appear on your last line only, never in prose).
- **Findings:** append `P266W3-RVW-n` stories to `scripts/ralph/prd-266w3.json` (n from 1; same shape: id, title, description, acceptanceCriteria with the cause, the change and the test, priority after the last story, passes false, notes), commit `git commit -s -m 'chore(prd-266): review findings → fix stories [skip ci]' -- scripts/ralph/prd-266w3.json`.

  Then `REVIEW_FINDINGS` as the last line.
