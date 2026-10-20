# Ralph Review Prompt — PRD-266 W1 Connect, the Automatos Go backend

You are a fresh-context **adversarial reviewer**. The build claims PRD-266 wave W1 is complete (US-003a, US-002, US-003b, US-004a, US-003c, US-004b, US-005a, US-001, US-005b, US-005c, US-006a, US-006b, US-007a, US-007b, US-008a, US-008b). This wave changes authentication, permissions and tenancy: review it as an attacker would. Find where:
- the pairing kernel differs from what CLI hosts did (code format, TTL, hash, token length), a copy survives in `services/cli_host_service.py`, a CLI-host test was edited, or a compare is not constant-time (US-003a);
- the revision does not chain onto the base head, a CREATE lacks `IF NOT EXISTS`, a pin did not move, the model and the DDL describe different tables, or `to_dict()` leaks a hash or the push key (US-002);
- a pairing code or a token is stored, logged or returned in clear beyond the one time; an expired, used or wrong code pairs; `/pair` answers outside local `paired` mode; the rate limits run after the code check, share one bucket for everyone, or let a wrong guess burn the real pairer's code; the QR payload is anything but `https://<web_url>/go/pair?…`; `/pairing-codes` is reachable by a non-admin or in saas (US-003b);
- an `amd_` token resolves in saas, reaches Clerk, resolves for a revoked row, outlives its rotation overlap, or lets `X-Workspace-ID` move it to another workspace; `principal.resolve_user_pk` trusts a claim source other than `local_operator` and `mobile_device`; a line inside `get_request_context_hybrid` changed or `hybrid.py` grew; `last_seen_at` is written on every request; a `dependency_overrides[get_request_context_hybrid]` in the tests stopped applying (US-004a);
- a device reads, patches, rotates or revokes another device; another workspace's id answers 403 or data instead of 404; a Clerk row holds a token hash; `push_key` comes back twice; revoke clears the hash so the next call cannot say `device_revoked` (US-003c);
- a resolver lets a device do what its bound user's role does not allow, or gives it every workspace's theme files; `notifications._resolve_user_id` is a second resolver instead of `resolve_user_pk`; the voice caller gained a fallback principal (US-004b);
- a datastore port can follow `API_PUBLISH_ADDRESS`/`WEB_PUBLISH_ADDRESS`; the compose bind test was weakened rather than updated; the boot check can be passed with a non-loopback address in `off`, runs in saas, or sits inside `run_stage`; `paired` mode leaves any no-credential path open on an authenticated route, or closes the API-key, CLI-host, webhook, `/health`, discovery or `/pair` lanes; any code trusts a source address for identity; the console script works in saas (US-005a);
- discovery leaks a secret or an env value other than the publishable key, lists a capability with no live route, advertises `api.automatos.app` from a local install, or emits an `oidc` mode (US-001);
- the browser token reaches anything but apiClient's token seam and the surfaces K8 names; a cookie is readable by the browser's scripts when it need not be, is not `SameSite=Strict`, or is sent in saas; a `device_revoked` leaves the token in storage; `lib/api-client.ts` grew (US-005b);
- the Phones & devices tab shows another workspace's devices, pairs in saas, or the docs describe something the code does not do (US-005c);
- `revoke` or any non-grant `require_workspace_admin` use widened to editors; viewers can list, answer, grant or deny; needs-you counts questions for a viewer (US-006a); the web duplicates role logic (US-006b);
- `ChatRequest` is defined twice, `api/chat.py` grew, spoken mode leaves markdown in a frame split across deltas, or a non-spoken frame changed (US-007a); the agent-info fix touched the 365-line function or changed another frame's bytes; `starting_after` pages across users or workspaces (US-007b);
- the Business gate can be bypassed on saas, blocks local, or misreads enterprise; Retell fields beyond the four pass through; `mint_web_call` is still over 50 code lines or lost a behaviour; an existing voice test was edited or its patches stopped applying (US-008a); a non-live call reaches the socket after accept, or a mid-call status change is ignored (US-008b);
- a story's evidence does not exist, an AC is marked DONE that the code does not meet, a W2/W3 piece or US-020 was built, a new dependency or third-party client was added, or a file over 800 lines grew (config.py's settings and the generated reports excepted).

You fix NOTHING in the code yourself.

## Scope

```
BASE=$(bash scripts/ralph/acceptance-prd266w1.sh --print-base)
git diff --stat $BASE..HEAD
git diff $BASE..HEAD -- . ':!scripts/ralph'
```

The base is the `origin/main` merge-base; the PRD commit (docs only) is already reviewed in PR #1257. Never file a finding against a line this diff does not touch, except where an AC says the base's behaviour had to change and it did not. Read, beside the diff: `scripts/ralph/prd-266w1.json` (binding: its `decisions` and each story's ACs and `notes`), `docs/PRDS/PRD-266-AUTOMATOS-GO-BACKEND.md` (§3, §4.1–4.3, §4.6–4.7, §4.10–4.11, §8), `AGENTS.md`, and the `ownerTest` list.

## Verification

- Run the **code-review** skill (or the code-reviewer agent) on the diff **in the foreground**, and the **security-reviewer** agent in the foreground on `orchestrator/core/auth/{pairing,credentials,device,principal,workspace_admin,workspace_permission}.py`, `orchestrator/services/mobile/`, `orchestrator/api/mobile_devices.py`, `orchestrator/api/well_known.py`, `orchestrator/core/security/rate_limiter.py`, `orchestrator/services/themes/membership.py`, `orchestrator/api/approval_grants.py`, `orchestrator/api/voice_retell.py`, `docker-compose.yml`, `frontend/components/local-auth-provider.tsx` and `frontend/components/settings/phones-devices/`. Ask the security reviewer explicitly about: token and code storage, constant-time compares, the rotation overlap, revocation, tenancy (404 never 403), the two public routes, rate-limit bucket keys and X-Forwarded-For, the paired-mode closure, the compose exposure, and the browser-held token.
- Run the **database-reviewer** agent in the foreground on `orchestrator/alembic/versions/prd266_mobile_devices.py` and `orchestrator/core/models/mobile_devices.py`.
- Run `bash scripts/ralph/acceptance-prd266w1.sh` yourself (it waits for the wave's one CI run on the tested commit and reads the PR body).
- Spot-check five `DONE` acceptance criteria at random against the code and the tests; always include one from US-004a, one from US-005a and one from US-003b or US-003c.
- No CI lane runs `next build` for this PR: read every new `'use client'` boundary and Next.js server route for what a build would refuse (a hook in a server component, a server-only import in a client file, a cookie API used on the client).
- **Nothing runs on the owner's machine** (no pytest, alembic, npm, vitest, tsc, next, docker, server, browser). Never end your turn with anything running in the background.

## Verdict

- **No CRITICAL/HIGH/MEDIUM:** a 7-line summary: (a) the kernel and the table: the CLI-host extraction, the revision and the pins; (b) pairing and the device routes: hashing, one-time return, rate limits, the QR link, tenancy; (c) the lane and the resolvers: saas refusal, rotation, revocation, the bound workspace, the decorator seam; (d) remote access: compose, the boot check, the paired-mode closure, the console script; (e) discovery and the web: no secrets, capabilities, pair a browser, Phones & devices, the docs; (f) editors, chat and voice: the answerer gate, spoken mode, agent-info, history paging, the Business gate, transport fields, the live-only socket; (g) the owner's next step: the frontend image build (no CI `next build`), then the `ownerTest` list.

  Then `REVIEW_PASS` as the last line (the words REVIEW_PASS and REVIEW_FINDINGS appear on your last line only, never in prose).
- **Findings:** append `P266W1-RVW-n` stories to `scripts/ralph/prd-266w1.json` (n from 1; same shape: id, title, description, acceptanceCriteria with the cause, the change and the test, priority after the last story, passes false, notes), commit `git commit -s -m 'chore(prd-266): review findings → fix stories [skip ci]' -- scripts/ralph/prd-266w1.json`.

  Then `REVIEW_FINDINGS` as the last line.
