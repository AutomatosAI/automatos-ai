# Ralph Review Prompt — PRD-251 Socials, Wave 0

You are a fresh-context **adversarial reviewer**. The build claims PRD-251 Wave 0 is complete. Find where:
- the gate leaks;
- one workspace can reach another's data or credentials;
- an approval can outlive the content it approved;
- a story's evidence does not exist.

You fix NOTHING in the code yourself.

## Scope

```
BASE=$(git merge-base HEAD origin/main)
git diff --stat $BASE..HEAD
git diff $BASE..HEAD
```

Diff against **`origin/main`**. Read:
- `scripts/ralph/prd-251.json` (binding);
- `docs/PRDS/PRD-251-SOCIALS.md` (D1–D17, plus the owner's answers in the status block).

## Hunt list: every item is a confirmed-risk class

1. **The gate (D1).**
   - Every route under `/api/socials` depends on `require_socials_enabled`.
   - The master switch off gives a 404. The master on and the workspace off still gives a 404.
   - `GET /api/workspaces/current` reports `socials.available` and `socials.enabled` truthfully.
   - Any socials route reachable with a switch off = CRITICAL.
   - A plan exposure key or a `plan_tiers.py` change = HIGH (owner: every plan gets Socials).
2. **Tenant isolation.**
   - Every socials query filters by the caller's `workspace_id`, and another workspace's post id returns 404. A query without that filter = CRITICAL.
   - The LinkedIn workaround:
     - it never reads another workspace's credential;
     - it never falls back to "the first active" credential;
     - it caches credentials AND tokens per workspace.
     A surviving process-global single credential or token = CRITICAL.
   - All three callers pass `workspace_id`: `tool_executor.py`, `recipe_executor.py` and `api/composio.py`.
3. **Approval integrity (D6).**
   - `content_hash` covers copy, variables, sources, format, template_id and media, over canonical JSON.
   - Any content edit voids an approval.
   - `publish_post` calls `assert_publishable` BEFORE anything else.
   - `approve` requires `socials:approve`; a viewer gets 403.
   - An approval that survives an edit = CRITICAL.
4. **Facts carry sources (D7).**
   - Approving with an unsourced claim and no override returns 422 naming the claim.
   - The override is stored and named in `review_log`.
   - A silent override = HIGH.
5. **The migration.**
   - Exactly one new revision, `prd251_socials`, chained onto `kb_multimodal_tables`.
   - Both head pins moved.
   - The downgrade drops exactly what the upgrade creates.
   - JSON columns use the JSONB variant; `idempotency_key` is unique.
   - A second new revision, or a `social_campaigns` table = HIGH.
6. **The image store.**
   - No single 1000-key page decides a lookup.
   - Legacy ids still resolve.
   - `Range` returns 206 with correct headers.
   - The body is streamed, not read whole.
7. **Video in Deliverables.** Registrable by agents, has an icon, and plays in the preview. No second media route.
8. **The tab (S0.5).**
   - Absent in BOTH shells when unavailable, and `?tab=socials` falls back to Outputs.
   - The single Turn-on / ask-an-admin card.
   - Calls go only through `apiClient`, never raw `fetch`.
   - Canonical terms (Deliverable, Auto, Command Center).
   - Classic and Studio both work.
9. **Scope and conventions.**
   - No Wave 1+ code: media-render, templates, voice, music, scheduler jobs, calendar source, publishers. Wave 1+ code = HIGH.
   - No new dependency.
   - No `os.getenv`/`os.environ` outside `config.py`, and no hardcoded values.
   - Every commit is DCO-signed.
   - No `node_modules` in the diff.
   - Pushes only to `feat/prd-251-socials`.
   - Route manifest updated for every new route.
10. **Tests prove behaviour.**
    - The publish-now 409 test proves the Composio executor was never called.
    - The LinkedIn test uses two workspaces and two credentials.
    - The lookup test uses 1,500 objects.
    - The tab tests cover all three gate states.
    - A test that asserts nothing, or an assertion deleted to make a suite pass = CRITICAL.

11. **The Composio deny list (US-009).**
    - Every path that executes a Composio action calls the one deny-list helper before any network call.
    - It holds with `AUTOMATOS_POLICY_PLANE=off`.
    - It reads a system setting seeded with the eight D16 slugs.
    - A path that skips it, or a hardcoded list, = CRITICAL.

## Verification

- Run the **code-review** skill (or the code-reviewer agent) on the diff. Any CRITICAL or HIGH it reports is a finding.
- Run `bash scripts/ralph/acceptance-prd251.sh` yourself. A green build with a red gate is a finding.
- Check CI: `gh run list --branch feat/prd-251-socials --limit 5`. A red required job caused by this diff is a finding. Arbitrate pre-existing red against `main`'s last run.
- Spot-check three `DONE` acceptance criteria at random against the code. Evidence that does not exist = CRITICAL.
- **Nothing runs a server, docker, a browser or a database on this machine.** The environment points the database and Redis at 127.0.0.1:1 on purpose.

## Verdict

- **No CRITICAL/HIGH/MEDIUM:** a 5-line summary noting:
  - (a) the gate states verified;
  - (b) tenant isolation traced for posts and for LinkedIn;
  - (c) the approval reset proven;
  - (d) the migration's single head;
  - (e) the owner's next step: test Wave 0 on the socials stack (`~/.automatos/socials-stack/socials-stack.sh up`) before Wave 1 is briefed.

  Final line: `REVIEW_PASS`
- **Findings:**
  1. Append `P251-RVW-n` stories to `scripts/ralph/prd-251.json`, each with a title, `file:line` evidence and mechanical ACs.
  2. Commit with `git commit -s -m 'chore(prd-251): review findings → fix stories'`, then push to `origin feat/prd-251-socials`.

  Final line: `REVIEW_FINDINGS`
