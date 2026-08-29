# e2e/ — Live scenario testing for Automatos

Scenario-driven end-to-end testing against a deployed environment (Railway prod with a
Clerk **test-mode** account). Born 2026-08-29 from the onboarding agent's proof that
live-account testing works; first pack covers the **Munder Difflin wave** (PRD-224..229)
plus canaries for PRD-231 and PRD-232.

## How it works

- **Scenarios are data** (`scenarios/*.yaml`): each one is a real user's journey —
  the utterances a person would actually type, followed by assertions on what the
  platform must now look like (board state, questions, notifications, log markers).
- **The runner drives them** (`run.py`): signs in with the Clerk test-mode account,
  plays each turn through the real chat API, then checks the assertions through the
  same public APIs the frontend uses. No mocks, no seams — if it passes here it
  works for a human.
- **Logs ride along**: with `railway` CLI logged in, the runner tails the API
  service during each scenario and captures matching `log_markers` into the report.
- **Playwright is surgical**, not the default: only UI-only assertions (bell
  deep-link lands on the right task, board renders the SSE move) get a browser.
  Conversation flows stay at API level — deterministic and fast.

## Auth (Clerk test mode)

Any email with the `+clerk_test` subaddress is a test account; verification code is
always `424242`; no emails are actually sent. Runner env (never commit values):

```
E2E_BASE_URL=https://<api host>
E2E_APP_URL=https://<frontend host>
E2E_CLERK_EMAIL=<the +clerk_test account>
E2E_CLERK_CODE=424242
```

## Running

```
python3 e2e/run.py --list                 # show scenarios + their status tags
python3 e2e/run.py S01 S02                # run specific scenarios
python3 e2e/run.py --pack munder          # run a whole pack
python3 e2e/run.py --pack munder --report e2e/reports/$(date +%F).md
```

**Coordination rule:** merges to main auto-deploy Railway and restart the API.
Do not run the pack (or merge) while another live tester's session matters,
unless the merge-freely policy is explicitly in force.

## Status tags

- `live` — expected to pass on current main.
- `canary-232` — expected to FAIL until PRD-232 deploys, then flip green. A
  canary passing "early" or failing "late" is itself a finding.
- `post-231` — meaningful only after the PRD-231 merge chain deploys.
- `manual` — needs a human step (e.g. Telegram answer); runner pauses and prompts.

## The loop

1. Run the pack → findings land in `reports/`.
2. Defect → branch → PR → CI green → merge (policy above) → Railway deploys.
3. Re-run the pack; canaries police the flips. Repeat until the pack is green.
