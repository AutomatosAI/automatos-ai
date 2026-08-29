# Ralph Review Prompt — PRD-230 Packages & Vertical Onboarding

Fresh-context **adversarial reviewer**. The build claims PRD-230 (+2 PRD-222 W0 fixes) complete. Find where the closure leaks a dependency, where an install isn't workspace-owned, where a seed references a ghost artifact, or where the chat metering fix doesn't actually meter. Fix NOTHING yourself.

## Scope

```
BASE=$(git merge-base HEAD origin/main)
git diff --stat $BASE..HEAD && git diff $BASE..HEAD
```

Contract: `scripts/ralph/prd-230.json`. Intent: the seeded PRD (§2 decisions are law: D1 registration, D2 closure, D3 ownership, D6 one-package-during-onboarding, D7 three steps).

## Hunt list

1. **The invariant (CRITICAL class).** Trace the installer end-to-end: any closure member that ends unregistered, any registration missing the workspace id, any artifact the workspace cannot edit, any second registration mechanism duplicating an existing pattern = finding. The agent-A test must go through the INSTALLER, not just the resolver.
2. **Closure honesty.** Cycles terminate; playbook→agent recursion works; composio app assignments become `required_connects`, never silent installs and never dropped.
3. **Chat metering fix is real (CRITICAL).** Don't trust the test alone: trace the chatbot call site → factory → `_tracking_ctx['trial']` → `record_trial_spend`. A fix that flags only some chat paths (streaming vs non-streaming, tool-turns) = finding. BYOK provably untouched.
4. **Doctrine v2 truthful.** Every capability claim in the section corresponds to something real (connect card, scan tool, Settings→Widget SDK two-step, marketplace tools from THIS wave); stage vocabulary matches `STAGE_ORDER` exactly; budget measured.
5. **Seeds resolve.** Every member ref in both Shopify packages resolves against real inventory (run the resolution, don't eyeball); no customer data; curation cited.
6. **One-package restriction** at the TOOL layer with honest copy; unrestricted post-onboarding; cannot be bypassed by `platform_install_marketplace_agent` looping a package's members (that path is allowed by design — but confirm the restriction copy doesn't lie about it).
7. **ONE migration**, additive only, single head. Route manifest hand-edited if any route was added (count bump).
8. **Load-bearing PRD-222 surfaces intact** (the acceptance checks them by name); trust guards + stage validator byte-untouched; walker green; no `os.getenv` outside config.
9. **Frontend build green** (popup/tab imports); no marketplace item hidden by the tab work; showcase row ordering tested.

## Verification

- code-review skill (or code-reviewer agent) on the diff — CRITICAL/HIGH = findings.
- `gh run list --branch ralph/prd-230-packages --workflow test.yml --limit 3` — NEW failures vs base = finding.
- `bash scripts/ralph/acceptance-prd230.sh` — non-zero = automatic CRITICAL.

## Verdict protocol

**Sentinel on the FINAL line, alone.**

- Clean → 5-line summary noting: (a) the agent-A invariant verified through the installer; (b) chat metering traced live; (c) both seed packages resolve fully; (d) Gerard's Q1-Q3 (tier visibility, vertical shortlist, tab naming) remain his calls. Final line: `REVIEW_PASS`
- Findings → append `P230-RVW-n` stories to `scripts/ralph/prd-230.json`, commit `chore(prd-230): review findings → fix stories`, push. Final line: `REVIEW_FINDINGS`
- Fix nothing; never force-push.
