# Ralph Review Prompt — PRD-251 Socials, Wave 1 (the video engine)

You are a fresh-context **adversarial reviewer**. The build claims PRD-251 Wave 1 is complete. Find where:
- a video can't actually be made;
- money can be spent without an estimate, a cap or a booking;
- a template hardcodes the brand;
- the GPL boundary leaks;
- a story's evidence does not exist.

You fix NOTHING in the code yourself.

## Scope

```
BASE=$(bash scripts/ralph/acceptance-prd251w1.sh --print-base)
git diff --stat $BASE..HEAD
git diff $BASE..HEAD
```

The base is the Wave 0 fork point while this branch is stacked on #782 (`feat/prd-251-socials`), and the `origin/main` merge-base once it sits on `main`. Wave 0's code is the base, not this wave's work: never file a finding against a line this diff does not touch. Read:
- `scripts/ralph/prd-251w1.json` (binding);
- `docs/PRDS/PRD-251-SOCIALS.md` (D1–D17 and the owner's Wave 1 answers);
- `docs/PRDS/PRD-251A-SOCIALS-REFERENCE-PIPELINE.md` (the proven settings and traps).

## Hunt list: every item is a confirmed-risk class

1. **The renderer works end to end (D3, S1.1).**
   - The CI `media-render` job builds the image and renders the fixture composition to an MP4 with an audio stream and the expected duration.
   - `hyperframes check` runs before every render, and a lint error returns 422 with the findings.
   - Hyperframes is pinned; telemetry and skills are off (`HYPERFRAMES_NO_TELEMETRY`, `DO_NOT_TRACK`, `HYPERFRAMES_SKIP_SKILLS`).
   - The Kokoro/espeak data path stays under 160 characters (boot assertion).
   - A render that "succeeds" on a skipped or mocked renderer = CRITICAL.
2. **Sizing, concurrency and quotas.**
   - Two renders at once overall, one per workspace, the rest queued.
   - Per-plan monthly minutes (Basic 10, Pro 60, Business 240) read from the plan-tier config (`_PLAN_TIERS_DEFAULTS`, overridable by `AUTOMATOS_PLAN_TIERS_JSON`), never hardcoded in logic.
   - A render past the quota is refused BEFORE `media-render` is called.
   - Enterprise and the local edition have no quota until set.
   - A quota bypass = HIGH.
3. **Money (D12, D13, D16).**
   - Every paid Composio call is estimated, checked against the post's cap and `socials.media_monthly_cap_usd`, submitted only under the cap, and booked on `LANE_MEDIA`.
   - Credit-billed toolkits book the balance difference.
   - Only allowlisted actions per toolkit, and the Wave 0 deny list is untouched.
   - Provider URLs are copied into our storage before a job is marked done.
   - Spend without an estimate or booking = CRITICAL. A weakened deny list = CRITICAL.
4. **No provider clients, no keys (D15).**
   - No direct HTTP client to Higgsfield, fal, ElevenLabs or fish.audio, and no new key storage or Settings → API Keys entries.
   - Paid tools are reached only through the workspace's Composio connection. A direct client = HIGH.
5. **Templates are data and brand-true (D4, D5).**
   - `document_templates.format` accepts `social_image` and `social_video` (ONE migration, chained onto the base's single head, its `EXPECTED_HEAD`; the head pins in `test_prd209` and `test_prd236_w1_routes` moved; `test_prd251_models`' chained-revisions assertion unweakened). A migration that makes a second head = HIGH.
   - The migration is create_all-first safe: it keeps existing tables, uses `IF NOT EXISTS` / `IF EXISTS`, and seeds insert-if-absent, and a test runs `create_all` first. A DDL step that fails on a create_all-first database = HIGH (the 89d89c250 crash-loop).
   - The seeded library includes the four reference templates (UI story, cinematic product, app promo, data story) ported from `docs/PRDS/prd251-reference/`.
   - No template carries a hex colour, font family or logo outside CSS-variable fallbacks. Changing the brand kit changes the render.
   - Every on-screen word is template text.
   - A hardcoded brand in a template = HIGH.
6. **The GPL boundary.** `phonemizer` and espeak-ng are installed and imported ONLY in `services/media-render`. Any import from the orchestrator = CRITICAL.
7. **Facts carry sources (D7).**
   - Claim variables resolve to real sources (Deliverable, report, document, URL, metric at a timestamp).
   - The unsourced override is recorded.
   - The infographic's figures are source-bound.
8. **Voice (D11).**
   - Kokoro by default. Fish Audio and ElevenLabs only through Composio.
   - No recordings; one file per line; timing flexes to the audio.
9. **Music (S1.6).**
   - Every track has licence and attribution fields.
   - A CC BY credit is appended to the post's default copy.
   - The mix normalises to −14 ± 1 LUFS.
10. **No skill edits in this repo (S1.9).** The owner's rule: skills are authored in `automatos-skills` first, then synced into this repo. Every file under `orchestrator/core/seeds/skills/` must carry the generated-file banner. A skill written by hand here, or any edit to the generated seed `orchestrator/core/seeds/platform-management-skill.md`, = MEDIUM (it forks the source of truth).
11. **Scope and conventions.**
    - No Wave 2+ code: scheduling, publishers, calendar, full composer. The exceptions are S3.5, S4.1 and S4.2, which the owner pulled into this wave.
    - No `os.getenv`/`os.environ` outside `config.py`, and `config-surface.json` is regenerated for new settings.
    - Every commit is DCO-signed.
    - No `node_modules` or generated video in the diff.
    - Route manifest updated.
    - Pushes only to the Wave 1 branch.
12. **The agent layer (US-115..US-120).**
    - **Thin tools.** The brand-kit tools share one persistence function with the REST route. The Socials tools call the Wave 0 service: no second status machine, content hash or approval reset.
    - **No way to publish.** No tool approves, schedules or publishes, and no socials tool schema has such a parameter. Violation = CRITICAL.
    - **The post gate.** In a Socials-on workspace, an agent's post call is refused on every agent path: the tool executor, the LinkedIn image workaround and playbook steps. Reads and uploads still run, a Socials-off workspace is unchanged, an unreadable setting fails closed, and the deny list is still checked first. Any bypass = CRITICAL.
    - **Built-in skills.** Auto's `platform-management` loads and refreshes exactly as before. Seed files carry the banner.
    - **The package.** Its agents are `Agent` rows, not `marketplace_items`. No publisher skill or `html-to-png` is attached. The playbook install path is tested, and skill links reconcile on every boot.
    - **Reuse.** A new tool, table or seeder where the story names an existing one = HIGH.

## Verification

- Run the **code-review** skill (or the code-reviewer agent) on the diff. Any CRITICAL or HIGH it reports is a finding.
- Run `bash scripts/ralph/acceptance-prd251w1.sh` yourself. A green build with a red gate is a finding.
- Check CI: `gh run list --branch feat/prd-251-w1-video-engine --limit 5`, and read the `media-render` job's log for the fixture render's duration and output checks.
- Spot-check three `DONE` acceptance criteria at random against the code. Evidence that does not exist = CRITICAL.
- **Nothing runs on this machine:** no server, docker, render, browser or database. CI is the evidence.

## Verdict

- **No CRITICAL/HIGH/MEDIUM:** a 5-line summary noting:
  - (a) the fixture render and its timing from CI;
  - (b) the money path traced for one footage call;
  - (c) the brand-kit swap proven;
  - (d) the GPL boundary;
  - (e) the owner's next step: sync the Socials skills (`scripts/sync-skills.py`), then on the socials stack with the media profile install the Socials package and have the Director re-make the four reference videos (Goal 8).

  Final line: `REVIEW_PASS`
- **Findings:**
  1. Append `P251W1-RVW-n` stories to `scripts/ralph/prd-251w1.json`, each with a title, `file:line` evidence and mechanical ACs.
  2. Commit with `git commit -s -m 'chore(prd-251): Wave 1 review findings → fix stories'`, then push to `origin feat/prd-251-w1-video-engine`.

  Final line: `REVIEW_FINDINGS`
