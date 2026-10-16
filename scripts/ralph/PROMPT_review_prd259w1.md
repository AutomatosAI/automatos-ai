# Ralph Review Prompt — PRD-259 Skins in Studio, Wave 1

You are a fresh-context **adversarial reviewer**. The build claims Wave 1 of PRD-259 (S-001, S-002, S-003, S-009, S-004a, S-004b, S-010a, S-010b) is complete. Find where:
- the frontend and backend validators disagree on any case, a token outside the 47 allowed names or a derived alias (`paper, bg-1, bg-2, fg, fg-2, fg-3`) gets through, a contrast pair can be dodged (one side set, the other not), or free text can carry `<`, `://` or `url(` (S-001);
- the migration is not chained onto the base's single head, a CREATE lacks `IF NOT EXISTS`, an existing table or revision changed, a head pin was not moved, or a column Wave 2 needs (status, rejection_reason, featured, install_count) is missing (S-002);
- a package member outside the allowed set, a `..`/absolute/backslash path, a symlink, a renamed file, an oversized font/mascot/preview or a zip bomb is accepted; a query is not filtered by `workspace_id`; another workspace can read a private skin's row or assets, or learns it exists (403 instead of 404); an asset outside the manifest is served; a header (nosniff, cache, checked type) is missing; the 10 MB transport cap still refuses a valid 15 MB package; storage uses anything but `core.storage` (S-003);
- `--radius` is not in both blocks, or other radius rules changed (S-009);
- `skinToCss` writes ANY character the skin carried instead of numbers Studio formatted; the selectors fail to outrank `.studio`/`.studio.dark`; a derived alias is set from the wrong token; the pre-paint script injects a cache that does not match `data-skin` or the grammar, uses `innerHTML`, or throws when storage is blocked; a font is loaded through a CSS `url()` or a raw `fetch('/api…')` (S-004a);
- a tampered cookie reaches `data-skin` or the class; a skin's base does not drive `isStudio` (a Studio layout swaps for a classic-based skin or the reverse); the tone lock overwrites the user's stored tone; the fallback leaves a broken skin active or shows no notice; `ThemeToggle` or a touched function is over its limit (S-004b);
- `darkMode` is not `['class', '.dark']`, the compile test would not catch a regression, an audit row is missing or still pending, a `dark:` class with a fixed palette colour survives without a 'blocked by the code-shape ratchet' reason, or a line was added inside a function already over the limit (S-010a/b);
- a story's evidence does not exist, or a `→ OWNER:` AC was claimed.

You fix NOTHING in the code yourself.

## Scope

```
BASE=$(bash scripts/ralph/acceptance-prd259w1.sh --print-base)
git diff --stat $BASE..HEAD
git diff $BASE..HEAD
```

The base is the `origin/main` merge-base. Never file a finding against a line this diff does not touch, except where a story's AC says the base's behaviour had to change and it did not. Read, beside the diff: `scripts/ralph/prd-259w1.json` (binding: its `decisions` D1–D6 and K1–K5, each story's ACs and `notes`), `docs/PRDS/PRD-259-SKINS-IN-STUDIO.md` §3–§5, §7, §10, the `ownerTest` list.

## Verification

- Run the **code-review** skill (or the code-reviewer agent) on the diff **in the foreground**, and the **security-reviewer** agent in the foreground on the S-003 and S-004a files (uploads, paths, tenant isolation, the CSS and pre-paint injection surface).
- Run `bash scripts/ralph/acceptance-prd259w1.sh` yourself (it waits for the wave's one CI run on the tested commit).
- Spot-check four `DONE` acceptance criteria at random against the code and the tests; always include one from S-003 and one from S-004a.
- **Nothing runs on the owner's machine** (no pytest, npm, vitest, tsc, docker, server, browser). Never end your turn with anything running in the background.

## Verdict

- **No CRITICAL/HIGH/MEDIUM:** a 6-line summary: (a) the validator: the corpus size and that both suites read it; (b) the migration: revision, down_revision, the two pins, Alembic from-zero in CI; (c) tenant isolation: the 404 tests and the asset headers; (d) the engine: the raw-string test, the selectors' specificity, the pre-paint grammar; (e) the dropdown: base → isStudio, the tone lock, the fallback; (f) radius and dark:, with the 'blocked by the code-shape ratchet' rows and the owner's next step (the `ownerTest` list: alembic upgrade, a Skin Kit file, screenshots).

  Then `REVIEW_PASS` as the last line (the words REVIEW_PASS and REVIEW_FINDINGS appear on your last line only, never in prose).
- **Findings:** append `P259-W1-RVW-n` stories to `scripts/ralph/prd-259w1.json` (n from 1; same shape: id, title, description, acceptanceCriteria with the cause, the change and the test, priority after the last story, passes false, notes), commit `git commit -s -m 'chore(prd-259): Wave 1 review findings → fix stories [skip ci]' -- scripts/ralph/prd-259w1.json`.

  Then `REVIEW_FINDINGS` as the last line.
