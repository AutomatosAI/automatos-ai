# Ralph Review Prompt — PRD-260 Play, a mascot theme in Automatos

You are a fresh-context **adversarial reviewer**. The build claims PRD-260 is complete (T-001, T-002a, T-002b, T-003, T-004, T-005a, T-005b, T-006, T-007a, T-007b, T-008a, T-008b, T-009, T-010). Find where:
- `'play'` is not a strict cookie value (a tampered cookie reaches the class), `<html>` does not get `studio play` from the server, `isStudio`/`useIsStudio()` is false under Play (a Studio page swaps to Classic's layout), switching away leaves the `play` class, or the defaults constants changed (T-001);
- the look engine accepts a non-catalogue key (`__proto__`, `toString`), an out-of-range glow or an unknown field; a Studio token is not an HSL triplet Studio formatted from numbers; a ground × accent pair under 4.5:1 (ring under 3:1) gets through the contrast test; a Play variable is not `--pl-*`; a guard test reads raw text instead of import syntax (T-002a);
- the pre-paint script runs outside `html.play`, injects anything a stored look carried instead of catalogue values, uses `innerHTML`, throws when storage is blocked, or is rendered under Classic/Studio; leaving Play leaves a look property on `<html>`, the tone forced, or the tone cookie; the stored tone is overwritten; Play's type comes through `next/font/google` or a literal family that next/font renamed (T-002b);
- the mascot reducer's priorities or one-shots differ from Play's tests; a core chat event maps to the wrong mascot event or none; Safari/iOS gets webm or Chrome gets mp4; a `<video>` renders under reduced motion (T-003);
- Play's nav is a second list of routes instead of `lib/studio-menu.ts`, or shows an item Studio hides for that edition, role or exposure; Studio and Classic (and Play below 1024 px) no longer render exactly what they rendered; MainLayout is over its limit (T-004);
- Play's chat does not send through `useChatWithReceipts` and the session store (a second data layer, Play's proxy or parser ported); `app/chat/page.tsx` was edited; Stop does not cancel the turn; receipts, tool lines, the Deliverable card, the snag box or the mission suggestion diverge from what the core does with the same data; a Deliverable id reaches a link unchecked (T-005a/b);
- `/conversations` renders under Classic/Studio instead of redirecting; grouping is wrong at the edges (T-006);
- the shelf or the detail fetch outside the core's hooks and authenticated paths; a second detail opens (Play's and DeliverableDeepLink's); `/deliverables/explorer` lost Studio's version (T-007a/b);
- Make it yours imports a look without `validateLook`, or the style sheet / editor render under Classic/Studio instead of the note (T-008a/b);
- `not-found.tsx` changed what Classic/Studio users see beyond the default 404 words; the asleep screen never clears (T-009);
- the docs describe something the code does not do, or the RUN checklist misses an ownerTest entry (T-010);
- a story's evidence does not exist, or a `→ OWNER:` AC was claimed.

You fix NOTHING in the code yourself.

## Scope

```
BASE=$(bash scripts/ralph/acceptance-prd260.sh --print-base)
git diff --stat $BASE..HEAD
git diff $BASE..HEAD -- . ':!frontend/public/auto' ':!frontend/app/fonts'
```

The base is the `origin/main` merge-base. The three cherry-picked commits (`git log --grep 'cherry picked from commit' $BASE..HEAD`) were reviewed in PRD-259 Wave 1: check only that they are the -x picks of cc08db61c, ad2d3fee7, 15c70c94d, unchanged apart from the sign-off. Never file a finding against a line this diff does not touch, except where a story's AC says the base's behaviour had to change and it did not. Read, beside the diff: `scripts/ralph/prd-260.json` (binding: its `decisions` D1–D11 and K1–K11, each story's ACs and `notes`), `docs/PRDS/PRD-260-PLAY-THEME.md`, the `ownerTest` list, and the read-only reference `scripts/ralph/prd260-spec/play-ref/` to compare a port with its source.

## Verification

- Run the **code-review** skill (or the code-reviewer agent) on the diff **in the foreground**, and the **security-reviewer** agent in the foreground on `frontend/lib/looks/{prepaint,transfer,validate}.ts`, `frontend/components/looks/look-provider.tsx`, `frontend/app/layout.tsx` and the Deliverable link/id handling in `frontend/components/play/`.
- Run `bash scripts/ralph/acceptance-prd260.sh` yourself (it waits for the wave's one CI run on the tested commit).
- Spot-check four `DONE` acceptance criteria at random against the code and the tests; always include one from T-002b and one from T-005a or T-005b.
- No CI lane ran `next build`: read every new server component, layout and `'use client'` boundary for what a build would refuse (a hook in a server component, a server-only import in a client file, a `next/font` call outside module scope).
- **Nothing runs on the owner's machine** (no npm, vitest, tsc, next, docker, server, browser). Never end your turn with anything running in the background.

## Verdict

- **No CRITICAL/HIGH/MEDIUM:** a 6-line summary: (a) the style axis: the cookie, the classes, isStudio under Play, the defaults untouched; (b) the look engine: the catalogue, the --pl- namespace, the Studio tokens, the 24-pair contrast test, the guard test; (c) first paint: the pre-paint script, the tone, the fonts, the clean-up on leaving Play; (d) Auto and the chat: the events mapping, the clip rule, useChatWithReceipts and the session store, receipts and the snag; (e) the frame and the pages: nav from the Studio menu, conversations, the shelf and detail, Make it yours, the style sheet, the 404 and asleep screens; (f) the owner's next step: the frontend image build (no CI `next build`), then the `ownerTest` list via docs/testing/PRD-260-PLAY-RUN-CHECKLIST.md.

  Then `REVIEW_PASS` as the last line (the words REVIEW_PASS and REVIEW_FINDINGS appear on your last line only, never in prose).
- **Findings:** append `P260-RVW-n` stories to `scripts/ralph/prd-260.json` (n from 1; same shape: id, title, description, acceptanceCriteria with the cause, the change and the test, priority after the last story, passes false, notes), commit `git commit -s -m 'chore(prd-260): review findings → fix stories [skip ci]' -- scripts/ralph/prd-260.json`.

  Then `REVIEW_FINDINGS` as the last line.
