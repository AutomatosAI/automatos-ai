# PRD-259 — `dark:` variant audit

Wave 1, stories S-010a (`components/chatbot/`) and S-010b (everything else). PRD-259 §7, decision D4.

## Why this audit exists

Tailwind 3.3.3's `darkVariants` (`node_modules/tailwindcss/lib/corePlugins.js`) knows only `'class'` and `'media'`; the `['variant', …]` form arrived in 3.4.1. `frontend/tailwind.config.ts` said `darkMode: ['variant', ['&:is(.dark *)']]`, so Tailwind added no dark variant and every `dark:` class generated nothing. S-010a sets `darkMode: ['class', '.dark']` (selector `:is(.dark &)`), proven by `frontend/lib/__tests__/tailwind-dark-variant.test.ts`. Studio Dark carries `.dark` too (`.studio.dark`), so each `dark:` class now applies in Classic Dark, Studio Dark and a skin's dark tone, and each one is seen for the first time.

## How the rows were found

One row per match of

```
git grep -noE "dark:[a-z!\[-][^ \"']*" f2be2abb0 -- 'frontend/*.ts' 'frontend/*.tsx'
```

on the wave's base (`origin/main` f2be2abb0): **133 uses in 22 files**. Line numbers are the base's.

## Verdicts

- **keep**: a token class (`text-foreground/90`, `bg-background/40`, …) that reads right in Classic Dark and Studio Dark. No change.
- **fix**: a fixed palette colour (`gray-*`, `slate-*`, `zinc-*`, `neutral-*`, `white`, `black`, a hex) or a class that fights the token beneath it. Replaced by the token class, or the `dark:` class dropped when the base class is already that token.
- **remove**: redundant (the same as the base class) or dead (`dark:prose-invert` and every `prose-*` class: `@tailwindcss/typography` is not installed; `plugins` is `tailwindcss-animate` only).
- **… — blocked by the code-shape ratchet (&lt;function&gt;, N lines)**: the use sits inside a function over the AGENTS.md limit, where `frontend/scripts/check-changed-lines-eslint.js` fails on any touched line (the finding spans the whole function). Only a pure line deletion is safe there, and a pure deletion cannot fix these. Not changed; listed for the PR body.
- **pending S-010b**: outside `components/chatbot/`, audited by S-010b.

Tally after S-010a: keep 32, fix 48, remove 13, pending 40 (of the fix and remove rows, 17 are blocked by the ratchet).

One remove is made without touching `Message`: `message.tsx:74`'s four `dark:prose-*` uses sat in the `proseClass` constant inside `Message` (409 lines). The line was deleted from `Message` (a pure deletion) and the constant, without the inert `dark:` classes, now sits at module scope under the same name.

## Rows

| file:line (base) | class | verdict | why |
|---|---|---|---|
| `app/dev/reset-onboarding/page.tsx:188` | `dark:text-amber-300` | pending S-010b | — |
| `app/dev/reset-onboarding/page.tsx:201` | `dark:text-red-400` | pending S-010b | — |
| `app/marketplace/widgets/[id]/page.tsx:578` | `dark:prose-invert` | pending S-010b | — |
| `components/activity/memory/memory-viewer.tsx:158` | `dark:prose-invert` | pending S-010b | — |
| `components/chatbot/artifact-metadata-panel.tsx:26` | `dark:text-muted-foreground` | remove | redundant: the base class is already `text-muted-foreground` |
| `components/chatbot/artifact-metadata-panel.tsx:26` | `dark:hover:text-white` | fix | fixed palette colour over the token; the base `hover:text-foreground` is already the token, so the dark: class is dropped |
| `components/chatbot/artifact-metadata-panel.tsx:47` | `dark:text-foreground/90` | keep | token class (`text-foreground/90`): follows Classic Dark, Studio Dark and a skin's dark tone |
| `components/chatbot/artifact-viewer.tsx:55` | `dark:text-white` | fix | fixed palette colour over the token; the base `text-foreground` is already the token, so the dark: class is dropped |
| `components/chatbot/artifact-viewer.tsx:73` | `dark:text-muted-foreground` | remove | redundant: the base class is already `text-muted-foreground` |
| `components/chatbot/artifact-viewer.tsx:73` | `dark:hover:text-white` | fix | fixed palette colour over the token; the base `hover:text-foreground` is already the token, so the dark: class is dropped |
| `components/chatbot/artifact-viewer.tsx:81` | `dark:text-muted-foreground` | remove | redundant: the base class is already `text-muted-foreground` |
| `components/chatbot/artifact-viewer.tsx:81` | `dark:hover:text-white` | fix | fixed palette colour over the token; the base `hover:text-foreground` is already the token, so the dark: class is dropped |
| `components/chatbot/artifact-viewer.tsx:89` | `dark:text-muted-foreground` | remove | redundant: the base class is already `text-muted-foreground` |
| `components/chatbot/artifact-viewer.tsx:89` | `dark:hover:text-white` | fix | fixed palette colour over the token; the base `hover:text-foreground` is already the token, so the dark: class is dropped |
| `components/chatbot/artifact-viewer.tsx:97` | `dark:text-muted-foreground` | remove | redundant: the base class is already `text-muted-foreground` |
| `components/chatbot/artifact-viewer.tsx:97` | `dark:hover:text-white` | fix | fixed palette colour over the token; the base `hover:text-foreground` is already the token, so the dark: class is dropped |
| `components/chatbot/chat.tsx:1118` | `dark:bg-background` | keep | token class (`bg-background`): follows Classic Dark, Studio Dark and a skin's dark tone |
| `components/chatbot/chat.tsx:1255` | `dark:bg-background` | keep | token class (`bg-background`): follows Classic Dark, Studio Dark and a skin's dark tone |
| `components/chatbot/markdown-components.tsx:38` | `dark:text-gray-100` | fix | fixed palette colour over the token; the base `text-foreground` is already the token, so the dark: class is dropped |
| `components/chatbot/markdown-components.tsx:41` | `dark:text-gray-100` | fix | fixed palette colour over the token; the base `text-foreground` is already the token, so the dark: class is dropped |
| `components/chatbot/markdown-components.tsx:43` | `dark:text-foreground/90` | keep | token class (`text-foreground/90`): follows Classic Dark, Studio Dark and a skin's dark tone |
| `components/chatbot/markdown-components.tsx:55` | `dark:text-gray-100` | fix | fixed palette colour over the token; the base `text-foreground` is already the token, so the dark: class is dropped |
| `components/chatbot/markdown-components.tsx:58` | `dark:text-gray-100` | fix | fixed palette colour over the token; the base `text-foreground` is already the token, so the dark: class is dropped |
| `components/chatbot/markdown-components.tsx:61` | `dark:text-gray-100` | fix | fixed palette colour over the token; the base `text-foreground` is already the token, so the dark: class is dropped |
| `components/chatbot/markdown-components.tsx:68` | `dark:bg-white/[0.06]` | fix | fixed palette colour over the token; the base `bg-secondary/60` is already the token, so the dark: class is dropped |
| `components/chatbot/markdown-components.tsx:68` | `dark:text-gray-200` | fix | fixed palette colour over the token; the base `text-foreground/90` is already the token, so the dark: class is dropped |
| `components/chatbot/markdown-components.tsx:68` | `dark:border-white/10` | fix | fixed palette colour over the token; the base `border-border/50` is already the token, so the dark: class is dropped |
| `components/chatbot/markdown-components.tsx:80` | `dark:text-foreground/90` | keep | token class (`text-foreground/90`): follows Classic Dark, Studio Dark and a skin's dark tone |
| `components/chatbot/markdown-components.tsx:85` | `dark:text-gray-100` | fix | fixed palette colour over the token; the base `text-foreground` is already the token, so the dark: class is dropped |
| `components/chatbot/markdown-components.tsx:88` | `dark:text-gray-100` | fix | fixed palette colour over the token; the base `text-foreground` is already the token, so the dark: class is dropped |
| `components/chatbot/markdown-components.tsx:91` | `dark:text-gray-100` | fix | fixed palette colour over the token; the base `text-foreground` is already the token, so the dark: class is dropped |
| `components/chatbot/markdown-components.tsx:97` | `dark:border-gray-800/60` | fix | fixed palette colour over the token; the base `border-border/60` is already the token, so the dark: class is dropped |
| `components/chatbot/markdown-components.tsx:97` | `dark:bg-background/40` | keep | token class (`bg-background/40`): follows Classic Dark, Studio Dark and a skin's dark tone |
| `components/chatbot/markdown-components.tsx:98` | `dark:divide-gray-800/70` | fix | fixed palette colour over the token; the base `divide-border/60` is already the token, so the dark: class is dropped |
| `components/chatbot/markdown-components.tsx:98` | `dark:text-gray-100` | fix | fixed palette colour over the token; the base `text-foreground` is already the token, so the dark: class is dropped |
| `components/chatbot/markdown-components.tsx:104` | `dark:bg-background/60` | keep | token class (`bg-background/60`): follows Classic Dark, Studio Dark and a skin's dark tone |
| `components/chatbot/markdown-components.tsx:104` | `dark:text-muted-foreground` | remove | redundant: the base class is already `text-muted-foreground` |
| `components/chatbot/markdown-components.tsx:109` | `dark:divide-gray-800/70` | fix | fixed palette colour over the token; the base `divide-border/50` is already the token, so the dark: class is dropped |
| `components/chatbot/markdown-components.tsx:112` | `dark:hover:bg-background/60` | keep | token class (`hover:bg-background/60`): follows Classic Dark, Studio Dark and a skin's dark tone |
| `components/chatbot/markdown-components.tsx:115` | `dark:text-foreground/90` | keep | token class (`text-foreground/90`): follows Classic Dark, Studio Dark and a skin's dark tone |
| `components/chatbot/markdown-components.tsx:120` | `dark:text-gray-200` | fix | fixed palette colour over the token; the base `text-foreground` is already the token, so the dark: class is dropped |
| `components/chatbot/markdown-components.tsx:143` | `dark:text-gray-100` | fix | fixed palette colour over the token; the base `text-foreground` is already the token, so the dark: class is dropped |
| `components/chatbot/message.tsx:74` | `dark:prose-invert` | remove | inert: every `prose*` class needs @tailwindcss/typography, which is not installed (plugins is tailwindcss-animate only); removed by moving `proseClass` out of `Message` to module scope (inside `Message` the line is a pure deletion) |
| `components/chatbot/message.tsx:74` | `dark:prose-headings:text-gray-100` | remove | inert: every `prose*` class needs @tailwindcss/typography, which is not installed (plugins is tailwindcss-animate only); removed by moving `proseClass` out of `Message` to module scope (inside `Message` the line is a pure deletion) |
| `components/chatbot/message.tsx:74` | `dark:prose-p:text-gray-100` | remove | inert: every `prose*` class needs @tailwindcss/typography, which is not installed (plugins is tailwindcss-animate only); removed by moving `proseClass` out of `Message` to module scope (inside `Message` the line is a pure deletion) |
| `components/chatbot/message.tsx:74` | `dark:prose-a:text-primary` | remove | inert: every `prose*` class needs @tailwindcss/typography, which is not installed (plugins is tailwindcss-animate only); removed by moving `proseClass` out of `Message` to module scope (inside `Message` the line is a pure deletion) |
| `components/chatbot/message.tsx:141` | `dark:text-foreground/90` | keep | token class (`text-foreground/90`): follows Classic Dark, Studio Dark and a skin's dark tone |
| `components/chatbot/message.tsx:312` | `dark:border-border/30` | remove — blocked by the code-shape ratchet (Message, 409 lines) | redundant: the base class is already `border-border/30`. NOT made: the line sits inside `Message` (409 lines, over the function limit), where only a pure line deletion passes the changed-lines gate |
| `components/chatbot/message.tsx:345` | `dark:text-foreground/90` | keep | token class (`text-foreground/90`): follows Classic Dark, Studio Dark and a skin's dark tone |
| `components/chatbot/message.tsx:348` | `dark:group-hover:text-foreground/90` | keep | token class (`group-hover:text-foreground/90`): follows Classic Dark, Studio Dark and a skin's dark tone |
| `components/chatbot/message.tsx:388` | `dark:text-gray-200` | fix — blocked by the code-shape ratchet (Message, 409 lines) | fixed palette colour over the token; the base `text-foreground` is already the token; the fix drops the dark: class. NOT made: the line sits inside `Message` (409 lines, over the function limit), where only a pure line deletion passes the changed-lines gate |
| `components/chatbot/message.tsx:393` | `dark:group-hover:text-foreground/90` | keep | token class (`group-hover:text-foreground/90`): follows Classic Dark, Studio Dark and a skin's dark tone |
| `components/chatbot/message.tsx:403` | `dark:text-foreground/90` | keep | token class (`text-foreground/90`): follows Classic Dark, Studio Dark and a skin's dark tone |
| `components/chatbot/message.tsx:438` | `dark:bg-background/50` | keep | token class (`bg-background/50`): follows Classic Dark, Studio Dark and a skin's dark tone |
| `components/chatbot/message.tsx:438` | `dark:text-foreground/90` | keep | token class (`text-foreground/90`): follows Classic Dark, Studio Dark and a skin's dark tone |
| `components/chatbot/sheet-artifact.tsx:90` | `dark:text-warning` | remove — blocked by the code-shape ratchet (SheetArtifact, 236 lines) | redundant: the base class is already `text-warning`. NOT made: the line sits inside `SheetArtifact` (236 lines, over the function limit), where only a pure line deletion passes the changed-lines gate |
| `components/chatbot/sheet-artifact.tsx:96` | `dark:border-gray-800/60` | fix — blocked by the code-shape ratchet (SheetArtifact, 236 lines) | fixed palette colour over the token; the base `border-border/60` is already the token; the fix drops the dark: class. NOT made: the line sits inside `SheetArtifact` (236 lines, over the function limit), where only a pure line deletion passes the changed-lines gate |
| `components/chatbot/sheet-artifact.tsx:96` | `dark:bg-background/40` | keep | token class (`bg-background/40`): follows Classic Dark, Studio Dark and a skin's dark tone |
| `components/chatbot/sheet-artifact.tsx:96` | `dark:text-gray-200` | fix — blocked by the code-shape ratchet (SheetArtifact, 236 lines) | fixed palette colour over the token; the base `text-foreground` is already the token; the fix drops the dark: class. NOT made: the line sits inside `SheetArtifact` (236 lines, over the function limit), where only a pure line deletion passes the changed-lines gate |
| `components/chatbot/sheet-artifact.tsx:127` | `dark:border-gray-800/60` | fix — blocked by the code-shape ratchet (SheetArtifact, 236 lines) | fixed palette colour over the token; the base `border-border/60` is already the token; the fix drops the dark: class. NOT made: the line sits inside `SheetArtifact` (236 lines, over the function limit), where only a pure line deletion passes the changed-lines gate |
| `components/chatbot/sheet-artifact.tsx:127` | `dark:bg-background/30` | keep | token class (`bg-background/30`): follows Classic Dark, Studio Dark and a skin's dark tone |
| `components/chatbot/sheet-artifact.tsx:170` | `dark:border-gray-800/60` | fix — blocked by the code-shape ratchet (SheetArtifact, 236 lines) | fixed palette colour over the token; the base `border-border/60` is already the token; the fix drops the dark: class. NOT made: the line sits inside `SheetArtifact` (236 lines, over the function limit), where only a pure line deletion passes the changed-lines gate |
| `components/chatbot/sheet-artifact.tsx:170` | `dark:bg-background/40` | keep | token class (`bg-background/40`): follows Classic Dark, Studio Dark and a skin's dark tone |
| `components/chatbot/sheet-artifact.tsx:172` | `dark:divide-gray-800/70` | fix — blocked by the code-shape ratchet (SheetArtifact, 236 lines) | fixed palette colour over the token; the base `divide-border/60` is already the token; the fix drops the dark: class. NOT made: the line sits inside `SheetArtifact` (236 lines, over the function limit), where only a pure line deletion passes the changed-lines gate |
| `components/chatbot/sheet-artifact.tsx:172` | `dark:text-gray-100` | fix — blocked by the code-shape ratchet (SheetArtifact, 236 lines) | fixed palette colour over the token; the base `text-foreground` is already the token; the fix drops the dark: class. NOT made: the line sits inside `SheetArtifact` (236 lines, over the function limit), where only a pure line deletion passes the changed-lines gate |
| `components/chatbot/sheet-artifact.tsx:173` | `dark:bg-background/60` | keep | token class (`bg-background/60`): follows Classic Dark, Studio Dark and a skin's dark tone |
| `components/chatbot/sheet-artifact.tsx:173` | `dark:text-muted-foreground` | remove — blocked by the code-shape ratchet (SheetArtifact, 236 lines) | redundant: the base class is already `text-muted-foreground`. NOT made: the line sits inside `SheetArtifact` (236 lines, over the function limit), where only a pure line deletion passes the changed-lines gate |
| `components/chatbot/sheet-artifact.tsx:182` | `dark:divide-gray-800/70` | fix — blocked by the code-shape ratchet (SheetArtifact, 236 lines) | fixed palette colour over the token; the base `divide-border/50` is already the token; the fix drops the dark: class. NOT made: the line sits inside `SheetArtifact` (236 lines, over the function limit), where only a pure line deletion passes the changed-lines gate |
| `components/chatbot/sheet-artifact.tsx:191` | `dark:hover:bg-background/60` | keep | token class (`hover:bg-background/60`): follows Classic Dark, Studio Dark and a skin's dark tone |
| `components/chatbot/sheet-artifact.tsx:196` | `dark:text-gray-200` | fix — blocked by the code-shape ratchet (SheetArtifact, 236 lines) | fixed palette colour over the token; the base `text-foreground` is already the token; the fix drops the dark: class. NOT made: the line sits inside `SheetArtifact` (236 lines, over the function limit), where only a pure line deletion passes the changed-lines gate |
| `components/chatbot/sheet-artifact.tsx:211` | `dark:border-gray-800/60` | fix — blocked by the code-shape ratchet (SheetArtifact, 236 lines) | fixed palette colour over the token; the base `border-border/60` is already the token; the fix drops the dark: class. NOT made: the line sits inside `SheetArtifact` (236 lines, over the function limit), where only a pure line deletion passes the changed-lines gate |
| `components/chatbot/sheet-artifact.tsx:211` | `dark:bg-background/30` | keep | token class (`bg-background/30`): follows Classic Dark, Studio Dark and a skin's dark tone |
| `components/chatbot/sheet-artifact.tsx:224` | `dark:text-foreground/90` | keep | token class (`text-foreground/90`): follows Classic Dark, Studio Dark and a skin's dark tone |
| `components/chatbot/sheet-artifact.tsx:241` | `dark:border-gray-800/60` | fix — blocked by the code-shape ratchet (SheetArtifact, 236 lines) | fixed palette colour over the token; the base `border-border/60` is already the token; the fix drops the dark: class. NOT made: the line sits inside `SheetArtifact` (236 lines, over the function limit), where only a pure line deletion passes the changed-lines gate |
| `components/chatbot/sheet-artifact.tsx:241` | `dark:bg-background/30` | keep | token class (`bg-background/30`): follows Classic Dark, Studio Dark and a skin's dark tone |
| `components/chatbot/sheet-artifact.tsx:242` | `dark:text-gray-200` | fix — blocked by the code-shape ratchet (SheetArtifact, 236 lines) | fixed palette colour over the token; the base `text-foreground` is already the token; the fix drops the dark: class. NOT made: the line sits inside `SheetArtifact` (236 lines, over the function limit), where only a pure line deletion passes the changed-lines gate |
| `components/chatbot/sheet-artifact.tsx:245` | `dark:border-gray-800/60` | fix — blocked by the code-shape ratchet (SheetArtifact, 236 lines) | fixed palette colour over the token; the base `border-border/60` is already the token; the fix drops the dark: class. NOT made: the line sits inside `SheetArtifact` (236 lines, over the function limit), where only a pure line deletion passes the changed-lines gate |
| `components/chatbot/sheet-artifact.tsx:245` | `dark:bg-background/60` | keep | token class (`bg-background/60`): follows Classic Dark, Studio Dark and a skin's dark tone |
| `components/chatbot/sheet-artifact.tsx:245` | `dark:text-gray-200` | fix — blocked by the code-shape ratchet (SheetArtifact, 236 lines) | fixed palette colour over the token; the base `text-foreground` is already the token; the fix drops the dark: class. NOT made: the line sits inside `SheetArtifact` (236 lines, over the function limit), where only a pure line deletion passes the changed-lines gate |
| `components/chatbot/text-artifact.tsx:75` | `dark:text-info/70` | keep | token class (`text-info/70`): follows Classic Dark, Studio Dark and a skin's dark tone |
| `components/chatbot/text-artifact.tsx:84` | `dark:border-gray-800/60` | fix | fixed palette colour over the token; the base `border-border/60` is already the token, so the dark: class is dropped |
| `components/chatbot/text-artifact.tsx:84` | `dark:bg-background/40` | keep | token class (`bg-background/40`): follows Classic Dark, Studio Dark and a skin's dark tone |
| `components/chatbot/text-artifact.tsx:86` | `dark:text-gray-100` | fix | fixed palette colour over the token; the base `text-foreground` is already the token, so the dark: class is dropped |
| `components/chatbot/text-artifact.tsx:90` | `dark:border-gray-800/60` | fix | fixed palette colour over the token; the base `border-border/60` is already the token, so the dark: class is dropped |
| `components/chatbot/text-artifact.tsx:90` | `dark:bg-background/40` | keep | token class (`bg-background/40`): follows Classic Dark, Studio Dark and a skin's dark tone |
| `components/chatbot/text-artifact.tsx:92` | `dark:text-gray-100` | fix | fixed palette colour over the token; the base `text-foreground` is already the token, so the dark: class is dropped |
| `components/chatbot/text-artifact.tsx:96` | `dark:border-gray-800/60` | fix | fixed palette colour over the token; the base `border-border/60` is already the token, so the dark: class is dropped |
| `components/chatbot/text-artifact.tsx:96` | `dark:bg-background/40` | keep | token class (`bg-background/40`): follows Classic Dark, Studio Dark and a skin's dark tone |
| `components/chatbot/text-artifact.tsx:98` | `dark:text-gray-100` | fix | fixed palette colour over the token; the base `text-foreground` is already the token, so the dark: class is dropped |
| `components/chatbot/text-artifact.tsx:102` | `dark:border-gray-800/60` | fix | fixed palette colour over the token; the base `border-border/60` is already the token, so the dark: class is dropped |
| `components/chatbot/text-artifact.tsx:102` | `dark:bg-background/40` | keep | token class (`bg-background/40`): follows Classic Dark, Studio Dark and a skin's dark tone |
| `components/chatbot/text-artifact.tsx:104` | `dark:text-gray-100` | fix | fixed palette colour over the token; the base `text-foreground` is already the token, so the dark: class is dropped |
| `components/chatbot/text-artifact.tsx:114` | `dark:text-foreground/90` | keep | token class (`text-foreground/90`): follows Classic Dark, Studio Dark and a skin's dark tone |
| `components/chatbot/text-artifact.tsx:123` | `dark:text-gray-200` | fix | fixed palette colour over the token; the base `text-foreground` is already the token, so the dark: class is dropped |
| `components/chatbot/text-artifact.tsx:149` | `dark:bg-background/70` | keep | token class (`bg-background/70`): follows Classic Dark, Studio Dark and a skin's dark tone |
| `components/chatbot/text-artifact.tsx:149` | `dark:border-gray-800/60` | fix | fixed palette colour over the token; the base `border-border/60` is already the token, so the dark: class is dropped |
| `components/chatbot/text-artifact.tsx:149` | `dark:text-gray-100` | fix | fixed palette colour over the token; the base `text-foreground` is already the token, so the dark: class is dropped |
| `components/layout/header.tsx:45` | `dark:text-white` | pending S-010b | — |
| `components/onboarding/power-up-card.tsx:143` | `dark:text-amber-500` | pending S-010b | — |
| `components/onboarding/power-up-card.tsx:210` | `dark:text-emerald-400` | pending S-010b | — |
| `components/onboarding/trial-balance-pill.tsx:41` | `dark:text-amber-500` | pending S-010b | — |
| `components/onboarding/trial-exhausted-banner.tsx:36` | `dark:text-amber-500` | pending S-010b | — |
| `components/shared/markdown-view.tsx:9` | `dark:prose-invert`,` | pending S-010b | — |
| `components/ui/alert.tsx:13` | `dark:border-destructive` | pending S-010b | — |
| `components/widgets/EmailWidget/EmailViewer.tsx:297` | `dark:prose-invert` | pending S-010b | — |
| `components/widgets/MemoryWidget/MemoryItem.tsx:39` | `dark:bg-blue-900` | pending S-010b | — |
| `components/widgets/MemoryWidget/MemoryItem.tsx:39` | `dark:text-info/80` | pending S-010b | — |
| `components/widgets/MemoryWidget/MemoryItem.tsx:45` | `dark:bg-purple-900` | pending S-010b | — |
| `components/widgets/MemoryWidget/MemoryItem.tsx:45` | `dark:text-agent/80` | pending S-010b | — |
| `components/widgets/MemoryWidget/MemoryItem.tsx:51` | `dark:bg-green-900` | pending S-010b | — |
| `components/widgets/MemoryWidget/MemoryItem.tsx:51` | `dark:text-green-300` | pending S-010b | — |
| `components/widgets/MemoryWidget/MemoryItem.tsx:57` | `dark:bg-warning` | pending S-010b | — |
| `components/widgets/MemoryWidget/MemoryItem.tsx:57` | `dark:text-warning` | pending S-010b | — |
| `components/widgets/MemoryWidget/MemoryItem.tsx:63` | `dark:bg-secondary` | pending S-010b | — |
| `components/widgets/MemoryWidget/MemoryItem.tsx:63` | `dark:text-foreground/90` | pending S-010b | — |
| `components/widgets/WorkflowWidget/StepDetail.tsx:157` | `dark:text-destructive/70` | pending S-010b | — |
| `components/widgets/WorkflowWidget/StepDetail.tsx:160` | `dark:text-destructive/80` | pending S-010b | — |
| `components/widgets/WorkflowWidget/WorkflowStatus.tsx:48` | `dark:bg-secondary` | pending S-010b | — |
| `components/widgets/WorkflowWidget/WorkflowStatus.tsx:48` | `dark:text-foreground/90` | pending S-010b | — |
| `components/widgets/WorkflowWidget/WorkflowStatus.tsx:55` | `dark:bg-blue-900` | pending S-010b | — |
| `components/widgets/WorkflowWidget/WorkflowStatus.tsx:55` | `dark:text-info/80` | pending S-010b | — |
| `components/widgets/WorkflowWidget/WorkflowStatus.tsx:63` | `dark:bg-yellow-900` | pending S-010b | — |
| `components/widgets/WorkflowWidget/WorkflowStatus.tsx:63` | `dark:text-yellow-300` | pending S-010b | — |
| `components/widgets/WorkflowWidget/WorkflowStatus.tsx:70` | `dark:bg-green-900` | pending S-010b | — |
| `components/widgets/WorkflowWidget/WorkflowStatus.tsx:70` | `dark:text-green-300` | pending S-010b | — |
| `components/widgets/WorkflowWidget/WorkflowStatus.tsx:77` | `dark:bg-destructive/20` | pending S-010b | — |
| `components/widgets/WorkflowWidget/WorkflowStatus.tsx:77` | `dark:text-destructive/80` | pending S-010b | — |
| `components/widgets/WorkflowWidget/WorkflowStatus.tsx:84` | `dark:bg-warning` | pending S-010b | — |
| `components/widgets/WorkflowWidget/WorkflowStatus.tsx:84` | `dark:text-warning` | pending S-010b | — |
| `components/widgets/WorkflowWidget/WorkflowSteps.tsx:118` | `dark:bg-blue-950/20` | pending S-010b | — |
| `components/widgets/WorkflowWidget/WorkflowSteps.tsx:147` | `dark:text-destructive` | pending S-010b | — |
| `components/widgets/WorkflowWidget/WorkflowSteps.tsx:160` | `dark:text-info` | pending S-010b | — |
| `components/widgets/WorkflowWidget/index.tsx:145` | `dark:text-destructive/80` | pending S-010b | — |
