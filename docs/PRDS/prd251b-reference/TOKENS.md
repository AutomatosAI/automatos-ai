# PRD-251B design tokens: the mockup's roles mapped to the app's Studio tokens

`studio.css` is the mockup's own stylesheet. It is **not copied** (B1). Every role below is built with the token or the `components/ui/*` component named here, so the Studio screens come out in the app's Studio Dark system, which is the same design language as the mockup. The app's tokens live in `frontend/app/globals.css` (`.studio.dark`, from line ~280); fonts bind through `next/font` in `frontend/app/layout.tsx` (`--font-geist-sans`, `--font-geist-mono`, `--font-newsreader`). Added 2026-10-02 evening on the owner's word ("lets not skip anything").

## Colour

| Mockup role (`studio.css`) | Mockup value | Build with | Note |
|---|---|---|---|
| `--bg` page | `#171614` | `bg-background` (30 12% 9%) | |
| `--surface` card | `#1E1C19` | `Card` from `components/ui/card` (`bg-card`, `border-border`) | |
| `--surface2` raised (buttons, segmented "on", pills) | `#262320` | `bg-secondary` (30 10% 16%) | |
| `--line` borders | `#35312B` | `border-border` (30 10% 22%) | |
| grid lines | `#2C2925` | `border-border/60` | |
| `--text` | `#F1EADF` | `text-foreground` (38 40% 90%) | |
| `--muted` (hints, `.h3`, chip keys) | `#A9A194` | `text-muted-foreground` (38 18% 62%) | |
| `--accent` (primary buttons, active underline, selected ring, today's day number, render label) | `#FF6E33` | `bg-accent` / `text-accent` / `border-accent` (15 80% 56%) | **Owner choice 1.** The mockup's hex is the plain dark theme's accent (16 100% 60%); the Studio theme's is one notch deeper. Default: the Studio token, no new colour. Lifting the Studio accent app-wide is one token (`.studio.dark --accent`), the owner's call. |
| accent hover `#FF8A5B` / `#FF8350` | | `hover:bg-accent/90`, `hover:text-accent` | |
| `--ink` text on accent | `#1A0E07` | `text-accent-foreground` (30 12% 9%) | |
| `--amber` (Needs you, the Queue count badge) | `#E5AE45` | `--warning` (47 70% 60%): `text-[hsl(var(--warning))]`, `border-[hsl(var(--warning))]`, `bg-[hsl(var(--warning)/0.10)]` | |
| `--blue` (Making) | `#7FA6F2` | `--info` (213 55% 68%) | |
| `--teal` (Scheduled, Approved, ok icons) | `#52BE9F` | `--success` (82 35% 52%, olive) | The app has no teal; olive is the brand's "good" colour. Known delta, by design. |
| `--red` (Skipped) | `#EA7563` | `text-muted-foreground` with the word **Skipped**; `Failed` uses `--destructive` | In the Studio theme `--destructive` equals the accent, so a red chip would read as "selected". The word carries the meaning (status is never colour alone). Known delta. |
| channel badge `.ch` | `#332F29` | `bg-muted` (30 8% 20%) `font-mono text-[10.5px] font-semibold rounded-md h-5 min-w-[24px] px-[5px]` | |
| segmented track `.seg` | `#191715` | `bg-background/60 rounded-xl p-[3px] border border-border`; a button `rounded-lg h-[38px] px-3.5`; "on" = `bg-secondary text-foreground ring-1 ring-inset ring-border` | Use the app's `Tabs`/`ToggleGroup` where one exists in `components/ui`; else buttons with `aria-pressed` |
| filter chip `.chipb` | `rgba(255,110,51,.13)` on | `rounded-full border border-border h-[38px] px-3.5 text-muted-foreground`; on = `border-accent bg-accent/15 text-foreground` | |
| inputs `.input .textarea .select` | `#191715` | `Input`, `Textarea`, `Select` from `components/ui` as they are | |
| drop zone `.drop` | dashed `#4A443C` | `border-dashed border-[1.5px] border-border rounded-xl min-h-[150px]` | |
| out-of-month day | `rgba(0,0,0,.22)` | `bg-black/20` | |
| today's day number | accent circle, ink text | `bg-accent text-accent-foreground rounded-full h-6 min-w-[24px] px-[7px] text-xs font-semibold` | |
| render frame placeholder `.render` | `#141210`, label mono 11px tracking .12em accent uppercase | only for the empty/placeholder state; real media goes through `FilePreview` | |

## Status chips (`.pc.st-*`): the word first, then the tone

| State | Word | Tone |
|---|---|---|
| draft with a slot | Planned | `border-dashed border-border text-muted-foreground bg-transparent` |
| rendering | Making | `border-[hsl(var(--info))] bg-[hsl(var(--info)/0.10)]`, key in info |
| needs_approval, changes_requested | Needs you | warning border + `bg-[hsl(var(--warning)/0.10)]`, key in warning |
| scheduled | Scheduled | `border-[hsl(var(--success)/0.7)]`, key in success |
| published, partially_published | Posted | `bg-card text-muted-foreground` |
| missed | Skipped | muted, see above |
| failed | Failed | `border-destructive`, key in destructive |

A chip: `rounded-lg border px-2 py-1.5 text-xs leading-[1.3] flex flex-col gap-[3px]`; line 1 and line 3 keys in `font-mono text-[10.5px] font-medium tracking-[.02em] text-muted-foreground`; the title truncates (`truncate`).

## Type

| Mockup | Build with |
|---|---|
| body 14px / 1.45 Geist | the app default (`--font-geist-sans`) |
| `.h1` Newsreader 400 32px / 1.1, tracking -.01em (26px on phone) | `font-serif font-normal text-[32px] leading-[1.1] tracking-[-0.01em]`; compact band: `text-[26px]` |
| `.h2` 15px 600 | `text-[15px] font-semibold` |
| `.h3` 11.5px 600 uppercase tracking .07em muted | `text-[11.5px] font-semibold uppercase tracking-[.07em] text-muted-foreground` |
| `.hint` 12.5px muted / 1.45 | `text-[12.5px] leading-[1.45] text-muted-foreground` |
| `.mono` Geist Mono | `font-mono` (`--font-geist-mono`) |
| `.pill` 12px 500 | `Badge` variant outline, `rounded-full h-6 px-2.5` |

## Shape, spacing, size

| Mockup | Build with |
|---|---|
| page `.wrap` max 1440, padding 24/32 (16 on phone) | `max-w-[1440px] mx-auto px-8 py-6`; compact band `px-4` |
| card radius 14, padding 18 | `Card` as it is (`rounded-xl p-[18px]`) |
| `.btn` radius 10, min-h 40; `.sm` 36; `.primary` = accent | `Button` from `components/ui/button`; sizes default/sm; **Owner choice 2:** primary actions (New post, Submit for approval, Approve · publishes, Review N posts) use the accent as the mockup does: `className="bg-accent text-accent-foreground hover:bg-accent/90"` (or an existing accent variant if `button.tsx` has one). The Studio's house CTA is cream on ink; the mockup chose orange inside Socials. Default: orange, per the approved mockup. |
| `.btn.ghost` | `Button variant="ghost"` |
| segmented 38px, nav items 44px, inputs 42px | 44px for every tapped control on phone (PRD-246; `studio-mobile-scope.test.ts`) |
| gaps: cards 12, grids 12/10, rows 10 | `gap-3`, `gap-2.5` |
| `.grid2/.grid3/.grid4` | `grid grid-cols-2|3|4 gap-3`; two columns below 1024 px |
| the split layouts (calendar 1fr/340px; editor 0.95fr/1.05fr; queue 320px/1fr) | `grid` with those tracks at ≥1024 px; stacked below (compact band) |
| mockup breakpoint 980 px | use the app's bands: 1023 px and 767 px only |
| icons | `lucide-react` (the app's set), 16/18 px, `stroke-2` |

## Where the compact CSS goes

Only inside the compact region of `frontend/app/globals.css` (between `── Studio compact` and `end of the compact region`), scoped `:is(.studio, <own root class>)`, with the two existing breakpoints. Anything else fails `studio-mobile-scope.test.ts`.
