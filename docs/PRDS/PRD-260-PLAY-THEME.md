# PRD-260: Play, a mascot theme in Automatos

Status: approved by Gerard, 14 Oct 2026 ("Go with it", plan agreed). One wave. Step 1 of 3.

## 1. Introduction

Automatos has two themes today, Classic and Studio (PRD-244: the Style axis, in one menu with the Tone). Each is a whole UI: the app frame and every page check the style and render their own version.

This PRD adds a third theme, **Play**: the fun UI built and tested in `automatos-play` (13–14 Oct). Gerard approved it from his screenshots of Play's chat and its "Make it yours" page. Play has:
- **Auto**, the mascot, living on the pages and reacting to what Auto is doing (12 animated states);
- Ember type and colours, Play's frame, chat, conversations and Deliverables;
- **Make it yours**: the look editor. The user picks a look (Ember, Obsidian, Aurora, Workshop, Daylight) and changes the accent, background, type, Auto's shell and light, and the glow. They can save, export and import looks.

**The 3-step plan (Gerard, 14 Oct):**
1. Ship this first mascot theme for real, inside Automatos. **This PRD.**
2. Then reverse-engineer the Design Kit from what shipped: the files, skills and guides a developer gives Claude Design, Claude Code or Codex to build a new theme (JARVIS, my business), then push it to the Marketplace for approval.
3. Users install approved themes and tweak them with each theme's look editor.

## 2. Goals

- The theme menu shows **Classic · Studio · Play**. Choosing Play turns the whole app into Play, with no flash on load.
- Play's chat, conversations and Deliverables pages, the look editor and the style sheet run on **Automatos's own data and hooks**: no proxy, no second data layer.
- Auto follows the real chat events Studio already reads.
- **Every other page still works** under Play: it shows Studio's version inside Play's frame, recoloured by the active look.
- Looks are saved per user, starting in the browser. Every look stays readable (WCAG AA), proven by tests.

## 3. How it fits (the Studio pattern)

- **Style:** `lib/ui-style.ts` gains `play`. The cookie `automatos-style` can be `play`. The server sets `<html class="studio play">`: `studio` so pages without a Play design render Studio's version, and `play` for Play's frame and the look variables.
- **`isStudio`:** the existing `useIsStudio()` stays true under Play, so every Studio page keeps rendering. A new `useIsPlay()` lets the frame and the Play-designed pages render Play's version.
- **The look reaches Studio pages:** the look engine (ported from Play's `lib/looks/`) emits Play's own variables **and** Studio's tokens (`--background`, `--foreground`, `--card`, `--primary`, `--accent`, `--muted`, `--border`, `--ring` and the rest, as HSL triplets), scoped to `html.play`. Studio pages pick up the look's colours with no change to them.
- **No flash:** the pre-paint script from Play (`lib/looks/prepaint.ts`) is applied in `app/layout.tsx` when the style is `play`.
- **Data:**
  - chat comes from `lib/chat/hooks.ts` (`useChat`) and the chat-session store;
  - Deliverables from `hooks/use-deliverables-api.ts`;
  - conversation history from `lib/chat/api.ts` (`getChatHistory`, `getChatMessages`).
  - Play's own proxy and stream parser are NOT ported, because Automatos's data layer replaces them.
- **Auto** is driven by Play's mascot reducer (`lib/mascot/`) fed from `useChat`'s events. The 12 clips ship under `frontend/public/auto/` (WebM, MP4 and PNG per state; about 4.5 MB). Rights are settled: Higgsfield's terms give ownership and commercial use (PRD-259 §10).
- **Phones (under 1024 px)** stay as today: Studio is desktop-only and falls back to Classic, and Play does the same in this PRD.

## 4. Pages in this ship

| Page | Under Play |
|---|---|
| Chat (`/chat`) | Play's chat: Auto's column with ring and caption, the first-open "Hi, I'm Auto." with example chips, tool lines, receipts, the Deliverable card, the snag box, Stop |
| Conversations | Play's list (Today, This week, Earlier; search; New chat), reachable from Play's frame |
| Deliverables (`/deliverables`, detail) | Play's shelf (cards, type chips, Load more) and detail (content, side panel, Ask Auto about this) |
| Make it yours (`/looks`) | the look editor from Play, exactly as in Gerard's screenshots |
| Style sheet (`/looks/style`) | Play's live style sheet |
| 404, backend asleep | Play's "This page wandered off" and "Automatos is asleep" |
| Everything else | Studio's version inside Play's frame, recoloured by the look |

## 5. Also in this wave

The two app-wide fixes found in PRD-259 Wave 1, cherry-picked from `feat/prd-259-w1-skins`:
- `--radius` defined globally (`cc08db61c`);
- `dark:` working again, `darkMode: ['class', '.dark']` plus the audit (`ad2d3fee7`, `15c70c94d`).

The colour-only skin format, its tables and its import from PRD-259 are NOT part of this. They stay on their branch, and Gerard decides on them later.

## 6. Non-goals

These are Gerard's to revisit, not deferred decisions:
- The Play designs for Missions, Agents and Knowledge (built in `automatos-play` wave 3; next).
- Phones.
- The Design Kit and the Marketplace for themes (steps 2 and 3).
- Saving looks per user on the server.

## 7. Open questions

1. Should Play be the default theme for new users, or stay opt-in?
2. Should the look editor also be reachable from the theme menu ("Customise look…")? This PRD adds it.

## Sources

- The Play app, read-only reference copy: `scripts/ralph/prd260-spec/play-ref/` (from `automatos-play` branch `ralph/play-pages-a`).
- Gerard's screenshots: the Play chat ("Hi, I'm Auto."), and "Make it yours" with Ember, Night, and Obsidian with a cobalt Auto.
