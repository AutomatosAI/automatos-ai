# The CSS layer model for themes (PRD-261 A-12c, PA-15)

## Decision

1. **Theme CSS stays unlayered.** The host never wraps a theme's `dist/theme.css` in a cascade
   layer such as `@layer automatos-theme`.
2. **Protected surfaces are protected by placement, not by layers.** The approval cards carry an
   inline `!important` guard (`frontend/components/theme-host/protected-surface.tsx`).
3. **The guard uses option (2) of PA-15:** no shadow root. The guard sits on the card's host
   element and on its inner wrapper.

## Why theme CSS is not in a layer

Core's CSS is unlayered in the browser. `frontend/app/globals.css:2-6` and `:272` use Tailwind 3's
`@tailwind` and `@layer base` directives. Tailwind compiles those directives away, so they produce
no native cascade layer, and its output reaches the page as ordinary unlayered rules.

In CSS Cascade 5, an unlayered normal declaration beats a normal declaration in any layer,
whatever their specificity. If theme CSS sat in `layer(automatos-theme)`, every theme rule would
lose to every core rule that sets the same property on the same element.

Play shows what that would break. `app/play-theme.css:204-209` recolours the page with
`:where(html.play) body { background: var(--pl-page-background) }`. Core's `globals.css:272` sets
`body { @apply bg-background }`. Both selectors have the same specificity, (0,0,1). Unlayered, the
cascade falls through to order of appearance; Play's sheet comes after core's, so Play's body
colour applies. Layered, Studio's `bg-background` would win on every page, whatever the order.

`e2e/browser/specs/protected-surfaces.spec.ts` proves this in the browser. Under the Play cookie,
it checks that `getComputedStyle(document.body).backgroundColor` follows `--pl-page-background`,
including when a sentinel colour is put on that token. Core's body rule never reads that token.

## What the server's scope check refuses

The check is `orchestrator/services/themes/css_scope.py` (A-06a, wave W-B1). Every rule in
`dist/theme.css` must sit under the theme's own `html.<htmlClass>`, so the sheet styles nothing once another theme
(or none) is active. The check refuses:

- a selector that does not start with `html.<htmlClass>`, or with `:where(…)` or `:is(…)` whose
  every argument does (the wrap form is D2-c, so both forms pass);
- a nested selector that uses `&` anywhere but at its start, such as `:not(&)`;
- an `@keyframes` name or an `@font-face` `font-family` that does not start with `<htmlClass>-`;
- `!important`, which is any `!` outside a string or a comment;
- `@import`, and every at-rule except `@media`, `@supports`, `@layer`, `@container`, `@font-face`
  and `@keyframes`;
- anything a browser could read differently from the tokenizer: an unclosed comment, string or
  block; a brace or semicolon inside brackets; a backslash escape of a letter or digit; an
  unquoted `url()` with anything a browser would end it on.

A theme may still put its own rules in a layer of its own. That only makes them lose to core.

## How the protected surfaces hold

`ProtectedSurface` wraps `ToolApprovalWidget` and `MissionApprovalWidget`. They are wrapped in
their own modules, so every place that draws them is covered: Studio's canvas and a chat reply
today, and a theme page through the SDK cards area once A-12a ships it.

The host (`[data-protected-surface]`) and the inner wrapper (`[data-protected-surface-inner]`)
each carry these declarations inline, each marked `!important`:

| Property | Value |
|---|---|
| `display` | `block` |
| `visibility` | `visible` |
| `opacity` | `1` |
| `pointer-events` | `auto` |
| `position` | `static` |
| `transform` | `none` |
| `filter` | `none` |
| `clip-path` | `none` |
| `overflow` | `visible` |

The first five are PA-15's. `static` also voids any `top` or `left` a theme sets, so a theme
cannot move the card off-screen that way.

An important declaration attached to the element comes before every stylesheet declaration in
the cascade. That includes important ones and layered ones, because element-attached styles are
sorted before layers. Theme CSS cannot be important at all, so no theme rule can hide these two
elements or switch off their clicks.

`visibility` and `pointer-events` inherit. A rule that takes them from an ancestor is therefore
undone at the host.

## Why option (2), not a shadow root

A shadow root would have to carry core's stylesheet inside it. Next serves that stylesheet as
hashed CSS chunks, so every chunk would have to be linked again inside each root.

Studio's scoped rules, such as `:is(.studio, .sh-chat) …` in `globals.css`, would also stop
matching. A selector inside a shadow tree cannot see the light-DOM `html.studio` above it.

The cards would render through a portal into each root as well. All of that changes how the
cards look and behave in every style, Classic included, to defend against the one attack that
option (2) leaves open: a scoped rule aimed at an element *inside* the card.

That attack stays a review item, below. If the hostile spec fails in CI on option (2), option (1)
is the fallback, and this page changes with it.

## What stays a review rule (PRD §4.4)

- **Ancestors.** `display: none`, `opacity: 0` or an off-screen move on an element around the card
  hides the card with it, and nothing inside the card can undo that. Core or the theme's frame
  draws that region, and the frame is reviewed. This narrows the A-12c proof line in the phase A
  plan (§2.6), which lists `display`, `opacity` and off-screen on the parents too. The spec takes
  only `visibility` and `pointer-events` from ancestors, because neither PA-15 option can beat the
  other three there. The narrowing waits for Gerard's acceptance in the W-A3 PR.
- **The card's own elements.** A scoped rule can still reach the card's buttons and text, under
  option (2).
- **In-page code.** A theme's JavaScript runs in the same page. It can remove the guard, the
  element or the card. For in-page code this is a review rule, not a guarantee.

The fixture's README (`e2e/browser/fixtures/hello-a12c/README.md`) says the same next to the
hostile sheet.
