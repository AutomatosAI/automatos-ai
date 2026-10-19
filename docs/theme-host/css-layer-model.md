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
- a selector that names a protected surface's attribute, `data-protected-surface` or
  `data-protected-surface-inner`, anywhere in it: in an attribute selector, inside `:has()` or
  `:not()`, in any case, escaped or not (A-12c, code `CSS_UNSCOPED`). So no theme rule can reach
  the approval cards, or the buttons and text inside them, by name;
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
| `transform`, `translate`, `scale`, `rotate` | `none` |
| `offset-path` | `none` |
| `filter` | `none` |
| `clip-path` | `none` |
| `mask`, `-webkit-mask` | `none` |
| `content-visibility` | `visible` |
| `contain` | `none` |
| `zoom` | `1` |
| `margin` | `0px` |
| `z-index` | `auto` |
| `width`, `height` | `100%` |
| `overflow` | `visible` |

The first five are PA-15's. `static` also voids any `top` or `left` a theme sets, so a theme
cannot move the card off-screen that way. The rest close every other property that hides, moves,
shrinks, cuts away or skips painting a box on its own: the transform properties and a motion
path, a mask, `content-visibility: hidden`, `contain` with a zero size, `zoom`, a negative margin,
a negative `z-index` on a flex item, and a zero width or height. `width` and `height` are the same
`100%` the wrappers' `h-full w-full` classes set, and both wrappers are the only child of a core
element (WidgetWrapper's `h-full` div on the canvas, the card cell in Play's chat), so no core
rule sets any of these on them and the guard changes no layout.

An important declaration attached to the element comes before every stylesheet declaration in
the cascade. That includes important ones and layered ones, because element-attached styles are
sorted before layers. Theme CSS cannot be important at all, so no theme rule can hide, move or
shrink these two elements or switch off their clicks.

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

The scope check refuses that attack when the rule names the card. A generic selector that
matches the card's elements stays a review item, below. If the hostile spec fails in CI on
option (2), option (1) is the fallback, and this page changes with it.

## What is guaranteed, and what stays a review rule (PRD §4.4)

Guaranteed, against theme CSS:

- **The card's own two boxes.** The host and the inner wrapper keep every value in the table
  above, whatever a stylesheet rule says, and `visibility` and `pointer-events` taken from an
  ancestor are given back at the host.
- **Rules that name the card.** The scope check refuses every selector that names
  `data-protected-surface` or `data-protected-surface-inner`, so a theme cannot aim a rule at the
  card, at anything inside it through it (`[data-protected-surface-inner] button`), or at its
  ancestors through it (`:has([data-protected-surface])`).

A review rule, not a guarantee:

- **Generic selectors.** A rule such as `html.<htmlClass> button` matches the card's buttons and
  text too, without naming the card. Under option (2) nothing stops that: no shadow root.
- **Ancestors.** `display: none`, `opacity: 0` or an off-screen move on an element around the card
  hides the card with it, and nothing inside the card can undo that. Core or the theme's frame
  draws that region, and the frame is reviewed. This narrows the A-12c proof line in the phase A
  plan (§2.6), which lists `display`, `opacity` and off-screen on the parents too. The spec takes
  only `visibility` and `pointer-events` from ancestors, because neither PA-15 option can beat the
  other three there. The narrowing waits for Gerard's acceptance in the W-A3 PR.
- **Overlays.** An element the theme draws over the card (its frame, a pseudo-element, a fixed
  layer) can cover it or take its clicks.
- **In-page code.** A theme's JavaScript runs in the same page. It can remove the guard, the
  element or the card. For in-page code this is a review rule, not a guarantee.

The fixture's README (`e2e/browser/fixtures/hello-a12c/README.md`) says the same next to the
hostile sheet.

## Host-owned AI label (PRD-264 EUAI-11a, EUAI-11b)

The in-app AI label ("Auto is an AI", EU AI Act Art. 50(1)) is protected differently from the approval cards.
It carries no inline `!important` guard: a guard would beat a theme's rule on the label itself, but not a
`display: none`, `opacity` or off-screen move on an element around it (the review rules above). So the host
renders the label outside the theme's page slot (`AiLabelGate`, `frontend/components/ai-label/ai-label.tsx`)
and checks at run time that a person can see it (`frontend/lib/ai-label/visibility.ts`): displayed, visible,
at least half opaque through all its ancestors, at least 4 px each way and wholly inside the viewport.

While the check fails, the chat is paused: every composer inside `AiLabelFence` is disabled, and the chat says
"This theme hides the AI notice, so chat is paused. Switch theme in Settings." A theme rule that hides, fades,
shrinks or moves the label therefore never leaves a chat running without it; it stops the chat. The fence
wraps only the composer, so a theme can't use the label to switch off the approval cards either.

Not checked, and so review rules for the label as for the cards: an element the theme draws over it (an
overlay, a pseudo-element, a fixed layer), and code the theme runs in the page.
