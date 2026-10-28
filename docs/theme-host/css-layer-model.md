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

Play shows what that would break. Play's package sheet (`dist/theme.css`, built in the theme kit
from `themes/play/src/app/play-theme.css`) recolours the page with
`:where(html.play) body { background: var(--pl-page-background) }`. Core's `globals.css:272` sets
`body { @apply bg-background }`. Both selectors have the same specificity, (0,0,1). Unlayered, the
cascade falls through to order of appearance; the host appends a theme's sheet to the end of
`<head>` (`frontend/lib/theme-host/look.ts`, `applyLook`), after core's, so Play's body colour
applies. Layered, Studio's `bg-background` would win on every page, whatever the order.

`e2e/browser/specs/protected-surfaces.spec.ts` proves this in the browser. Under installed Play
(the package), it checks that `getComputedStyle(document.body).backgroundColor` follows
`--pl-page-background`, including when a sentinel colour is put on that token. Core's body rule
never reads that token.

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
- a selector that names the host's AI label, `data-ai-label` (which covers its copy's
  `data-ai-label-caption`), the same way (PRD-264 EUAI-11a, code `CSS_UNSCOPED`);
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
today, and a theme page through the SDK cards area once A-12a ships it. The voice button the host
draws for themes (the SDK's `voice.VoiceButton`, `frontend/components/voice/voice-button.tsx`) puts
its "AI voice assistant" caption and its stop control (and mute and the device for a call) in one
too, named `voice-control`, so no theme rule hides them while the microphone is open.

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
renders the label outside the theme's page slot (`AiLabelGate`, `frontend/components/ai-label/ai-label.tsx`),
checks at run time that a person can see it, and, when they can't, draws the same words again in a layer of
its own that theme CSS can't reach. The chat keeps working: nothing is disabled and nothing is paused. Since
28 Oct 2026 this replaces the chat lock (the old `AiLabelFence` pause and its "chat is paused" notice): the
product rule is that the AI label is a small caption and never locks the composer.

### The check (`frontend/lib/ai-label/visibility.ts`)

A false alarm costs nothing but the copy, so the check is strict: what it can't confirm reads as hidden. The
label's words sit in their own element (`data-ai-label-text`), and the label counts as seen only when:

| Reason when it fails | What the check reads |
|---|---|
| `display`, `visibility`, `opacity` | No `display: none` on the words or around them; the words visible; their opacity times their ancestors' at least 0.5 |
| `paint` | No `clip-path`, `mask-image`, `-webkit-mask-image` or `filter`, and no `content-visibility: hidden`, on the words, the label or any ancestor |
| `font` | A font size of at least 9 px |
| `color` | The fill (`-webkit-text-fill-color`, else `color`) is not transparent and has at least 3:1 against what is behind it: the first opaque `background-color` up the ancestors with the translucent ones between painted over it, or the page's own white (and black too, where the root allows a dark scheme). A colour in any space is read: `rgb()` and `color(srgb)` as text, any other through a 1 by 1 canvas |
| `size`, `viewport` | The label's box at least 4 px each way and inside the viewport |
| `text` | The words themselves, measured as a Range: at least 9 px tall and 0.3 em per character wide, wholly inside the viewport, and inside the label's own box when the label clips its overflow. So `text-indent`, a tiny or clipped box, `letter-spacing` and a scale transform all fail |
| `covered` | Hit-testing at the start, middle and end of the words finds the words' own element on top. The label is `pointer-events: none`, which hit-testing skips, so the label and its words take `pointer-events: auto` for the probe only and get their own value back. An ancestor's clipping, an overlay of the theme's, and the label's own `::before` or `::after` (which hits as the label, not as its words) all fail |

Where the browser can't measure text (no Range boxes) or hit-test (no `elementsFromPoint`), the label reads as
hidden. The check runs on mount, when `<html>`'s classes or attributes or `<head>`'s elements change, on resize
and every two seconds, and logs a theme-host warning the first time it finds the label hidden and again only
when the way changes: `[theme-host] the active theme hides the page AI label (color); the host draws it in its
own layer instead`.

### The copy (`frontend/components/ai-label/protected-caption.tsx`, `protected-caption-parts.ts`)

- **A closed shadow root.** The words sit in a closed shadow root of an element the host appends to `<body>`
  itself (`data-ai-label-caption`), outside the theme's page, frame and slots. No selector in a document
  stylesheet matches inside a shadow root, and `host.shadowRoot` is null to a theme's code. The words carry
  `all: initial` and their own values inline, every one `!important`, so nothing they would inherit from a
  theme's `<html>` or `<body>` reaches them, and they read no custom property (their colours are the system's
  `Canvas` and `CanvasText`). They wrap inside the viewport, and an agent name past 40 characters is cut with
  an ellipsis (`frontend/lib/ai-label/copy.ts`), so " is an AI" is never lost.
- **The cards' guard on its element.** The element in the page's DOM carries `PROTECTED_SURFACE_GUARD` inline,
  every value `!important`, but fixed, empty (0 by 0) and click-through, and also `will-change`, `perspective`,
  `backdrop-filter` and `container-type` closed. As for the cards, an important declaration attached to the
  element beats every stylesheet declaration.
- **The browser's top layer.** The words are a manual popover, shown with `showPopover()`: above every
  `z-index`, and out of reach of what an ancestor does to its descendants (opacity, filter, transform,
  clip-path, overflow). Where the browser has no popover, they stay a fixed element.
- **It heals itself.** A MutationObserver on `<body>`'s children, the host's attributes and the whole shadow
  tree rebuilds the copy from scratch the moment it is removed, moved or changed (its words, its guard). Its
  own `toggle` event re-shows it after a `hidePopover()`. `beforetoggle` and `toggle` caught at the document in
  the capture phase, and `fullscreenchange`, raise it again (hide, then show, which puts it last in the top
  layer) whenever something else enters the top layer. A watchdog on the same two-second tick checks it is
  whole, rebuilds it if not, and raises it, for whatever no event reports (a dialog in a theme's own shadow
  root). A theme that undoes every repair gets five immediate repairs a second and the watchdog's after that,
  so it can't make the page spin. Each kind of tampering logs one warning: `[theme-host] something in the page
  removed the AI label's protected copy; the host put it back`.

Auto's floating widget draws its own label in its panel (`WidgetAiLabelFrame`), checked the same way, and its
copy at the top of the panel, kept inside the viewport.

### What the review still flags

The scope check refuses every selector that names `data-ai-label` (the label, its words' `data-ai-label-text`,
and the copy's `data-ai-label-caption`), as it refuses the cards' names. The LLM stage's "Deception" category
names content placed over, hiding or imitating the host's AI label, and code that removes, moves or restyles
it (`orchestrator/services/themes/llm_stage.py`).

### The SDK

`cards.AiLabelFence` stays exported, because installed themes (Play, HUD) wrap their composers in it, but it
draws its children alone and disables nothing. The SDK's chat (EUAI-11b): Send and Try again on
`useChatController(conversation)` ask the label store (`frontend/lib/ai-label/store.ts`) at the moment they are
called, and return false, sending nothing, only when no page label is mounted
(`frontend/lib/theme-sdk/internal/ai-label-gate.ts`): a theme that opens a chat in its frame on a page with no
label starts no turn. A hidden label is still on screen, in the host's layer, so it never stops a turn. Auto's
widget's label never clears a theme's chat. Stop always runs.

### Guaranteed at run time, and what stays a review rule

Guaranteed against theme CSS: every way of hiding the label in the check's table brings the copy, and no
stylesheet rule hides, fades, moves, clips, covers or recolours the copy. Guaranteed against a theme's code
once the copy exists: removing, moving or editing it, or hiding it with `hidePopover()`, is undone at once, and
a popover, dialog or fullscreen element shown later is put back under it.

Review rules, not guarantees:

- **Paint the check doesn't read:** a background image behind the words, and a font of the theme's whose
  glyphs are blank (its `@font-face` family must start with the theme's class, so the review sees it).
- **Code the theme runs before the copy exists:** wrapping `attachShadow`, `showPopover` or
  `MutationObserver` to reach or quiet the copy, or a loop that undoes every repair (it can't spin the page,
  but each undo shows for up to two seconds). Sandboxing a theme's code is not something the host does; the
  review reads that code, and installed themes are signed.
- **A top-layer element of the theme's opened in its own shadow root** sits above the copy until the next
  watchdog tick, two seconds at most.
