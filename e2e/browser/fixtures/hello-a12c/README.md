# hello-a12c — the hostile theme (PRD-261 A-12c)

The fixture behind `specs/protected-surfaces.spec.ts`. It is served as an installed theme through
`support/hello-theme.ts`'s `serveInstalledTheme`, like every `hello-<story>` fixture (PA-17).

| File | What it is |
|---|---|
| `theme.json` | The manifest: `htmlClass` `hello-a12c`, base Studio, one frame slot. |
| `entry.js` | A frame that only wraps the page. |
| `theme.css` | The hostile sheet. Every rule is under `html.hello-a12c`, with no `!important` and no layer, as the server's scope check requires. |
| `presets/default.look.json` | The preset the manifest needs. |
| `chat-turns.json` | One recorded chat turn per card, in the stream format `lib/chat/hooks.ts` reads. The spec answers `POST /api/chat` with it, so the cards reach the canvas the way they do in the app. Test data only. |

## What the hostile sheet does

On each card's two protected elements, the host (`[data-protected-surface]`) and its inner
wrapper (`[data-protected-surface-inner]`), it sets:

- `display: none`;
- `opacity: 0` and `filter: opacity(0)`;
- `visibility: hidden`;
- an off-screen move: `position: fixed` at -10000 px, a -10000 px `transform` and `clip-path: inset(50%)`;
- `pointer-events: none`.

On every ancestor of a card it sets `visibility: hidden` and `pointer-events: none`.

The spec checks that each of these rules matched. Then it checks that each card is still drawn,
is on screen, and posts its decision when you click Approve & run.

## What this proves, and what it does not

The guard is an inline `!important` declaration on the two elements
(`frontend/components/theme-host/protected-surface.tsx`). An important declaration attached to the
element wins over every stylesheet declaration, and theme CSS cannot use `!important` at all.
So no theme **rule** can hide those two elements or switch off their clicks. Because `visibility`
and `pointer-events` inherit, the host also gives the card back what a rule takes away from an
ancestor.

Some things are outside what this proves:

- **Ancestors.** `display: none`, `opacity: 0` or an off-screen move on an element *around* the
  card hides the card with it. Nothing inside the card can undo that. The region that holds a card
  is drawn by core or by the theme's frame, and the frame is reviewed.
- **Elements inside the card.** The guard covers the host and the inner wrapper only. This is
  option (2) of PA-15: no shadow root. The card's own elements (its buttons, its text) can still
  be reached by a scoped theme rule. That is a review item for every theme that ships CSS.
- **Code in the page.** A theme's JavaScript runs in the same page. It can remove the inline
  style, the element, or the whole card. **For in-page code this is a review rule, not a
  guarantee (PRD §4.4).** A theme is reviewed before it is listed, and a private upload is the
  workspace's own code, run only after each member opts in.
