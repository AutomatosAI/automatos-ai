# hello-a12a — the host-cards theme (PRD-261 A-12a)

The fixture behind `specs/host-cards.spec.ts`. It is served as an installed theme through
`support/hello-theme.ts`'s `serveInstalledTheme`, like every `hello-<story>` fixture (PA-17).

| File | What it is |
|---|---|
| `theme.json` | The manifest: `htmlClass` `hello-a12a`, base Studio, one frame slot. |
| `entry.js` | A frame whose header renders the SDK's `HeaderUtilities`, and the route `hello-a12a` (served at `/x/hello-a12a`) that places the SDK's `ToolApproval` for one grant and shows what its `onDecided` reported. It imports only `react`, `react/jsx-runtime` and `@automatos/theme-sdk`. |
| `theme.css` | The theme's own stylesheet, every rule under `html.hello-a12a`, on the theme's own elements only. |
| `presets/default.look.json` | The preset the manifest needs. |

## What the spec proves

- A theme's frame can render the host's header utilities (Help and the docs, notifications, the
  theme menu, the account) from the SDK alone (PRD §4.4).
- A theme places the host's tool approval card instead of drawing its own: the card is the host's
  protected surface (`data-protected-surface="tool-approval"`), and **Approve & run** posts the
  host's own decision, `POST /api/v1/approval-grants/12101/grant`, which the spec answers. The card
  then shows the host's outcome line and tells the theme through `onDecided`.

The grant is test data; the spec answers the decision itself, so no real grant is touched.
