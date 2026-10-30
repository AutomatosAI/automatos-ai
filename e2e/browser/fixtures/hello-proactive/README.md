# hello-proactive — the proactive defaults theme (PRD-268 US-C21f)

The fixture behind `specs/proactive-defaults.spec.ts`. It is served as an installed theme through
`support/hello-theme.ts`'s `serveInstalledTheme`, like every `hello-<story>` fixture (PA-17).

| File | What it is |
|---|---|
| `theme.json` | The manifest: `htmlClass` `hello-proactive`, base Studio, SDK `^1.3.0`, one frame slot. |
| `entry.js` | A plain frame, and the route `hello-proactive` (served at `/x/hello-proactive`) that places the SDK 1.3.0 proactive defaults with fixture data: `BriefCard`, `SuggestionCard` (the card and the pinned line), `StatusLabel` (one row per display state), `ReadinessList` (one item per state, plus a browser row), `CalibrationCard` and `ReplyChips`. It imports only `react`, `react/jsx-runtime` and `@automatos/theme-sdk`. |
| `theme.css` | The theme's own stylesheet, every rule under `html.hello-proactive`, on the theme's own elements only. The frame's ground, ink and rule are the host's tokens, so it follows the tone. |
| `presets/default.look.json` | The preset the manifest needs: the route's gap and width. |

## What the spec proves

- A theme places every proactive default from the SDK alone, and each draws the host's own words,
  glyphs and controls, under Studio light and Studio dark, with a screenshot of each.
- Auto's words (the brief's greeting, prose and rows, the suggestion's title and detail, the
  calibration lines and offers, the chips) carry `data-ai-generated`, and each card that holds them
  shows the tiny "Auto is an AI" caption once.
- Nothing starts, checks, rates or speaks on render: no proactive route, readiness check, brief
  feedback or speech is called. Listen reads the brief the host holds (the spec answers the day's
  message), and it waits to be pressed.

Every handler only writes what the card handed the theme into the route's last line; the data is
test data, and the spec answers every backend call the cards could make, so nothing real is touched.

Classic is not here: the host runs every installed theme on Studio (`lib/ui-style.ts`
`rendersStudio`, `lib/theme-host/manifest.ts` accepts only `base: "studio"`), and under Classic it
serves no theme route (`app/x/[...path]/page.tsx` sends the member to the chat).
