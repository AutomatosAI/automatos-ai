# hello-a12h — a theme's Command Center page reading its data through the SDK (PRD-261 A-12h)

The fixture behind `specs/pages-command-center.spec.ts`. It is served as an installed theme through
`support/hello-theme.ts`'s `serveInstalledTheme`, like every `hello-<story>` fixture (PA-17).

| File | What it is |
|---|---|
| `theme.json` | The manifest: `htmlClass` `hello-a12h`, base Studio, slot `frame@1`. |
| `entry.js` | A frame that only wraps the page, and `registerPage('/command-center')`: a section that reads the stats, the Needs-you number, the feed, the schedule, the open questions and the watches through `@automatos/theme-sdk` (`pages/command-center`), opens the board's event stream with `useBoardEventStream`, and counts the `board-changed` events it pushes (`useHostEvent`). |
| `theme.css` | Two rules, both under `html.hello-a12h`. |
| `presets/default.look.json` | The preset the manifest needs. |

The spec intercepts every one of those backend reads and the stream, so each value the page shows
came through the SDK's hooks and the host's backend client. `lib/theme-sdk/__tests__/hello-a12h-fixture.test.ts`
holds the entry to the shared modules and to names a built SDK area exports.
