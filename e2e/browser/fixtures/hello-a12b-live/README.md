# hello-a12b-live — a placement slot on a live page (PRD-261 A-12b)

The fixture behind `specs/placement-slots.spec.ts`. It is served as an installed theme through
`support/hello-theme.ts`'s `serveInstalledTheme`, like every `hello-<story>` fixture (PA-17).

| File | What it is |
|---|---|
| `theme.json` | The manifest: `htmlClass` `hello-a12b-live`, base Studio, slots `frame@1` and `page-header.mascot@1`. |
| `entry.js` | A frame that only wraps the page, and a `page-header.mascot@1` component that shows the header's title. |
| `theme.css` | Two rules, both under `html.hello-a12b-live`. |
| `presets/default.look.json` | The preset the manifest needs. |

The spec opens `/documents`, which renders the shared `PageHeader` under both bases
(`docs/theme-host/placement-reach.md`), and checks that the mascot draws inside the header's
`data-theme-slot="page-header.mascot@1"` host, next to the page title.
