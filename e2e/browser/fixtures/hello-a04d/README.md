# hello-a04d — a theme page on every Q3 route (PRD-261 A-04d)

The fixture behind `specs/theme-pages.spec.ts`. It is served as an installed theme through
`support/hello-theme.ts`'s `serveInstalledTheme`, like every `hello-<story>` fixture (PA-17).

| File | What it is |
|---|---|
| `theme.json` | The manifest: `htmlClass` `hello-a04d`, base Studio, slot `frame@1`. |
| `entry.js` | A frame that only wraps the page, and `registerPage` on every route of the host's page list (`/chat`, `/deliverables`, `/assignments`, `/missions/[id]`, `/agents`, `/documents`, `/playbooks`, `/command-center`, `/`): a section that names its route. |
| `entry.frame-only.js` | The same frame and no pages, served in place of `entry.js` for the core-page checks. |
| `theme.css` | Two rules, both under `html.hello-a04d`. |
| `presets/default.look.json` | The preset the manifest needs. |

The spec opens each of the eight Q3 pages (missions on both of its routes) under the install's
style: with `entry.js` the theme's page renders inside the theme's frame; with
`entry.frame-only.js` the core page renders there, unchanged (home forwards to `/chat`).
