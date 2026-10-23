# hello-a04b — a theme route at /x/hello (PRD-261 A-04b)

The fixture behind `specs/theme-routes.spec.ts`. It is served as an installed theme through
`support/hello-theme.ts`'s `serveInstalledTheme`, like every `hello-<story>` fixture (PA-17).

| File | What it is |
|---|---|
| `theme.json` | The manifest: `htmlClass` `hello-a04b`, base Studio, slot `frame@1`. |
| `entry.js` | A frame that only wraps the page, and `registerRoute('hello', …)`: a section that says hello. |
| `theme.css` | Two rules, both under `html.hello-a04b`. |
| `presets/default.look.json` | The preset the manifest needs. |

The spec opens `/x/hello` under the install's style and checks that the route renders inside the
theme's frame. Under the same style it also checks `/chat` at 1440 px and 800 px and that a cold
load mounts the theme's frame once (until W-A8 those cases ran under builtin Play);
`specs/play-parity/play-routes.spec.ts` holds installed Play, the theme kit's package, to the same.
