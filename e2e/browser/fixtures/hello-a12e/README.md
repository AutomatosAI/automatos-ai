# hello-a12e — missions pages that read through the SDK (PRD-261 A-12e)

The fixture behind `specs/pages-assignments.spec.ts`. It is served as an installed theme through
`support/hello-theme.ts`'s `serveInstalledTheme`, like every `hello-<story>` fixture (PA-17).

| File | What it is |
|---|---|
| `theme.json` | The manifest: `htmlClass` `hello-a12e`, base Studio, slot `frame@1`. |
| `entry.js` | A frame that only wraps the page, and `registerPage` on both missions routes (Gerard's Q3 "missions", D0). `/assignments` lists the Missions (`useMissions`, each linked with `missionHref` and `HostLink`) and the recommendations (`useRecommendedAssignments`); `/missions/[id]` reads the id from `useHostPathname`, the Mission from `useMission`, and pauses it with `usePauseMission`. |
| `theme.css` | Two rules, both under `html.hello-a12e`. |
| `presets/default.look.json` | The preset the manifest needs. |
| `responses.json` | Test data only: the bodies the spec answers `GET /api/missions`, `GET /api/missions/<id>`, `POST /api/missions/<id>/pause` and `GET /api/assignments/recommended` with, in the shapes of `frontend/types/missions.ts` and `hooks/use-assignments-api.ts`. |

The entry imports only `react/jsx-runtime` and `@automatos/theme-sdk`, so whatever the pages draw came
through the SDK's `pages/assignments` area, the core hooks it wraps and the host's backend client.
`frontend/lib/theme-sdk/__tests__/hello-a12e-fixture.test.ts` holds the files to that.
