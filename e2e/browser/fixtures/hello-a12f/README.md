# hello-a12f — a knowledge page that reads through the SDK (PRD-261 A-12f)

The fixture behind `specs/pages-documents.spec.ts`. It is served as an installed theme through
`support/hello-theme.ts`'s `serveInstalledTheme`, like every `hello-<story>` fixture (PA-17).

| File | What it is |
|---|---|
| `theme.json` | The manifest: `htmlClass` `hello-a12f`, base Studio, slot `frame@1`. |
| `entry.js` | A frame that only wraps the page, and `registerPage('/documents')` (Gerard's Q3 "knowledge", D0). The page draws the documents (`useDocuments`, a Delete button each that runs `useDeleteDocument`), the total (`useDocumentStats`), the teams with their counts (`useTeams`, `useDocumentTeamCounts`), the cloud connections (`useCloudConnections`) and the database sources (`useDatabaseKnowledge`). |
| `theme.css` | Two rules, both under `html.hello-a12f`. |
| `presets/default.look.json` | The preset the manifest needs. |
| `responses.json` | Test data only: the bodies the spec answers `GET /api/documents/`, `GET /api/documents/analytics`, `GET /api/teams`, `GET /api/documents/team-counts`, `GET /api/cloud-documents/connections`, `GET /api/knowledge/sources/database/` and its templates list with, in the shapes the backend sends. |

The entry imports only `react/jsx-runtime` and `@automatos/theme-sdk`, so whatever the page draws came
through the SDK's `pages/documents` area, the core hooks it wraps and the host's backend client.
`frontend/lib/theme-sdk/__tests__/hello-a12f-fixture.test.ts` holds the files to that.
