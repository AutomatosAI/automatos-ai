# hello-a12g — a playbooks page that reads through the SDK (PRD-261 A-12g)

The fixture behind `specs/pages-playbooks.spec.ts`. It is served as an installed theme through
`support/hello-theme.ts`'s `serveInstalledTheme`, like every `hello-<story>` fixture (PA-17).

| File | What it is |
|---|---|
| `theme.json` | The manifest: `htmlClass` `hello-a12g`, base Studio, slot `frame@1`. |
| `entry.js` | A frame that only wraps the page, and `registerPage('/playbooks')` (Gerard's Q3 "playbooks", D0). The page lists the playbooks (`useWorkflowPlaybooks`), each with an Open button; the open playbook shows its steps (`useWorkflowPlaybook`), its runs (`usePlaybookExecutions`) and a Run button (`useExecutePlaybook`) that shows the new run's id. |
| `theme.css` | Two rules, both under `html.hello-a12g`. |
| `presets/default.look.json` | The preset the manifest needs. |
| `responses.json` | Test data only: the bodies the spec answers `GET /api/workflow-recipes`, `GET /api/workflow-recipes/<template_id>`, `GET /api/workflow-recipes/<template_id>/executions` and `POST /api/workflow-recipes/<template_id>/execute` with, in the shapes of `WorkflowTemplate.to_dict` and `RecipeExecution.to_dict`. |

The entry imports only `react`, `react/jsx-runtime` and `@automatos/theme-sdk`, so whatever the page
draws came through the SDK's `pages/playbooks` area, the core hooks it wraps
(`hooks/use-playbook-api.ts`) and the host's backend client.
`frontend/lib/theme-sdk/__tests__/hello-a12g-fixture.test.ts` holds the files to that.
