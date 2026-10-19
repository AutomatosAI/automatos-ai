# hello-a12d — a theme's agents page on the SDK (PRD-261 A-12d)

The fixture behind `specs/pages-agents.spec.ts`. It is served as an installed theme through
`support/hello-theme.ts`'s `serveInstalledTheme`, like every `hello-<story>` fixture (PA-17).

| File | What it is |
|---|---|
| `theme.json` | The manifest: `htmlClass` `hello-a12d`, base Studio, slot `frame@1`. |
| `entry.js` | A frame that only wraps the page, and `registerPage('/agents', …)`: a section drawn from the SDK's `pages/agents` area alone (`useAgents()` for the roster, `useAgentStats()` for the counts) with a Refresh button that calls the roster's `refetch`. |
| `responses.json` | Test data: the answers the spec gives `GET /api/agents/` and `GET /api/system/agent-statistics`. The stats count seven agents and the roster lists two, so each number on the page shows which read it came from. |
| `theme.css` | Two rules, both under `html.hello-a12d`. |
| `presets/default.look.json` | The preset the manifest needs. |

The spec opens `/agents` under the install's style: the theme's page renders inside its frame with
the intercepted roster and counts, never the core Agent Management page, and Refresh reads the
roster again through the host's backend client.
