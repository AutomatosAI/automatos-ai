# Placement slots: which pages reach them

PRD-261 A-12b (PRD §4.3 item 6, §4.6 way 2). A theme can put Auto, or its own art, on a core page
without cloning the page. It does this by registering a component for one of three placement slots.
The slots live in two shared primitives:

| Slot | Where it draws | Props (`components/theme-host/placement-slot.tsx`) |
|---|---|---|
| `page-header.art@1` | `PageHeader`: the first child of the header row, before the title block | `{ base, title, titleAccent, eyebrow }` |
| `page-header.mascot@1` | `PageHeader`: after the title block, before the actions | `{ base, title, titleAccent, eyebrow }` |
| `empty-state.mascot@1` | `EmptyState`: in place of the icon tile | `{ base, title, description }` |

`base` is `'studio'` when the Studio layout renders (`useIsStudio()`), and `'classic'` otherwise.
A placed component draws inside `<div data-theme-slot="<slot id>">`. If it throws, the core markup
draws in its place (nothing for the header slots, the icon tile for the empty state), and the host
is told, as it is when a frame throws. If no component is registered for a slot, the primitive
renders exactly the DOM it rendered before the slots existed.
`components/theme-host/__tests__/placement-slot.test.tsx` checks this against the markup as HTML.

The reach is limited. A slot only shows on a page that renders the shared `PageHeader` or `EmptyState`,
and many pages have their own Studio components that use neither. **On a page with no slot, a theme
places Auto through a `pages` entry** (`registerPage`, A-04b/A-04d). A page the active theme owns
(its own `pages` entry) shows the theme's page, so no core slot draws there either.

## How this list was made

This list comes from reading the code at the W-A1 merge (`286957d64`), not from running it. For each
`app/**/page.tsx`, the imports were walked across `app/` and `components/`, skipping
`components/play/**` (whose `PageHeader` is a different component) and `workspace-skills-tab.tsx`
(whose `EmptyState` is a local function). A page counts when a file it reaches renders
`<PageHeader` or `<EmptyState` imported from `components/shared`. Each page's base was then read from
its `useIsStudio()` branch. A change to any of these pages, or a new user of the two primitives,
should update this file in the same pull request.

## The Q3 pages (the pages a theme may also replace)

| Route | Studio base | Classic base |
|---|---|---|
| `/chat` | none: `StudioChatShell` | none: the chat uses neither |
| `/deliverables` | `empty-state.mascot@1` only, on the Templates tab when there are no templates (`DeliverablesStudio` → `TemplateStudio` → `TemplateCards`); the Blog tab's Studio variant has its own head | `PageHeader` (the page head, and the Blog tab's head in `deliverables-blogs.tsx`); `EmptyState` on the Templates tab when empty |
| `/assignments` | none: `StudioAssignmentsHub` | `PageHeader` (`assignments-page.tsx`); `EmptyState` on the Playbooks tab when empty (`assignments-playbooks-grid.tsx` → `workflows/playbooks-tab.tsx`) |
| `/missions` | none: it redirects to `/assignments?tab=missions` | none: the same redirect |
| `/missions/[id]` | `PageHeader` (`mission-detail-page.tsx`, no Studio branch) | `PageHeader` |
| `/agents` | none: `AgentManagementStudio` | `PageHeader` (`agent-management.tsx`) |
| `/documents` (knowledge) | `PageHeader` (`document-management.tsx` renders it in both bases) | `PageHeader` |
| `/playbooks` | none: it redirects to `/assignments?tab=playbooks` | `PageHeader` (`PlaybooksPanel.tsx`) |
| `/command-center` | none: `CommandCenterShell` | `PageHeader` (`activity-page.tsx`) |
| `/` (home) | none: it redirects to `/chat` | none: the same redirect |

## Every other page with a slot

None of these has a Studio branch, so each one reaches its slot under both bases.

| Route | Slot reached | Through |
|---|---|---|
| `/tools` | Classic only: `PageHeader`. The Studio variant draws its own `cc-headrow` (`tools-dashboard.tsx`, `variant === 'studio'`) | `tools-dashboard.tsx` |
| `/analytics` | `PageHeader` | `analytics-page.tsx` |
| `/marketplace` | `PageHeader` | `marketplace-homepage.tsx` |
| `/marketplace/developer`, `/marketplace/publish`, `/marketplace/widgets` | `PageHeader` | the page files |
| `/team` | `PageHeader` | `team-management.tsx` |
| `/settings` | `PageHeader`; `EmptyState` in System settings (admins only) on the Audit logs and System prompts tabs when they are empty | `SettingsPanel.tsx` → `SystemSettingsTab.tsx` → `CredentialAuditTab.tsx`, `SystemPromptsTab.tsx` |
| `/settings/notifications`, `/settings/profile` | `PageHeader` | the page files (`local-profile-form.tsx` in the local edition) |
| `/activity/execution` | `PageHeader` | the page file |
| `/deliverables/explorer` | `PageHeader` | the page file |
| `/admin/plugins`, `/admin/plugins/upload`, `/admin/workspaces` | `PageHeader` | the page files |
| `/looks`, `/looks/style` | `PageHeader`, on the plain note shown under Classic and Studio | `components/looks/play-only-route.tsx` (until A-04b) |

So under a theme on the Studio base, Auto reaches `/missions/[id]`, `/documents` and the empty
Templates tab of `/deliverables` through slots. To reach `/chat`, `/assignments`, `/agents`,
`/command-center` or `/`, it needs a `pages` entry.

## Where the registrations come from

`PlacementSlot` (`placement-slot.tsx`) reads the active theme's registered slots from the theme
host context, through `useThemeHost()`. The theme host provider (`theme-host-provider.tsx`) puts
them there: they are the `slots` map of the theme's load result (`ThemeRegistrations.slots`,
`lib/theme-host/types.ts`), which the loader has already limited to `THEME_SLOT_IDS`. With no
active theme the host passes `null`, and every slot draws its fallback, so the page has the DOM it
had before. The browser lane proves a registered `page-header.mascot@1` draws on a live page
(`e2e/browser/specs/placement-slots.spec.ts`, fixture `fixtures/hello-a12b-live/`).

## The AI label is not a slot (PRD-264 EUAI-11a, EUAI-11b)

The in-app AI label ("Auto is an AI", or "{agent name} is an AI") is drawn by the host, never through a
placement slot or a `pages` entry: `AiLabelGate` renders it in `app/chat/layout.tsx` after the route switch,
so it shows on `/chat` whether the core's page, Play's chat or a theme's own `/chat` page is on screen, and on
Play's Conversations page. No slot id, prop or manifest field reaches it, so a theme can neither replace nor
remove it, and a theme page never renders a second copy. Every `/x/` route that uses the `chat` area renders
the same gate in its layout (the PRD-261 wave that adds `/x/`). The rule and its runtime check are in
`css-layer-model.md` and `sdk-surface-map.md` ("Host-owned AI label").
