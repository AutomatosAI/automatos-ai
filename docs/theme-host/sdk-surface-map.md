# The theme SDK's surface map (PRD-261 A-10c, PA-16)

Every core module Play's folders import, and every data hook the Q3 pages read, mapped name by name to what a
theme uses instead once Play is a package. A theme imports one module from the host, `@automatos/theme-sdk`
(`frontend/lib/theme-sdk/index.ts`); it never imports `@/…`, `next/*`, zustand stores, react-query, the backend
client or sonner (PRD §4.3 item 7).

Each row says one of four things in its **Map** column:

- **SDK**: the theme reaches what the module gives through the names in **SDK names**, written `area.name`.
  Each one is planned in `frontend/lib/theme-sdk/surface/<area>.json` (`pages/agents` is `pages-agents.json`).
  An area marked `stub` there throws "not in this SDK build" until its lane lands; the lane fills bodies and
  never renames. A row can be SDK for some of its names and copied or host-only for the rest; **Why** says which.
- **copied into Play**: pure utilities, types and Play's own code that lives in core folders today. Play's
  package carries its own copy (types also come from the SDK's `.d.ts`, A-10b).
- **host-only**: the host keeps it and does the job itself; **Why** says what a theme does instead.
- **shared module**: the host's own instance, served to the theme through the import map
  (`frontend/lib/theme-host/shared-modules.json`).

`frontend/lib/theme-sdk/__tests__/surface-map.test.ts` holds this file to the code: every `area.name` here is in
its surface file, every surface name is somewhere here, and a scan of `components/{play,looks,mascot}` and
`lib/{play,looks,mascot}` (their runtime files, not their tests) finds no `@/` import and no package import that
has no row. A row ending in `/*` covers every module under that folder.

**Changing the surface.** A new SDK name needs a row here and a name in its surface file in the same commit (the
area lanes of W-A4 and W-A4P add none; a missing one is a seam-review item, PA-16). The SDK is first published by
W-A3, so the names A-10c added on top of A-10a's need no version bump; any change after W-A3 merges bumps
`sdkVersion` (PRD §4.3 item 7).

Paths in **Read by** are under `frontend/`: `c/` is `components/`, `l/` is `lib/`.

## 1. The core modules Play's folders import (74: 61 in §0, W-A1's three, the rows W-A2 added and PRD-264's one)

| Module | Read by | Map | SDK names | Why |
|---|---|---|---|---|
| `@/components/ai-label/ai-label-fence` | c/play/chat/message-box.tsx | host-only | — | PRD-264 EUAI-11a: the host's AI label fence around the composer, which disables it while a theme hides the label. The host keeps it; once Play is a package, the SDK's chat area pauses its composer the same way (see "Host-owned AI label" below). |
| `@/components/chatbot/mission-created-card` | c/play/chat/mission-suggestion.tsx | SDK | `cards.MissionCreatedCard` | The host's card for the Mission the chat created. |
| `@/components/chatbot/mission-suggestion-card` | c/play/chat/mission-suggestion.tsx | SDK | `cards.MissionSuggestionCard` | The host's card offering a Mission. |
| `@/components/chatbot/task-card` | c/play/chat/reply-cards.tsx | SDK | `cards.TaskCard` | The host's task card. |
| `@/components/chatbot/withdraw-offer-card` | c/play/chat/reply-cards.tsx | SDK | `cards.WithdrawOfferCard` | The host's withdraw-offer card. |
| `@/components/icons/deliverable-icon` | l/play/shelf.ts | SDK | `deliverables.deliverableLabel` | A deliverable type's label; the icons themselves are the theme's own. |
| `@/components/layout/main-layout` | c/play/chat/play-chat.tsx, c/play/deliverables/play-shelf.tsx, c/looks/play-only-route.tsx | host-only | — | The host draws the frame: a theme registers `frame@1` and the host's ThemeSlot wraps every theme page in it (A-04a, A-04b). |
| `@/components/layout/studio-header` | c/play/shell/play-frame.tsx | SDK | `cards.HeaderUtilities` | The header utilities (docs and help, the theme menu, notifications, the account) every frame must show (PRD §4.4). |
| `@/components/local/first-run-nudge` | c/play/chat/play-chat.tsx | SDK | `cards.FirstRunNudge` | The local edition's first-run nudge. |
| `@/components/onboarding/trial-exhausted-banner` | c/play/chat/turn-end.tsx | SDK | `cards.TrialExhaustedBanner` | The host's trial banner. |
| `@/components/shared/markdown-view` | c/play/chat/play-markdown.tsx, c/play/deliverables/deliverable-content.tsx | SDK | `cards.MarkdownView` | The host's sanitising Markdown renderer; a theme never renders untrusted Markdown itself. |
| `@/components/settings/themes/theme-labels` | c/looks/play-only-route.tsx | host-only | — | The same note: its Switch to Play skips an install of Play's package that is turned off (`isDisabled`), as the host's theme menu does (A-02c). A theme never lists or picks installs. |
| `@/components/shared/page-header` | c/looks/play-only-route.tsx | host-only | — | Only the "Looks are part of Play" note uses it; that note is the host's plain `/looks` page under Classic and Studio (PA-14, A-04b). |
| `@/components/ui/button` | c/looks/play-only-route.tsx | host-only | — | Same note as the row above; a theme's own buttons are its own. |
| `@/components/widgets/FileWidget/FilePreview` | c/play/deliverables/deliverable-content.tsx, c/play/deliverables/shelf-card.tsx | SDK | `deliverables.useAuthenticatedBlobUrl` | A file the backend serves only with the caller's token, as a blob URL; a theme cannot send the token. The preview component itself is `cards.FilePreview`. |
| `@/components/widgets/MissionApprovalWidget` | c/play/chat/approval-cards.tsx | SDK | `cards.MissionApproval` | A protected surface: the host draws it (A-12c). |
| `@/components/widgets/ToolApprovalWidget` | c/play/chat/approval-cards.tsx | SDK | `cards.ToolApproval` | A protected surface: the host draws it (A-12c). |
| `@/components/widgets/types` | c/play/chat/approval-cards.tsx, l/chat/approvals.ts | copied into Play | — | Types only (`ToolApprovalWidgetData`, `MissionApprovalWidgetData`, `MissionApprovalTask`, `WidgetMetadata`). |
| `@/components/workspace-provider` | c/play/deliverables/play-shelf.tsx, c/play/chat/turn-end.tsx, c/play/shell/play-sidebar.tsx, c/play/shell/play-nav.ts, c/looks/account-look-scope.tsx; /documents (document-management.tsx) | SDK | `context.useWorkspace`, `navigation.useCoreNav` | The workspace a theme may see; the plan exposure (`WorkspaceExposure`) is applied inside `useCoreNav`. account-look-scope is host code from A-13c (the look runtime). |
| `@/components/workspace/gallery-view/made-by` | l/play/detail.ts | SDK | `cards.MadeBy` | Who made a deliverable; Play reads the source labels (`MADE_BY_SOURCE`) to build the line, which `MadeBy` gives whole. |
| `@/contexts/role-context` | c/play/shell/play-sidebar.tsx | SDK | `context.useRole` | The caller's workspace and platform roles. |
| `@/contexts/ui-style-context` | c/looks/look-provider.tsx, c/looks/play-only-route.tsx | host-only | — | The host decides which theme is active and mounts its look runtime (A-13c); a theme reads its tone with `context.useTone` and its look with `looks.useLook`. |
| `@/hooks/use-api-file-download` | c/play/deliverables/detail-panel.tsx, c/play/deliverables/deliverable-content.tsx | SDK | `deliverables.useFileDownload` | Downloads a generated file through the backend client. |
| `@/hooks/use-auto-mascot` | c/play/chat/play-chat-view.tsx | copied into Play | — | Play's own mascot driver in core's `hooks/` (§0); it moves with Play (A-13b, A-16b). |
| `@/hooks/use-backend-health` | c/play/chat/play-chat-view.tsx, c/play/edges/asleep-gate.tsx, c/play/shell/play-sidebar.tsx, c/play/shell/connection-pill.tsx, c/looks/account-look-scope.tsx | SDK | `context.useHealth` | Whether the backend answers (its `BackendConnection` comes with the SDK's `HostHealth` type). `BACKEND_HEALTH_QUERY_KEY` is read only by account-look-scope, host code from A-13c. |
| `@/hooks/use-backend-mascot` | c/play/edges/asleep-gate.tsx | copied into Play | — | Play's own mascot code in core's `hooks/`; its health input comes from `context.useHealth`. |
| `@/hooks/use-board-event-stream` | c/play/chat/play-chat.tsx; /command-center (command-center-shell.tsx) | SDK | `pages/command-center.useBoardEventStream` | Opens the board's event stream; its changes reach a theme as `events.useHostEvent('board-changed')`. |
| `@/hooks/use-conversation-list` | c/play/conversations/play-conversations.tsx | SDK | `chat.useConversationList` | The caller's conversations, newest first. |
| `@/hooks/use-deliverables-api` | c/play/deliverables/*, l/play/shelf.ts, l/play/detail.ts, l/play/seen.ts | SDK | `deliverables.useDeliverables`, `deliverables.useDeliverable`, `deliverables.DELIVERABLE_FEED_FILTERS` | The list, one deliverable, and the feed's default filters (`FEED_DEFAULT_FILTERS`). Its types (`Deliverable`, `FilterState`, the responses) come from the SDK's `.d.ts`. |
| `@/hooks/use-mascot-player` | c/mascot/auto-mascot.tsx | copied into Play | — | Play's mascot player (A-13b moves it into `lib/mascot`). |
| `@/hooks/use-media-preferences` | c/looks/look-provider.tsx, c/mascot/auto-mascot.tsx | SDK | `context.useMediaPreferences` | `usePrefersReducedMotion` is `useMediaPreferences().reducedMotion`. `useVideoFormat` is copied into Play: it reads Play's own `lib/mascot/media`. |
| `@/hooks/use-notice-seq` | c/looks/look-import.tsx, c/looks/look-manager.tsx, c/looks/look-notice.tsx, c/looks/style/style-screen.tsx | copied into Play | — | A pure React helper (a ref counter) for live-region notices. |
| `@/hooks/use-online` | c/play/edges/asleep-screen.tsx | SDK | `context.useOnline` | Whether the browser is online. |
| `@/hooks/use-play-conversation` | c/play/chat/play-chat.tsx | SDK | `chat.useChatController` | The conversation on screen; its logic is `lib/chat-controller/conversation.ts` (A-11a), and this hook adds only Play's `?draft=` link and its fresh-chat count. |
| `@/hooks/use-play-turn` | c/play/chat/play-chat-view.tsx | SDK | `chat.useChatController` | The turn's approvals, deliverables and suggestion: Play's adapter over the controller's `useChatTurn` (`lib/chat-controller/turn.ts`, A-11a). |
| `@/hooks/use-recent-chats` | c/play/shell/play-sidebar.tsx | SDK | `chat.useRecentChats` | The latest conversations for a sidebar. |
| `@/hooks/use-recheck-countdown` | c/play/edges/asleep-screen.tsx | copied into Play | — | A pure timer for the asleep screen. |
| `@/hooks/use-refresh-history-on-turn-end` | l/chat-controller/controller.ts | SDK | `chat.useChatController` | The controller refreshes the history lists when a turn ends; a theme has no query client. |
| `@/hooks/use-shelf-mascot` | c/play/deliverables/play-shelf.tsx | copied into Play | — | Play's own mascot code; its reads go through `context.useHealth` and `deliverables.useDeliverables`. |
| `@/hooks/use-start-chat` | c/play/conversations/play-conversations.tsx, c/play/deliverables/shelf-body.tsx, c/play/shell/play-sidebar.tsx | SDK | `chat.useStartChat` | New chat: a fresh draft in the host's chat session, then the chat page. |
| `@/hooks/use-stopped-reply` | l/chat-controller/controller.ts | SDK | `chat.useChatController` | Stop keeps the partial reply (FX-30); the controller owns it (A-11a). |
| `@/hooks/use-studio-theme` | c/play/conversations/play-conversations.tsx, c/looks/play-only-route.tsx | host-only | — | The host renders a theme's pages and slots only when the layout is `theme` (ThemeSlot, A-04b); a theme never branches on the layout or the width. play-route-switch.tsx and not-found-view.tsx, which also read it, are gone: A-04b moved their job into ThemeSlot, host code. |
| `@/lib/auth-edition` | c/play/shell/play-nav.ts, c/play/edges/asleep-gate.tsx | SDK | `context.useEdition`, `navigation.useCoreNav` | `isLocal` is `useEdition() === 'local'`; the edition's nav filters run inside `useCoreNav`. |
| `@/lib/auth-hooks` | c/looks/account-look-scope.tsx | host-only | — | Whose looks to show is the host look runtime's job (A-13c); a theme reads the active look with `looks.useLook`. |
| `@/lib/chat-controller/*` | c/play/chat/play-chat-view.tsx, hooks/use-play-conversation.ts, hooks/use-play-turn.ts | SDK | `chat.useChatController` | The controller itself (A-11a); a theme reaches it only through the SDK (A-11b). |
| `@/lib/chat/approvals` | c/play/chat/approval-cards.tsx, l/chat-controller/approvals.ts, l/chat-controller/view-model.ts | SDK | `chat.useChatController` | A-11a moved `components/play/chat/approvals.ts` here; the view model carries the filed approvals. |
| `@/lib/chat/deliverables` | c/play/chat/auto-reply.tsx, c/play/chat/deliverable-card.tsx, c/play/chat/message-list.tsx, c/play/deliverables/play-shelf.tsx, c/play/deliverables/shelf-card.tsx, c/looks/style/style-components.tsx, l/play/detail.ts, l/chat-controller/turn.ts, l/chat-controller/view-model.ts | SDK | `chat.useChatController` | A-11a moved the deliverables parser here; the view model carries each turn's deliverables. |
| `@/lib/chat/message-text` | c/play/chat/auto-reply.tsx, c/play/chat/message-list.tsx, c/play/chat/play-chat-view.tsx, c/play/chat/reply-cards.tsx, l/chat-controller/turn.ts | SDK | `chat.useChatController` | A-11a moved the message-text parser here; the view model carries each reply's text, and since A-14a what else a reply carries for Play's chat (`HostChatMessage`: `toolCalls`, `provenance`, `savedError`, `limitReached`, `reasoningChars`, `withdrawOffer`, `linkedDeliverables`; fields, no new SDK name). |
| `@/lib/chat/receipts` | c/play/chat/receipts.tsx, c/play/chat/auto-reply.tsx, l/chat/deliverables.ts, l/mascot/events.ts | SDK | `chat.useChatController`, `chat.receiptLine`, `chat.receiptLink`, `chat.movedSomethingOnTheBoard` | Each reply's receipts, live ones included (`receiptsOf`, `liveReceiptsFromToolData`, `LIVE_KEY`), come on the controller's view model; the line, the checked link (`safeLink`) and the board test stay the host's so every theme tells the same truth. `NO_ACTIONS` and `NOTHING_ON_THE_BOARD` are wording, copied into Play. |
| `@/lib/chat/suggestion` | c/play/chat/mission-suggestion.tsx, c/play/chat/turn-end.tsx, l/chat-controller/turn.ts, l/chat-controller/view-model.ts | SDK | `chat.useChatController` | A-11a moved the Mission-suggestion parser here; the view model carries the suggestion. |
| `@/lib/chat/tool-calls` | l/chat/message-text.ts | host-only | — | Called by the message parser A-11a moved into `lib/chat`; task cards come on the controller's view model. |
| `@/lib/chat/use-chat-with-receipts` | l/chat-controller/controller.ts, l/chat-controller/view-model.ts | SDK | `chat.useChatController` | The stream itself. |
| `@/lib/cookies` | l/looks/tone.ts | host-only | — | The host forces the tone from the theme's tone policy (`ThemeTonePolicy`); a theme never writes cookies. |
| `@/lib/deliverables/download` | c/play/deliverables/deliverable-content.tsx, c/play/deliverables/detail-panel.tsx, c/play/deliverables/play-deliverable.tsx, l/play/detail.ts | SDK | `deliverables.deliverableDownloadUrl`, `deliverables.deliverableFileUrl`, `deliverables.deliverableFilename`, `deliverables.fileDownloadUrl` | The URLs go through the backend client's base URL. `DOWNLOAD_FAILED` is wording, copied into Play. |
| `@/lib/deliverables/tabs` | c/play/deliverables/play-shelf.tsx | SDK | `deliverables.visibleDeliverableTabs`, `deliverables.DELIVERABLE_TAB_LABELS`, `navigation.DELIVERABLE_EXPLORER_HREF` | The tabs this workspace shows, their labels (`TAB_LABELS`) and the explorer's route (`EXPLORER_HREF`). |
| `@/lib/download-text` | c/looks/look-manager.tsx | copied into Play | — | A pure browser helper (a Blob URL on a throwaway link). |
| `@/lib/global-search-event` | c/play/shell/play-frame.tsx | SDK | `events.openGlobalSearch` | Opens the host's ⌘K search. |
| `@/lib/nav-exposure` | c/play/shell/play-nav.ts | SDK | `navigation.useCoreNav` | The plan-tier filter runs inside `useCoreNav`. |
| `@/lib/page-context` | l/chat-controller/controller.ts | SDK | `chat.useChatController` | The controller sends the page's context with each turn; a theme does not build it. |
| `@/lib/studio-menu` | c/play/shell/play-nav.ts | SDK | `navigation.useCoreNav`, `navigation.resolveActiveNavId` | The Studio menu after the host's filters, without icons (`StudioMenuItem` is the SDK's `CoreNavItem`). |
| `@/lib/text/plain-line` | l/play/draft.ts | copied into Play | — | Pure text helpers (`oneLine`). `UNSAFE_TEXT` is read by the look runtime's `lib/look-kit/validate.ts` (moved out of `lib/looks` by A-13a). |
| `@/lib/theme-host/boot-sanitize` | l/looks/tone.ts | host-only | — | The boot data's tone parser; the host applies the tone before first paint (A-03). |
| `@/lib/theme-host/boot-types` | l/looks/prepaint.ts | host-only | — | Builtin Play's first-paint tables (A-03, PA-7); an installed theme ships them as data in its package, not code. |
| `@/lib/theme-host/builtin/play-entry` | c/looks/look-provider.tsx, c/looks/play-only-route.tsx | host-only | — | Builtin Play's registry entry (PA-7); deleted at A-19. |
| `@/lib/theme-host/choice-sync` | c/looks/play-only-route.tsx | host-only | — | The same note: Switch to Play is the theme menu's pick (`useChooseStyle`: stored on the server, put back when refused, PA-12) and asks for the workspace's installs (`useWantInstalledThemes`, `packageIdOfBuiltin`), A-02c. A theme never chooses the workspace's theme. |
| `@/lib/theme-host/registry` | c/looks/play-only-route.tsx | host-only | — | The same note: the style an install of Play's package is picked by (`themeStyleOf`); the registry of themes is the host's. |
| `@/lib/theme-host/registry-types` | c/looks/play-only-route.tsx | host-only | — | The same note: types only (`UiStyle`), the host's style axis (A-02a). |
| `@/lib/ticket-links` | c/play/*, c/looks/*, l/play/draft.ts, l/chat/deliverables.ts, l/chat-controller/view-model.ts | SDK | `navigation.CHAT_HREF`, `navigation.chatHref`, `navigation.DELIVERABLES_HREF`, `navigation.DELIVERABLE_PARAM`, `navigation.LOOKS_HREF`, `navigation.LOOK_STYLE_HREF` | Core's route builders. |
| `@/lib/trial` | c/play/chat/turn-end.tsx | SDK | `chat.useChatController`, `context.useTrial` | A turn that ends on an exhausted trial says so on the view model (today `errorCode === TRIAL_EXHAUSTED_CODE`); the ledger is `useTrial`. |
| `@/lib/ui-style` | l/looks/tone.ts | host-only | — | The tone cookie's age; the host owns the cookie (PA-12) and forces the tone. |
| `@/lib/utils` | c/looks/look-manager.tsx | copied into Play | — | A pure utility (`copyToClipboard`). |
| `@/stores/chat-session-store` | c/play/shell/play-sidebar.tsx, l/chat-controller/controller.ts, l/chat-controller/conversation.ts | SDK | `chat.useChatController`, `chat.useRecentChats` | A store is never exposed; the controller tells the host a new chat's id, and the active chat id comes with the recent list. |
| `@/stores/mission-store` | c/play/chat/mission-suggestion.tsx | SDK | `chat.useChatController` | The Mission the chat is planning comes on the view model. |
| `@/types` | c/play/chat/*, hooks/use-play-turn.ts, l/chat-controller/*, l/chat/deliverables.ts, l/chat/message-text.ts, l/chat/suggestion.ts, l/play/conversations.ts, l/mascot/types.ts | copied into Play | — | Types only (`ChatMessage`, `Chat`, `Receipt`, `ToolCall`, `TaskCardData`); the SDK's own types are in its `.d.ts` (A-10b). |

## 2. Imports the lanes in flight will add

The scan runs on every lane's branch, so the modules a planned move puts into Play's imports are mapped before
the move lands.

| Module | Read by | Map | SDK names | Why |
|---|---|---|---|---|
| `@/stores/play-chat-store` | hooks/use-start-chat.ts, hooks/use-play-conversation.ts | SDK | `chat.useStartChat` | "Show a fresh chat" is part of New chat. |
| `@/hooks/use-celebrate-new` | hooks/use-shelf-mascot.ts | copied into Play | — | Play's own code in core's `hooks/` (§0); it moves with Play. |
| `@/hooks/use-idle-timeout` | hooks/use-auto-mascot.ts | copied into Play | — | A pure timer the mascot driver uses; it moves with it. |
| `@/lib/look-kit/*` | `lib/looks` (`engine.ts` binds Play's catalogue, prefix and look spec; the pure helpers directly), `components/looks` through `lib/looks/engine` after A-13a | copied into Play | — | The look runtime is core-owned in Phase A and the kit vendors it by version (PA-9); the active look and saving are `looks.useLook` and `looks.saveLook`. |
| `@/lib/theme-host/builtin/*` | `lib/looks` after A-09b | host-only | — | Builtin Play's registration files (PA-7: `play-look-keys.ts`, `play-boot.ts` and the rest); deleted at A-19. |
| `@/components/theme-host/*` | Play's frame and cards after A-12c, A-13c | host-only | — | The host's own components (slots, the protected-surface guard, the look runtime); a theme gets the cards through `cards.*`. |
| `@/lib/api-client` | Play's tests today; `lib/looks/store.ts` if A-09b reads the server directly | host-only | — | A theme never holds the backend client (PRD §4.3 item 7); every call it needs is an SDK hook (looks: `looks.listLooks`, `looks.saveLook`, A-13c). |

## 3. The data hooks the Q3 pages read (§0)

The top-level reads of each page's shell. Hooks behind a page's sub-tabs (governance, heartbeats, board tasks,
analytics, …) are not in SDK v1: a theme page that needs one keeps core's own tab inside `cards.CorePage`, or the
seam owner adds a row and a name here first. Chat and deliverables read the hooks in §1.

| Module | Read by | Map | SDK names | Why |
|---|---|---|---|---|
| `@/hooks/use-agent-api` | /agents (agent-management-studio.tsx) | SDK | `pages/agents.useAgents`, `pages/agents.useAgentStats` | The agent list and its stats strip (A-12d). |
| `@/hooks/use-page-api` | /agents, /assignments, /command-center, /documents (each `page.tsx`) | host-only | — | It tags the page's requests with the page name; the host's page layout keeps calling it around a theme page (A-04b). |
| `@/hooks/use-missions-api` | /assignments missions tab (assignments-hub.tsx, missions-body.tsx), /missions/[id] (mission-detail-page.tsx, mission-field-panel.tsx, mission-field-inspector.tsx, mission-step-check-toggle.tsx) | SDK | `pages/assignments.useMissions`, `pages/assignments.useMission`, `pages/assignments.useApproveMission`, `pages/assignments.useRejectMission`, `pages/assignments.usePauseMission`, `pages/assignments.useResumeMission`, `pages/assignments.useCancelMission`, `pages/assignments.useReplanMission`, `pages/assignments.useRerunMission`, `pages/assignments.useSaveAsRoutine`, `pages/assignments.useMissionField`, `pages/assignments.useFieldQuery`, `pages/assignments.useUpdateMissionSettings` | The missions list and one Mission with its actions, its field and its step-check setting (A-12e). |
| `@/hooks/use-assignments-api` | /assignments (assignments-page.tsx) | SDK | `pages/assignments.useRecommendedAssignments` | The recommended assignments (A-12e). |
| `@/hooks/use-playbook-api` | /assignments?tab=playbooks (playbooks-body.tsx, run-playbook-dialog.tsx); /playbooks forwards there under Studio | SDK | `pages/playbooks.useWorkflowPlaybooks`, `pages/playbooks.useWorkflowPlaybook`, `pages/playbooks.useExecutePlaybook`, `pages/playbooks.useCreatePlaybook`, `pages/playbooks.useUpdatePlaybook`, `pages/playbooks.useDeletePlaybook`, `pages/playbooks.usePlaybookExecutions` | The playbooks and their runs (A-12g). Classic's PlaybooksPanel calls `apiClient.listWorkflowRecipes` itself; the SDK wraps the hooks, not that call. |
| `@/hooks/use-document-api` | /documents (document-management.tsx) | SDK | `pages/documents.useDocuments`, `pages/documents.useDocumentStats`, `pages/documents.useUploadDocument`, `pages/documents.useDeleteDocument` | The knowledge documents (A-12f). |
| `@/hooks/use-teams` | /documents (document-management.tsx) | SDK | `pages/documents.useTeams`, `pages/documents.useDocumentTeamCounts` | The teams filter (A-12f). |
| `@/hooks/use-database-knowledge` | /documents (document-management.tsx) | SDK | `pages/documents.useDatabaseKnowledge` | Database sources (A-12f). |
| `@/hooks/use-cloud-storage` | /documents (document-management.tsx) | SDK | `pages/documents.useCloudConnections`, `pages/documents.useTriggerSync`, `pages/documents.useSelectRootFolder` | Cloud folders and their sync (A-12f). |
| `@/hooks/use-activity-api` | /command-center (command-center-shell.tsx) | SDK | `pages/command-center.useActivityFeed`, `pages/command-center.useActivityStats`, `pages/command-center.useActivitySchedule` | The activity feed, its stats and the schedule (A-12h). |
| `@/hooks/use-approval-grants` | /command-center (command-center-shell.tsx) | SDK | `pages/command-center.useQuestions` | The questions waiting for the owner (A-12h). |
| `@/hooks/use-needs-you` | /command-center (command-center-shell.tsx) | SDK | `pages/command-center.useNeedsYou` | The needs-you count (A-12h). |
| `@/hooks/use-watches-api` | /command-center (command-center-shell.tsx) | SDK | `pages/command-center.useWatches` | The watch list (A-12h). |

## 4. The packages Play's folders import

| Module | Read by | Map | SDK names | Why |
|---|---|---|---|---|
| `react` | every component | shared module | — | The host's React, through the import map; a theme never bundles its own. |
| `@automatos/theme-sdk` | the Play modules W-A5 ports (A-14a–f), c/play/chat/*, hooks/use-play-conversation.ts and hooks/use-play-turn.ts first (A-14a) | shared module | — | The SDK itself: the host's own instance through the import map (`lib/theme-host/shared-modules.json`); core code reaches it by the tsconfig.json path and the vitest alias to `lib/theme-sdk/index.ts`. Every SDK row above is reached through it. |
| `next/link` | c/play/*, c/looks/* | SDK | `navigation.HostLink` | The host's client-side link; an https href opens in a new tab. |
| `next/navigation` | c/play/conversations/play-conversations.tsx, c/play/deliverables/play-shelf.tsx, c/play/shell/play-sidebar.tsx | SDK | `navigation.useHostRouter`, `navigation.useHostPathname`, `navigation.useHostSearchParams` | The App Router, the path and the query string; play-route-switch read it too, before A-04b made it host code (ThemeSlot, components/theme-host/theme-slot.tsx) and deleted the file. |
| `next/dynamic` | c/play/*-lazy.tsx | host-only | — | It needs Next's compiler; a theme splits its own bundle with `React.lazy` and `import()`. |
| `lucide-react` | c/play/shell/play-frame.tsx, c/play/shell/play-sidebar.tsx | copied into Play | — | Icons are bundled into the theme's package. |
| `@tanstack/react-query` | c/looks/account-look-scope.tsx | host-only | — | A theme has no query client; account-look-scope is host code from A-13c, and every query a theme needs is an SDK hook. |
| `sonner` | l/chat-controller/conversation.ts, hooks/use-api-file-download.ts | SDK | `notices.notify`, `notices.dismissNotice` | Only the host's toaster renders notices (A-10a); a theme never bundles sonner. |
| `zustand` | the stores above | host-only | — | A store is never exposed; each one's data comes through its area's hook. |

## 5. SDK names no import above maps to

| SDK names | What a theme uses them for |
|---|---|
| `theme.SDK_VERSION`, `theme.defineTheme` | Registering: the entry module's default export is `defineTheme({ register })` (A-01). |
| `context.useTone` | The light or dark tone the page is painted in. |
| `navigation.BOARD_HREF`, `navigation.QUESTIONS_HREF`, `navigation.missionHref`, `navigation.questionHref`, `navigation.ticketHref` | Core's routes for the board, the questions, a Mission and a ticket. |
| `events.HOST_EVENT_NAMES`, `events.onHostEvent`, `events.useHostEvent` | Following the board's and the chat's changes by the SDK's names, never the window events'. |
| `cards.CorePage`, `cards.FilePreview` | Wrapping core's own page inside a theme's page; the host's file preview. |
| `looks.useLook`, `looks.saveLook`, `looks.listLooks`, `looks.exportLook`, `looks.importLook` | The active look and the saved looks; Play's own `lib/looks` store is the look runtime the host mounts from A-13c (PA-9). |
| `assets.assetUrl` | The URL of a file in the theme's package (Play's mascot clips move from `public/auto` into it, A-16b). |

## Host-owned AI label (PRD-264 EUAI-11a, EUAI-11b)

EU AI Act Art. 50(1) and (5): a person chatting in the app is told, at first exposure and for the whole
session, that they are talking to an AI. The host draws that notice; no theme draws it, moves it or turns it
off, and no manifest field, SDK name or setting does either.

- **Where the host draws it.** `app/chat/layout.tsx` wraps `ThemeSlot` (the theme's page slot) in
  `AiLabelGate` (`frontend/components/ai-label/ai-label.tsx`), which renders the label after it, outside it. So
  the default chat (with Mission Mode), Play's chat and any theme's `/chat` show it, once. Every `/x/` theme
  route (`app/x/[...path]/page.tsx`, Play's Conversations among them) renders the same gate around
  `MainLayout`, on the left (Auto's widget takes the bottom right), so the label sits outside the theme's frame
  and route. Every theme route gets it: a manifest's `uses` is checked (`lib/theme-host/package-manifest.ts`)
  but not handed to the host, so the host can't yet tell which routes use the `chat` area.
  `components/theme-host/__tests__/theme-slot-route.test.tsx` proves the gate is there. Auto's floating
  widget, which `MainLayout` draws on every page but `/chat` (theme-drawn pages among them), carries its own
  label inside its panel (`WidgetAiLabelFrame`, `frontend/components/ai-label/widget-ai-label.tsx`), checked
  the same way; while it is hidden, the widget's input and starter chips are disabled.
- **What it says.** "Auto is an AI", or "{agent name} is an AI" while the owner's picked agent answers
  (`frontend/lib/ai-label/copy.ts`), in theme tokens, with `role="note"` and the same words for screen readers.
- **What happens if a theme hides it.** The label checks on mount, whenever `<html>`'s classes or attributes or
  `<head>`'s elements change (a theme or tone switch, a theme stylesheet arriving), on resize and every two
  seconds that it is displayed, visible, opaque, not zero-sized and inside the viewport
  (`frontend/lib/ai-label/visibility.ts`). While it isn't, every composer inside `AiLabelFence` is disabled
  (the default chat's input through `FencedMultimodalInput`, Play's message box) and the chat says "This theme
  hides the AI notice, so chat is paused. Switch theme in Settings." The fence wraps the composer only, never a
  whole chat screen, so the approval cards in a reply stay clickable (they are protected surfaces).
- **The SDK's chat area** (`frontend/lib/theme-sdk/chat.ts`, a stub today) takes the rule when A-11b fills it:
  a theme's composer built on `chat.useChatController` must pause the same way while the label is hidden, and
  the chat stream must not start until the host's label is mounted. A-11b's tests prove both.
- **Proof.** `frontend/lib/theme-host/__tests__/ai-label.test.ts` renders the chat layout with the default chat,
  builtin Play's chat and a fixture theme that owns `/chat`, then with hostile fixture sheets that set
  `display:none`, `visibility:hidden`, `opacity:0`, zero size or an off-screen position on the label: in each
  hiding case the input is disabled. `frontend/lib/theme-host/__tests__/ai-label-widget.test.tsx` does the
  same for Auto's floating widget.
