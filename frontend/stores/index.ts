/**
 * Stores Entry Point
 *
 * Exports all Zustand stores used in the application.
 */

export {
  useWorkspaceStore,
  useWidgets,
  useWidgetIds,
  useActiveWidgetId,
  useLayoutMode,
  useIsChatCollapsed,
  useChatPanelWidth,
  useIsWidgetTrayOpen,
  useWidget,
  useWidgetPosition,
  useWidgetSize,
  // US-006: Workspace persistence selectors
  useCurrentWorkspaceId,
  useSavedWorkspaces,
  useIsWorkspaceLoading,
  useIsWorkspaceSaving,
  useHasUnsavedChanges,
  useLastSaved,
} from './workspace-store'

export type {
  SavedWorkspaceSummary,
  SavedWorkspaceDetail,
} from './workspace-store'
