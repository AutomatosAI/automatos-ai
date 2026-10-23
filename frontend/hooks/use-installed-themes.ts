/**
 * PRD-261 A-07d — the workspace's installed themes, its theme settings and every write on them.
 * ==============================================================================================
 *
 * React Query over the apiClient theme methods (plan §2.2). The server enforces every rule
 * (manage, who may install, the private-upload switch, the per-package opt-in, step-up, 404 for
 * another workspace's install); these hooks only call it and refresh what changed. The writes the
 * server guards with step-up (import, opt-in, default, update, rollback, settings) go through
 * `useStepUp` (lib/theme-host/api.ts), so SaaS shows Clerk's "confirm it's you" and runs the call
 * once more; the local edition calls straight through.
 *
 * Choosing a theme stores the member's choice on the server (PA-12, the truth), then hands the
 * style to the style context, which writes the `automatos-style` cookie and, for an installed
 * theme, reloads (PA-13: its first paint comes from the server).
 *
 * Query keys are scoped by workspace id, so switching workspaces clears the cache. A-02c reads
 * `useInstalledThemes` for the menu and ⌘K entries, but only once one of them asks
 * (`wantInstalledThemes`, choice-sync.ts): until then it passes `enabled: false` and reads only
 * what Settings › Appearance already fetched, so an ordinary page asks for no installs.
 */

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { toast } from 'sonner'

import { useWorkspace } from '@/components/workspace-provider'
import { useUiStyleOptional } from '@/contexts/ui-style-context'
import { apiClient } from '@/lib/api-client'
import type { InstalledTheme, ThemeSettings, ThemeSettingsChange, ThemeUpdateComparison, ThemeUpload } from '@/lib/api-client'
import { themeErrorMessage, useStepUp } from '@/lib/theme-host/api'
import { themeStyleOf } from '@/lib/theme-host/registry'

export const themeQueryKeys = {
  all: (workspaceId: string | null) => ['themes', workspaceId] as const,
  installed: (workspaceId: string | null) => ['themes', workspaceId, 'installed'] as const,
  settings: (workspaceId: string | null) => ['themes', workspaceId, 'settings'] as const,
  active: (workspaceId: string | null) => ['themes', workspaceId, 'active'] as const,
  compare: (workspaceId: string | null, installId: string, packageId: string) =>
    ['themes', workspaceId, 'compare', installId, packageId] as const,
}

/** Refusals that mean the list on screen is out of date: refetch it, then show the server's words. */
const STALE_STATUSES: ReadonlySet<number> = new Set([404, 409])

function useWorkspaceId(): string | null {
  return useWorkspace().workspace?.id ?? null
}

function statusOf(error: unknown): number | undefined {
  const status = (error as { status?: unknown } | null | undefined)?.status
  return typeof status === 'number' ? status : undefined
}

/**
 * True when a write was refused because what the dialog showed is out of date (404 gone, 409 moved
 * on). The hook already refetched the list; the dialog closes so the next try starts from it.
 */
export function isStaleThemeRefusal(error: unknown): boolean {
  const status = statusOf(error)
  return status !== undefined && STALE_STATUSES.has(status)
}

function useInvalidateThemes(): () => Promise<void> {
  const workspaceId = useWorkspaceId()
  const queryClient = useQueryClient()
  return () => queryClient.invalidateQueries({ queryKey: themeQueryKeys.all(workspaceId) })
}

// ============= QUERIES =============

export interface InstalledThemesOptions {
  /** False reads the cache only (no request); the default fetches. */
  readonly enabled?: boolean
}

/** The workspace's active installs, each with its opt-in, revocation and "needs update" state. */
export function useInstalledThemes({ enabled = true }: InstalledThemesOptions = {}) {
  const workspaceId = useWorkspaceId()
  return useQuery<InstalledTheme[]>({
    queryKey: themeQueryKeys.installed(workspaceId),
    enabled: enabled && !!workspaceId,
    queryFn: async () => (await apiClient.getInstalledThemes()).installs,
    staleTime: 15_000,
  })
}

/** The workspace's theme settings and what this caller may do (A-07e). */
export function useThemeSettings() {
  const workspaceId = useWorkspaceId()
  return useQuery<ThemeSettings>({
    queryKey: themeQueryKeys.settings(workspaceId),
    enabled: !!workspaceId,
    queryFn: () => apiClient.getThemeSettings(),
    staleTime: 15_000,
  })
}

/** The `choice` field of GET /api/themes/active, or null when it has none. */
export function choiceOf(active: unknown): string | null {
  const choice = (active as { choice?: unknown } | null | undefined)?.choice
  return typeof choice === 'string' ? choice : null
}

/** Where GET /api/themes/active's choice came from (A-07b): the member's own, the workspace default, or Studio for neither. */
export type ChoiceSource = 'choice' | 'default' | 'fallback'
const CHOICE_SOURCES: readonly string[] = ['choice', 'default', 'fallback']

/** The `source` field of GET /api/themes/active, or null when it has none (a backend before it). */
export function choiceSourceOf(active: unknown): ChoiceSource | null {
  const source = (active as { source?: unknown } | null | undefined)?.source
  return typeof source === 'string' && CHOICE_SOURCES.includes(source) ? (source as ChoiceSource) : null
}

/** The member's current choice here, as the server holds it ('classic', 'studio' or an install id). */
export function useActiveThemeChoice() {
  const workspaceId = useWorkspaceId()
  return useQuery<string | null>({
    queryKey: themeQueryKeys.active(workspaceId),
    enabled: !!workspaceId,
    queryFn: async () => choiceOf(await apiClient.getActiveTheme()),
    staleTime: 15_000,
  })
}

/** What updating `install` to its pending package changes; fetched only while `enabled`. */
export function useThemeUpdateComparison(install: InstalledTheme, enabled: boolean) {
  const workspaceId = useWorkspaceId()
  const packageId = install.pendingPackageId ?? ''
  return useQuery<ThemeUpdateComparison>({
    queryKey: themeQueryKeys.compare(workspaceId, install.installId, packageId),
    enabled: enabled && !!workspaceId && !!packageId,
    queryFn: () => apiClient.compareThemeUpdate(install.installId, packageId),
  })
}

// ============= WRITES =============

/** One theme write: the server's sentence on a refusal, a refetch when the list was stale, a refresh on success. */
function useThemeWrite<Vars, Result>(run: (vars: Vars) => Promise<Result>, done: string | null) {
  const invalidate = useInvalidateThemes()
  return useMutation<Result, unknown, Vars>({
    mutationFn: run,
    onSuccess: async () => {
      if (done) toast.success(done)
      await invalidate()
    },
    onError: async (error) => {
      toast.error(themeErrorMessage(error))
      if (isStaleThemeRefusal(error)) await invalidate()
    },
  })
}

/** Upload a .autotheme (manage, who may install, the private-upload switch, step-up). */
export function useImportTheme() {
  const importTheme = useStepUp(apiClient.importTheme.bind(apiClient))
  return useThemeWrite<File, ThemeUpload>((file) => importTheme(file), null)
}

/** Opt in to the exact package the dialog showed (step-up; 409 when it is no longer current). */
export function useOptInTheme() {
  const optIn = useStepUp(apiClient.optInTheme.bind(apiClient))
  return useThemeWrite<{ installId: string; packageId: string }, unknown>(
    ({ installId, packageId }) => optIn(installId, packageId),
    null,
  )
}

/** Make an install the workspace default, or clear it with null (manage + step-up; 409 for unreviewed). */
export function useSetDefaultTheme() {
  const setDefault = useStepUp(apiClient.setDefaultTheme.bind(apiClient))
  return useThemeWrite<string | null, unknown>((installId) => setDefault(installId), 'Workspace default saved.')
}

/** Move an install to the package the compare showed (manage + step-up). */
export function useUpdateThemeInstall() {
  const update = useStepUp(apiClient.updateThemeInstall.bind(apiClient))
  return useThemeWrite<{ installId: string; packageId: string }, InstalledTheme>(
    ({ installId, packageId }) => update(installId, packageId),
    'Theme updated.',
  )
}

/** Swap an install back to its previous package (manage + step-up). */
export function useRollbackTheme() {
  const rollback = useStepUp(apiClient.rollbackTheme.bind(apiClient))
  return useThemeWrite<string, InstalledTheme>((installId) => rollback(installId), 'Theme rolled back.')
}

/** Soft-uninstall: looks stay, members who chose it see the workspace default (manage). */
export function useUninstallTheme() {
  return useThemeWrite<string, unknown>((installId) => apiClient.uninstallTheme(installId), 'Theme uninstalled.')
}

/** Change the keys sent, all or none (manage + step-up). */
export function usePutThemeSettings() {
  const putSettings = useStepUp(apiClient.putThemeSettings.bind(apiClient))
  return useThemeWrite<ThemeSettingsChange, ThemeSettings>((changes) => putSettings(changes), 'Theme settings saved.')
}

/**
 * Use an installed theme: the choice is stored on the server first, then the style context writes
 * the cookie and reloads into it. No step-up: the opt-in already gates private code (A-07b).
 */
export function useChooseTheme() {
  const styleContext = useUiStyleOptional()
  return useThemeWrite<string, unknown>(async (installId) => {
    const answer = await apiClient.putThemeChoice(installId)
    styleContext?.setStyle(themeStyleOf(`install:${installId}`))
    return answer
  }, null)
}
