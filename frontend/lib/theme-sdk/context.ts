/**
 * PRD-261 A-10a — SDK area `context`: what a theme may know about where it is drawn. The
 * workspace, the caller's roles, the edition, the trial, whether the browser is online, whether
 * the backend answers, the user's media preferences and the light/dark tone.
 *
 * Each hook reads the host's own provider (the theme renders inside the host's React tree, on
 * the host's React) and hands back a small, theme-neutral copy: never the provider's whole value,
 * so a theme sees no webhook key, plan limits or settings, and the host can change its providers
 * without changing the SDK. Outside a provider each hook answers "unknown" (null, or the same
 * fail-open default the host's own chrome uses), never throws. useThemeLayout is the host's own
 * width source for the active theme (hooks/use-studio-theme.ts), added by W-A5's seam owner for a
 * theme route that must leave a phone to the host's page. The one exception is useHealth: it
 * reads react-query, so it needs the host's QueryClientProvider (components/providers.tsx) above it, as
 * every page and theme the host draws has; in an isolated mount without one, react-query throws.
 */
import { useMemo } from 'react'
import { useTheme } from 'next-themes'

import { useWorkspaceOptional, type TrialState, type Workspace } from '@/components/workspace-provider'
import { useSystemRoleOptional } from '@/contexts/role-context'
import { useBackendHealth, type BackendHealth, type BackendHealthOptions } from '@/hooks/use-backend-health'
import { usePrefersReducedMotion } from '@/hooks/use-media-preferences'
import { useThemeLayout as useHostThemeLayout } from '@/hooks/use-studio-theme'
import { authEdition, type AuthEdition } from '@/lib/auth-edition'
import type { ThemeLayout } from '@/lib/theme-host/registry-types'

export { useOnline } from '@/hooks/use-online'

export type HostEdition = AuthEdition
export type HostWorkspaceRole = Workspace['role']
export type HostSystemRole = 'super_admin' | 'admin' | 'user'
export type HostTone = 'light' | 'dark'
export type HostHealth = BackendHealth
export type HostHealthOptions = BackendHealthOptions
export type HostThemeLayout = ThemeLayout

/** The workspace the page is in: who it is, its plan, and whether Socials is offered in it. */
export interface HostWorkspace {
  readonly id: string
  readonly name: string
  readonly slug: string
  readonly plan: Workspace['plan']
  /**
   * The platform offers Socials to this workspace (`socials.available` on GET /api/workspaces/current,
   * PRD-251 D1), as the host's Deliverables page reads it. False while the workspace doesn't say.
   */
  readonly socialsAvailable: boolean
}

export interface HostWorkspaceState {
  /** Null while it loads, when nobody is signed in, or outside the host's provider. */
  readonly workspace: HostWorkspace | null
  readonly isLoading: boolean
}

export interface HostRole {
  /** The caller's role in this workspace; null until the workspace answers. */
  readonly workspaceRole: HostWorkspaceRole | null
  /** The caller's platform role; null until the session answers. */
  readonly systemRole: HostSystemRole | null
  /** A platform admin. Unknown is not an admin. */
  readonly isAdmin: boolean
  /** False for a workspace viewer: write affordances render disabled. Open while unknown, as the host's are. */
  readonly canEdit: boolean
}

export interface HostTrial {
  readonly state: TrialState
  readonly grantedUsd: number
  readonly spentUsd: number
}

export interface HostMediaPreferences {
  /** The user asks for reduced motion (and always on the server, so a first paint never animates). */
  readonly reducedMotion: boolean
}

const HOST_TONES: readonly string[] = ['light', 'dark']

/** The workspace the page is in, as a theme may see it. */
export function useWorkspace(): HostWorkspaceState {
  const context = useWorkspaceOptional()
  const source = context?.workspace ?? null
  const isLoading = context?.isLoading ?? false
  const id = source?.id
  const name = source?.name
  const slug = source?.slug
  const plan = source?.plan
  const socialsAvailable = source?.socials?.available === true
  return useMemo(() => {
    const workspace = id !== undefined && name !== undefined && slug !== undefined && plan !== undefined
      ? Object.freeze({ id, name, slug, plan, socialsAvailable })
      : null
    return Object.freeze({ workspace, isLoading })
  }, [id, name, slug, plan, socialsAvailable, isLoading])
}

/** The caller's workspace and platform roles. */
export function useRole(): HostRole {
  const workspaceContext = useWorkspaceOptional()
  const roleContext = useSystemRoleOptional()
  const workspaceRole = workspaceContext?.workspace?.role ?? null
  const systemRole = roleContext && !roleContext.isLoading ? roleContext.systemRole : null
  const isAdmin = roleContext?.isAdmin === true
  const canEdit = workspaceContext?.canEdit ?? true
  return useMemo(
    () => Object.freeze({ workspaceRole, systemRole, isAdmin, canEdit }),
    [workspaceRole, systemRole, isAdmin, canEdit],
  )
}

/** 'local' (one operator, no sign-in) or 'saas'. Fixed for the life of the build. */
export function useEdition(): HostEdition {
  return authEdition
}

/** The workspace's trial ledger; null when it has none or has not loaded. */
export function useTrial(): HostTrial | null {
  const trial = useWorkspaceOptional()?.workspace?.onboarding?.trial ?? null
  const state = trial?.state
  const grantedUsd = trial?.granted_usd
  const spentUsd = trial?.spent_usd
  return useMemo(
    () => (state !== undefined && grantedUsd !== undefined && spentUsd !== undefined
      ? Object.freeze({ state, grantedUsd, spentUsd })
      : null),
    [state, grantedUsd, spentUsd],
  )
}

/**
 * Whether the backend answers. `poll: true` makes this caller the page's one poller (the asleep
 * screen); every other caller reads the same answer. A frozen copy of the host's answer: the
 * connection, whether an ask is in flight, the backend's host name when it can be read, and
 * `check`. Needs the host's QueryClientProvider (see the header).
 */
export function useHealth(options: HostHealthOptions = {}): HostHealth {
  const { connection, checking, host, check } = useBackendHealth(options)
  return useMemo(
    () => Object.freeze(host === undefined ? { connection, checking, check } : { connection, checking, host, check }),
    [connection, checking, host, check],
  )
}

/** The user's media preferences. */
export function useMediaPreferences(): HostMediaPreferences {
  const reducedMotion = usePrefersReducedMotion()
  return useMemo(() => Object.freeze({ reducedMotion }), [reducedMotion])
}

/** The tone the page is painted in, forced or chosen; null before the host knows it (the server). */
export function useTone(): HostTone | null {
  const { forcedTheme, resolvedTheme } = useTheme()
  const tone = forcedTheme ?? resolvedTheme
  return tone !== undefined && HOST_TONES.includes(tone) ? (tone as HostTone) : null
}

/**
 * Where the active theme draws at this width: 'theme' at or above its minWidth, 'fallback' below it (or
 * when no theme is active), 'pending' on the server and while hydrating, before the width is known
 * (render nothing of the theme and send nowhere until then). The host's own value, the one its
 * ThemeSlot and frame read (PRD-260 P260-FX-25, PRD-261 A-02a).
 */
export function useThemeLayout(): HostThemeLayout {
  return useHostThemeLayout()
}
