'use client'

/**
 * PRD-261 E3c (plan §2.2.2) — the host says whose theme state the SDK's `storage` area reads and
 * writes (lib/theme-sdk/storage.ts, over lib/theme-sdk/internal/theme-state-runtime.ts).
 *
 * It publishes the scope `{themeId, userKey, workspaceKey}` only while all of these hold:
 * - the active theme is an installed one with its theme id (its registry entry, which the root
 *   layout built from the boot);
 * - the theme host loads or runs it (THEME_HOST_RUNNING_STATUSES), so a failed load or safe mode
 *   clears it in the commit the host stops, before the host releases the entry;
 * - the user is known: Clerk has loaded a signed-in user with an id, or, in the local edition
 *   (signed in with no user id), its one user `local`.
 * `workspaceKey` is the WorkspaceProvider's workspace id, and null while it loads, after it failed
 * and outside a provider: then `within: 'workspace'` state is not ready, and `account` state is.
 * Anything else (Classic, Studio, a builtin theme, safe mode, signed out) clears the scope, so the
 * SDK reads not-ready and never touches the browser's storage.
 *
 * It renders its children and nothing else, whatever the scope: the tree above the page keeps one
 * shape, so a style switched in place never remounts the page below (M69). The host's look runtime
 * mounts it (./theme-look-runtime.tsx), which renders its children for every style. It reads no
 * looks and no builtin theme: the scope is the host's, the same for every installed theme.
 */
import { useEffect, type ReactNode } from 'react'

import { useWorkspaceOptional } from '@/components/workspace-provider'
import { useActiveTheme } from '@/hooks/use-studio-theme'
import { isLocal } from '@/lib/auth-edition'
import { useUser } from '@/lib/auth-hooks'
import type { ThemeEntry } from '@/lib/theme-host/registry-types'
import { THEME_HOST_RUNNING_STATUSES } from '@/lib/theme-host/types'
import { clearThemeStateScope, publishThemeStateScope } from '@/lib/theme-sdk/internal/theme-state-runtime'

import { useThemeHost } from './theme-host-context'

/** The user part of the scope in the local edition, whose one user is signed in with no id. */
export const LOCAL_EDITION_USER_KEY = 'local'

type WorkspaceState = ReturnType<typeof useWorkspaceOptional>

function nonEmpty(value: unknown): string | null {
  return typeof value === 'string' && value.length > 0 ? value : null
}

/** An installed theme's id; null for Classic and Studio (no entry), a builtin theme, or no id. */
function installedThemeId(theme: ThemeEntry | null): string | null {
  return theme !== null && theme.kind === 'installed' ? nonEmpty(theme.themeId) : null
}

/** The installed theme's id while the host loads or runs it; null in every other state. */
function useRunningThemeId(): string | null {
  const theme = useActiveTheme()
  const status = useThemeHost()?.status ?? null
  const running = status !== null && THEME_HOST_RUNNING_STATUSES.includes(status)
  return running ? installedThemeId(theme) : null
}

/** The signed-in user's id, `local` in the local edition; null while Clerk loads, signed out, or with no id. */
function useUserKey(): string | null {
  const { user, isLoaded, isSignedIn } = useUser()
  if (!isLoaded || isSignedIn !== true) return null
  return isLocal ? LOCAL_EDITION_USER_KEY : nonEmpty(user?.id)
}

/**
 * The workspace's id once it loaded; null while it loads, when it never loaded, and outside a
 * provider. A failed refetch keeps the workspace the provider still holds, as the look scope does.
 */
function workspaceKeyOf(workspaceCtx: WorkspaceState): string | null {
  if (workspaceCtx === null || workspaceCtx.isLoading) return null
  return nonEmpty(workspaceCtx.workspace?.id)
}

export interface ThemeStateScopeProps {
  readonly children: ReactNode
}

export function ThemeStateScope({ children }: ThemeStateScopeProps) {
  const themeId = useRunningThemeId()
  const userKey = useUserKey()
  const workspaceKey = workspaceKeyOf(useWorkspaceOptional())
  // A new scope replaces the old one in one step (the runtime tells every subscriber once); none clears it.
  useEffect(() => {
    if (themeId === null || userKey === null) clearThemeStateScope()
    else publishThemeStateScope({ themeId, userKey, workspaceKey })
  }, [themeId, userKey, workspaceKey])
  // The scope goes with the host: nothing reads a theme's state after the tree above the page unmounts.
  useEffect(() => () => clearThemeStateScope(), [])
  // One shape for every scope: the page below never remounts on a switch (M69).
  return <>{children}</>
}
