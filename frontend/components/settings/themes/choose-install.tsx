'use client'

/**
 * PRD-261 A-07d — choosing an installed theme, wherever a member chooses one: "Use this theme" in
 * Settings › Appearance › Themes and in the Marketplace, the theme menu and ⌘K (the last two through
 * lib/theme-host/choice-sync.ts `useChooseStyle`).
 *
 * A private version this member has not opted in to (`optIn.required`) opens the opt-in dialog first
 * (opt-in-dialog.tsx, with the server's wording and the row as it was when chosen). Only once the
 * server recorded the opt-in is the choice stored, and the page reloads into the theme
 * (hooks/use-installed-themes.ts `useChooseTheme`); cancelling chooses nothing. Any other install is
 * chosen at once. Stored without the opt-in, the choice reloaded the page into a theme the host then
 * refused to run, and Studio showed instead.
 *
 * Mounted once (components/providers.tsx), inside the workspace and query providers and around every
 * page and ⌘K, so one dialog serves them all, the header's menu included.
 */
import { createContext, useCallback, useContext, useMemo, useState, type ReactNode } from 'react'

import { useChooseTheme } from '@/hooks/use-installed-themes'
import type { InstalledTheme } from '@/lib/api-client'

import { OptInDialog } from './opt-in-dialog'
import { needsOptIn } from './theme-labels'

export interface ChooseInstall {
  /** Chooses the install for this member, after the opt-in dialog when its version needs one. */
  readonly chooseInstall: (install: InstalledTheme) => void
  /** True while a choice is being stored (the page reloads once it is). */
  readonly choosing: boolean
}

const ChooseInstallContext = createContext<ChooseInstall | null>(null)

export function ChooseInstallProvider({ children }: { children: ReactNode }) {
  const { mutate: choose, isLoading: choosing } = useChooseTheme()
  const [optInFor, setOptInFor] = useState<InstalledTheme | null>(null)

  const chooseInstall = useCallback(
    (install: InstalledTheme) => {
      if (needsOptIn(install)) setOptInFor(install)
      else choose(install.installId)
    },
    [choose],
  )
  const value = useMemo(() => ({ chooseInstall, choosing }), [chooseInstall, choosing])

  return (
    <ChooseInstallContext.Provider value={value}>
      {children}
      <OptInDialog
        install={optInFor}
        onClose={() => setOptInFor(null)}
        onOptedIn={(install) => choose(install.installId)}
      />
    </ChooseInstallContext.Provider>
  )
}

/** The chooser, or null outside ChooseInstallProvider (an isolated mount, which has no dialog to open). */
export function useChooseInstallOptional(): ChooseInstall | null {
  return useContext(ChooseInstallContext)
}

/** The chooser: Settings and the Marketplace render inside ChooseInstallProvider, as every page does. */
export function useChooseInstall(): ChooseInstall {
  const chooser = useContext(ChooseInstallContext)
  if (chooser === null) throw new Error('useChooseInstall must be used within a ChooseInstallProvider')
  return chooser
}
