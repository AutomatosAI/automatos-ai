'use client'

/**
 * PRD-261 C-04b — "Use this theme" for an install the Themes tab added (or found) in this workspace.
 *
 * The same path as A-07d's installed list, reused rather than copied: the row comes from
 * `useInstalledThemes` (so the opt-in state is the server's), a version the member has not opted
 * in to opens A-07d's `OptInDialog` with the server's wording, and the choice is stored through
 * `useChooseTheme`, which reloads into the theme. A reviewed Marketplace version needs no opt-in
 * (services/themes/optin.py), so for a listing install the dialog shows only when the server
 * says it is required.
 */
import { useState } from 'react'

import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { OptInDialog } from '@/components/settings/themes/opt-in-dialog'
import { USE_THEME } from '@/components/settings/themes/installed-row'
import { IN_USE, isDisabled } from '@/components/settings/themes/theme-labels'
import { useActiveThemeChoice, useChooseTheme, useInstalledThemes } from '@/hooks/use-installed-themes'
import type { InstalledTheme } from '@/lib/api-client'

export function UseInstalledTheme({ installId }: { installId: string }) {
  const installed = useInstalledThemes()
  const active = useActiveThemeChoice()
  const choose = useChooseTheme()
  const [optInFor, setOptInFor] = useState<InstalledTheme | null>(null)
  const install = installed.data?.find((row) => row.installId === installId)

  if (!install) return null
  if (active.data === installId) return <Badge>{IN_USE}</Badge>

  const use = () => {
    if (install.optIn.required) setOptInFor(install)
    else choose.mutate(install.installId)
  }

  return (
    <>
      <Button size="sm" disabled={choose.isLoading || isDisabled(install)} onClick={use}>
        {USE_THEME}
      </Button>
      <OptInDialog
        install={optInFor}
        onClose={() => setOptInFor(null)}
        onOptedIn={(row) => choose.mutate(row.installId)}
      />
    </>
  )
}
