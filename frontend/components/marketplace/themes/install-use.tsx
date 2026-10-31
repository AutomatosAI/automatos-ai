'use client'

/**
 * PRD-261 C-04b — "Use this theme" for an install the Themes tab added (or found) in this workspace.
 *
 * The same path as A-07d's installed list, reused rather than copied: the row comes from
 * `useInstalledThemes` (so the opt-in state is the server's), and the choice goes through the one
 * chooser Settings, the theme menu and ⌘K share (components/settings/themes/choose-install.tsx): a
 * version the member has not opted in to opens A-07d's opt-in dialog with the server's wording, and
 * the choice is stored once the server recorded the opt-in, then the page reloads into the theme. A
 * reviewed Marketplace version needs no opt-in (services/themes/optin.py), so for a listing install
 * the dialog shows only when the server says it is required.
 */
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { useChooseInstall } from '@/components/settings/themes/choose-install'
import { USE_THEME } from '@/components/settings/themes/installed-row'
import { IN_USE, isDisabled } from '@/components/settings/themes/theme-labels'
import { useActiveThemeChoice, useInstalledThemes } from '@/hooks/use-installed-themes'

export function UseInstalledTheme({ installId }: { installId: string }) {
  const installed = useInstalledThemes()
  const active = useActiveThemeChoice()
  const { chooseInstall, choosing } = useChooseInstall()
  const install = installed.data?.find((row) => row.installId === installId)

  if (!install) return null
  if (active.data === installId) return <Badge>{IN_USE}</Badge>

  return (
    <Button size="sm" disabled={choosing || isDisabled(install)} onClick={() => chooseInstall(install)}>
      {USE_THEME}
    </Button>
  )
}
