'use client'

/**
 * PRD-261 A-07d — Settings › Appearance › Themes (PRD §5).
 *
 * Owners and admins (`workspace:manage`) get Install theme with the scan report, the installed
 * list with update (compare and apply), rollback, uninstall and the workspace default, and the
 * workspace's theme settings (A-07e). Every other member sees the list and the choice only. The
 * role decides which controls show; the server enforces every gate (and step-up) itself.
 */
import { Palette } from 'lucide-react'

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { useWorkspace } from '@/components/workspace-provider'
import { useThemeSettings } from '@/hooks/use-installed-themes'

import { InstallTheme } from './install-theme'
import { InstalledList } from './installed-list'
import { canManageThemes } from './theme-labels'
import { ThemeSettingsCard } from './theme-settings'

export function AppearanceTab() {
  const { workspace } = useWorkspace()
  const canManage = canManageThemes(workspace?.role)
  const settings = useThemeSettings()

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Palette className="h-5 w-5" aria-hidden />
          Themes
        </CardTitle>
        <CardDescription>
          A theme changes how Automatos looks for you in this workspace. A private theme runs code nobody at
          Automatos reviewed, so each member opts in to it before it runs for them.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        {canManage && <InstallTheme settings={settings.data} />}
        <InstalledList canManage={canManage} />
        {canManage && settings.data && <ThemeSettingsCard settings={settings.data} />}
      </CardContent>
    </Card>
  )
}
