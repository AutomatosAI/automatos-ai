/**
 * Socials — the platform master switch (PRD-251 S0.1, super-admin surface)
 * ========================================================================
 *
 * The `socials.enabled` system setting, on the same by-category plane as
 * every other tab (the Auto Live pattern). OFF hides the Socials tab in
 * Deliverables and 404s every /api/socials route, platform-wide. ON lets
 * each workspace turn Socials on for itself. No env var, no redeploy.
 */

import React, { useEffect, useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Label } from '@/components/ui/label'
import { Badge } from '@/components/ui/badge'
import { Switch } from '@/components/ui/switch'
import { Loader2, Megaphone, Save } from 'lucide-react'
import { SystemSetting } from '@/lib/api/system-settings'

export const SOCIALS_ENABLED_KEY = 'enabled'

interface SocialsSettingsTabProps {
  settings: SystemSetting[]
  onSave: (updates: Record<string, string>) => void
  saving: boolean
  onReset: () => void
}

function isOn(setting: SystemSetting | undefined): boolean {
  const value = setting?.value ?? setting?.default_value ?? 'false'
  return value === 'true'
}

export default function SocialsSettingsTab({
  settings,
  onSave,
  saving,
  onReset,
}: SocialsSettingsTabProps) {
  const setting = settings.find((s) => s.key === SOCIALS_ENABLED_KEY)
  const [enabled, setEnabled] = useState(isOn(setting))

  useEffect(() => {
    setEnabled(isOn(setting))
  }, [setting])

  if (!setting) {
    return (
      <Card>
        <CardContent className="py-8 text-center text-sm text-muted-foreground">
          Socials settings have not been seeded yet — run the pending migrations.
        </CardContent>
      </Card>
    )
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Megaphone className="h-5 w-5" />
          Socials
          <Badge variant={enabled ? 'default' : 'secondary'}>{enabled ? 'ON' : 'OFF'}</Badge>
        </CardTitle>
        <CardDescription>
          The platform master switch. OFF hides the Socials tab in Deliverables for every
          workspace. ON lets each workspace owner or admin turn Socials on for their own
          workspace.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="flex items-center justify-between rounded-xl border border-border/50 px-4 py-3">
          <div>
            <Label htmlFor="socials-enabled" className="font-medium">
              Socials enabled
            </Label>
            <p className="text-xs text-muted-foreground mt-0.5">
              {enabled
                ? 'Workspaces can turn Socials on for themselves.'
                : 'No workspace sees Socials — platform-wide.'}
            </p>
          </div>
          <Switch id="socials-enabled" checked={enabled} onCheckedChange={setEnabled} />
        </div>

        <div className="flex items-center gap-2">
          <Button
            onClick={() => onSave({ [SOCIALS_ENABLED_KEY]: enabled ? 'true' : 'false' })}
            disabled={saving}
          >
            {saving ? (
              <>
                <Loader2 className="w-4 h-4 mr-2 animate-spin" /> Saving…
              </>
            ) : (
              <>
                <Save className="w-4 h-4 mr-2" /> Save Socials settings
              </>
            )}
          </Button>
          <Button variant="outline" onClick={onReset} disabled={saving}>
            Reset to defaults
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}
