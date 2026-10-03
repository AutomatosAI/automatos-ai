'use client'

/**
 * Socials for this workspace — the WORKSPACE switch (PRD-251 D1; PRD-251B US-B106).
 * ==================================================================================
 *
 * Not the platform master: that is Settings → System Settings → Socials
 * (SocialsSettingsTab, super-admin only). This card is the workspace's own switch,
 * `workspace.settings.socials.enabled`, written through PUT /api/workspaces/current/socials
 * by an owner or admin (workspace:manage); anyone else reads it. Off means invisible:
 * the Socials tab, its calendar, Auto's Socials tools and the marketplace package go
 * with it, and come back on the next request when it is turned on again.
 *
 * While the master is off the card only says so, read-only: a workspace cannot turn
 * on what the platform has not.
 */
import { Share2 } from 'lucide-react'

import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Label } from '@/components/ui/label'
import { Switch } from '@/components/ui/switch'
import { useWorkspace } from '@/components/workspace-provider'
import { canTurnOnSocials } from '@/components/deliverables/socials/socials-status'
import { useSetWorkspaceSocials } from '@/hooks/use-socials-api'

export const MASTER_OFF_TEXT = 'Socials is not switched on for this platform. The platform\u2019s super-admin turns it on in Settings \u2192 System Settings.'
export const ASK_AN_ADMIN_TEXT = 'A workspace owner or admin turns Socials on or off.'

export function WorkspaceSocialsCard() {
  const { workspace } = useWorkspace()
  const setSocials = useSetWorkspaceSocials()
  const available = workspace?.socials?.available ?? false
  const enabled = available && (workspace?.socials?.enabled ?? false)
  const canManage = canTurnOnSocials(workspace?.role)

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Share2 className="h-5 w-5" aria-hidden />
          Socials for this workspace
          <Badge variant={enabled ? 'default' : 'secondary'}>{enabled ? 'ON' : 'OFF'}</Badge>
        </CardTitle>
        <CardDescription>
          On-brand posts for your channels, each one approved before it goes out. Off hides the
          Socials tab and its calendar, Auto&apos;s Socials tools and the Socials marketplace package
          for this workspace; nothing already drafted is deleted.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        {!available ? (
          <p className="text-sm text-muted-foreground" data-testid="socials-master-off">
            {MASTER_OFF_TEXT}
          </p>
        ) : (
          <div className="flex items-center justify-between rounded-xl border border-border/50 px-4 py-3">
            <div>
              <Label htmlFor="workspace-socials-enabled" className="font-medium">
                Socials enabled
              </Label>
              <p className="mt-0.5 text-xs text-muted-foreground">
                {enabled ? 'This workspace drafts, approves and publishes posts.' : 'This workspace does not see Socials.'}
              </p>
            </div>
            <Switch
              id="workspace-socials-enabled"
              checked={enabled}
              disabled={!canManage || setSocials.isLoading}
              onCheckedChange={(on) => setSocials.mutate(on)}
            />
          </div>
        )}
        {available && !canManage && <p className="text-xs text-muted-foreground">{ASK_AN_ADMIN_TEXT}</p>}
      </CardContent>
    </Card>
  )
}
