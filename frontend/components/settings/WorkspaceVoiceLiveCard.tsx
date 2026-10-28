'use client'

/**
 * Auto Live for this workspace — the WORKSPACE switch (Settings → Voice → Auto Live).
 * ======================================================================================
 *
 * Not the platform's own setup: that is Settings → System Settings → Voice
 * (VoiceLiveArmingTab, super-admin only — the Retell credentials and the platform-wide
 * on/off). This card is the workspace's own switch, `workspace.settings.voice_live.enabled`,
 * written through `PUT /api/workspaces/current/voice-live` by an owner or admin
 * (`workspace:manage`); anyone else reads it. A workspace stays off until it is turned
 * on here, whatever the platform has set up.
 *
 * `workspace.voiceLive.available` is the SaaS plan gate (D34: Auto Live needs a plan
 * with the voice family; the local edition is never plan-gated) — not the platform
 * switch or the Retell credentials, which this card cannot see or change. The switch
 * stays visible but disabled, with the plain reason, while the plan blocks it or the
 * viewer cannot manage the workspace (never hidden: a member still sees the state).
 */
import { Mic } from 'lucide-react'

import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Label } from '@/components/ui/label'
import { Switch } from '@/components/ui/switch'
import { useWorkspace } from '@/components/workspace-provider'
import { useSetWorkspaceVoiceLive } from '@/hooks/use-voice-live-settings'
import { LIVE_REASON_SENTENCES } from '@/lib/voice/live/refusal'
import { canManageVoiceLive } from '@/lib/voice/live/workspace-switch'

export const ASK_AN_ADMIN_TEXT = 'A workspace owner or admin turns Auto Live on or off.'

export function WorkspaceVoiceLiveCard() {
  const { workspace } = useWorkspace()
  const setVoiceLive = useSetWorkspaceVoiceLive()
  const planAllows = workspace?.voiceLive?.available ?? false
  const enabled = workspace?.voiceLive?.enabled ?? false
  const canManage = canManageVoiceLive(workspace?.role)
  const disabled = !planAllows || !canManage || setVoiceLive.isLoading

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Mic className="h-5 w-5" aria-hidden />
          Auto Live for this workspace
          <Badge variant={enabled ? 'default' : 'secondary'}>{enabled ? 'ON' : 'OFF'}</Badge>
        </CardTitle>
        <CardDescription>
          A real-time voice call with Auto, back and forth. Off hides the Live chip under the
          chat box and any theme&apos;s voice call button for this workspace.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="flex items-center justify-between rounded-xl border border-border/50 px-4 py-3">
          <div>
            <Label htmlFor="workspace-voice-live-enabled" className="font-medium">
              Auto Live enabled
            </Label>
            <p className="mt-0.5 text-xs text-muted-foreground">
              {enabled
                ? 'This workspace can start a live call with Auto.'
                : 'This workspace does not see Auto Live.'}
            </p>
          </div>
          <Switch
            id="workspace-voice-live-enabled"
            checked={enabled}
            disabled={disabled}
            onCheckedChange={(on) => setVoiceLive.mutate(on)}
          />
        </div>
        {!planAllows ? (
          <p className="text-sm text-muted-foreground" data-testid="voice-live-plan-blocked">
            {LIVE_REASON_SENTENCES.voice_live_plan_required}
          </p>
        ) : !canManage ? (
          <p className="text-xs text-muted-foreground">{ASK_AN_ADMIN_TEXT}</p>
        ) : null}
      </CardContent>
    </Card>
  )
}
