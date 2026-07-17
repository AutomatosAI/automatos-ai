/**
 * Voice — Auto Live arming (PRD-207 S7, super-admin surface)
 * ===========================================================
 *
 * The platform master switch + the masked Retell credentials, riding the
 * same by-category system-settings plane as every other tab. Arming and
 * disarming is THIS page — a toggle, no env var, no redeploy; OFF kills
 * minting AND in-flight webhook turns platform-wide, instantly.
 */

import React, { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Badge } from '@/components/ui/badge'
import { Switch } from '@/components/ui/switch'
import { Loader2, Mic, Save, Shield } from 'lucide-react'
import { SystemSetting } from '@/lib/api/system-settings'

interface VoiceLiveArmingTabProps {
  settings: SystemSetting[]
  onSave: (updates: Record<string, string>) => void
  saving: boolean
  onReset: () => void
}

const CRED_FIELDS: Array<{ key: string; label: string; hint: string }> = [
  {
    key: 'retell_api_key',
    label: 'Retell API key',
    hint: 'Used server-side to mint web-call tokens. Never reaches the browser.',
  },
  {
    key: 'retell_webhook_secret',
    label: 'Retell webhook signing key',
    hint: 'Verifies x-retell-signature on inbound webhooks (Retell signs with your API key unless configured otherwise). Empty = webhooks fail closed.',
  },
  {
    key: 'retell_agent_id',
    label: 'Retell agent id',
    hint: 'The Retell agent Auto Live mints web calls against.',
  },
]

export default function VoiceLiveArmingTab({
  settings,
  onSave,
  saving,
  onReset,
}: VoiceLiveArmingTabProps) {
  const [formData, setFormData] = useState<Record<string, string>>({})

  React.useEffect(() => {
    const initial: Record<string, string> = {}
    settings.forEach((setting) => {
      initial[setting.key] =
        setting.value !== null && setting.value !== undefined
          ? setting.value
          : setting.default_value || ''
    })
    setFormData(initial)
  }, [settings])

  const setField = (key: string, value: string) =>
    setFormData((prev) => ({ ...prev, [key]: value }))

  const enabled = (formData['live_enabled'] || 'false') === 'true'
  const armed = CRED_FIELDS.every((f) => (formData[f.key] || '').trim().length > 0)
  const findSetting = (key: string) => settings.find((s) => s.key === key)

  if (settings.length === 0) {
    return (
      <Card>
        <CardContent className="py-8 text-center text-sm text-muted-foreground">
          Voice settings have not been seeded yet — run the pending migrations.
        </CardContent>
      </Card>
    )
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Mic className="h-5 w-5" />
            Auto Live — real-time voice
            <Badge variant={enabled ? 'default' : 'secondary'}>
              {enabled ? 'ON' : 'OFF'}
            </Badge>
            {!armed && (
              <Badge variant="outline" className="text-warning border-warning/40">
                not armed
              </Badge>
            )}
          </CardTitle>
          <CardDescription>
            The platform master switch. OFF refuses every new call AND kills in-flight
            speech instantly — workspaces also need their own toggle before members can
            go live.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="flex items-center justify-between rounded-xl border border-border/50 px-4 py-3">
            <div>
              <Label htmlFor="voice-live-enabled" className="font-medium">
                Live voice enabled
              </Label>
              <p className="text-xs text-muted-foreground mt-0.5">
                {enabled
                  ? 'Minting and webhooks are allowed wherever a workspace opts in.'
                  : 'Nothing mints, nothing speaks — platform-wide.'}
              </p>
            </div>
            <Switch
              id="voice-live-enabled"
              checked={enabled}
              onCheckedChange={(checked) => setField('live_enabled', checked ? 'true' : 'false')}
            />
          </div>

          <div className="space-y-4">
            <div className="flex items-center gap-2 text-sm font-medium">
              <Shield className="h-4 w-4" />
              Retell credentials
              <span className="text-xs font-normal text-muted-foreground">
                (stored masked — values shown only to admins)
              </span>
            </div>
            {CRED_FIELDS.map((field) => {
              const meta = findSetting(field.key)
              return (
                <div key={field.key}>
                  <Label htmlFor={`voice-${field.key}`}>{field.label}</Label>
                  <Input
                    id={`voice-${field.key}`}
                    type={meta?.is_sensitive ? 'password' : 'text'}
                    value={formData[field.key] || ''}
                    onChange={(e) => setField(field.key, e.target.value)}
                    placeholder={meta?.is_sensitive ? '••••••••' : ''}
                    className="mt-1 font-mono"
                    autoComplete="off"
                  />
                  <p className="text-xs text-muted-foreground mt-1">{field.hint}</p>
                </div>
              )
            })}
          </div>

          <div className="flex items-center gap-2">
            <Button onClick={() => onSave(formData)} disabled={saving}>
              {saving ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" /> Saving…
                </>
              ) : (
                <>
                  <Save className="w-4 h-4 mr-2" /> Save voice settings
                </>
              )}
            </Button>
            <Button variant="outline" onClick={onReset} disabled={saving}>
              Reset to defaults
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
