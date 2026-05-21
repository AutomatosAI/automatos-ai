'use client'

import { useEffect, useState } from 'react'
import Link from 'next/link'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Switch } from '@/components/ui/switch'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import {
  listCallbackChannels,
  sendCallbackTest,
  updateSiteSettings,
  type CallbackTestResponse,
} from '@/lib/sites/api'
import { PLATFORM_LABEL, type ChannelConnection } from '@/lib/channels/types'
import type {
  Site,
  CallbackSettings,
  CallbackDestination,
  CallbackPlatform,
} from '@/lib/sites/types'

type DestinationChoice = CallbackPlatform | 'off'

const DEFAULTS: CallbackSettings = {
  enabled: false,
  destinations: [],
}

/**
 * Coerce a possibly-legacy persisted destinations array down to the
 * single dropdown value the new UI exposes. We only render one
 * destination — multi-destination fan-out lives in the dispatcher but
 * is not exposed in the merchant UI yet. If multiple were saved, the
 * first valid one wins.
 */
function pickPrimaryDestination(
  raw: unknown,
): CallbackDestination | null {
  if (!Array.isArray(raw)) return null
  for (const d of raw) {
    if (!d || typeof d !== 'object') continue
    const platform = (d as { platform?: string }).platform
    if (platform === 'telegram' || platform === 'slack' || platform === 'whatsapp' || platform === 'webhook') {
      return {
        platform,
        channel_id: (d as { channel_id?: string }).channel_id,
        webhook_url: (d as { webhook_url?: string }).webhook_url,
      }
    }
  }
  return null
}

export function CallbackPanel({
  site,
  onUpdated,
}: {
  site: Site
  onUpdated: (s: Site) => void
}) {
  const initialDestination = pickPrimaryDestination(site.settings.callback?.destinations)
  const initial: CallbackSettings = {
    ...DEFAULTS,
    enabled: site.settings.callback?.enabled ?? false,
    destinations: initialDestination ? [initialDestination] : [],
  }
  const [draft, setDraft] = useState<CallbackSettings>(initial)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [testing, setTesting] = useState(false)
  const [testResult, setTestResult] = useState<CallbackTestResponse | null>(null)
  const [testError, setTestError] = useState<string | null>(null)

  // Connected channels — same source as heartbeat's "Report To". Only
  // platforms with an active workspace connection appear in the dropdown.
  const [channels, setChannels] = useState<ChannelConnection[] | null>(null)
  useEffect(() => {
    let cancelled = false
    listCallbackChannels()
      .then((c) => { if (!cancelled) setChannels(c) })
      .catch(() => { if (!cancelled) setChannels([]) })
    return () => { cancelled = true }
  }, [])

  const currentDest: CallbackDestination | null = draft.destinations[0] ?? null
  const choice: DestinationChoice = currentDest === null ? 'off' : currentDest.platform

  const connectedPlatforms = new Set<string>(
    (channels ?? []).filter((c) => c.status === 'active').map((c) => c.platform),
  )

  function setChoice(next: DestinationChoice) {
    if (next === 'off') {
      setDraft((d) => ({ ...d, destinations: [] }))
      return
    }
    const existing = currentDest && currentDest.platform === next ? currentDest : null
    setDraft((d) => ({
      ...d,
      destinations: [
        {
          platform: next,
          channel_id: existing?.channel_id,
          webhook_url: existing?.webhook_url,
        },
      ],
    }))
  }

  function patchCurrent(patch: Partial<CallbackDestination>) {
    if (!currentDest) return
    setDraft((d) => ({
      ...d,
      destinations: [{ ...currentDest, ...patch }],
    }))
  }

  // Validation matches what the backend will accept.
  const destValid = (() => {
    if (!currentDest) return true // "Off" is fine
    if (currentDest.platform === 'webhook') {
      const url = currentDest.webhook_url?.trim() ?? ''
      return url.startsWith('http://') || url.startsWith('https://')
    }
    return connectedPlatforms.has(currentDest.platform)
  })()

  const dirty = JSON.stringify(draft) !== JSON.stringify(initial)
  const canSave = dirty && !saving && destValid
  const canTest =
    !testing &&
    !dirty &&
    initial.destinations.length > 0 &&
    initial.enabled

  async function save() {
    setSaving(true)
    setError(null)
    try {
      const updated = await updateSiteSettings(site.id, { callback: draft })
      onUpdated(updated)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Save failed')
    } finally {
      setSaving(false)
    }
  }

  async function runTest() {
    setTesting(true)
    setTestResult(null)
    setTestError(null)
    try {
      const resp = await sendCallbackTest(site.id)
      setTestResult(resp)
    } catch (e: unknown) {
      setTestError(e instanceof Error ? e.message : 'Test failed')
    } finally {
      setTesting(false)
    }
  }

  return (
    <Card className="glass-card">
      <CardHeader className="flex flex-row items-center justify-between">
        <div>
          <CardTitle className="text-base">Callback handoff</CardTitle>
          <p className="text-xs text-muted-foreground mt-1">
            When a customer asks to be called back, route the request to a channel
            your team already monitors.
          </p>
        </div>
        <Switch
          checked={draft.enabled}
          onCheckedChange={(v) => setDraft((d) => ({ ...d, enabled: v }))}
          aria-label="Enable callback handoff"
        />
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-2">
          <Label>Send to</Label>
          <Select value={choice} onValueChange={(v) => setChoice(v as DestinationChoice)}>
            <SelectTrigger><SelectValue /></SelectTrigger>
            <SelectContent>
              <SelectItem value="off">Off — accept but don't deliver</SelectItem>
              {(['telegram', 'slack', 'whatsapp'] as const).map((p) => {
                const connected = connectedPlatforms.has(p)
                return (
                  <SelectItem key={p} value={p} disabled={!connected}>
                    {PLATFORM_LABEL[p] ?? p}
                    {!connected && ' — not connected'}
                  </SelectItem>
                )
              })}
              <SelectItem value="webhook">Webhook URL</SelectItem>
            </SelectContent>
          </Select>
          <p className="text-xs text-muted-foreground">
            {choice === 'off' &&
              'Submissions are recorded but no message is sent. Useful for staging.'}
            {choice === 'telegram' &&
              'Routed to your Telegram bot — uses the chat captured when you /start-ed it.'}
            {choice === 'slack' &&
              'Routed to your Slack workspace (channel below, or the workspace default).'}
            {choice === 'whatsapp' &&
              'Routed via your connected WhatsApp channel.'}
            {choice === 'webhook' &&
              'POSTed as JSON to your URL. Same shape as heartbeat webhooks.'}
          </p>
          {choice !== 'off' && choice !== 'webhook' && !connectedPlatforms.has(choice) && (
            <p className="text-xs text-destructive">
              No active {PLATFORM_LABEL[choice] ?? choice} channel in this workspace.{' '}
              <Link href="/settings" className="underline">
                Connect one under Settings → Channels
              </Link>{' '}
              first.
            </p>
          )}
        </div>

        {choice === 'slack' && (
          <div className="space-y-2">
            <Label>Slack channel ID (optional)</Label>
            <Input
              value={currentDest?.channel_id ?? ''}
              onChange={(e) => patchCurrent({ channel_id: e.target.value })}
              placeholder="C01ABCDEF — leave blank to use workspace default"
            />
          </div>
        )}

        {choice === 'webhook' && (
          <div className="space-y-2">
            <Label>Webhook URL</Label>
            <Input
              type="url"
              value={currentDest?.webhook_url ?? ''}
              onChange={(e) => patchCurrent({ webhook_url: e.target.value })}
              placeholder="https://hooks.slack.com/... or any endpoint"
            />
          </div>
        )}

        {error && <p className="text-xs text-destructive">{error}</p>}

        <div className="flex flex-wrap items-center justify-between gap-2 pt-2">
          <Button
            variant="ghost"
            size="sm"
            disabled={!canTest}
            onClick={runTest}
            title={
              dirty
                ? 'Save your changes first, then send a test.'
                : !initial.enabled
                ? 'Enable callbacks before testing.'
                : initial.destinations.length === 0
                ? 'Pick a destination before testing.'
                : 'Fires a dummy callback through the configured destination.'
            }
          >
            {testing ? 'Sending test…' : 'Send test'}
          </Button>
          <div className="flex gap-2">
            <Button
              variant="outline"
              disabled={!dirty || saving}
              onClick={() => setDraft(initial)}
            >
              Reset
            </Button>
            <Button disabled={!canSave} onClick={save}>
              {saving ? 'Saving…' : 'Save'}
            </Button>
          </div>
        </div>

        {testError && (
          <p className="text-xs text-destructive">{testError}</p>
        )}

        {testResult && (
          <div className="space-y-1 rounded-md border border-border/40 bg-muted/30 p-3 text-xs">
            <p className="text-foreground font-medium">
              Test dispatched — request id {testResult.request_id}
            </p>
            {testResult.results.length === 0 ? (
              <p className="text-muted-foreground">
                No destinations were attempted.
              </p>
            ) : (
              <ul className="space-y-0.5">
                {testResult.results.map((r, i) => (
                  <li
                    key={i}
                    className={r.success ? 'text-emerald-500' : 'text-destructive'}
                  >
                    {r.success ? '✓' : '✗'}{' '}
                    {PLATFORM_LABEL[r.platform ?? r.destination_type] ??
                      r.platform ??
                      r.destination_type}
                    {r.target ? ` → ${r.target}` : ''}{' '}
                    ({r.latency_ms}ms)
                    {r.error ? ` — ${r.error}` : ''}
                  </li>
                ))}
              </ul>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
