'use client'

import { useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Switch } from '@/components/ui/switch'
import { Button } from '@/components/ui/button'
import { ChannelDestinationPicker } from './destinations/ChannelDestinationPicker'
import {
  sendCallbackTest,
  updateSiteSettings,
  type CallbackTestResponse,
} from '@/lib/sites/api'
import type { Site, CallbackSettings, CallbackDestination } from '@/lib/sites/types'

const DEFAULTS: CallbackSettings = {
  enabled: false,
  destinations: [],
}

export function CallbackPanel({
  site,
  onUpdated,
}: {
  site: Site
  onUpdated: (s: Site) => void
}) {
  const initial: CallbackSettings = {
    ...DEFAULTS,
    ...(site.settings.callback ?? {}),
    // Drop legacy destination shapes silently — they were never shipped
    // outside the user's test workspace, but be defensive.
    destinations: ((site.settings.callback?.destinations as CallbackDestination[] | undefined) ?? []).filter(
      (d) => d?.type === 'channel_connection',
    ),
  }
  const [draft, setDraft] = useState<CallbackSettings>(initial)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [testing, setTesting] = useState(false)
  const [testResult, setTestResult] = useState<CallbackTestResponse | null>(null)
  const [testError, setTestError] = useState<string | null>(null)

  const dirty = JSON.stringify(draft) !== JSON.stringify(initial)
  const targetsValid = draft.destinations.every((d) => d.target.trim().length > 0)
  const canSave = dirty && !saving && targetsValid
  // Test runs against what's saved on the server, not the unsaved draft.
  const canTest =
    !testing &&
    !dirty &&
    initial.destinations.length > 0 &&
    initial.destinations.every((d) => d.target.trim().length > 0)

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
        <ChannelDestinationPicker
          value={draft.destinations}
          onChange={(destinations) => setDraft((d) => ({ ...d, destinations }))}
        />

        {!targetsValid && draft.destinations.length > 0 && (
          <p className="text-xs text-destructive">
            Every destination needs a target (channel ID, chat ID, or phone number).
          </p>
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
                : initial.destinations.length === 0
                ? 'Add at least one destination before testing.'
                : 'Fires a dummy callback through every configured destination.'
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
                    {r.platform ?? r.destination_type}
                    {r.target ? ` → ${r.target}` : ''}
                    {' '}({r.latency_ms}ms)
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
