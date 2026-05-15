'use client'

import { useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Switch } from '@/components/ui/switch'
import { Label } from '@/components/ui/label'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { updateSiteSettings } from '@/lib/sites/api'
import type { Site, CartIdleSettings } from '@/lib/sites/types'

const DEFAULTS: CartIdleSettings = {
  enabled: false,
  idle_seconds: 90,
  greeting: 'Any questions before you check out?',
  frequency_cap: { scope: 'session', max_pops: 1 },
}

const IDLE_MIN = 10
const IDLE_MAX = 600

export function CartIdlePanel({
  site,
  onUpdated,
}: {
  site: Site
  onUpdated: (s: Site) => void
}) {
  if (!site.capabilities.has_cart) {
    return (
      <Card className="glass-card">
        <CardHeader>
          <CardTitle className="text-base">Cart-idle proactive</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">
            This Site type ({site.type}) doesn't expose cart events, so cart-idle
            isn't available. Connect a Shopify storefront to enable it.
          </p>
        </CardContent>
      </Card>
    )
  }

  const initial: CartIdleSettings = {
    ...DEFAULTS,
    ...(site.settings.cart_idle ?? {}),
  }
  const [draft, setDraft] = useState<CartIdleSettings>(initial)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const dirty = JSON.stringify(draft) !== JSON.stringify(initial)
  const idleInRange = draft.idle_seconds >= IDLE_MIN && draft.idle_seconds <= IDLE_MAX
  const greetingValid = draft.greeting.trim().length > 0
  const canSave = dirty && !saving && idleInRange && greetingValid

  async function save() {
    setSaving(true)
    setError(null)
    try {
      const updated = await updateSiteSettings(site.id, { cart_idle: draft })
      onUpdated(updated)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Save failed')
    } finally {
      setSaving(false)
    }
  }

  return (
    <Card className="glass-card">
      <CardHeader className="flex flex-row items-center justify-between">
        <div>
          <CardTitle className="text-base">Cart-idle proactive</CardTitle>
          <p className="text-xs text-muted-foreground mt-1">
            Pop the widget when a shopper has items in cart but goes idle.
          </p>
        </div>
        <Switch
          checked={draft.enabled}
          onCheckedChange={(v) => setDraft((d) => ({ ...d, enabled: v }))}
          aria-label="Enable cart-idle"
        />
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-1">
            <Label htmlFor="cart-idle-seconds" className="text-xs text-muted-foreground">
              Idle threshold (seconds)
            </Label>
            <Input
              id="cart-idle-seconds"
              type="number"
              min={IDLE_MIN}
              max={IDLE_MAX}
              value={draft.idle_seconds}
              onChange={(e) =>
                setDraft((d) => ({ ...d, idle_seconds: Number(e.target.value) }))
              }
              aria-invalid={!idleInRange}
            />
            <p className={`text-[10px] ${idleInRange ? 'text-muted-foreground' : 'text-destructive'}`}>
              Between {IDLE_MIN} and {IDLE_MAX} seconds.
            </p>
          </div>
        </div>

        <div className="space-y-1">
          <Label htmlFor="cart-idle-greeting" className="text-xs text-muted-foreground">
            Greeting
          </Label>
          <Input
            id="cart-idle-greeting"
            value={draft.greeting}
            onChange={(e) => setDraft((d) => ({ ...d, greeting: e.target.value }))}
            aria-invalid={!greetingValid}
          />
          {!greetingValid && (
            <p className="text-[10px] text-destructive">Greeting can't be empty.</p>
          )}
        </div>

        {error && <p className="text-xs text-destructive">{error}</p>}

        <div className="flex justify-end gap-2 pt-2">
          <Button variant="outline" disabled={!dirty || saving} onClick={() => setDraft(initial)}>
            Reset
          </Button>
          <Button disabled={!canSave} onClick={save}>
            {saving ? 'Saving…' : 'Save'}
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}
