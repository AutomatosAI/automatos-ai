'use client';

import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { updateSiteSettings } from '@/lib/sites/api';
import type { Site, CartIdleSettings } from '@/lib/sites/types';

const DEFAULTS: CartIdleSettings = {
  enabled: false,
  idle_seconds: 90,
  greeting: 'Any questions before you check out?',
  frequency_cap: { scope: 'session', max_pops: 1 },
};

export function CartIdlePanel({
  site,
  onUpdated,
}: {
  site: Site;
  onUpdated: (s: Site) => void;
}) {
  // Capability gate — hide entirely if Site has no cart.
  if (!site.capabilities.has_cart) return null;

  const initial: CartIdleSettings = {
    ...DEFAULTS,
    ...(site.settings.cart_idle ?? {}),
  };
  const [draft, setDraft] = useState<CartIdleSettings>(initial);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const dirty = JSON.stringify(draft) !== JSON.stringify(initial);

  async function save() {
    setSaving(true);
    setError(null);
    try {
      const updated = await updateSiteSettings(site.id, { cart_idle: draft });
      onUpdated(updated);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Save failed');
    } finally {
      setSaving(false);
    }
  }

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle className="text-base">Cart-idle proactive</CardTitle>
        <Switch
          checked={draft.enabled}
          onCheckedChange={(v) => setDraft((d) => ({ ...d, enabled: v }))}
        />
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-1">
            <Label htmlFor="cart-idle-seconds" className="text-xs text-gray-500">
              Idle threshold (seconds)
            </Label>
            <Input
              id="cart-idle-seconds"
              type="number"
              min={10}
              max={600}
              value={draft.idle_seconds}
              onChange={(e) =>
                setDraft((d) => ({ ...d, idle_seconds: Number(e.target.value) || 90 }))
              }
            />
          </div>
        </div>
        <div className="space-y-1">
          <Label htmlFor="cart-idle-greeting" className="text-xs text-gray-500">
            Greeting
          </Label>
          <Input
            id="cart-idle-greeting"
            value={draft.greeting}
            onChange={(e) => setDraft((d) => ({ ...d, greeting: e.target.value }))}
          />
        </div>

        {error && <p className="text-xs text-red-600">{error}</p>}

        <div className="flex justify-end gap-2 pt-2">
          <Button variant="outline" disabled={!dirty || saving} onClick={() => setDraft(initial)}>
            Reset
          </Button>
          <Button disabled={!dirty || saving} onClick={save}>
            {saving ? 'Saving…' : 'Save'}
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
