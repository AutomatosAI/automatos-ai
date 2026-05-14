'use client';

import { useState } from 'react';
import { Trash2, Plus } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { updateSiteSettings } from '@/lib/sites/api';
import type {
  Site,
  CallbackSettings,
  CallbackDestination,
  CallbackDestinationType,
} from '@/lib/sites/types';

const DEFAULTS: CallbackSettings = {
  enabled: false,
  destinations: [],
  working_hours_only: true,
  working_hours: {
    tz: 'Europe/London',
    monday: { start: '09:00', end: '17:00' },
    tuesday: { start: '09:00', end: '17:00' },
    wednesday: { start: '09:00', end: '17:00' },
    thursday: { start: '09:00', end: '17:00' },
    friday: { start: '09:00', end: '17:00' },
    saturday: 'closed',
    sunday: 'closed',
  },
  sla_hours: 4,
  team_capacity: 'limited',
  intent_phrases: ['speak to someone', 'call me back', 'talk to a human'],
  rate_limit_per_hour: 100,
};

export function CallbackPanel({
  site,
  onUpdated,
}: {
  site: Site;
  onUpdated: (s: Site) => void;
}) {
  const initial: CallbackSettings = {
    ...DEFAULTS,
    ...(site.settings.callback ?? {}),
  };
  const [draft, setDraft] = useState<CallbackSettings>(initial);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const dirty = JSON.stringify(draft) !== JSON.stringify(initial);

  async function save() {
    setSaving(true);
    setError(null);
    try {
      const updated = await updateSiteSettings(site.id, { callback: draft });
      onUpdated(updated);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Save failed');
    } finally {
      setSaving(false);
    }
  }

  function addDestination(type: CallbackDestinationType) {
    const blank: CallbackDestination = { type };
    setDraft((d) => ({ ...d, destinations: [...d.destinations, blank] }));
  }

  function updateDestination(idx: number, patch: Partial<CallbackDestination>) {
    setDraft((d) => ({
      ...d,
      destinations: d.destinations.map((dest, i) =>
        i === idx ? { ...dest, ...patch } : dest,
      ),
    }));
  }

  function removeDestination(idx: number) {
    setDraft((d) => ({
      ...d,
      destinations: d.destinations.filter((_, i) => i !== idx),
    }));
  }

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle className="text-base">Callback handoff</CardTitle>
        <Switch
          checked={draft.enabled}
          onCheckedChange={(v) => setDraft((d) => ({ ...d, enabled: v }))}
        />
      </CardHeader>
      <CardContent className="space-y-5">
        {/* Destinations */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <Label className="text-xs text-gray-500">
              Send callbacks to ({draft.destinations.length})
            </Label>
            <div className="flex gap-1">
              <Button size="sm" variant="outline" onClick={() => addDestination('email')}>
                <Plus className="w-3 h-3 mr-1" /> Email
              </Button>
              <Button size="sm" variant="outline" onClick={() => addDestination('slack_webhook')}>
                <Plus className="w-3 h-3 mr-1" /> Slack
              </Button>
              <Button size="sm" variant="outline" onClick={() => addDestination('crm_webhook')}>
                <Plus className="w-3 h-3 mr-1" /> CRM
              </Button>
              {site.type === 'shopify' && (
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => addDestination('shopify_customer_note')}
                >
                  <Plus className="w-3 h-3 mr-1" /> Shopify note
                </Button>
              )}
            </div>
          </div>

          {draft.destinations.length === 0 && (
            <p className="text-xs text-gray-400 italic">
              No destinations — callbacks will be accepted but not delivered until you add one.
            </p>
          )}

          <div className="space-y-2">
            {draft.destinations.map((dest, idx) => (
              <div
                key={idx}
                className="flex items-center gap-2 p-2 border rounded bg-gray-50"
              >
                <Badge variant="outline" className="text-xs shrink-0">
                  {dest.type}
                </Badge>
                {dest.type === 'email' && (
                  <Input
                    placeholder="sales@example.com"
                    value={dest.address ?? ''}
                    onChange={(e) => updateDestination(idx, { address: e.target.value })}
                  />
                )}
                {dest.type === 'slack_webhook' && (
                  <Input
                    placeholder="https://hooks.slack.com/services/..."
                    value={dest.url ?? ''}
                    onChange={(e) => updateDestination(idx, { url: e.target.value })}
                  />
                )}
                {dest.type === 'crm_webhook' && (
                  <>
                    <Input
                      placeholder="https://crm.example.com/webhook"
                      value={dest.url ?? ''}
                      onChange={(e) => updateDestination(idx, { url: e.target.value })}
                    />
                    <Input
                      placeholder="Bearer ... (optional)"
                      value={dest.auth_header ?? ''}
                      onChange={(e) => updateDestination(idx, { auth_header: e.target.value })}
                    />
                  </>
                )}
                {dest.type === 'shopify_customer_note' && (
                  <span className="text-xs text-gray-500 flex-1">
                    Notes append to matching Shopify customer records.
                  </span>
                )}
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={() => removeDestination(idx)}
                >
                  <Trash2 className="w-3 h-3" />
                </Button>
              </div>
            ))}
          </div>
        </div>

        {/* SLA + capacity */}
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-1">
            <Label htmlFor="cb-sla" className="text-xs text-gray-500">
              Response SLA (working hours)
            </Label>
            <Input
              id="cb-sla"
              type="number"
              min={1}
              max={48}
              value={draft.sla_hours}
              onChange={(e) =>
                setDraft((d) => ({ ...d, sla_hours: Number(e.target.value) || 4 }))
              }
            />
          </div>
          <div className="space-y-1">
            <Label htmlFor="cb-capacity" className="text-xs text-gray-500">
              Team capacity
            </Label>
            <select
              id="cb-capacity"
              className="w-full border rounded px-2 py-2 text-sm bg-white"
              value={draft.team_capacity}
              onChange={(e) =>
                setDraft((d) => ({
                  ...d,
                  team_capacity: e.target.value as 'limited' | 'normal',
                }))
              }
            >
              <option value="limited">Limited (softens phrasing)</option>
              <option value="normal">Normal</option>
            </select>
          </div>
        </div>

        {/* Working hours summary */}
        <div className="space-y-1">
          <Label className="text-xs text-gray-500">Working hours</Label>
          <div className="text-sm text-gray-700">
            {draft.working_hours_only ? (
              <>
                {draft.working_hours.tz} ·
                {(['monday','tuesday','wednesday','thursday','friday','saturday','sunday'] as const)
                  .map((d) => {
                    const v = draft.working_hours[d];
                    return v === 'closed'
                      ? null
                      : ` ${d.slice(0, 3)} ${v.start}-${v.end}`;
                  })
                  .filter(Boolean)
                  .join(',')}
              </>
            ) : (
              <span className="text-gray-500 italic">24/7 — no working-hours filter</span>
            )}
          </div>
          <Button
            size="sm"
            variant="link"
            className="px-0"
            onClick={() =>
              setDraft((d) => ({ ...d, working_hours_only: !d.working_hours_only }))
            }
          >
            {draft.working_hours_only ? 'Switch to 24/7' : 'Restrict to working hours'}
          </Button>
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
