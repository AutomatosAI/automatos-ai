'use client';

import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { updateSiteSettings } from '@/lib/sites/api';
import type { Site, ProactiveSettings } from '@/lib/sites/types';

const DEFAULTS: ProactiveSettings = {
  enabled: false,
  page_types: ['product'],
  triggers: [{ type: 'time_on_page', seconds: 20 }],
  frequency_cap: { scope: 'session', max_pops: 1 },
  greeting_source: 'agent_with_canned_fallback',
  canned_fallback: 'Need a hand finding the right product?',
  agent_timeout_ms: 30000,
  popup_style: 'corner_bubble',
  respect_consent: true,
  dismissal_persistence: 'session',
};

export function ProactivePanel({
  site,
  onUpdated,
}: {
  site: Site;
  onUpdated: (s: Site) => void;
}) {
  const initial: ProactiveSettings = {
    ...DEFAULTS,
    ...(site.settings.widget_proactive ?? {}),
  };
  const [draft, setDraft] = useState<ProactiveSettings>(initial);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const dirty = JSON.stringify(draft) !== JSON.stringify(initial);

  async function save() {
    setSaving(true);
    setError(null);
    try {
      const updated = await updateSiteSettings(site.id, { widget_proactive: draft });
      onUpdated(updated);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Save failed');
    } finally {
      setSaving(false);
    }
  }

  const togglePageType = (page: string) => {
    setDraft((d) => ({
      ...d,
      page_types: d.page_types.includes(page)
        ? d.page_types.filter((p) => p !== page)
        : [...d.page_types, page],
    }));
  };

  const seconds = draft.triggers[0]?.seconds ?? 20;
  const setSeconds = (n: number) =>
    setDraft((d) => ({
      ...d,
      triggers: [{ type: 'time_on_page', seconds: Math.max(5, Math.min(120, n)) }],
    }));

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle className="text-base">Proactive engagement</CardTitle>
        <Switch
          checked={draft.enabled}
          onCheckedChange={(v) => setDraft((d) => ({ ...d, enabled: v }))}
        />
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-2">
          <Label className="text-xs text-gray-500">Fire on</Label>
          <div className="flex gap-2 flex-wrap">
            {['product', 'collection', 'page', 'home'].map((pt) => (
              <button
                key={pt}
                type="button"
                onClick={() => togglePageType(pt)}
                className={`px-3 py-1 rounded-full text-xs border ${
                  draft.page_types.includes(pt)
                    ? 'bg-indigo-100 border-indigo-300 text-indigo-900'
                    : 'bg-white border-gray-200 text-gray-600 hover:bg-gray-50'
                }`}
              >
                {pt}
              </button>
            ))}
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-1">
            <Label htmlFor="proactive-seconds" className="text-xs text-gray-500">
              After (seconds)
            </Label>
            <Input
              id="proactive-seconds"
              type="number"
              min={5}
              max={120}
              value={seconds}
              onChange={(e) => setSeconds(Number(e.target.value))}
            />
          </div>
          <div className="space-y-1">
            <Label htmlFor="proactive-greeting" className="text-xs text-gray-500">
              Canned fallback
            </Label>
            <Input
              id="proactive-greeting"
              value={draft.canned_fallback}
              onChange={(e) => setDraft((d) => ({ ...d, canned_fallback: e.target.value }))}
            />
          </div>
        </div>

        {error && (
          <p className="text-xs text-red-600">{error}</p>
        )}

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
