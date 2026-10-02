'use client'

/**
 * PRD-251B US-B109 — the editor's When: the date, time and timezone of the post's slot
 * (PUT /slot on save, US-B105). Once approved it publishes then; unapproved by then, it
 * is skipped and nothing posts.
 */
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { browserTimezone } from '@/lib/social-time'
import type { EditorSlot } from './editor-model'
import { EditorCard, Hint } from './editor-ui'

export const WHEN_HINT = 'Once approved it publishes at this time; if nobody approves it by then, it is skipped and nothing posts.'
const FALLBACK_ZONES = ['UTC', 'Europe/London', 'Europe/Lisbon', 'Europe/Dublin', 'America/New_York', 'America/Los_Angeles']
const SELECT = 'h-[42px] w-full rounded-md border border-input bg-background px-2 text-sm'

/** Every timezone the browser knows, with the ones in use first. */
export function timezoneChoices(current: string | null): string[] {
  const supported = (Intl as unknown as { supportedValuesOf?: (key: string) => string[] }).supportedValuesOf
  const all = typeof supported === 'function' ? supported('timeZone') : FALLBACK_ZONES
  const first = [current, browserTimezone(), 'UTC'].filter((z): z is string => !!z)
  return Array.from(new Set([...first, ...all]))
}

export function EditorWhenCard({ slot, onChange }: { slot: EditorSlot | null; onChange: (slot: EditorSlot | null) => void }) {
  const current = slot ?? { date: '', time: '', timezone: browserTimezone() }
  const set = (changes: Partial<EditorSlot>) => {
    const next = { ...current, ...changes }
    onChange(next.date || next.time ? next : null)
  }
  return (
    <EditorCard label="When">
      <div className="grid gap-3 sm:grid-cols-3">
        <div className="flex flex-col gap-1">
          <Label htmlFor="socials-editor-date">Date</Label>
          <Input id="socials-editor-date" type="date" value={current.date} onChange={(e) => set({ date: e.target.value })} />
        </div>
        <div className="flex flex-col gap-1">
          <Label htmlFor="socials-editor-time">Time</Label>
          <Input id="socials-editor-time" type="time" value={current.time} onChange={(e) => set({ time: e.target.value })} />
        </div>
        <div className="flex flex-col gap-1">
          <Label htmlFor="socials-editor-zone">Timezone</Label>
          <select id="socials-editor-zone" className={SELECT} value={current.timezone} onChange={(e) => set({ timezone: e.target.value })}>
            {timezoneChoices(slot?.timezone ?? null).map((zone) => (
              <option key={zone} value={zone}>{zone}</option>
            ))}
          </select>
        </div>
      </div>
      <Hint>{WHEN_HINT}</Hint>
    </EditorCard>
  )
}
