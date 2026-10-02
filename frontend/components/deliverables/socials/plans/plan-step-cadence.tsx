'use client'

/**
 * PRD-251B US-B207, step 2 — Cadence (Plan.dc.html): one row per channel group, each with its
 * format (a video with a length its templates offer, and the template or "Let Auto pick"),
 * how often and at what time. Nothing is made now: the plan only books the slots. The box
 * below sums the posts per channel over the plan's days and the render minutes its videos
 * need (images and text need none).
 */
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import type { SocialChannel, SocialTemplateSummary } from '@/lib/api-client'
import type { SocialWeekday } from '@/lib/socials-plan-types'
import { useSocialChannels } from '@/hooks/use-socials-composer'
import { useSocialTemplates } from '@/hooks/use-socials-editor'
import { channelLabel } from '../socials-status'
import { lengthLabel } from '../studio/socials-calendar-model'
import {
  OFTEN_LABELS, WEEKDAY_LABELS, WEEKDAYS, cadenceSummary, daysFor, newRow, oftenOf,
  type DraftRow, type Often, type PlanDraft,
} from './plan-model'
import { PlanStepHeading, PlanSummaryBox } from './plan-ui'

const SELECT = 'h-[38px] rounded-md border border-input bg-background px-2 text-sm'
const CHIP = 'h-[32px] rounded-full border border-border px-3 text-[13px] text-muted-foreground'
const CHIP_ON = 'border-accent bg-accent/15 text-foreground'
export const FORMAT_CHOICES = [
  { value: 'image', label: 'Image' },
  { value: 'carousel', label: 'Carousel' },
  { value: 'video', label: 'Video' },
  { value: 'fact_card', label: 'Fact card' },
  { value: 'text', label: 'Text only' },
] as const
export const NO_CHANNELS = 'Connect a channel in Composio first: the plan posts only to connected channels.'

interface RowProps {
  row: DraftRow
  index: number
  channels: ReadonlyArray<SocialChannel>
  videoTemplates: ReadonlyArray<SocialTemplateSummary>
  onChange: (row: DraftRow) => void
  onRemove: (() => void) | null
}

function lengthsOf(row: DraftRow, templates: ReadonlyArray<SocialTemplateSummary>): number[] {
  const chosen = templates.find((template) => template.id === row.templateId)
  const all = chosen ? chosen.durations : templates.flatMap((template) => template.durations)
  return Array.from(new Set(all)).sort((a, b) => a - b)
}

function VideoChoices({ row, index, videoTemplates, onChange }: Omit<RowProps, 'channels' | 'onRemove'>) {
  const lengths = lengthsOf(row, videoTemplates)
  return (
    <>
      <select aria-label={`Row ${index + 1} template`} className={SELECT} value={row.templateId ?? ''}
        onChange={(e) => onChange({ ...row, templateId: e.target.value || null, lengthSeconds: null })}>
        <option value="">Let Auto pick</option>
        {videoTemplates.map((template) => <option key={template.id} value={template.id}>{template.name}</option>)}
      </select>
      <select aria-label={`Row ${index + 1} length`} className={SELECT} value={row.lengthSeconds ?? ''}
        onChange={(e) => onChange({ ...row, lengthSeconds: e.target.value ? Number(e.target.value) : null })}>
        <option value="">Any length</option>
        {lengths.map((seconds) => <option key={seconds} value={seconds}>{lengthLabel(seconds)}</option>)}
      </select>
    </>
  )
}

function CadenceRowEditor({ row, index, channels, videoTemplates, onChange, onRemove }: RowProps) {
  const often = oftenOf(row.days)
  const toggle = (toolkit: string) =>
    onChange({ ...row, channels: row.channels.includes(toolkit) ? row.channels.filter((c) => c !== toolkit) : [...row.channels, toolkit] })
  return (
    <div role="group" aria-label={`Cadence row ${index + 1}`} className="flex flex-col gap-2 rounded-xl border border-border bg-card p-3">
      <div className="flex flex-wrap gap-1.5" aria-label="Channels">
        {channels.map((channel) => (
          <button key={channel.toolkit} type="button" aria-pressed={row.channels.includes(channel.toolkit)}
            className={`${CHIP} ${row.channels.includes(channel.toolkit) ? CHIP_ON : ''}`} onClick={() => toggle(channel.toolkit)}>
            {channel.label || channelLabel(channel.toolkit)}
          </button>
        ))}
      </div>
      <div className="flex flex-wrap items-center gap-2">
        <select aria-label={`Row ${index + 1} format`} className={SELECT} value={row.format}
          onChange={(e) => onChange({ ...row, format: e.target.value, templateId: null, lengthSeconds: null })}>
          {FORMAT_CHOICES.map((choice) => <option key={choice.value} value={choice.value}>{choice.label}</option>)}
        </select>
        {row.format === 'video' && <VideoChoices row={row} index={index} videoTemplates={videoTemplates} onChange={onChange} />}
        <select aria-label={`Row ${index + 1} how often`} className={SELECT} value={often}
          onChange={(e) => onChange({ ...row, days: daysFor(e.target.value as Often, row.days) })}>
          {(Object.keys(OFTEN_LABELS) as Often[]).map((key) => <option key={key} value={key}>{OFTEN_LABELS[key]}</option>)}
        </select>
        {often === 'weekly' && (
          <select aria-label={`Row ${index + 1} day`} className={SELECT} value={row.days[0]}
            onChange={(e) => onChange({ ...row, days: [e.target.value as SocialWeekday] })}>
            {WEEKDAYS.map((day) => <option key={day} value={day}>{WEEKDAY_LABELS[day]}</option>)}
          </select>
        )}
        <Input aria-label={`Row ${index + 1} time`} type="time" className="h-[38px] w-[120px]" value={row.time}
          onChange={(e) => onChange({ ...row, time: e.target.value })} />
        {onRemove && <Button type="button" variant="ghost" size="sm" onClick={onRemove}>Remove</Button>}
      </div>
    </div>
  )
}

export function PlanStepCadence({ draft, set }: { draft: PlanDraft; set: (changes: Partial<PlanDraft>) => void }) {
  const { data: channels } = useSocialChannels()
  const { data: templates } = useSocialTemplates('video')
  const summary = cadenceSummary(draft)
  const replace = (index: number, row: DraftRow) => set({ cadence: draft.cadence.map((r, i) => (i === index ? row : r)) })
  const remove = (index: number) => set({ cadence: draft.cadence.filter((_, i) => i !== index) })
  const perChannel = summary.perChannel.map((entry) => `${entry.posts} ${channelLabel(entry.channel)}`).join(', ')
  return (
    <div className="flex flex-col gap-4">
      <PlanStepHeading title="Cadence" lead="How often each channel gets a post, in what format and at what time. Nothing is made now: the plan only books the slots." />
      {!(channels ?? []).length && <p className="m-0 text-sm text-muted-foreground">{NO_CHANNELS}</p>}
      {draft.cadence.map((row, index) => (
        <CadenceRowEditor key={row.id ?? `new-${index}`} row={row} index={index} channels={channels ?? []} videoTemplates={templates ?? []}
          onChange={(next) => replace(index, next)} onRemove={draft.cadence.length > 1 ? () => remove(index) : null} />
      ))}
      <Button type="button" variant="outline" className="self-start" onClick={() => set({ cadence: [...draft.cadence, newRow()] })}>
        Add a channel
      </Button>
      <PlanSummaryBox label="Cadence summary">
        <p className="m-0">
          Over {summary.days} days: <strong>{summary.posts}</strong> posts{perChannel ? ` (${perChannel})` : ''}, about{' '}
          {summary.days ? Math.round((summary.posts / summary.days) * 10) / 10 : 0} a day.
        </p>
        <p className="m-0 text-muted-foreground">
          Render minutes for the whole plan: about {summary.renderMinutes} ({summary.videos} videos, one 9:16 render each). Images use none.
        </p>
      </PlanSummaryBox>
    </div>
  )
}
