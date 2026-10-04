/**
 * PRD-251B US-B207 — the Plan page's model (pure): the five steps (Plan.dc.html), the form's
 * draft and how it maps to and from a plan, the cadence's "how often" presets, the visual
 * mix and late-approval choices, the cadence summary (posts per channel over the plan's
 * days and the render minutes its videos need) and the plan's status line.
 */
import type {
  SocialLatePolicy,
  SocialPlan,
  SocialPlanInput,
  SocialPlanRhythm,
  SocialPlanSources,
  SocialWeekday,
} from '@/lib/socials-plan-types'

export const PLAN_STEPS = ['Goal and dates', 'Cadence', 'What to research', 'Making and approving', 'Content bank'] as const
export const WEEKDAYS: ReadonlyArray<SocialWeekday> = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']
export const WEEKDAY_LABELS: Record<SocialWeekday, string> = {
  mon: 'Monday', tue: 'Tuesday', wed: 'Wednesday', thu: 'Thursday', fri: 'Friday', sat: 'Saturday', sun: 'Sunday',
}
export const DEFAULT_PLAN_DAYS = 35
/** PRD-251C (C5): the server's default repeat window, and the most it takes. */
export const DEFAULT_REPEAT_AFTER_DAYS = 60
export const MAX_REPEAT_AFTER_DAYS = 365

/** PRD-251C (C1): a new plan makes its week on Sunday at 17:00, researches the day before
 * (O1, O3, C4), and reminds at 20:00 the evening before a day's posts (C3). */
export const DEFAULT_BATCH_DAY: SocialWeekday = 'sun'
export const DEFAULT_BATCH_DATE = 25
export const MAX_BATCH_DATE = 28
export const NEW_PLAN_MAKE_TIME = '17:00'
export const DEFAULT_REMIND_AT = '20:00'

export function dayBefore(day: SocialWeekday): SocialWeekday {
  return WEEKDAYS[(WEEKDAYS.indexOf(day) + WEEKDAYS.length - 1) % WEEKDAYS.length]
}

/** What the repeat window's field says, as a whole number of days the server takes. */
export function repeatDays(value: string): number {
  const days = Math.round(Number(value))
  return Number.isFinite(days) ? Math.min(MAX_REPEAT_AFTER_DAYS, Math.max(1, days)) : 1
}
const MS_PER_DAY = 86_400_000
const SECONDS_PER_MINUTE = 60

export type Often = 'daily' | 'weekdays' | 'mwf' | 'weekly' | 'custom'
export const OFTEN_LABELS: Record<Often, string> = {
  daily: 'Every day', weekdays: 'Weekdays', mwf: 'Mon, Wed, Fri', weekly: 'Once a week', custom: 'Chosen days',
}
const OFTEN_DAYS: Record<Exclude<Often, 'weekly' | 'custom'>, SocialWeekday[]> = {
  daily: [...WEEKDAYS],
  weekdays: ['mon', 'tue', 'wed', 'thu', 'fri'],
  mwf: ['mon', 'wed', 'fri'],
}

export interface DraftRow {
  id?: string
  channels: string[]
  format: string
  lengthSeconds: number | null
  templateId: string | null
  days: SocialWeekday[]
  time: string
}

export type MixKey = 'templates' | 'mixed' | 'ai'
export const MIX_PRESETS: ReadonlyArray<{ key: MixKey; label: string; mix: Record<string, number>; note: string }> = [
  { key: 'templates', label: 'Templates only (free)', mix: { templates: 100 }, note: 'Brand templates only. No AI tool is called.' },
  {
    key: 'mixed', label: 'Templates + AI images', mix: { templates: 50, ai_images: 50 },
    note: 'Templates for most posts, with AI images where a template has an image slot, styled by your brand-kit references.',
  },
  { key: 'ai', label: 'AI first', mix: { ai_images: 100 }, note: 'AI images first, with the words added on top by a template.' },
]

export const LATE_CHOICES: ReadonlyArray<{ value: SocialLatePolicy; label: string }> = [
  { value: 'skip', label: 'Skip the slot' },
  { value: 'next_slot', label: 'Move it to the next free slot' },
]

export interface PlanDraft {
  name: string
  goal: string
  audience: string
  startsOn: string
  endsOn: string
  timezone: string
  cadence: DraftRow[]
  sources: Omit<SocialPlanSources, 'never_say'>
  neverSay: string
  researchEnabled: boolean
  researchDay: SocialWeekday
  researchTime: string
  /** PRD-251C: research adds no topic close to a post of the last this-many days. */
  repeatAfterDays: number
  rhythm: SocialPlanRhythm
  batchDay: SocialWeekday
  batchDate: number
  remindAt: string
  makeTime: string
  imagesEarly: number
  videosEarly: number
  mix: MixKey
  latePolicy: SocialLatePolicy
}

export function oftenOf(days: ReadonlyArray<SocialWeekday>): Often {
  const sorted = WEEKDAYS.filter((day) => days.includes(day))
  const same = (preset: SocialWeekday[]) => preset.length === sorted.length && preset.every((day, i) => day === sorted[i])
  if (same(OFTEN_DAYS.daily)) return 'daily'
  if (same(OFTEN_DAYS.weekdays)) return 'weekdays'
  if (same(OFTEN_DAYS.mwf)) return 'mwf'
  return sorted.length === 1 ? 'weekly' : 'custom'
}

/** The days a "how often" choice means; a weekly row keeps its day (Monday by default). */
export function daysFor(often: Often, current: ReadonlyArray<SocialWeekday>): SocialWeekday[] {
  if (often === 'weekly') return [current[0] ?? 'mon']
  if (often === 'custom') return current.length ? [...current] : ['mon']
  return [...OFTEN_DAYS[often]]
}

function isoDay(date: Date): string {
  return date.toISOString().slice(0, 10)
}

export function newRow(): DraftRow {
  return { channels: [], format: 'image', lengthSeconds: null, templateId: null, days: [...OFTEN_DAYS.weekdays], time: '09:00' }
}

export function emptyDraft(timezone: string, today: Date = new Date()): PlanDraft {
  return {
    name: '', goal: '', audience: '', timezone,
    startsOn: isoDay(today), endsOn: isoDay(new Date(today.getTime() + (DEFAULT_PLAN_DAYS - 1) * MS_PER_DAY)),
    cadence: [newRow()],
    sources: { knowledge: true, deliverables: true, website: true, github: false, notes: '' },
    neverSay: '', researchEnabled: true, researchDay: dayBefore(DEFAULT_BATCH_DAY), researchTime: '06:00', repeatAfterDays: DEFAULT_REPEAT_AFTER_DAYS,
    rhythm: 'weekly', batchDay: DEFAULT_BATCH_DAY, batchDate: DEFAULT_BATCH_DATE, remindAt: DEFAULT_REMIND_AT,
    makeTime: NEW_PLAN_MAKE_TIME, imagesEarly: 0, videosEarly: 1, mix: 'templates', latePolicy: 'skip',
  }
}

export function mixOf(visualMix: Record<string, number> | null | undefined): MixKey {
  const found = MIX_PRESETS.find((preset) => JSON.stringify(preset.mix) === JSON.stringify(visualMix ?? {}))
  return found?.key ?? 'templates'
}

export function draftFromPlan(plan: SocialPlan): PlanDraft {
  const base = emptyDraft(plan.timezone || 'UTC')
  return {
    ...base,
    name: plan.name, goal: plan.goal ?? '', audience: plan.audience ?? '',
    startsOn: plan.starts_on ?? base.startsOn, endsOn: plan.ends_on ?? base.endsOn,
    cadence: plan.cadence.map((row) => ({
      id: row.id, channels: [...row.channels], format: row.format, lengthSeconds: row.length_seconds,
      templateId: row.template_id, days: [...row.days], time: row.time,
    })),
    sources: { knowledge: plan.sources.knowledge, deliverables: plan.sources.deliverables, website: plan.sources.website, github: plan.sources.github, notes: plan.sources.notes ?? '' },
    neverSay: (plan.sources.never_say ?? []).join(', '),
    researchEnabled: plan.research.enabled, researchDay: plan.research.day, researchTime: plan.research.time,
    repeatAfterDays: plan.research.repeat_after_days ?? DEFAULT_REPEAT_AFTER_DAYS,
    rhythm: plan.make.rhythm ?? 'daily', batchDay: plan.make.batch_day ?? DEFAULT_BATCH_DAY,
    batchDate: plan.make.batch_date ?? DEFAULT_BATCH_DATE, remindAt: plan.make.remind_at ?? DEFAULT_REMIND_AT,
    makeTime: plan.make.time, imagesEarly: plan.make.image_days_early ?? 0, videosEarly: plan.make.video_days_early,
    mix: mixOf(plan.make.visual_mix), latePolicy: plan.late_policy,
  }
}

function phrases(text: string): string[] {
  return text.split(',').map((phrase) => phrase.trim()).filter(Boolean)
}

export function inputFromDraft(draft: PlanDraft): SocialPlanInput {
  return {
    name: draft.name.trim(), goal: draft.goal.trim() || null, audience: draft.audience.trim() || null,
    starts_on: draft.startsOn, ends_on: draft.endsOn, timezone: draft.timezone,
    cadence: draft.cadence.map((row) => ({
      ...(row.id ? { id: row.id } : {}), channels: row.channels, format: row.format,
      length_seconds: row.format === 'video' ? row.lengthSeconds : null, template_id: row.templateId, days: row.days, time: row.time,
    })),
    sources: { ...draft.sources, never_say: phrases(draft.neverSay) },
    make: {
      time: draft.makeTime, image_days_early: draft.imagesEarly, video_days_early: draft.videosEarly,
      visual_mix: MIX_PRESETS.find((preset) => preset.key === draft.mix)?.mix ?? { templates: 100 },
      rhythm: draft.rhythm, batch_day: draft.batchDay, batch_date: draft.batchDate, remind_at: draft.remindAt,
    },
    research: { enabled: draft.researchEnabled, day: draft.researchDay, time: draft.researchTime, repeat_after_days: draft.repeatAfterDays },
    late_policy: draft.latePolicy,
  }
}

/** What the plan still needs before it can be saved, in the form's words; empty when ready. */
export function missingFields(draft: PlanDraft): string[] {
  const missing: string[] = []
  if (!draft.name.trim()) missing.push('a name')
  if (!draft.startsOn || !draft.endsOn) missing.push('its dates')
  if (!draft.cadence.length || draft.cadence.some((row) => !row.channels.length)) missing.push('a channel for every cadence row')
  return missing
}

/** The plan's local days, first to last. */
export function planDays(startsOn: string, endsOn: string): string[] {
  const start = new Date(`${startsOn}T00:00:00Z`).getTime()
  const end = new Date(`${endsOn}T00:00:00Z`).getTime()
  if (!Number.isFinite(start) || !Number.isFinite(end) || end < start) return []
  return Array.from({ length: Math.round((end - start) / MS_PER_DAY) + 1 }, (_, i) => isoDay(new Date(start + i * MS_PER_DAY)))
}

function weekdayOf(day: string): SocialWeekday {
  return WEEKDAYS[(new Date(`${day}T00:00:00Z`).getUTCDay() + 6) % 7]
}

export interface CadenceSummary {
  days: number
  perChannel: Array<{ channel: string; posts: number }>
  posts: number
  videos: number
  renderMinutes: number
}

/** Over the plan's days: each channel's posts, all posts, the videos and their render minutes
 * (one 9:16 render of each video at its length; images and text need none). */
export function cadenceSummary(draft: Pick<PlanDraft, 'startsOn' | 'endsOn' | 'cadence'>): CadenceSummary {
  const days = planDays(draft.startsOn, draft.endsOn)
  const perChannel = new Map<string, number>()
  let posts = 0
  let videos = 0
  let seconds = 0
  for (const row of draft.cadence) {
    const slots = days.filter((day) => row.days.includes(weekdayOf(day))).length
    posts += slots
    row.channels.forEach((channel) => perChannel.set(channel, (perChannel.get(channel) ?? 0) + slots))
    if (row.format === 'video') {
      videos += slots
      seconds += slots * (row.lengthSeconds ?? 30)
    }
  }
  const channels = Array.from(perChannel, ([channel, count]) => ({ channel, posts: count }))
  return { days: days.length, perChannel: channels, posts, videos, renderMinutes: Math.round((seconds / SECONDS_PER_MINUTE) * 10) / 10 }
}

/** "Running · day 10 of 35", "Paused · day 10 of 35", "Starts in 3 days", "Ended". */
export function statusLine(plan: Pick<SocialPlan, 'status' | 'starts_on' | 'ends_on'>, today: Date = new Date()): string {
  if (plan.status === 'ended') return 'Ended'
  const days = plan.starts_on && plan.ends_on ? planDays(plan.starts_on, plan.ends_on) : []
  const index = days.indexOf(isoDay(today))
  if (days.length && isoDay(today) < days[0]) {
    const wait = planDays(isoDay(today), days[0]).length - 1
    return `Starts in ${wait} ${wait === 1 ? 'day' : 'days'}`
  }
  if (index < 0) return plan.status === 'paused' ? 'Paused' : 'Finished'
  return `${plan.status === 'paused' ? 'Paused' : 'Running'} · day ${index + 1} of ${days.length}`
}
