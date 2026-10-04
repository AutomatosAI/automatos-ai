/**
 * PRD-251C US-C202 — a plan's rhythm on the Plan page (pure): the three choices, what each
 * one means in a line, the batch day's pull on the research day (C4: a weekly plan researches
 * the day before its batch, until the owner picks another day), and when the next batch is
 * made, in the plan's timezone.
 */
import type { SocialPlanRhythm, SocialWeekday } from '@/lib/socials-plan-types'
import { WEEKDAY_LABELS, dayBefore, type PlanDraft } from './plan-model'

export const RHYTHM_CHOICES: ReadonlyArray<{ value: SocialPlanRhythm; label: string }> = [
  { value: 'daily', label: 'Daily' },
  { value: 'weekly', label: 'Weekly' },
  { value: 'monthly', label: 'Monthly' },
]

const ORDINAL_SUFFIXES: Record<number, string> = { 1: 'st', 2: 'nd', 3: 'rd', 21: 'st', 22: 'nd', 23: 'rd' }

export function ordinal(day: number): string {
  return `${day}${ORDINAL_SUFFIXES[day] ?? 'th'}`
}

/** What the rhythm means, in a line: when posts are made and how they are approved. */
export function rhythmSummary(draft: Pick<PlanDraft, 'rhythm' | 'batchDay' | 'batchDate' | 'makeTime'>): string {
  if (draft.rhythm === 'weekly') {
    return `Every ${WEEKDAY_LABELS[draft.batchDay]} at ${draft.makeTime} the next 7 days' posts are made, and you approve the week in one go in the Queue.`
  }
  if (draft.rhythm === 'monthly') {
    return `On the ${ordinal(draft.batchDate)} of each month at ${draft.makeTime} the next month's posts are made, and you approve them in one go in the Queue.`
  }
  return 'Each post is made on its day, or the day before, at the make time, and you approve it in the Queue.'
}

/** A new batch day; the research day moves with it while it was the day before (C4). */
export function batchDayChange(draft: Pick<PlanDraft, 'batchDay' | 'researchDay'>, batchDay: SocialWeekday): Partial<PlanDraft> {
  return draft.researchDay === dayBefore(draft.batchDay) ? { batchDay, researchDay: dayBefore(batchDay) } : { batchDay }
}

/** "The next batch is made Sun 18 Oct, 17:00." in the plan's timezone; null without one. */
export function nextBatchLine(at: string | null | undefined, timezone: string): string | null {
  const when = at ? new Date(at) : null
  if (!when || Number.isNaN(when.getTime())) return null
  const options: Intl.DateTimeFormatOptions = { weekday: 'short', day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' }
  try {
    return `The next batch is made ${when.toLocaleString('en-GB', { ...options, timeZone: timezone || 'UTC' })}.`
  } catch {
    return `The next batch is made ${when.toLocaleString('en-GB', { ...options, timeZone: 'UTC' })} UTC.`
  }
}
