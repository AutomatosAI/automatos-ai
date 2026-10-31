/**
 * PRD-251B US-B208 — a plan's slots on the Socials calendar (pure). A slot that is planned and
 * not made yet is a dashed chip with its topic once one is pinned to its day; dragging it moves
 * the slot (slot_overrides), never the cadence. On the grid it is a `social` item whose post id
 * is `plan-slot:<plan id>:<slot key>`, so the calendar's one drag path can tell it from a post.
 * A made slot is its post, which the calendar already shows.
 *
 * P256-FIX-SOC-5 (F436): a planned slot also says why it has no post yet, from the server's phase:
 * "made Wed 21 Oct, 13:44" (its batch has not come), "being made now", or why it is stalled, late
 * or skipped.
 */
import type { ScheduleItem } from '@/hooks/use-activity-api'
import type { SocialPlan, SocialPlanSlot, SocialPlanSlotPhase, SocialPlanSlotsResponse } from '@/lib/socials-plan-types'
import { wallInZone } from '@/lib/social-time'
import { occurrence, type CalEvent, type WindowSpan } from '@/components/command-center/calendar-model'
import { ALL_CHANNELS, SOCIAL_EVENT_MIN, VIDEO_ONLY, type ChannelFilter } from './socials-calendar-model'

export const PLAN_SLOT_PREFIX = 'plan-slot:'
const UUID_LENGTH = 36

export interface PlannedSlot {
  planId: string
  planName: string
  timezone: string
  slot: SocialPlanSlot
}

export function plannedId(planId: string, key: string): string {
  return `${PLAN_SLOT_PREFIX}${planId}:${key}`
}

/** The plan and slot key a planned item's id names, or null for a post's own id. */
export function parsePlannedId(id: string | null | undefined): { planId: string; key: string } | null {
  if (!id || !id.startsWith(PLAN_SLOT_PREFIX)) return null
  const rest = id.slice(PLAN_SLOT_PREFIX.length)
  const planId = rest.slice(0, UUID_LENGTH)
  const key = rest.slice(UUID_LENGTH + 1)
  return planId.length === UUID_LENGTH && key ? { planId, key } : null
}

/** The planned (not made) slots of the active plans, from their slots answers. */
export function plannedSlots(answers: ReadonlyArray<SocialPlanSlotsResponse>, plans: ReadonlyArray<SocialPlan>): PlannedSlot[] {
  const byId = new Map(plans.map((plan) => [plan.id, plan]))
  return answers.flatMap((answer) => {
    const plan = byId.get(answer.plan_id)
    if (!plan) return []
    return answer.slots
      .filter((slot) => slot.state === 'planned')
      .map((slot) => ({ planId: plan.id, planName: plan.name, timezone: plan.timezone || 'UTC', slot }))
  })
}

const SHORT_DAYS = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
const SHORT_MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
const PHASE_LABELS: Partial<Record<SocialPlanSlotPhase, string>> = { stalled: 'Stalled', late: 'Late', skipped: 'Skipped' }
export const BEING_MADE = 'being made now'

/** "Wed 21 Oct, 13:44": a UTC instant on the wall clock of `tz`. */
export function zonedDayTime(iso: string, tz: string): string {
  const wall = wallInZone(iso, tz)
  const day = new Date(`${wall.slice(0, 10)}T00:00:00Z`)
  return `${SHORT_DAYS[day.getUTCDay()]} ${day.getUTCDate()} ${SHORT_MONTHS[day.getUTCMonth()]}, ${wall.slice(11, 16)}`
}

/** P256-FIX-SOC-5: why a planned slot has no post yet, in a line; null when the server says nothing. */
export function plannedPhaseLine(planned: PlannedSlot): string | null {
  const { phase, why, batch } = planned.slot
  if (phase === 'waiting' && batch?.made_at) return `made ${zonedDayTime(batch.made_at, planned.timezone)}`
  if (phase === 'building') return BEING_MADE
  const label = phase ? PHASE_LABELS[phase] : undefined
  return label && why ? `${label}: ${why}` : null
}

export function plannedTitle(planned: PlannedSlot): string {
  const title = planned.slot.topic?.title ?? `Planned: ${planned.planName}`
  const line = plannedPhaseLine(planned)
  return line ? `${title} · ${line}` : title
}

export function plannedItem(planned: PlannedSlot): ScheduleItem {
  const id = plannedId(planned.planId, planned.slot.key)
  return {
    id, name: plannedTitle(planned), type: 'social', next_run_at: planned.slot.at, frequency: '',
    agent_name: null, agent_id: null, post_id: id, timezone: planned.timezone, status: 'planned',
  }
}

export function plannedMatches(planned: PlannedSlot, filter: ChannelFilter): boolean {
  if (filter === ALL_CHANNELS) return true
  if (filter === VIDEO_ONLY) return planned.slot.format === 'video'
  return planned.slot.channels.includes(filter)
}

export function plannedEvents(planned: ReadonlyArray<PlannedSlot>, span: WindowSpan, filter: ChannelFilter): CalEvent[] {
  return planned.flatMap((entry) => {
    const at = new Date(entry.slot.at)
    if (!plannedMatches(entry, filter) || at < span.start || at > span.end) return []
    return [occurrence(plannedItem(entry), at, SOCIAL_EVENT_MIN, false)]
  })
}

/** The rail's plan line: "2 posts a day · 18 of 24 topics unused". */
export function planRailLine(plan: Pick<SocialPlan, 'cadence' | 'bank'>): string {
  const perWeek = plan.cadence.reduce((sum, row) => sum + row.days.length, 0)
  const perDay = Math.round((perWeek / 7) * 10) / 10
  return `${perDay} ${perDay === 1 ? 'post' : 'posts'} a day · ${plan.bank.unused} of ${plan.bank.topics} topics unused`
}
