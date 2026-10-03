/**
 * PRD-251B US-B208 — a plan's slots on the Socials calendar (pure). A slot that is planned and
 * not made yet is a dashed chip with its topic once one is pinned to its day; dragging it moves
 * the slot (slot_overrides), never the cadence. On the grid it is a `social` item whose post id
 * is `plan-slot:<plan id>:<slot key>`, so the calendar's one drag path can tell it from a post.
 * A made slot is its post, which the calendar already shows.
 */
import type { ScheduleItem } from '@/hooks/use-activity-api'
import type { SocialPlan, SocialPlanSlot, SocialPlanSlotsResponse } from '@/lib/socials-plan-types'
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

export function plannedTitle(planned: PlannedSlot): string {
  return planned.slot.topic?.title ?? `Planned: ${planned.planName}`
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
