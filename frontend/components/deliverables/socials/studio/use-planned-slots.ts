'use client'

/**
 * PRD-251B US-B208 — the active plans and their planned (not made) slots in the calendar's
 * window: one slots query per active plan, over the Month or Week the calendar shows.
 */
import { useMemo } from 'react'

import { windowSpanFor } from '@/components/command-center/calendar-view'
import { useSocialPlans, useSocialPlansSlots } from '@/hooks/use-socials-plans'
import type { SocialPlan } from '@/lib/socials-plan-types'
import { plannedSlots, type PlannedSlot } from './plan-calendar-model'

export function usePlannedSlots(mode: 'month' | 'week', anchor: Date): { plans: SocialPlan[]; planned: PlannedSlot[] } {
  const { data } = useSocialPlans()
  const plans = useMemo(() => (data?.plans ?? []).filter((plan) => plan.status === 'active'), [data])
  const span = windowSpanFor(mode, anchor)
  const start = span.start.toISOString()
  const end = new Date(span.end.getTime() + 1).toISOString()
  const answers = useSocialPlansSlots(plans.map((plan) => plan.id), start, end)
  return { plans, planned: plannedSlots(answers, plans) }
}
