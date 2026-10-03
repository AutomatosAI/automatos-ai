'use client'

/**
 * PRD-253 Wave P — the plan a session in Plan mode presented, on its ticket.
 *
 * `runtime_ref.session_plans` holds every plan the ticket's sessions presented,
 * oldest first (orchestrator/services/session_plans.py). While the latest one
 * waits for the operator, the ticket says so and shows the plan. The answer is
 * the Plan card in Command Center → Questions, or a Telegram reply.
 */

import Link from 'next/link'

export interface SessionPlan {
  round: number
  plan: string
  waiting: boolean
  answer: string | null
}

export function latestSessionPlan(ref: Record<string, any> | null | undefined): SessionPlan | null {
  const raw = ref?.session_plans
  if (!Array.isArray(raw)) return null
  const plans = raw.filter(
    (p): p is Record<string, unknown> => !!p && typeof p === 'object' && typeof p.plan === 'string' && p.plan.trim() !== '',
  )
  const latest = plans[plans.length - 1]
  if (!latest) return null
  return {
    round: Number(latest.version) || plans.length,
    plan: String(latest.plan),
    waiting: !latest.answered_at,
    answer: typeof latest.answer === 'string' ? latest.answer : null,
  }
}

export function SessionPlanPanel({ runtimeRef }: { runtimeRef: Record<string, any> | null | undefined }) {
  const plan = latestSessionPlan(runtimeRef)
  if (!plan) return null
  return (
    <div
      data-testid="session-plan"
      className={plan.waiting
        ? 'rounded-md border border-[hsl(var(--warning))]/40 bg-[hsl(var(--warning))]/10 p-2 text-xs space-y-1'
        : 'text-xs space-y-1'}
    >
      <p className={plan.waiting ? 'font-medium' : 'text-muted-foreground'}>
        {plan.waiting ? `Waiting for your approval of its plan (round ${plan.round})` : `Its plan (round ${plan.round})`}
        {!plan.waiting && plan.answer ? ` — your answer: ${plan.answer}` : ''}
      </p>
      <pre className="whitespace-pre-wrap font-sans max-h-48 overflow-y-auto">{plan.plan}</pre>
      {plan.waiting && (
        <p className="text-muted-foreground">
          Answer it on the Plan card in{' '}
          <Link href="/command-center?tab=questions" className="underline underline-offset-2 hover:text-foreground">
            Command Center → Questions
          </Link>{' '}
          or on Telegram: Approve starts the work, Reject sends the ticket to review, and your own words have it
          revise the plan. The whole plan is in plan.md among the ticket&apos;s deliverables.
        </p>
      )}
    </div>
  )
}
