'use client'

/**
 * PRD-251C (C9, US-C407; C7, US-C404) — on a saved plan: its health (what needs the owner now,
 * each with the one action that fixes it) and Auto's proposals from its results, each applied
 * by one click through the plan's PUT and never by itself. Nothing shows when all is well.
 */
import Link from 'next/link'

import { Button } from '@/components/ui/button'
import { BRAND_KIT_HREF } from '@/lib/deliverables/tabs'
import type { SocialPlan } from '@/lib/socials-plan-types'
import type { SocialPlanHealthItem } from '@/lib/socials-results-types'
import { useResearchSocialPlan } from '@/hooks/use-socials-plans'
import { useApplyProposal, useSocialPlanHealth, useSocialPlanProposals } from '@/hooks/use-socials-results'

/** Tools & Integrations, where a channel is connected through Composio. */
export const TOOLS_HREF = '/tools'
const CARD = 'flex flex-col gap-2.5 rounded-xl border border-border bg-card p-3.5'
const ROW = 'flex flex-wrap items-start justify-between gap-3'
const LINK = 'text-sm font-medium text-foreground underline-offset-4 hover:underline'

interface InsightsProps {
  planId: string
  canEdit: boolean
  onCadence: () => void
  onApplied: (plan: SocialPlan) => void
}

function HealthAction({ item, planId, canEdit, onCadence }: { item: SocialPlanHealthItem } & Omit<InsightsProps, 'onApplied'>) {
  const research = useResearchSocialPlan()
  const { kind, label } = item.action
  if (kind === 'connect') return <Link href={TOOLS_HREF} className={LINK}>{label}</Link>
  if (kind === 'ai_tools') return <Link href={BRAND_KIT_HREF as any} className={LINK}>{label}</Link>
  if (kind === 'cadence') return <Button type="button" size="sm" variant="outline" onClick={onCadence}>{label}</Button>
  return (
    <Button type="button" size="sm" variant="outline" disabled={!canEdit || research.isLoading} onClick={() => research.mutate(planId)}>
      {label}
    </Button>
  )
}

export function PlanInsights({ planId, canEdit, onCadence, onApplied }: InsightsProps) {
  const health = useSocialPlanHealth(planId)
  const proposals = useSocialPlanProposals(planId)
  const apply = useApplyProposal(planId, onApplied)
  const items = health.data?.items ?? []
  const proposed = proposals.data?.proposals ?? []
  if (!items.length && !proposed.length) return null
  return (
    <div className="grid items-start gap-3 lg:grid-cols-2">
      {items.length > 0 && (
        <section aria-label="Plan health" className={CARD}>
          <h2 className="m-0 text-sm font-semibold text-foreground">Plan health</h2>
          <ul className="m-0 flex list-none flex-col gap-2.5 p-0">
            {items.map((item) => (
              <li key={item.id} aria-label={item.title} className={ROW}>
                <div className="flex min-w-0 flex-col">
                  <span className="text-sm font-medium text-foreground">{item.title}</span>
                  <span className="text-[13px] text-muted-foreground">{item.detail}</span>
                </div>
                <HealthAction item={item} planId={planId} canEdit={canEdit} onCadence={onCadence} />
              </li>
            ))}
          </ul>
        </section>
      )}
      {proposed.length > 0 && (
        <section aria-label="Auto's proposals" className={CARD}>
          <h2 className="m-0 text-sm font-semibold text-foreground">Auto proposes</h2>
          <ul className="m-0 flex list-none flex-col gap-2.5 p-0">
            {proposed.map((proposal) => (
              <li key={proposal.id} aria-label={proposal.title} className={ROW}>
                <div className="flex min-w-0 flex-col">
                  <span className="text-sm font-medium text-foreground">{proposal.title}</span>
                  <span className="text-[13px] text-muted-foreground">{proposal.why}</span>
                </div>
                <Button type="button" size="sm" disabled={!canEdit || apply.isLoading} onClick={() => apply.mutate(proposal)}>Apply</Button>
              </li>
            ))}
          </ul>
        </section>
      )}
    </div>
  )
}
