'use client'

/**
 * PRD-251B B2, B6 — the Studio's Plans view: the workspace's plans (each with its state, its
 * dates, its cadence in a line and its content bank's counts), and a plan's page at
 * ?view=plans&plan=<id> (or plan=new). Campaigns that are not plans keep their own view
 * below the plans, so a series approval still has its place. Plan with Auto opens a new plan,
 * where Auto drafts it from what the person says.
 */
import { Button } from '@/components/ui/button'
import type { Workspace } from '@/components/workspace-provider'
import type { SocialPost } from '@/lib/api-client'
import type { SocialPlan } from '@/lib/socials-plan-types'
import { useSocialPlans } from '@/hooks/use-socials-plans'
import { SocialsCampaigns } from '../socials-campaigns-view'
import { canAuthorPosts, canDeletePlan, channelLabel } from '../socials-status'
import { NEW_PLAN, type GoTo } from '../studio/studio-route'
import { AUTO_TITLE } from './plan-auto-model'
import { OFTEN_LABELS, cadenceSummary, oftenOf, statusLine } from './plan-model'
import { SocialsPlanPage } from './plan-page'

export const NO_PLANS = 'No plans yet. A plan books a cadence of posts over its dates; each post is made on its day from the content bank.'
export const PLAN_WITH_AUTO_LEAD =
  "Say what this week's socials should do, in your own words, and Auto drafts the plan from your knowledge, your website and your ideas."

export function cadenceLine(plan: Pick<SocialPlan, 'cadence'>): string {
  return plan.cadence
    .map((row) => `${row.channels.map(channelLabel).join(' + ')} ${row.format === 'text' ? 'text' : row.format} · ${OFTEN_LABELS[oftenOf(row.days)].toLowerCase()} at ${row.time}`)
    .join(' · ')
}

function PlanCard({ plan, onOpen }: { plan: SocialPlan; onOpen: () => void }) {
  const summary = cadenceSummary({ startsOn: plan.starts_on ?? '', endsOn: plan.ends_on ?? '', cadence: plan.cadence.map((row) => ({ ...row, lengthSeconds: row.length_seconds, templateId: row.template_id })) })
  return (
    <article aria-label={plan.name} className="flex flex-col gap-2 rounded-xl border border-border bg-card p-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 className="m-0 text-[17px] font-semibold text-foreground">{plan.name}</h3>
        <span className="rounded-full border border-border px-2.5 py-0.5 text-[12px] text-muted-foreground">{statusLine(plan)}</span>
      </div>
      <p className="m-0 text-[13px] text-muted-foreground">{plan.starts_on} – {plan.ends_on} · {plan.timezone} · {summary.posts} posts</p>
      <p className="m-0 text-[13px] text-foreground">{cadenceLine(plan)}</p>
      <div className="flex flex-wrap items-center justify-between gap-2">
        <span className="text-[12.5px] text-muted-foreground">Content bank: {plan.bank.topics} topics, {plan.bank.unused} unused</span>
        <Button type="button" variant="outline" size="sm" onClick={onOpen}>Open plan</Button>
      </div>
    </article>
  )
}

function PlanWithAutoStart({ onStart }: { onStart: () => void }) {
  return (
    <section aria-label={AUTO_TITLE} className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-dashed border-border bg-card/60 p-4">
      <p className="m-0 max-w-[72ch] text-sm text-muted-foreground">{PLAN_WITH_AUTO_LEAD}</p>
      <Button type="button" onClick={onStart}>{AUTO_TITLE}</Button>
    </section>
  )
}

interface PlansViewProps {
  role: Workspace['role']
  posts: SocialPost[]
  planId: string | null
  go: GoTo
}

export function SocialsPlansView({ role, posts, planId, go }: PlansViewProps) {
  const { data } = useSocialPlans()
  const canEdit = canAuthorPosts(role)
  if (planId) {
    return (
      <SocialsPlanPage key={planId} planId={planId === NEW_PLAN ? null : planId} canEdit={canEdit}
        onSaved={(plan) => go({ view: 'plans', plan: plan.id, post: null })}
        canDelete={canDeletePlan(role)} onDeleted={() => go({ view: 'plans', plan: null, post: null })} />
    )
  }
  const plans = data?.plans ?? []
  return (
    <div className="flex flex-col gap-6">
      <section aria-label="Plans" className="flex flex-col gap-3">
        {canEdit && <PlanWithAutoStart onStart={() => go({ view: 'plans', plan: NEW_PLAN, post: null })} />}
        {plans.length === 0 && <p className="m-0 text-sm text-muted-foreground">{NO_PLANS}</p>}
        <div className="grid gap-3 lg:grid-cols-2">
          {plans.map((plan) => <PlanCard key={plan.id} plan={plan} onOpen={() => go({ view: 'plans', plan: plan.id, post: null })} />)}
        </div>
      </section>
      <SocialsCampaigns role={role} posts={posts} plansExcluded />
    </div>
  )
}
