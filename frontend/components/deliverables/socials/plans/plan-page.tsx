'use client'

/**
 * PRD-251B US-B207 — the Plan page (Plan.dc.html): the plan's name and state, its dates and
 * timezone, Delete (owners and admins, 3 Oct 2026), Pause (or Resume) and Save; the five
 * steps on the left (Goal and dates, Cadence, What to research, Making and approving, Content
 * bank) and the chosen one on the right, with Back and Next. A new plan is created on Save,
 * then opens as itself; the content bank fills once the plan exists. A new plan starts with
 * Plan with Auto: Auto drafts the steps from what the person says, and saving that draft also
 * adds Auto's ideas to the bank and starts research. PRD-251C (US-C404, US-C407): a saved plan
 * shows its health and Auto's proposals above the steps (plan-insights.tsx).
 */
import { useEffect, useState } from 'react'

import { Button } from '@/components/ui/button'
import type { SocialPlan, SocialPlanAutoDraft } from '@/lib/socials-plan-types'
import { browserTimezone } from '@/lib/social-time'
import { cn } from '@/lib/utils'
import { useSaveDraftedPlan } from '@/hooks/use-socials-plan-draft'
import { useDeleteSocialPlan, useSaveSocialPlan, useSetSocialPlanStatus, useSocialPlan } from '@/hooks/use-socials-plans'
import { DeleteAskedFirst } from '../studio/delete-asked-first'
import { PlanAutoNotes } from './plan-auto-notes'
import { PlanInsights } from './plan-insights'
import { autoChanges, researchAsked, topicInputs } from './plan-auto-model'
import { PLAN_STEPS, draftFromPlan, emptyDraft, inputFromDraft, missingFields, statusLine, type PlanDraft } from './plan-model'
import { PlanStepBank } from './plan-step-bank'
import { PlanStepCadence } from './plan-step-cadence'
import { PlanStepGoal } from './plan-step-goal'
import { PlanStepMaking } from './plan-step-making'
import { PlanStepResearch } from './plan-step-research'
import { PlanWithAuto } from './plan-with-auto'

// PRD-251C: the header's line follows the plan's rhythm.
const RHYTHM_LINES: Record<PlanDraft['rhythm'], string> = {
  daily: 'the plan sets the rhythm and the topics; each post is made on its day.',
  weekly: 'the plan sets the rhythm and the topics; each week is made at once and approved in one sitting.',
  monthly: 'the plan sets the rhythm and the topics; each month is made at once and approved in one sitting.',
}

export const PLAN_DELETE_CONFIRM =
  'Delete this plan? Its content bank goes with it, and it cannot be undone. The posts it made stay, as ordinary posts.'
const STEP_BUTTON = 'flex min-h-[44px] items-center gap-2.5 rounded-lg px-3 text-left text-sm font-medium'

interface PlanPageProps {
  /** A plan's id, or null for a new plan. */
  planId: string | null
  canEdit: boolean
  onSaved: (plan: SocialPlan) => void
  /** An owner or admin: the plan's page offers Delete. */
  canDelete?: boolean
  onDeleted?: () => void
}

function DeletePlan({ planId, onDeleted }: { planId: string; onDeleted?: () => void }) {
  const remove = useDeleteSocialPlan()
  return (
    <DeleteAskedFirst noun="plan" question={PLAN_DELETE_CONFIRM} busy={remove.isLoading}
      onConfirm={() => remove.mutate({ planId }, { onSuccess: () => onDeleted?.() })} />
  )
}

interface StepBodyProps {
  step: number
  draft: PlanDraft
  set: (c: Partial<PlanDraft>) => void
  planId: string | null
  auto: AutoState
  /** PRD-251C: the saved plan's next batch, for the Making step. */
  nextBatchAt?: string | null
}

function StepBody({ step, draft, set, planId, auto, nextBatchAt }: StepBodyProps) {
  if (step === 0) return <PlanStepGoal draft={draft} set={set} />
  if (step === 1) return <PlanStepCadence draft={draft} set={set} />
  if (step === 2) return <PlanStepResearch draft={draft} set={set} />
  if (step === 3) return <PlanStepMaking draft={draft} set={set} nextBatchAt={nextBatchAt} />
  return <PlanStepBank planId={planId} ideas={auto.draft?.topics} onLeaveOut={auto.leaveOut} />
}

function StepsNav({ step, onStep }: { step: number; onStep: (step: number) => void }) {
  return (
    <nav aria-label="Plan steps" className="flex flex-col gap-1 rounded-xl border border-border bg-card p-2.5">
      {PLAN_STEPS.map((label, index) => (
        <button key={label} type="button" aria-label={label} aria-pressed={index === step} onClick={() => onStep(index)}
          className={cn(STEP_BUTTON, index === step ? 'bg-secondary text-foreground' : 'text-muted-foreground')}>
          <span className="text-[12px] text-muted-foreground">{index + 1}</span>
          {label}
        </button>
      ))}
    </nav>
  )
}

function usePlanDraft(plan: SocialPlan | undefined): [PlanDraft, (changes: Partial<PlanDraft>) => void] {
  const [draft, setDraft] = useState<PlanDraft>(() => (plan ? draftFromPlan(plan) : emptyDraft(browserTimezone())))
  const loadedId = plan?.id
  useEffect(() => {
    if (plan) setDraft(draftFromPlan(plan))
    // Reload the form when another plan (or the saved one) arrives, never while typing.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [loadedId])
  return [draft, (changes) => setDraft((current) => ({ ...current, ...changes }))]
}

interface AutoState {
  /** Auto's last draft of this new plan, with the ideas the person has not left out. */
  draft: SocialPlanAutoDraft | null
  adopt: (draft: SocialPlanAutoDraft) => void
  leaveOut: (title: string) => void
}

function useAutoDraft(set: (changes: Partial<PlanDraft>) => void, onAdopted: () => void): AutoState {
  const [draft, setDraft] = useState<SocialPlanAutoDraft | null>(null)
  return {
    draft,
    adopt: (drafted) => {
      set(autoChanges(drafted))
      setDraft(drafted)
      onAdopted()
    },
    leaveOut: (title) => setDraft((current) => (current ? { ...current, topics: current.topics.filter((t) => t.title !== title) } : current)),
  }
}

/** Save the plan: a new one drafted by Auto also takes its ideas into the bank and starts research. */
function usePlanSave(planId: string | null, draft: PlanDraft, auto: SocialPlanAutoDraft | null, onSaved: (plan: SocialPlan) => void) {
  const save = useSaveSocialPlan()
  const adopt = useSaveDraftedPlan()
  const run = () => {
    const input = inputFromDraft(draft)
    if (planId || !auto) {
      save.mutate({ planId, input }, { onSuccess: onSaved })
      return
    }
    adopt.mutate({ input, topics: topicInputs(auto.topics), research: researchAsked(draft.sources) }, { onSuccess: ({ plan }) => onSaved(plan) })
  }
  return { run, busy: save.isLoading || adopt.isLoading }
}

export function SocialsPlanPage({ planId, canEdit, onSaved, canDelete = false, onDeleted }: PlanPageProps) {
  const { data: plan } = useSocialPlan(planId)
  const [draft, set] = usePlanDraft(plan)
  const [step, setStep] = useState(planId ? 1 : 0)
  const auto = useAutoDraft(set, () => setStep(0))
  const save = usePlanSave(planId, draft, auto.draft, onSaved)
  const status = useSetSocialPlanStatus()
  const missing = missingFields(draft)
  const last = step === PLAN_STEPS.length - 1
  const saveButton = (
    <Button type="button" disabled={!canEdit || save.busy || missing.length > 0 || plan?.status === 'ended'} onClick={save.run}>
      Save plan
    </Button>
  )
  return (
    <div className="socials-plan flex flex-col gap-5">
      <header className="flex flex-wrap items-end justify-between gap-4">
        <div className="flex flex-col gap-1.5">
          <div className="flex flex-wrap items-center gap-3">
            <h1 className="m-0 text-[26px] font-semibold text-foreground">{draft.name || 'New plan'}</h1>
            {plan && <span className="rounded-full border border-border px-2.5 py-0.5 text-[12px] text-muted-foreground">{statusLine(plan)}</span>}
          </div>
          <p className="m-0 text-sm text-muted-foreground">
            {draft.startsOn} – {draft.endsOn} · {draft.timezone} · {RHYTHM_LINES[draft.rhythm]}
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {plan && canDelete && <DeletePlan planId={plan.id} onDeleted={onDeleted} />}
          {plan && plan.status !== 'ended' && canEdit && (
            <Button type="button" variant="outline" disabled={status.isLoading}
              onClick={() => status.mutate({ planId: plan.id, action: plan.status === 'paused' ? 'resume' : 'pause' })}>
              {plan.status === 'paused' ? 'Resume plan' : 'Pause plan'}
            </Button>
          )}
          {saveButton}
        </div>
      </header>
      {!planId && <PlanWithAuto canEdit={canEdit} drafted={!!auto.draft} onDrafted={auto.adopt} />}
      {auto.draft && <PlanAutoNotes warnings={auto.draft.warnings} />}
      {missing.length > 0 && <p className="m-0 text-[12.5px] text-muted-foreground">Before saving, the plan needs {missing.join(', ')}.</p>}
      {plan && <PlanInsights planId={plan.id} canEdit={canEdit} onCadence={() => setStep(1)} onApplied={(saved) => set(draftFromPlan(saved))} />}
      <div className="grid items-start gap-5 lg:grid-cols-[260px_minmax(0,1fr)]">
        <StepsNav step={step} onStep={setStep} />
        <section aria-label={PLAN_STEPS[step]} className="flex min-h-[520px] flex-col gap-5 rounded-xl border border-border bg-background p-5">
          <StepBody step={step} draft={draft} set={set} planId={planId} auto={auto} nextBatchAt={plan?.next_batch_at} />
          <footer className="mt-auto flex gap-2 border-t border-border pt-3.5">
            <Button type="button" variant="ghost" disabled={step === 0} onClick={() => setStep(step - 1)}>Back</Button>
            {last ? saveButton : <Button type="button" variant="outline" onClick={() => setStep(step + 1)}>Next</Button>}
          </footer>
        </section>
      </div>
    </div>
  )
}
