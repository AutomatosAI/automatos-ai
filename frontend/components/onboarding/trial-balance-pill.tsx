'use client'

import { useWorkspace } from '@/components/workspace-provider'
import { isSaaS } from '@/lib/auth-edition'

/**
 * PRD-222 US-014 (W1·S9) — the trial balance pill.
 *
 * A small, always-honest balance chip ("$3.40 of $5.00 trial") reading the
 * trial ledger straight off the US-002 workspace snapshot
 * (`workspaces.onboarding.trial`) — never a second bookkeeping source. Amber
 * once the trial is warned or exhausted; hidden entirely once the workspace
 * converts (paid) or was never granted a trial. Mounted on the chat surface and
 * the Command Center.
 *
 * `className` lets each mount position the pill (absolute in chat, inline in the
 * Command Center action cluster) without the component owning layout.
 */
export function TrialBalancePill({ className = '' }: { className?: string }) {
  const { workspace } = useWorkspace()
  const trial = workspace?.onboarding?.trial ?? null

  // PRD-233 S7: plan/trial copy is a hosted-edition surface — never in local,
  // whatever a future local seed puts in onboarding.trial.
  if (!isSaaS) return null

  // No pill for a converted (paid) workspace or one that never got a trial.
  if (!trial || trial.state === 'converted') return null

  const remaining = Math.max(0, trial.granted_usd - trial.spent_usd)
  const warn = trial.state === 'warned' || trial.state === 'exhausted'

  return (
    <span
      data-testid="trial-balance-pill"
      data-state={trial.state}
      title="Trial credit remaining"
      className={[
        'inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-medium whitespace-nowrap',
        warn
          ? 'border-amber-500/40 bg-amber-500/10 text-amber-600 dark:text-amber-500'
          : 'border-border bg-muted/40 text-muted-foreground',
        className,
      ].join(' ')}
    >
      <span
        aria-hidden="true"
        className={`h-1.5 w-1.5 rounded-full ${warn ? 'bg-amber-500' : 'bg-emerald-500'}`}
      />
      ${remaining.toFixed(2)} of ${trial.granted_usd.toFixed(2)} trial
    </span>
  )
}
