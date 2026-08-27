'use client'

import { AlertTriangle } from 'lucide-react'
import { useWorkspace } from '@/components/workspace-provider'
import { TRIAL_EXHAUSTED_CODE } from '@/lib/trial'
import { PowerUpCard } from './power-up-card'

/**
 * PRD-222 US-014 (W1·S9) — the deterministic exhausted state.
 *
 * Renders with ZERO model/LLM dependency (a broke workspace can still take a
 * key): a static banner embedding the US-013 power-up card. Two triggers, both
 * deterministic:
 *   1. `workspace.onboarding.trial.state === 'exhausted'` — the US-002 snapshot,
 *      which the backend flips on the request that trips the gate; and
 *   2. `errorCode === 'trial_exhausted'` — the typed error a live request
 *      returns the instant it is blocked, before the snapshot refreshes.
 *
 * The only network call reachable from this surface is the credentials POST
 * inside PowerUpCard — the banner copy and layout never touch a model.
 */
export function TrialExhaustedBanner({ errorCode }: { errorCode?: string | null }) {
  const { workspace } = useWorkspace()
  const state = workspace?.onboarding?.trial?.state

  const exhausted = state === 'exhausted' || errorCode === TRIAL_EXHAUSTED_CODE
  if (!exhausted) return null

  return (
    <div
      data-testid="trial-exhausted-banner"
      role="status"
      className="bg-amber-500/5 border border-amber-500/30 rounded-xl p-4 space-y-3 max-w-md"
    >
      <div className="flex items-center gap-2">
        <AlertTriangle className="w-4 h-4 text-amber-600 dark:text-amber-500 shrink-0" />
        <span className="text-sm font-medium text-foreground">
          Your trial credit is used up
        </span>
      </div>
      <p className="text-sm text-foreground/80 leading-snug">
        Connect your own AI key to keep Auto running — everything you built stays
        exactly where it is.
      </p>
      {/* The credentials card is the way out; it works with no model call. */}
      <PowerUpCard embedded />
    </div>
  )
}
