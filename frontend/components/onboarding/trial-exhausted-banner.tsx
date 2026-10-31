'use client'

import Link from 'next/link'
import { AlertTriangle } from 'lucide-react'
import { isWorkPaused, useBillingSummary } from '@/hooks/use-billing-summary'
import { CREDITS_EXHAUSTED_CODE } from '@/lib/credits'
import { PowerUpCard } from './power-up-card'

/** Where "Top up" goes: Settings → Billing. */
export const TOP_UP_HREF = '/settings?tab=billing'

const PAUSED_TITLE = 'Your credits are used up'
const PAUSED_BODY =
  "Work on Automatos's AI keys is paused. Add your own AI key to keep going — everything you built stays where it is."
const OWN_KEYS_PLAN_BODY =
  'On the Own keys plan the AI runs on your own keys. Add your Anthropic, OpenAI or OpenRouter key to keep going.'

/**
 * PRD-265 US-008 — the paused state: the workspace's credits cannot pay for one
 * more ask on Automatos's AI keys.
 *
 * The name is kept because `TrialExhaustedBanner` is the theme SDK's public card
 * name (`cards.TrialExhaustedBanner`, lib/theme-sdk/host-cards/chrome.tsx); since
 * PRD-265 the trial is credits in the ledger and the banner is about credits.
 *
 * Renders on either trigger:
 *   1. the billing summary says work is paused and the workspace has no keys of
 *      its own (`isWorkPaused`; a workspace on its own keys keeps working); or
 *   2. `errorCode === 'credits_exhausted'`, the typed code a refused request
 *      returns the instant it is blocked, before the summary catches up.
 *
 * The way out is the embedded power-up card (add a key), which needs no model
 * call. "Top up" shows only while paid plans are open (`billing_open`): before
 * then there is nothing to buy. The banner itself makes no model call; the only
 * requests reachable from it are the summary GET and the power-up card's
 * credentials POST.
 */
export function TrialExhaustedBanner({ errorCode }: { errorCode?: string | null }) {
  const { data: summary } = useBillingSummary()

  const paused = isWorkPaused(summary) || errorCode === CREDITS_EXHAUSTED_CODE
  if (!paused) return null

  const ownKeysPlan = summary?.platform_ai === false

  return (
    <div
      data-testid="trial-exhausted-banner"
      role="status"
      className="bg-warning/5 border border-warning/30 rounded-xl p-4 space-y-3 max-w-md"
    >
      <div className="flex items-center gap-2">
        <AlertTriangle className="w-4 h-4 text-warning shrink-0" />
        <span className="text-sm font-medium text-foreground">{PAUSED_TITLE}</span>
      </div>
      <p className="text-sm text-foreground/80 leading-snug">
        {ownKeysPlan ? OWN_KEYS_PLAN_BODY : PAUSED_BODY}
      </p>
      {summary?.billing_open && (
        <Link
          href={TOP_UP_HREF}
          data-testid="trial-exhausted-top-up"
          className="inline-flex items-center rounded-md border border-warning/40 px-3 py-1.5 text-xs font-medium text-foreground hover:bg-warning/10 transition-colors"
        >
          Top up
        </Link>
      )}
      {/* The credentials card is the way out; it works with no model call. */}
      <PowerUpCard embedded />
    </div>
  )
}
