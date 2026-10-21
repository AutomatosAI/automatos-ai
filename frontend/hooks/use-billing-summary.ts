/**
 * PRD-265 US-008 — the workspace's plan and credits, as its members see them.
 *
 * One endpoint serves the header balance, the paused banner and the theme SDK's
 * trial (`GET /api/v1/billing/summary`, orchestrator/api/billing.py). Every
 * reader goes through this hook, so they share one cached response and cannot
 * disagree. Hosted edition only: the local edition never meters (the route
 * answers 404 there), so the query never runs locally.
 *
 * 1 credit = 1 US cent of AI usage at Automatos prices.
 */
import { useEffect } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { apiClient } from '@/lib/api-client'
import { isSaaS } from '@/lib/auth-edition'

export const BILLING_SUMMARY_PATH = '/api/v1/billing/summary'

/** The summary is asked again this often, and counts as fresh for half that. */
const REFETCH_INTERVAL_MS = 60_000
const STALE_TIME_MS = 30_000

/**
 * A reply landed in a chat that the open page did not stream itself: a
 * background post, or a turn that finished after the browser left
 * (hooks/use-board-event-stream.ts). Either may have spent credits.
 */
const CHAT_CHANGED_EVENT = 'automatos:chat-changed'

/** Credits by bucket (spent in this order: trial, monthly, top-up). */
export interface BillingBalance {
  trial: number
  monthly: number
  topup: number
  /** What the workspace's plan can spend now; null when it is not metered. */
  spendable: number | null
}

export interface BillingSummary {
  /** The plan's key (`basic` is the Own keys plan). */
  plan: string | null
  plan_name: string | null
  price_usd: number | null
  /** "beta" while paid plans are not open; null once they are. */
  price_label: string | null
  /** Paid plans and top-ups can be bought (the `BILLING_OPEN` switch). */
  billing_open: boolean
  status: 'active' | 'paused'
  renewal_date: string | null
  balance: BillingBalance
  /** The one-time trial grant, in credits. */
  trial_credits: number
  /** The plan may run on Automatos's AI keys once its trial is spent (false on Own keys). */
  platform_ai: boolean
  /** Credits pay for platform-funded work; false for an allowance plan (enterprise). */
  metered: boolean
  /** Platform-funded work is paused: no credits left. Own-key calls still run. */
  paused: boolean
  /** The balance is running low (and not yet paused). */
  low: boolean
  /** The workspace has switched on AI keys of its own. */
  own_keys: boolean
}

export const billingSummaryQueryKeys = {
  all: ['billing', 'summary'] as const,
}

/**
 * Platform-funded work is paused and the workspace has no keys of its own to
 * carry on with: what the paused banner and the balance pill's "Paused" stand
 * for. A workspace on its own keys keeps working at zero credits
 * (orchestrator/services/credits/routing.py), so nothing it does is paused.
 */
export function isWorkPaused(summary: BillingSummary | null | undefined): boolean {
  return summary?.paused === true && summary.own_keys !== true
}

/**
 * The caller's workspace plan and credits. Polled every minute, refreshed when
 * the window regains focus, and at once when a reply lands in a chat from the
 * background (`automatos:chat-changed`).
 */
export function useBillingSummary() {
  const queryClient = useQueryClient()

  const query = useQuery<BillingSummary>({
    queryKey: billingSummaryQueryKeys.all,
    queryFn: () => apiClient.request<BillingSummary>(BILLING_SUMMARY_PATH),
    enabled: isSaaS,
    refetchInterval: REFETCH_INTERVAL_MS,
    staleTime: STALE_TIME_MS,
    refetchOnWindowFocus: true,
  })

  useEffect(() => {
    if (typeof window === 'undefined' || !isSaaS) return
    // Several surfaces mount this hook; one refetch in flight is enough.
    const onChatChanged = () => {
      void queryClient.invalidateQueries({ queryKey: billingSummaryQueryKeys.all }, { cancelRefetch: false })
    }
    window.addEventListener(CHAT_CHANGED_EVENT, onChatChanged)
    return () => window.removeEventListener(CHAT_CHANGED_EVENT, onChatChanged)
  }, [queryClient])

  return query
}
