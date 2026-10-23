/**
 * PRD-265 US-011 — the plans a workspace can be on, and what its caller may do about them.
 *
 * `GET /api/v1/billing/plans` (orchestrator/api/billing.py), any member: the plan catalogue
 * with display names and prices, the workspace's Stripe subscription, the top-up offer, and
 * three switches the Billing tab obeys: `billing_open` (the public copy), `checkout_available`
 * (Stripe is configured and the caller may buy now) and `can_manage` (owner or admin).
 * Hosted edition only: the route answers 404 in the local edition, so the query never runs there.
 */
import { useQuery } from '@tanstack/react-query'
import { apiClient } from '@/lib/api-client'
import { isSaaS } from '@/lib/auth-edition'

export const BILLING_PLANS_PATH = '/api/v1/billing/plans'

/** The catalogue changes rarely; a checkout return asks again on its own schedule. */
const STALE_TIME_MS = 30_000

export type SubscriptionStatus = 'active' | 'trialing' | 'past_due' | 'unpaid' | 'canceled' | 'incomplete'

/** The statuses in which Stripe is still billing the workspace for its plan. */
const LIVE_STATUSES: ReadonlySet<string> = new Set(['active', 'trialing', 'past_due', 'unpaid'])

export interface BillingSubscription {
  /** null when the workspace has never subscribed. */
  status: SubscriptionStatus | null
  cancel_at_period_end: boolean
  current_period_end: string | null
  /** The workspace has a Stripe customer, so the portal (invoices, card, cancel) can open. */
  has_billing_account: boolean
}

export interface BillingPlan {
  /** `basic` is the Own keys plan. */
  key: string
  /** The display name ("Own keys", "Pro", …): the only name the UI shows. */
  name: string
  price_usd: number | null
  monthly_credits: number | null
  platform_ai: boolean
  /** Sold through Checkout by its owner; Business is by invite and Enterprise is not sold here. */
  self_serve: boolean
  coming_soon: boolean
  seats?: number | null
  /** 0 means unlimited. */
  max_agents?: number | null
  render_minutes_month?: number | null
  voice?: boolean
}

export interface BillingTopUpOffer {
  credits: number
  price_usd: number
  valid_days: number
  /** The workspace has an active subscription on a plan with platform AI (Pro, Business). */
  available: boolean
}

export interface BillingPlans {
  billing_open: boolean
  checkout_available: boolean
  can_manage: boolean
  current_plan: string | null
  subscription: BillingSubscription
  plans: BillingPlan[]
  topup: BillingTopUpOffer
}

export const billingPlansQueryKeys = {
  all: ['billing', 'plans'] as const,
}

/** Stripe is still billing this workspace (including a renewal it is retrying). */
export function hasLiveSubscription(subscription: BillingSubscription | null | undefined): boolean {
  return typeof subscription?.status === 'string' && LIVE_STATUSES.has(subscription.status)
}

/** The plans, the workspace's subscription and what the caller may buy. */
export function useBillingPlans() {
  return useQuery<BillingPlans>({
    queryKey: billingPlansQueryKeys.all,
    queryFn: () => apiClient.request<BillingPlans>(BILLING_PLANS_PATH),
    enabled: isSaaS,
    staleTime: STALE_TIME_MS,
    refetchOnWindowFocus: true,
  })
}
