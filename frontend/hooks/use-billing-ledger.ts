/**
 * PRD-265 US-008 / US-011 — the workspace's credit ledger, as its owner and admins see it.
 *
 * `GET /api/v1/billing/ledger?page=&page_size=` (orchestrator/api/billing.py): one page of
 * the ledger, newest first (a line per metered call, plus grants, top-ups, refunds and
 * expiries), and the last 30 days' credits by model. Owner/admin only: anyone else gets 403,
 * which the Billing tab reads as "no ledger section". Hosted edition only.
 *
 * 1 credit = 1 US cent of AI usage at Automatos prices; credits are whole numbers, signed
 * (a debit is negative).
 */
import { useQuery } from '@tanstack/react-query'
import { apiClient, type ApiHttpError } from '@/lib/api-client'
import { isSaaS } from '@/lib/auth-edition'

export const BILLING_LEDGER_PATH = '/api/v1/billing/ledger'
/** Rows per page in the Billing tab (the API allows up to 200). */
export const LEDGER_PAGE_SIZE = 25

const STALE_TIME_MS = 30_000
const FORBIDDEN = 403
/** One retry for a failed read, none for a refusal: a 403 will not change on a second ask. */
const MAX_RETRIES = 1

export interface LedgerItem {
  id: number | string
  at: string | null
  kind: string
  bucket: string | null
  /** Signed: a grant or top-up is positive, a debit negative. */
  credits: number
  /** What the movement was, in the ledger's own words ("AI usage", "Top-up", …). */
  what: string
  /** The model a metered call ran on; null for every other movement. */
  model: string | null
}

export interface LedgerModelUsage {
  model: string | null
  calls: number
  /** Credits spent on this model in the window (positive). */
  credits: number
}

export interface BillingLedger {
  items: LedgerItem[]
  total: number
  page: number
  page_size: number
  usage: { since: string; by_model: LedgerModelUsage[] }
}

export const billingLedgerQueryKeys = {
  all: ['billing', 'ledger'] as const,
  page: (page: number, pageSize: number) => ['billing', 'ledger', page, pageSize] as const,
}

/** The ledger page's address. */
export function billingLedgerPath(page: number, pageSize: number = LEDGER_PAGE_SIZE): string {
  const params = new URLSearchParams({ page: String(page), page_size: String(pageSize) })
  return `${BILLING_LEDGER_PATH}?${params.toString()}`
}

/** The API refused the caller (not an owner or admin). */
export function isForbidden(error: unknown): boolean {
  return (error as ApiHttpError | null | undefined)?.status === FORBIDDEN
}

/** One page of the ledger (1-based); the previous page stays on screen while the next loads. */
export function useBillingLedger(page: number, pageSize: number = LEDGER_PAGE_SIZE) {
  return useQuery<BillingLedger, ApiHttpError>({
    queryKey: billingLedgerQueryKeys.page(page, pageSize),
    queryFn: () => apiClient.request<BillingLedger>(billingLedgerPath(page, pageSize)),
    enabled: isSaaS,
    staleTime: STALE_TIME_MS,
    keepPreviousData: true,
    retry: (failures, error) => !isForbidden(error) && failures < MAX_RETRIES,
  })
}
