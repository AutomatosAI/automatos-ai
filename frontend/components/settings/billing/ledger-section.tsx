'use client'

/**
 * PRD-265 US-011 — the credit ledger on the Billing tab, for the workspace's owner and admins:
 * a paged table of credit movements, newest first, and the last 30 days' credits by model
 * (`GET /api/v1/billing/ledger`). The route refuses everyone else with 403, and then the
 * section is not shown at all; any other failure says so in one line.
 */
import { useState } from 'react'
import { ChevronLeft, ChevronRight } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { isForbidden, useBillingLedger, type BillingLedger } from '@/hooks/use-billing-ledger'
import { CREDIT_UNIT } from './billing-format'
import { LedgerTable, UsageByModel } from './ledger-table'

export const LEDGER_FAILED = 'The ledger could not be loaded. Try again in a moment.'
const FIRST_PAGE = 1

/** How many pages the ledger has (at least one). */
export function pageCount(ledger: Pick<BillingLedger, 'total' | 'page_size'>): number {
  return Math.max(FIRST_PAGE, Math.ceil(ledger.total / Math.max(1, ledger.page_size)))
}

interface LedgerPagerProps {
  readonly page: number
  readonly pages: number
  readonly loading: boolean
  readonly onPage: (page: number) => void
}

function LedgerPager({ page, pages, loading, onPage }: LedgerPagerProps) {
  if (pages <= FIRST_PAGE) return null
  return (
    <div className="flex items-center justify-between gap-2">
      <span className="text-xs text-muted-foreground">
        Page {page} of {pages}
      </span>
      <div className="flex gap-2">
        <Button type="button" size="sm" variant="outline" disabled={loading || page <= FIRST_PAGE} onClick={() => onPage(page - 1)}>
          <ChevronLeft className="mr-1 h-4 w-4" aria-hidden="true" />
          Previous
        </Button>
        <Button type="button" size="sm" variant="outline" disabled={loading || page >= pages} onClick={() => onPage(page + 1)}>
          Next
          <ChevronRight className="ml-1 h-4 w-4" aria-hidden="true" />
        </Button>
      </div>
    </div>
  )
}

function LedgerBody({ ledger, page, onPage }: { ledger: ReturnType<typeof useBillingLedger>; page: number; onPage: (page: number) => void }) {
  if (ledger.isLoading) return <p className="text-sm text-muted-foreground">Loading the ledger…</p>
  if (ledger.isError || !ledger.data) return <p role="alert" className="text-sm text-destructive">{LEDGER_FAILED}</p>
  return (
    <>
      <LedgerTable items={ledger.data.items} />
      <LedgerPager page={page} pages={pageCount(ledger.data)} loading={ledger.isFetching} onPage={onPage} />
      <UsageByModel usage={ledger.data.usage?.by_model ?? []} />
    </>
  )
}

export function LedgerSection() {
  const [page, setPage] = useState(FIRST_PAGE)
  const ledger = useBillingLedger(page)
  if (isForbidden(ledger.error)) return null

  return (
    <section aria-labelledby="billing-ledger-title" data-testid="billing-ledger" className="space-y-3">
      <div className="space-y-1">
        <h3 id="billing-ledger-title" className="text-xs uppercase tracking-wide text-muted-foreground">
          Credit ledger
        </h3>
        <p className="text-xs text-muted-foreground">{CREDIT_UNIT}</p>
      </div>
      <LedgerBody ledger={ledger} page={page} onPage={setPage} />
    </section>
  )
}
