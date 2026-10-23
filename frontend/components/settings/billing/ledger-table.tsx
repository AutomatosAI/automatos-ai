'use client'

/**
 * PRD-265 US-011 — the ledger's rows and the last 30 days by model, as tables.
 *
 * A row is one credit movement: when, what ("AI usage", "Top-up", "Refund", …), the model for
 * a metered call, the credits (signed: a grant is +, a debit −) and the bucket it moved.
 */
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import type { LedgerItem, LedgerModelUsage } from '@/hooks/use-billing-ledger'
import { bucketLabel, formatCredits, formatMoment, signedCredits } from './billing-format'

export const NO_LEDGER_ROWS = 'No credit movements yet.'
export const NO_USAGE = 'No AI usage in the last 30 days.'
const NO_MODEL = '—'

export function LedgerTable({ items }: { items: readonly LedgerItem[] }) {
  if (items.length === 0) return <p className="text-sm text-muted-foreground">{NO_LEDGER_ROWS}</p>
  return (
    <div className="rounded-xl border border-border">
      <Table data-testid="billing-ledger-table">
        <TableHeader>
          <TableRow>
            <TableHead>Date</TableHead>
            <TableHead>What</TableHead>
            <TableHead>Model</TableHead>
            <TableHead className="text-right">Credits</TableHead>
            <TableHead>Bucket</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {items.map((item) => (
            <TableRow key={item.id}>
              <TableCell className="whitespace-nowrap text-muted-foreground">{formatMoment(item.at)}</TableCell>
              <TableCell className="text-foreground">{item.what}</TableCell>
              <TableCell className="text-muted-foreground">{item.model || NO_MODEL}</TableCell>
              <TableCell className={item.credits > 0 ? 'text-right tabular-nums text-success' : 'text-right tabular-nums text-foreground'}>
                {signedCredits(item.credits)}
              </TableCell>
              <TableCell className="text-muted-foreground">{bucketLabel(item.bucket)}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  )
}

export function UsageByModel({ usage }: { usage: readonly LedgerModelUsage[] }) {
  return (
    <section aria-labelledby="billing-usage-title" className="space-y-2">
      <h4 id="billing-usage-title" className="text-xs uppercase tracking-wide text-muted-foreground">
        Last 30 days, by model
      </h4>
      {usage.length === 0 ? (
        <p className="text-sm text-muted-foreground">{NO_USAGE}</p>
      ) : (
        <div className="rounded-xl border border-border">
          <Table data-testid="billing-usage-table">
            <TableHeader>
              <TableRow>
                <TableHead>Model</TableHead>
                <TableHead className="text-right">Calls</TableHead>
                <TableHead className="text-right">Credits</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {usage.map((row) => (
                <TableRow key={row.model ?? NO_MODEL}>
                  <TableCell className="text-foreground">{row.model || NO_MODEL}</TableCell>
                  <TableCell className="text-right tabular-nums text-muted-foreground">{formatCredits(row.calls)}</TableCell>
                  <TableCell className="text-right tabular-nums text-foreground">{formatCredits(row.credits)}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      )}
    </section>
  )
}
