'use client'

/**
 * PRD-265 W2 — the workspace's invoices in the Billing tab: number, date, amount, status and
 * "Download PDF". The PDF comes from our API through apiClient and is saved from the page
 * (hooks/use-billing-payment.ts), so the browser never visits Stripe for it.
 */
import { useState } from 'react'
import { Download, Loader2 } from 'lucide-react'
import { toast } from 'sonner'
import { Button } from '@/components/ui/button'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { billingErrorMessage } from '@/lib/billing-errors'
import { downloadInvoicePdf, INVOICE_DOWNLOAD_FAILED, useInvoices, type Invoice } from '@/hooks/use-billing-payment'
import { formatDay, formatMoney } from './billing-format'

export const NO_INVOICES = 'No invoices yet.'
export const INVOICES_FAILED = "The invoices couldn't be loaded. Try again in a moment."

const STATUS_LABELS: Readonly<Record<string, string>> = {
  paid: 'Paid',
  open: 'Due',
  draft: 'Draft',
  void: 'Void',
  uncollectible: 'Unpaid',
}

export function invoiceStatusLabel(status: string): string {
  return STATUS_LABELS[status] ?? status
}

function InvoiceRow({ invoice, busy, onDownload }: { invoice: Invoice; busy: boolean; onDownload: (invoice: Invoice) => void }) {
  const label = invoice.number || invoice.id
  return (
    <TableRow>
      <TableCell className="text-foreground">{label}</TableCell>
      <TableCell className="whitespace-nowrap text-muted-foreground">{formatDay(invoice.created)}</TableCell>
      <TableCell className="text-right tabular-nums text-foreground">{formatMoney(invoice.total_cents, invoice.currency)}</TableCell>
      <TableCell className="text-muted-foreground">{invoiceStatusLabel(invoice.status)}</TableCell>
      <TableCell className="text-right">
        <Button
          type="button"
          size="sm"
          variant="ghost"
          aria-label={`Download PDF of invoice ${label}`}
          disabled={busy}
          onClick={() => onDownload(invoice)}
        >
          {busy ? <Loader2 className="mr-1 h-4 w-4 animate-spin" aria-hidden="true" /> : <Download className="mr-1 h-4 w-4" aria-hidden="true" />}
          Download PDF
        </Button>
      </TableCell>
    </TableRow>
  )
}

function InvoiceList({ invoices, downloading, onDownload }: { invoices: Invoice[]; downloading: string | null; onDownload: (invoice: Invoice) => void }) {
  if (invoices.length === 0) return <p className="text-sm text-muted-foreground">{NO_INVOICES}</p>
  return (
    <div className="rounded-xl border border-border">
      <Table data-testid="billing-invoices-table">
        <TableHeader>
          <TableRow>
            <TableHead>Invoice</TableHead>
            <TableHead>Date</TableHead>
            <TableHead className="text-right">Amount</TableHead>
            <TableHead>Status</TableHead>
            <TableHead className="text-right">PDF</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {invoices.map((invoice) => (
            <InvoiceRow key={invoice.id} invoice={invoice} busy={downloading === invoice.id} onDownload={onDownload} />
          ))}
        </TableBody>
      </Table>
    </div>
  )
}

export function InvoicesTable() {
  const invoices = useInvoices(true)
  const [downloading, setDownloading] = useState<string | null>(null)

  const download = async (invoice: Invoice) => {
    setDownloading(invoice.id)
    try {
      await downloadInvoicePdf(invoice)
    } catch (error) {
      toast.error(billingErrorMessage(error, INVOICE_DOWNLOAD_FAILED))
    } finally {
      setDownloading(null)
    }
  }

  return (
    <section aria-labelledby="billing-invoices-title" className="space-y-2">
      <h4 id="billing-invoices-title" className="text-xs uppercase tracking-wide text-muted-foreground">
        Invoices
      </h4>
      {invoices.isLoading && <p className="text-sm text-muted-foreground">Loading invoices…</p>}
      {invoices.isError && <p role="alert" className="text-sm text-destructive">{INVOICES_FAILED}</p>}
      {invoices.data && <InvoiceList invoices={invoices.data} downloading={downloading} onDownload={(invoice) => void download(invoice)} />}
    </section>
  )
}
