'use client'

/**
 * PRD-255 US-007 — the locale on the Brand kit page: the currency amounts print in (an ISO
 * 4217 code; empty prints none, FR-7: no renderer adds a currency the kit doesn't have) and
 * the date style, each with a line saying what a document will show.
 */
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import type { BrandDateStyle, BrandKit } from '@/components/documents/blocks/types'
import { SELECT_CLASS, SectionTitle } from './brand-kit-inputs'

const CURRENCY_CODE_LENGTH = 3
const DAY_FIRST: BrandDateStyle = 'd MMMM yyyy'
const MONTH_FIRST: BrandDateStyle = 'MMMM d, yyyy'
// Every kit's default, as the server answers a kit without one.
const DEFAULT_DATE_STYLE = DAY_FIRST
// The date each style is shown with: 6 October 2026 (months count from 0).
const SAMPLE_DATE = new Date(2026, 9, 6)
const MONTH = new Intl.DateTimeFormat('en', { month: 'long' })

/** ``date`` in ``style``, as a document prints it. */
export function formatDateStyle(date: Date, style: BrandDateStyle): string {
  const month = MONTH.format(date)
  return style === MONTH_FIRST
    ? `${month} ${date.getDate()}, ${date.getFullYear()}`
    : `${date.getDate()} ${month} ${date.getFullYear()}`
}

export const DATE_STYLES: readonly BrandDateStyle[] = [DAY_FIRST, MONTH_FIRST]

interface BrandKitLocaleProps {
  kit: BrandKit
  patch: (p: Partial<BrandKit>) => void
}

export function BrandKitLocale({ kit, patch }: BrandKitLocaleProps) {
  if (kit.date_style === undefined) return null
  const currency = kit.currency ?? ''
  return (
    <section aria-label="Locale">
      <SectionTitle title="Locale">How amounts and dates print.</SectionTitle>
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
        <div>
          <Label htmlFor="brand-currency" className="text-xs">Currency (ISO code, such as GBP)</Label>
          <Input
            id="brand-currency" value={currency} maxLength={CURRENCY_CODE_LENGTH} placeholder="None" className="font-mono uppercase"
            onChange={(e) => patch({ currency: e.target.value.trim().toUpperCase() })}
          />
          <p className="mt-1 text-xs text-muted-foreground">
            {currency ? `Amounts print in ${currency}.` : 'Amounts print with no currency.'}
          </p>
        </div>
        <div>
          <Label htmlFor="brand-date-style" className="text-xs">Date style</Label>
          <select
            id="brand-date-style" className={SELECT_CLASS} value={kit.date_style ?? DEFAULT_DATE_STYLE}
            onChange={(e) => patch({ date_style: e.target.value as BrandDateStyle })}
          >
            {DATE_STYLES.map((style) => <option key={style} value={style}>{formatDateStyle(SAMPLE_DATE, style)}</option>)}
          </select>
        </div>
      </div>
    </section>
  )
}
