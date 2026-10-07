'use client'

/**
 * PRD-255 US-007 — the locale on the Brand kit page: the currency amounts print in (an ISO
 * 4217 code; empty prints none, FR-7: no renderer adds a currency the kit doesn't have) and
 * the date style, each with a line saying what a document will show.
 *
 * The country (Gerard, 7 Oct): an empty currency or date style is the country's (GB: GBP and
 * 6 October 2026; US: USD and October 6, 2026), shown as the placeholder and the first date
 * option, so a kit with a country prints its amounts in its currency and Auto never asks.
 */
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import type { BrandDateStyle, BrandKit } from '@/components/documents/blocks/types'
import { SELECT_CLASS, SectionTitle } from './brand-kit-inputs'
import { COUNTRIES, countryLocale } from './country-locale'

const CURRENCY_CODE_LENGTH = 3
const DAY_FIRST: BrandDateStyle = 'd MMMM yyyy'
const MONTH_FIRST: BrandDateStyle = 'MMMM d, yyyy'
// Empty, every kit's default: the country's date style, else day first.
const FROM_COUNTRY: BrandDateStyle = ''
// The date each style is shown with: 6 October 2026 (months count from 0).
const SAMPLE_DATE = new Date(2026, 9, 6)
const MONTH = new Intl.DateTimeFormat('en', { month: 'long' })

/** ``date`` in ``style``, as a document prints it (empty: day first). */
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

function CountryField({ country, patch }: { country: string; patch: BrandKitLocaleProps['patch'] }) {
  return (
    <div>
      <Label htmlFor="brand-country" className="text-xs">Country</Label>
      <select
        id="brand-country" className={SELECT_CLASS} value={country}
        onChange={(e) => patch({ country: e.target.value })}
      >
        <option value="">None</option>
        {COUNTRIES.map(({ code, name }) => <option key={code} value={code}>{name}</option>)}
      </select>
      <p className="mt-1 text-xs text-muted-foreground">An empty currency or date style takes the country&apos;s.</p>
    </div>
  )
}

/** What the currency line says: the kit's own code, else the country's, else none. */
function currencyNote(currency: string, fromCountry: string | undefined): string {
  if (currency) return `Amounts print in ${currency}.`
  if (fromCountry) return `Amounts print in ${fromCountry}, the country's currency.`
  return 'Amounts print with no currency.'
}

export function BrandKitLocale({ kit, patch }: BrandKitLocaleProps) {
  if (kit.date_style === undefined) return null
  const country = kit.country ?? ''
  const fromCountry = countryLocale(country)
  const currency = kit.currency ?? ''
  const sample = formatDateStyle(SAMPLE_DATE, fromCountry?.dateStyle ?? DAY_FIRST)
  return (
    <section aria-label="Locale">
      <SectionTitle title="Locale">How amounts and dates print.</SectionTitle>
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
        <CountryField country={country} patch={patch} />
        <div>
          <Label htmlFor="brand-currency" className="text-xs">Currency (ISO code, such as GBP)</Label>
          <Input
            id="brand-currency" value={currency} maxLength={CURRENCY_CODE_LENGTH}
            placeholder={fromCountry?.currency ?? 'None'} className="font-mono uppercase"
            onChange={(e) => patch({ currency: e.target.value.trim().toUpperCase() })}
          />
          <p className="mt-1 text-xs text-muted-foreground">{currencyNote(currency, fromCountry?.currency)}</p>
        </div>
        <div>
          <Label htmlFor="brand-date-style" className="text-xs">Date style</Label>
          <select
            id="brand-date-style" className={SELECT_CLASS} value={kit.date_style ?? FROM_COUNTRY}
            onChange={(e) => patch({ date_style: e.target.value as BrandDateStyle })}
          >
            <option value={FROM_COUNTRY}>{fromCountry ? `${sample} (the country's)` : `${sample} (default)`}</option>
            {DATE_STYLES.map((style) => <option key={style} value={style}>{formatDateStyle(SAMPLE_DATE, style)}</option>)}
          </select>
        </div>
      </div>
    </section>
  )
}
