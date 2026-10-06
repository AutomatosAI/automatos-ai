/**
 * The kit's country and what it gives (Gerard, 7 Oct): an empty currency or date style is the
 * country's. Mirrors orchestrator/modules/documents/country_locale.py, which the server prints
 * with; the Locale card shows the country's values as placeholders before a save.
 */
import type { BrandDateStyle } from '@/components/documents/blocks/types'

const DAY: BrandDateStyle = 'd MMMM yyyy'
const MONTH: BrandDateStyle = 'MMMM d, yyyy'
// The euro area in 2026 (Bulgaria joined on 1 January 2026).
const EURO_AREA = [
  'AT', 'BE', 'BG', 'CY', 'DE', 'EE', 'ES', 'FI', 'FR', 'GR', 'HR',
  'IE', 'IT', 'LT', 'LU', 'LV', 'MT', 'NL', 'PT', 'SI', 'SK',
] as const

interface CountryLocale {
  currency: string
  dateStyle: BrandDateStyle
}

const OTHER_COUNTRIES: Record<string, CountryLocale> = {
  GB: { currency: 'GBP', dateStyle: DAY },
  US: { currency: 'USD', dateStyle: MONTH },
  CA: { currency: 'CAD', dateStyle: MONTH },
  AU: { currency: 'AUD', dateStyle: DAY },
  NZ: { currency: 'NZD', dateStyle: DAY },
  CZ: { currency: 'CZK', dateStyle: DAY },
  DK: { currency: 'DKK', dateStyle: DAY },
  HU: { currency: 'HUF', dateStyle: DAY },
  PL: { currency: 'PLN', dateStyle: DAY },
  RO: { currency: 'RON', dateStyle: DAY },
  SE: { currency: 'SEK', dateStyle: DAY },
  CH: { currency: 'CHF', dateStyle: DAY },
  NO: { currency: 'NOK', dateStyle: DAY },
  IS: { currency: 'ISK', dateStyle: DAY },
  AE: { currency: 'AED', dateStyle: DAY },
  BR: { currency: 'BRL', dateStyle: DAY },
  CN: { currency: 'CNY', dateStyle: DAY },
  HK: { currency: 'HKD', dateStyle: DAY },
  IN: { currency: 'INR', dateStyle: DAY },
  JP: { currency: 'JPY', dateStyle: DAY },
  KR: { currency: 'KRW', dateStyle: DAY },
  MX: { currency: 'MXN', dateStyle: DAY },
  SG: { currency: 'SGD', dateStyle: DAY },
  ZA: { currency: 'ZAR', dateStyle: DAY },
}

export const COUNTRY_LOCALES: Readonly<Record<string, CountryLocale>> = {
  ...Object.fromEntries(EURO_AREA.map((code) => [code, { currency: 'EUR', dateStyle: DAY }])),
  ...OTHER_COUNTRIES,
}

const REGION_NAMES = new Intl.DisplayNames(['en'], { type: 'region' })

/** A country's name in English ("United Kingdom" for GB); the code itself when there is none. */
export function countryName(code: string): string {
  return REGION_NAMES.of(code) ?? code
}

/** Every country the kit takes, as `{ code, name }`, by name. */
export const COUNTRIES = Object.keys(COUNTRY_LOCALES)
  .map((code) => ({ code, name: countryName(code) }))
  .sort((a, b) => a.name.localeCompare(b.name))

/** What the kit's country gives: its currency and date style; nothing without a country. */
export function countryLocale(country: string | undefined): CountryLocale | undefined {
  return country ? COUNTRY_LOCALES[country] : undefined
}
