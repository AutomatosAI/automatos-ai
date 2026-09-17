'use client'

/**
 * PRD-244 W1 — the Command Centre's period, kept from the legacy page's
 * selector: one value drives the stats strip, the Summary tab's read and the
 * Activity stream. A plain <select> styled as a cc-btn; the options are the
 * ones the activity API accepts.
 */
export const PERIOD_OPTIONS = [
  { value: '1d', label: '1 Day' },
  { value: '7d', label: '7 Days' },
  { value: '30d', label: '30 Days' },
  { value: '90d', label: '90 Days' },
] as const

export type Period = (typeof PERIOD_OPTIONS)[number]['value']

export function isPeriod(value: string): value is Period {
  return PERIOD_OPTIONS.some((o) => o.value === value)
}

export function PeriodSelect({ value, onChange }: { value: Period; onChange: (next: Period) => void }) {
  return (
    <select
      className="cc-btn cc-period"
      aria-label="Period"
      value={value}
      onChange={(e) => {
        const next = e.target.value
        if (isPeriod(next)) onChange(next)
      }}
    >
      {PERIOD_OPTIONS.map((o) => (
        <option key={o.value} value={o.value}>
          {o.label}
        </option>
      ))}
    </select>
  )
}
