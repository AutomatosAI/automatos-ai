/**
 * PRD-255 Wave 1, US-007 — the locale on the Brand kit page: the currency as an ISO code
 * (empty: amounts print with no currency, FR-7) and the date style, each shown as a
 * document prints it. The country (7 Oct) gives an empty currency and date style.
 */
import { useState } from 'react'
import { describe, it, expect, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'

import type { BrandKit } from '@/components/documents/blocks/types'
import { BrandKitLocale, formatDateStyle } from '../brand-kit-locale'
import { designKit } from './brand-kit-design-kit'

let latest: BrandKit | null = null

function Harness({ start }: { start: BrandKit }) {
  const [kit, setKit] = useState(start)
  latest = kit
  return <BrandKitLocale kit={kit} patch={(p) => setKit((k) => ({ ...k, ...p }))} />
}

afterEach(() => { cleanup(); latest = null })

describe('the locale', () => {
  it('says amounts print with no currency until the kit has one, and stores the code upper case', () => {
    render(<Harness start={designKit()} />)
    expect(screen.getByText('Amounts print with no currency.')).toBeInTheDocument()
    fireEvent.change(screen.getByLabelText(/^Currency/), { target: { value: 'gbp' } })
    expect(latest?.currency).toBe('GBP')
    expect(screen.getByText('Amounts print in GBP.')).toBeInTheDocument()
  })

  it('offers each date style as a date printed in it, and stores the choice', () => {
    render(<Harness start={designKit()} />)
    const select = screen.getByLabelText('Date style')
    expect(screen.getByRole('option', { name: '6 October 2026' })).toBeInTheDocument()
    expect(screen.getByRole('option', { name: 'October 6, 2026' })).toBeInTheDocument()
    fireEvent.change(select, { target: { value: 'MMMM d, yyyy' } })
    expect(latest?.date_style).toBe('MMMM d, yyyy')
  })

  it('takes a country, whose currency and date style show until the kit sets its own', () => {
    render(<Harness start={designKit({ date_style: '' })} />)
    const country = screen.getByLabelText('Country')
    expect(screen.getByRole('option', { name: 'United Kingdom' })).toBeInTheDocument()
    expect(screen.getByRole('option', { name: '6 October 2026 (default)' })).toBeInTheDocument()
    fireEvent.change(country, { target: { value: 'US' } })
    expect(latest?.country).toBe('US')
    expect(screen.getByLabelText(/^Currency/)).toHaveAttribute('placeholder', 'USD')
    expect(screen.getByText("Amounts print in USD, the country's currency.")).toBeInTheDocument()
    expect(screen.getByRole('option', { name: "October 6, 2026 (the country's)" })).toBeInTheDocument()
    fireEvent.change(country, { target: { value: 'IE' } })
    expect(screen.getByLabelText(/^Currency/)).toHaveAttribute('placeholder', 'EUR')
    expect(screen.getByRole('option', { name: "6 October 2026 (the country's)" })).toBeInTheDocument()
    fireEvent.change(screen.getByLabelText(/^Currency/), { target: { value: 'gbp' } })
    expect(screen.getByText('Amounts print in GBP.')).toBeInTheDocument()  // a currency the kit sets wins
  })

  it('formats a date in either style', () => {
    const date = new Date(2026, 0, 2)
    expect(formatDateStyle(date, 'd MMMM yyyy')).toBe('2 January 2026')
    expect(formatDateStyle(date, 'MMMM d, yyyy')).toBe('January 2, 2026')
  })
})
