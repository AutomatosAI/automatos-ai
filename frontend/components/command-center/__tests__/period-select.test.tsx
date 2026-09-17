import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'

import { PeriodSelect, PERIOD_OPTIONS, isPeriod } from '../period-select'

describe('PeriodSelect (PRD-244 W1)', () => {
  it('offers the four periods the activity API accepts and reports a change', () => {
    const onChange = vi.fn()
    render(<PeriodSelect value="1d" onChange={onChange} />)
    const select = screen.getByLabelText('Period') as HTMLSelectElement
    expect(Array.from(select.options).map((o) => o.value)).toEqual(['1d', '7d', '30d', '90d'])
    expect(PERIOD_OPTIONS.map((o) => o.label)).toEqual(['1 Day', '7 Days', '30 Days', '90 Days'])
    fireEvent.change(select, { target: { value: '30d' } })
    expect(onChange).toHaveBeenCalledWith('30d')
  })

  it('accepts only known periods', () => {
    expect(isPeriod('7d')).toBe(true)
    expect(isPeriod('2w')).toBe(false)
  })
})
