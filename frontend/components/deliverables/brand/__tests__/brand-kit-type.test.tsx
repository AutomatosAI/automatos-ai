/**
 * PRD-255 Wave 1, US-007 — the type scale on the Brand kit page: every step's size, line
 * height and weight, a live sample line per step in the kit's fonts, and a change to one
 * step that leaves the others as they were.
 */
import { useState } from 'react'
import { describe, it, expect, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent, within } from '@testing-library/react'

import type { BrandKit } from '@/components/documents/blocks/types'
import { BrandKitType, TYPE_STEP_LABELS } from '../brand-kit-type'
import { designKit } from './brand-kit-design-kit'

let latest: BrandKit | null = null

function Harness({ start }: { start: BrandKit }) {
  const [kit, setKit] = useState(start)
  latest = kit
  return <BrandKitType kit={kit} patch={(p) => setKit((k) => ({ ...k, ...p }))} />
}

afterEach(() => { cleanup(); latest = null })

describe('the type scale', () => {
  it('shows each of the seven steps with a sample line at its size, line height and weight', () => {
    render(<Harness start={designKit()} />)
    const section = screen.getByRole('region', { name: 'Type' })
    expect(Object.keys(TYPE_STEP_LABELS)).toHaveLength(7)
    const h1 = within(section).getByTestId('type-h1')
    expect(within(h1).getByLabelText('Size (pt)')).toHaveValue(22)
    expect(within(h1).getByLabelText('Line (pt)')).toHaveValue(28)
    expect(within(h1).getByLabelText('Weight')).toHaveValue(700)
    const sample = within(section).getByTestId('type-sample-h1')
    expect(sample.style.fontSize).toBe('22pt')
    expect(sample.style.lineHeight).toBe('28pt')
    expect(sample.style.fontWeight).toBe('700')
  })

  it('sets the headings\' samples in the heading font and the body\'s in the body font', () => {
    render(<Harness start={designKit()} />)
    expect(screen.getByTestId('type-sample-display').style.fontFamily).toContain('Brand Serif')
    expect(screen.getByTestId('type-sample-body').style.fontFamily).toContain('Inter')
  })

  it('changes one step, field by field, and the sample follows', () => {
    render(<Harness start={designKit()} />)
    const h2 = screen.getByTestId('type-h2')
    fireEvent.change(within(h2).getByLabelText('Size (pt)'), { target: { value: '18' } })
    expect(screen.getByTestId('type-sample-h2').style.fontSize).toBe('18pt')
    expect(latest?.type_scale?.h2).toEqual({ size_pt: 18, line_pt: 22, weight: 600 })
    expect(latest?.type_scale?.body).toEqual(designKit().type_scale?.body)
  })

  it('ignores an emptied box rather than storing nothing', () => {
    render(<Harness start={designKit()} />)
    fireEvent.change(within(screen.getByTestId('type-body')).getByLabelText('Size (pt)'), { target: { value: '' } })
    expect(latest?.type_scale?.body.size_pt).toBe(10.5)
  })

  it('shows nothing for a kit from a backend without a type scale', () => {
    render(<Harness start={designKit({ type_scale: undefined })} />)
    expect(screen.queryByRole('region', { name: 'Type' })).toBeNull()
  })
})
