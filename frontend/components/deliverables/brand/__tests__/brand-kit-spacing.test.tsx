/**
 * PRD-255 Wave 1, US-007 — spacing and the logo's rules on the Brand kit page: the unit, the
 * page margin, the letterhead logo's height, clear space and least size, and a page drawn to
 * scale that follows them.
 */
import { useState } from 'react'
import { describe, it, expect, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'

import type { BrandKit } from '@/components/documents/blocks/types'
import { BrandKitSpacing } from '../brand-kit-spacing'
import { designKit } from './brand-kit-design-kit'

// The preview draws A4 (210 mm) 168 px wide.
const PX_PER_MM = 168 / 210
let latest: BrandKit | null = null

function Harness({ start, logoUrl = null }: { start: BrandKit; logoUrl?: string | null }) {
  const [kit, setKit] = useState(start)
  latest = kit
  return <BrandKitSpacing kit={kit} logoUrl={logoUrl} patch={(p) => setKit((k) => ({ ...k, ...p }))} />
}

afterEach(() => { cleanup(); latest = null })

describe('spacing and logo', () => {
  it('shows the unit, the margin and the logo rules', () => {
    render(<Harness start={designKit()} />)
    expect(screen.getByLabelText('Spacing unit (pt)')).toHaveValue(4)
    expect(screen.getByLabelText('Page margin (mm)')).toHaveValue(18)
    expect(screen.getByLabelText('Letterhead logo height (mm)')).toHaveValue(16)
    expect(screen.getByLabelText('Clear space (logo heights)')).toHaveValue(0.5)
    expect(screen.getByLabelText('Least logo size (mm)')).toHaveValue(8)
  })

  it('draws the margin and the logo\'s clear space to scale, and follows a change', () => {
    render(<Harness start={designKit()} />)
    expect(screen.getByTestId('preview-margin').style.top).toBe(`${18 * PX_PER_MM}px`)
    // Clear space: half the logo's 16 mm height.
    expect(screen.getByTestId('preview-clear-space').style.padding).toBe(`${0.5 * (16 * PX_PER_MM)}px`)

    fireEvent.change(screen.getByLabelText('Page margin (mm)'), { target: { value: '25' } })
    expect(screen.getByTestId('preview-margin').style.left).toBe(`${25 * PX_PER_MM}px`)
    expect(latest?.page_margin_mm).toBe(25)
  })

  it('changes one logo rule and keeps the others', () => {
    render(<Harness start={designKit()} />)
    fireEvent.change(screen.getByLabelText('Letterhead logo height (mm)'), { target: { value: '20' } })
    expect(latest?.logo_rules).toEqual({ letterhead_mm: 20, clear_space: 0.5, min_mm: 8 })
    expect(screen.getByTestId('preview-clear-space').style.padding).toBe(`${0.5 * (20 * PX_PER_MM)}px`)
  })

  it('shows the uploaded logo in the preview', () => {
    render(<Harness start={designKit()} logoUrl="blob:logo" />)
    expect(screen.getByAltText('Letterhead logo')).toHaveAttribute('src', 'blob:logo')
  })
})
