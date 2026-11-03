/**
 * PRD-270 US-270-05a (§5.5) — a listing a Sync with Automatos Cloud wrote (`source` 'cloud') carries a
 * secondary "From Automatos Cloud" badge beside Official; a local listing (`source` null or absent)
 * never does.
 */
import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import { ListingBadges } from '../listing-badges'
import { CLOUD_BADGE, OFFICIAL_BADGE } from '../listing-format'
import { listing } from './listing-fixtures'

const OFFICIAL = { name: 'Automatos', official: true }

describe('the From Automatos Cloud badge', () => {
  it('is the PRD’s words', () => {
    expect(CLOUD_BADGE).toBe('From Automatos Cloud')
  })

  it('shows on a Cloud listing, after Official', () => {
    const { container } = render(<ListingBadges listing={listing({ source: 'cloud', publisher: OFFICIAL })} />)

    expect(screen.getByText(CLOUD_BADGE)).toBeInTheDocument()
    const badges = Array.from(container.firstElementChild?.children ?? []).map((badge) => badge.textContent)
    expect(badges.slice(0, 2)).toEqual([OFFICIAL_BADGE, CLOUD_BADGE])
  })

  it('shows on a Cloud listing that is not Official', () => {
    render(<ListingBadges listing={listing({ source: 'cloud' })} />)

    expect(screen.getByText(CLOUD_BADGE)).toBeInTheDocument()
    expect(screen.queryByText(OFFICIAL_BADGE)).not.toBeInTheDocument()
  })

  it.each([
    ['null', { source: null }],
    ['absent', {}],
  ])('never shows on a local listing (source %s)', (_label, overrides) => {
    render(<ListingBadges listing={listing({ ...overrides, publisher: OFFICIAL })} />)

    expect(screen.getByText(OFFICIAL_BADGE)).toBeInTheDocument()
    expect(screen.queryByText(CLOUD_BADGE)).not.toBeInTheDocument()
  })
})
