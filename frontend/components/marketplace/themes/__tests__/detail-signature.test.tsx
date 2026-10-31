/**
 * PRD-261 C-05k — the listing detail's Signature fact (listings.py `signature`, C-05b) and the
 * Official badge: "Signed by Automatos" only for a `verified` signature, and Official only from
 * `publisher.official` (QB-21), never from the signature alone.
 */
import { render, screen, within } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import { SIGNATURE_FACT, SIGNATURE_WORDS, SIGNED_BY_AUTOMATOS } from '@/components/settings/themes/theme-labels'
import type { ThemeSignatureVerdict } from '@/lib/api-client'

import { DetailFacts } from '../detail-sections'
import { ListingBadges } from '../listing-badges'
import { OFFICIAL_BADGE } from '../listing-format'
import { UNSIGNED, listing, listingDetail } from './listing-fixtures'

const VERIFIED: ThemeSignatureVerdict = { state: 'verified', keyId: 'ed25519:0123456789abcdef' }
const INVALID: ThemeSignatureVerdict = { state: 'invalid', keyId: 'ed25519:fedcba9876543210' }

function facts() {
  return screen.getByLabelText('What this theme changes')
}

describe('the Signature fact', () => {
  it('a verified signature reads "Signed by Automatos"', () => {
    render(<DetailFacts detail={listingDetail({ signature: VERIFIED })} />)
    expect(within(facts()).getByText(SIGNATURE_FACT)).toBeInTheDocument()
    expect(within(facts()).getByText(SIGNED_BY_AUTOMATOS)).toBeInTheDocument()
  })

  it.each([
    ['invalid', INVALID, SIGNATURE_WORDS.invalid],
    ['unsigned', UNSIGNED, SIGNATURE_WORDS.unsigned],
  ])('an %s signature says so, never "Signed by Automatos"', (_case, signature, words) => {
    render(<DetailFacts detail={listingDetail({ signature })} />)
    expect(within(facts()).getByText(words)).toBeInTheDocument()
    expect(within(facts()).queryByText(SIGNED_BY_AUTOMATOS)).toBeNull()
  })
})

describe('the Official badge comes only from publisher.official', () => {
  it('a verified signature from a publisher that is not official shows no Official badge', () => {
    render(<ListingBadges listing={listing({ signature: VERIFIED, publisher: { name: 'acme', official: false } })} />)
    expect(screen.queryByText(OFFICIAL_BADGE)).toBeNull()
  })

  it('an official publisher shows the badge, whatever this card says of the signature', () => {
    render(<ListingBadges listing={listing({ signature: UNSIGNED, publisher: { name: 'automatos', official: true } })} />)
    expect(screen.getByText(OFFICIAL_BADGE)).toBeInTheDocument()
  })
})
