/**
 * 2026-10-09: Opus 5.5, Sonnet 5.5 and Haiku 5.5 had no catalogue price, and the
 * model dropdown called them "free". They cost money on the owner's key.
 */
import { describe, expect, it } from 'vitest'

import { PRICE_UNKNOWN, priceNote } from '../model-price'

describe('priceNote', () => {
  it('says free only for a route that is free', () => {
    expect(priceNote({ is_free: true, price_known: true })).toBe(' · free')
  })

  it('says the price is unknown for an unpriced route, never free', () => {
    expect(priceNote({ is_free: false, price_known: false })).toBe(' · price unknown')
  })

  it('says nothing for a priced route, or a server that predates price_known', () => {
    expect(priceNote({ is_free: false, price_known: true })).toBe('')
    expect(priceNote({ is_free: false })).toBe('')
    expect(priceNote({})).toBe('')
  })

  it('names the unknown price the same way on the marketplace card', () => {
    expect(PRICE_UNKNOWN).toBe('Price unknown')
  })
})
