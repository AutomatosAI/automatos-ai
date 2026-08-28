/**
 * PRD-222 W2·S1b (US-024) — marketplace plan-label chip logic.
 */
import { describe, it, expect } from 'vitest'
import { itemRequiredDepth, planChipForItem } from '@/lib/marketplace-exposure'

describe('itemRequiredDepth', () => {
  it('reads an explicit numeric depth from metadata', () => {
    expect(itemRequiredDepth({ metadata: { marketplace_depth: 3 } })).toBe(3)
    expect(itemRequiredDepth({ metadata: { min_depth: 2 } })).toBe(2)
  })

  it('maps a named min_plan / tier to a depth', () => {
    expect(itemRequiredDepth({ metadata: { min_plan: 'pro' } })).toBe(2)
    expect(itemRequiredDepth({ metadata: { tier: 'business' } })).toBe(3)
  })

  it('defaults to 1 (all tiers) when unspecified', () => {
    expect(itemRequiredDepth({ metadata: {} })).toBe(1)
    expect(itemRequiredDepth({})).toBe(1)
  })
})

describe('planChipForItem', () => {
  const proItem = { metadata: { min_plan: 'pro' } }
  const bizItem = { metadata: { tier: 'business' } }

  it('chips items beyond the basic depth (1)', () => {
    expect(planChipForItem(proItem, 1)).toBe('Pro')
    expect(planChipForItem(bizItem, 1)).toBe('Business')
  })

  it('drops the chip once the tier can reach the item', () => {
    expect(planChipForItem(proItem, 2)).toBeNull() // pro reaches pro items
    expect(planChipForItem(bizItem, 2)).toBe('Business') // ...but not business
    expect(planChipForItem(bizItem, 3)).toBeNull() // business reaches all
  })

  it('never chips a depth-1 item (available to all)', () => {
    expect(planChipForItem({ metadata: {} }, 1)).toBeNull()
  })
})
