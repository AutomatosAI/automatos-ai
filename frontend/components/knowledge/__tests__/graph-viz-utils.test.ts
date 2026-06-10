import { describe, it, expect } from 'vitest'
import {
  idOf,
  colorForType,
  colorForCommunity,
  GRAPH_PALETTE,
  NEUTRAL_TYPE_COLOR,
  NEUTRAL_COMMUNITY_COLOR,
} from '../graph-viz-utils'

// ---------------------------------------------------------------------------
// idOf — null-guarded link-endpoint id extraction (the 4 crash sites in
// BusinessGraphVisualization all route through this).
// ---------------------------------------------------------------------------

describe('idOf', () => {
  it('returns the id of a resolved node object (post-simulation shape)', () => {
    expect(idOf({ id: 'node-1' })).toBe('node-1')
  })

  it('returns a bare string id unchanged (pre-simulation shape)', () => {
    expect(idOf('node-2')).toBe('node-2')
  })

  it('does NOT throw on null — the typeof null === "object" trap', () => {
    // Before the guard, `typeof null === "object"` is true so `(null).id`
    // threw "Cannot read properties of null" → the GraphErrorBoundary fallback.
    expect(() => idOf(null)).not.toThrow()
    expect(idOf(null)).toBeUndefined()
  })

  it('returns undefined for undefined and for an object without an id', () => {
    expect(idOf(undefined)).toBeUndefined()
    expect(idOf({})).toBeUndefined()
  })
})

// ---------------------------------------------------------------------------
// colorForType — deterministic hash over a DATA palette. Same type → same
// colour, every time (BINDING Q26: no hardcoded per-type hex).
// ---------------------------------------------------------------------------

describe('colorForType', () => {
  it('is deterministic — the same type always maps to the same colour', () => {
    expect(colorForType('shopify_product')).toBe(colorForType('shopify_product'))
    expect(colorForType('person')).toBe(colorForType('person'))
    // Arbitrary, never-seen types are handled too (not just hardcoded ones).
    expect(colorForType('some_unforeseen_entity_type')).toBe(
      colorForType('some_unforeseen_entity_type'),
    )
  })

  it('always returns a colour drawn from the shared DATA palette', () => {
    for (const t of ['shopify_product', 'concept', 'metric', 'rule', 'xyz']) {
      expect(GRAPH_PALETTE).toContain(colorForType(t))
    }
  })

  it('distinguishes at least some distinct types (not a single flat colour)', () => {
    const colours = new Set(
      ['concept', 'entity', 'process', 'metric', 'rule', 'person', 'org'].map(
        colorForType,
      ),
    )
    expect(colours.size).toBeGreaterThan(1)
  })

  it('falls back to the neutral colour for an absent type', () => {
    expect(colorForType(undefined)).toBe(NEUTRAL_TYPE_COLOR)
    expect(colorForType(null)).toBe(NEUTRAL_TYPE_COLOR)
    expect(colorForType('')).toBe(NEUTRAL_TYPE_COLOR)
  })
})

// ---------------------------------------------------------------------------
// colorForCommunity — stable colour per community id, neutral when unclustered.
// ---------------------------------------------------------------------------

describe('colorForCommunity', () => {
  it('is deterministic and palette-bound for a community id', () => {
    expect(colorForCommunity(3)).toBe(colorForCommunity(3))
    expect(GRAPH_PALETTE).toContain(colorForCommunity(3))
  })

  it('wraps around the palette by modulo (large ids never overflow)', () => {
    expect(colorForCommunity(GRAPH_PALETTE.length)).toBe(colorForCommunity(0))
  })

  it('returns the neutral colour for an unclustered node (null/undefined)', () => {
    expect(colorForCommunity(null)).toBe(NEUTRAL_COMMUNITY_COLOR)
    expect(colorForCommunity(undefined)).toBe(NEUTRAL_COMMUNITY_COLOR)
  })
})
