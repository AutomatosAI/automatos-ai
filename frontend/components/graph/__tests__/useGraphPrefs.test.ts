import { describe, it, expect, beforeEach } from 'vitest'
import {
  readGraphPrefs,
  writeGraphPrefs,
  DEFAULT_GRAPH_PREFS,
} from '../useGraphPrefs'

// Pure-store coverage for the per-user+workspace graph prefs (PRD-165 S1,
// BINDING Q25). The React hook is a thin wrapper over these; the persistence
// contract lives here.

describe('graph prefs store', () => {
  beforeEach(() => {
    window.localStorage.clear()
  })

  it('defaults to a collapsed legend (Q25: chips collapsed by default)', () => {
    expect(DEFAULT_GRAPH_PREFS.legendCollapsed).toBe(true)
    expect(readGraphPrefs('ws-1', 'knowledge').legendCollapsed).toBe(true)
  })

  it('round-trips prefs for a workspace + surface', () => {
    writeGraphPrefs('ws-1', 'knowledge', {
      legendCollapsed: false,
      colorMode: 'type',
      hiddenTypes: ['shopify_variant'],
      hiddenRelations: ['variant_of'],
    })
    const read = readGraphPrefs('ws-1', 'knowledge')
    expect(read.legendCollapsed).toBe(false)
    expect(read.colorMode).toBe('type')
    expect(read.hiddenTypes).toEqual(['shopify_variant'])
    expect(read.hiddenRelations).toEqual(['variant_of'])
  })

  it('keeps prefs distinct per workspace and per surface', () => {
    writeGraphPrefs('ws-1', 'knowledge', { ...DEFAULT_GRAPH_PREFS, colorMode: 'type' })
    // A different workspace and a different surface both stay on defaults.
    expect(readGraphPrefs('ws-2', 'knowledge').colorMode).toBe('community')
    expect(readGraphPrefs('ws-1', 'codegraph').colorMode).toBe('community')
    expect(readGraphPrefs('ws-1', 'knowledge').colorMode).toBe('type')
  })

  it('survives a corrupt/partial saved blob by falling back to defaults', () => {
    window.localStorage.setItem('automatos:graph-prefs:ws-1:knowledge', '{ not json')
    expect(readGraphPrefs('ws-1', 'knowledge')).toEqual(DEFAULT_GRAPH_PREFS)

    window.localStorage.setItem(
      'automatos:graph-prefs:ws-1:knowledge',
      JSON.stringify({ colorMode: 'bogus', hiddenTypes: 'not-an-array' }),
    )
    const read = readGraphPrefs('ws-1', 'knowledge')
    expect(read.colorMode).toBe('community') // unknown value → default
    expect(read.hiddenTypes).toEqual([]) // wrong type → default
    expect(read.legendCollapsed).toBe(true) // missing → default
  })
})
