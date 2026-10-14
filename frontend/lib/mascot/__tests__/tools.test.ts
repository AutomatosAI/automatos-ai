// @vitest-environment node
/**
 * PRD-260 D6 — which state a tool call shows, ported from Automatos Play's
 * tests/mascot/tools.test.ts: exact names first, then the patterns, the rest
 * working.
 */
import { describe, expect, it } from 'vitest'

import { READING_TOOLS, SEARCHING_TOOLS, toolActivity, WORKING_TOOLS } from '@/lib/mascot/tools'

describe('toolActivity', () => {
  it('matches every listed name exactly', () => {
    for (const name of Array.from(SEARCHING_TOOLS)) expect({ name, activity: toolActivity(name) }).toEqual({ name, activity: 'searching' })
    for (const name of Array.from(READING_TOOLS)) expect({ name, activity: toolActivity(name) }).toEqual({ name, activity: 'reading' })
    for (const name of Array.from(WORKING_TOOLS)) expect({ name, activity: toolActivity(name) }).toEqual({ name, activity: 'working' })
  })

  it("falls back to the patterns for tools the lists don't know", () => {
    expect(toolActivity('platform_web_screenshot')).toBe('searching')
    expect(toolActivity('platform_browse_marketplace_themes')).toBe('searching')
    expect(toolActivity('platform_summarise_document')).toBe('reading')
    expect(toolActivity('platform_graph_neighbors')).toBe('reading')
    expect(toolActivity('recall_memories')).toBe('reading')
  })

  it('calls everything else working, writes that match a reading pattern included', () => {
    expect(toolActivity('generate_document')).toBe('working')
    expect(toolActivity('platform_store_memory')).toBe('working')
    expect(toolActivity('platform_query_data')).toBe('working')
    expect(toolActivity('query_database')).toBe('working')
    expect(toolActivity('')).toBe('working')
  })
})
