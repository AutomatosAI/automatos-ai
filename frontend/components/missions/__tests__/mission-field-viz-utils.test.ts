import { describe, it, expect } from 'vitest'
import {
  patternsToFieldGraph,
  colorForAgent,
  dimColor,
  FIELD_CORE_ID,
  MAX_PATTERN_NODES,
} from '../mission-field-viz-utils'
import type { FieldPattern } from '@/hooks/use-missions-api'

function pat(over: Partial<FieldPattern>): FieldPattern {
  return {
    id: 'p1',
    key: 'finding',
    value: 'some content',
    agent_id: 1,
    strength: 1,
    decayed_strength: 0.5,
    access_count: 0,
    created_at: '2026-06-01T00:00:00Z',
    last_accessed: '2026-06-01T00:00:00Z',
    is_archived: false,
    ...over,
  }
}

describe('patternsToFieldGraph', () => {
  it('returns only the core hub for an empty field', () => {
    const { nodes, links } = patternsToFieldGraph([])
    expect(nodes).toHaveLength(1)
    expect(nodes[0].id).toBe(FIELD_CORE_ID)
    expect(nodes[0].kind).toBe('core')
    expect(links).toHaveLength(0)
  })

  it('renders nodes + edges for a live AND a completed (archived) mission', () => {
    const patterns = [
      pat({ id: 'a', agent_id: 1, key: 'live-finding', is_archived: false }),
      pat({ id: 'b', agent_id: 1, key: 'archived-finding', is_archived: true }),
      pat({ id: 'c', agent_id: 2, key: 'other-agent', is_archived: false }),
    ]
    const { nodes, links } = patternsToFieldGraph(patterns)

    // core + 2 agents + 3 patterns
    expect(nodes.filter((n) => n.kind === 'core')).toHaveLength(1)
    expect(nodes.filter((n) => n.kind === 'agent')).toHaveLength(2)
    expect(nodes.filter((n) => n.kind === 'pattern')).toHaveLength(3)

    // core→agent links (2) + agent→pattern links (3)
    expect(links).toHaveLength(5)
    const ids = new Set(nodes.map((n) => n.id))
    expect(ids.has('agent:1')).toBe(true)
    expect(ids.has('agent:2')).toBe(true)
    expect(ids.has('pattern:a')).toBe(true)
    expect(ids.has('pattern:b')).toBe(true)

    // archived pattern carries the flag and a dimmed colour (≠ its live sibling).
    const live = nodes.find((n) => n.id === 'pattern:a')!
    const archived = nodes.find((n) => n.id === 'pattern:b')!
    expect(archived.archived).toBe(true)
    expect(live.archived).toBe(false)
    expect(archived.color).not.toBe(live.color)
  })

  it('produces no dangling links — every endpoint id is a node', () => {
    const patterns = [pat({ id: 'a', agent_id: 5 }), pat({ id: 'b', agent_id: 9 })]
    const { nodes, links } = patternsToFieldGraph(patterns)
    const ids = new Set(nodes.map((n) => n.id))
    for (const l of links) {
      expect(ids.has(l.source)).toBe(true)
      expect(ids.has(l.target)).toBe(true)
    }
  })

  it('caps pattern nodes and keeps the strongest', () => {
    const patterns = Array.from({ length: MAX_PATTERN_NODES + 50 }, (_, i) =>
      pat({ id: `p${i}`, agent_id: 1, decayed_strength: i / 1000 }),
    )
    const { nodes } = patternsToFieldGraph(patterns)
    const patternNodes = nodes.filter((n) => n.kind === 'pattern')
    expect(patternNodes).toHaveLength(MAX_PATTERN_NODES)
    // strongest (highest index → highest decayed_strength) is kept
    expect(patternNodes.some((n) => n.id === `pattern:p${patterns.length - 1}`)).toBe(true)
  })

  it('does not mutate the input array order', () => {
    const patterns = [
      pat({ id: 'a', decayed_strength: 0.1 }),
      pat({ id: 'b', decayed_strength: 0.9 }),
    ]
    patternsToFieldGraph(patterns)
    expect(patterns.map((p) => p.id)).toEqual(['a', 'b'])
  })
})

describe('colorForAgent / dimColor', () => {
  it('is deterministic — same agent id always maps to the same colour', () => {
    expect(colorForAgent(3)).toBe(colorForAgent(3))
  })

  it('handles negative and large agent ids without going out of range', () => {
    expect(colorForAgent(-1)).toMatch(/^#[0-9a-f]{6}$/i)
    expect(colorForAgent(9999)).toMatch(/^#[0-9a-f]{6}$/i)
  })

  it('dims a hex colour toward black as an rgb() string', () => {
    expect(dimColor('#ffffff')).toBe('rgb(102, 102, 102)')
  })
})
