/**
 * Pure helpers for the mission field force-graph visualisation.
 *
 * Extracted from mission-field-viz.tsx so the pattern→graph transform and the
 * colour logic can be unit-tested without dragging next/dynamic +
 * react-force-graph (WebGL) into the test runner — same split S6 used for the
 * knowledge graph (graph-viz-utils.ts).
 */
import type { FieldPattern } from '@/hooks/use-missions-api'
import { GRAPH_PALETTE, NEUTRAL_TYPE_COLOR } from '../graph/graph-viz-utils'

// The hub node every agent links to. A constant id so links resolve.
export const FIELD_CORE_ID = '__field_core__'

// Render cap: a force simulation with thousands of nodes never settles and
// pins the main thread. Keep the strongest patterns. This is a perf guard,
// not a surfaced metric.
export const MAX_PATTERN_NODES = 200

export type FieldNodeKind = 'core' | 'agent' | 'pattern'

export interface FieldGraphNode {
  id: string
  label: string
  kind: FieldNodeKind
  color: string
  val: number
  agentId?: number
  archived?: boolean
  strength?: number
}

export interface FieldGraphLink {
  source: string
  target: string
}

export interface FieldGraph {
  nodes: FieldGraphNode[]
  links: FieldGraphLink[]
}

/**
 * Stable colour for an agent id over the shared graph palette. The palette is
 * DATA and assignment is a deterministic modulo (BINDING Q26) — no per-agent
 * hex literal, and any agent id gets a consistent hue across renders.
 */
export function colorForAgent(agentId: number): string {
  const len = GRAPH_PALETTE.length
  return GRAPH_PALETTE[((agentId % len) + len) % len]
}

/** Fade a hex colour toward black — archived patterns read as "settled/quiet"
 * without introducing a separate hardcoded colour. Returns an rgb() string so
 * both the 2D canvas and the 3D (three.js) renderer accept it. */
export function dimColor(hex: string): string {
  const m = hex.replace('#', '')
  const f = 0.4
  const r = Math.round(parseInt(m.slice(0, 2), 16) * f)
  const g = Math.round(parseInt(m.slice(2, 4), 16) * f)
  const b = Math.round(parseInt(m.slice(4, 6), 16) * f)
  return `rgb(${r}, ${g}, ${b})`
}

function agentLabel(agentId: number): string {
  return agentId === 0 ? 'System' : `Agent ${agentId}`
}

/**
 * Build a force-graph dataset from the field patterns: a central field hub,
 * one node per contributing agent (linked to the hub), and one node per
 * pattern (linked to its agent). Archived patterns are kept but dimmed, so a
 * completed mission's field still renders its history. Pure — never mutates
 * the input.
 */
export function patternsToFieldGraph(patterns: FieldPattern[]): FieldGraph {
  const nodes: FieldGraphNode[] = [
    {
      id: FIELD_CORE_ID,
      label: 'Shared Field',
      kind: 'core',
      color: NEUTRAL_TYPE_COLOR,
      val: 8,
    },
  ]
  const links: FieldGraphLink[] = []

  // Unique agents in first-seen order; count their patterns for node sizing.
  const orderedAgents: number[] = []
  const seen = new Set<number>()
  const countByAgent = new Map<number, number>()
  for (const p of patterns) {
    countByAgent.set(p.agent_id, (countByAgent.get(p.agent_id) ?? 0) + 1)
    if (!seen.has(p.agent_id)) {
      seen.add(p.agent_id)
      orderedAgents.push(p.agent_id)
    }
  }

  for (const agentId of orderedAgents) {
    const count = countByAgent.get(agentId) ?? 0
    nodes.push({
      id: `agent:${agentId}`,
      label: agentLabel(agentId),
      kind: 'agent',
      color: colorForAgent(agentId),
      val: 3 + Math.min(count, 12),
      agentId,
    })
    links.push({ source: FIELD_CORE_ID, target: `agent:${agentId}` })
  }

  // Strongest patterns first so the cap keeps the most meaningful ones.
  const ranked = [...patterns]
    .sort((a, b) => b.decayed_strength - a.decayed_strength)
    .slice(0, MAX_PATTERN_NODES)

  for (const p of ranked) {
    const base = colorForAgent(p.agent_id)
    nodes.push({
      id: `pattern:${p.id}`,
      label: p.key,
      kind: 'pattern',
      color: p.is_archived ? dimColor(base) : base,
      val: 1 + p.decayed_strength * 3,
      agentId: p.agent_id,
      archived: p.is_archived,
      strength: p.decayed_strength,
    })
    links.push({ source: `agent:${p.agent_id}`, target: `pattern:${p.id}` })
  }

  return { nodes, links }
}
