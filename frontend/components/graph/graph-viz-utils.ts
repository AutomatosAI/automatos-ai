/**
 * Pure helpers shared by every graph surface on the GraphView shell
 * (PRD-165 S1). Engine-agnostic: no react-force-graph / reactflow imports, so
 * they unit-test without dragging a renderer into the test runner and are
 * shared by the KG force view, the codegraph DAG, the mission DAG, and the
 * field viz (PRD-166) — node colours and legend swatches never drift.
 *
 * Moved here from components/knowledge/graph-viz-utils.ts in PRD-165: the
 * palette is no longer KG-specific, it's the shell's.
 */

// ── Link-endpoint id extraction ──────────────────────────────────────────
// react-force-graph mutates link.source / link.target from a string id into
// the resolved node object once the simulation runs, so call sites see one
// shape before the sim and the other after. Worse: `typeof null === "object"`
// is true in JS, so a malformed (null) endpoint slips into the object branch
// and `.id` throws "Cannot read properties of null" — surfacing as the
// "Couldn't render the graph view" GraphErrorBoundary fallback. idOf guards
// both: null/undefined → undefined (never a throw); object → its id; string →
// itself.
export function idOf(endpoint: unknown): string | undefined {
  if (endpoint == null) return undefined
  if (typeof endpoint === 'object') {
    return (endpoint as { id?: string }).id
  }
  return endpoint as string
}

// ── Deterministic colour palette (DATA — BINDING Q26) ────────────────────
// 24-colour wheel spread so every adjacent pair is visually distinct. The
// palette is DATA; colour assignment is algorithmic (a stable hash of the type
// string), so ANY node type — not just a handful of hardcoded Shopify ones —
// gets a stable, distinct colour. No per-type hex literals.
export const GRAPH_PALETTE: readonly string[] = [
  '#ff5e3a', '#10e89e', '#38bdf8', '#c084fc', '#fbbf24', '#f472b6',
  '#22d3ee', '#fb7185', '#a3e635', '#818cf8', '#facc15', '#2dd4bf',
  '#f97316', '#06b6d4', '#d946ef', '#84cc16', '#0ea5e9', '#ec4899',
  '#eab308', '#14b8a6', '#a855f7', '#f59e0b', '#6366f1', '#22c55e',
]

// Neutral fallbacks: an absent type / an unclustered node.
export const NEUTRAL_TYPE_COLOR = '#94a3b8'
export const NEUTRAL_COMMUNITY_COLOR = '#64748b'

// 32-bit string hash → palette index. Deterministic and stable across runs:
// the same key always lands on the same colour. The double-mod keeps the index
// in range even when the 32-bit accumulator goes negative.
function paletteIndex(key: string, modulo: number): number {
  let h = 0
  for (let i = 0; i < key.length; i++) {
    h = (Math.imul(h, 31) + key.charCodeAt(i)) | 0
  }
  return ((h % modulo) + modulo) % modulo
}

/** Stable colour for a node `file_type` — same type always maps to the same hue. */
export function colorForType(type: string | null | undefined): string {
  if (!type) return NEUTRAL_TYPE_COLOR
  return GRAPH_PALETTE[paletteIndex(type, GRAPH_PALETTE.length)]
}

/** Stable colour for a community id — null (unclustered) gets the neutral hue. */
export function colorForCommunity(community: number | null | undefined): string {
  if (community == null) return NEUTRAL_COMMUNITY_COLOR
  const len = GRAPH_PALETTE.length
  return GRAPH_PALETTE[((community % len) + len) % len]
}

// ── Chrome from CSS vars (PRD-165 S1: palette-from-CSS-vars) ──────────────
// The categorical NODE palette above is intentionally fixed data (Q26). The
// shell *chrome* — most importantly the canvas background — must read theme
// tokens so the graph is legible in light theme too (the old hardcoded
// `#0a0d14` / `#111827` were invisible-text traps). Resolve a CSS custom
// property at call time with a dark fallback so it stays correct before the
// var is defined and under SSR/jsdom (where getComputedStyle is unreliable).
export function cssVar(name: string, fallback: string): string {
  if (typeof window === 'undefined' || typeof document === 'undefined') return fallback
  try {
    const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim()
    return v || fallback
  } catch {
    return fallback
  }
}

/** Canvas/background colour for force renderers — theme token, dark fallback. */
export function graphCanvasBackground(): string {
  return cssVar('--graph-canvas', '#0a0d14')
}
