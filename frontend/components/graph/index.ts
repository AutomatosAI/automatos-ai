/**
 * The shared GraphView shell (PRD-165 S1). One import surface for every graph
 * view — Knowledge Graph, CodeGraph, mission DAG, field viz (PRD-166).
 */
export { GraphView, type GraphViewProps } from './GraphView'
export { GraphLegend, type LegendChip, type LegendSection } from './GraphLegend'
export { GraphErrorBoundary } from './GraphErrorBoundary'
export {
  useGraphPrefs,
  readGraphPrefs,
  writeGraphPrefs,
  DEFAULT_GRAPH_PREFS,
  type GraphPrefs,
  type GraphColorMode,
  type GraphSurface,
} from './useGraphPrefs'
export {
  idOf,
  cssVar,
  graphCanvasBackground,
  colorForType,
  colorForCommunity,
  GRAPH_PALETTE,
  NEUTRAL_TYPE_COLOR,
  NEUTRAL_COMMUNITY_COLOR,
} from './graph-viz-utils'
