'use client'

import { useCallback, useEffect, useRef, useState, type ReactNode } from 'react'
import { Loader2, Maximize2, Minimize2 } from 'lucide-react'
import { GraphErrorBoundary } from './GraphErrorBoundary'

/**
 * The one GraphView shell every graph surface sits on (PRD-165 S1).
 *
 * It owns the chrome that used to be copy-pasted per surface: the bordered
 * container, a toolbar row, a top-right controls overlay, fullscreen, a legend
 * overlay slot, an optional left sidebar, an optional right node-detail
 * side-panel, and consistent loading / empty / error states (the error state
 * via a built-in GraphErrorBoundary). The actual renderer — react-force-graph
 * for the KG/field, ReactFlow for the codegraph / mission DAGs — is the
 * `children`, so the shell is engine-agnostic.
 *
 * Consumed by: the Knowledge Graph (S1/S2), CodeGraph (S4), the mission DAG,
 * and the field viz (PRD-166).
 */

export interface GraphViewProps {
  /** The renderer canvas (force-graph / ReactFlow / etc.). */
  children: ReactNode
  /** Row above the graph — typically a stats bar. */
  toolbar?: ReactNode
  /** Top-right overlay inside the graph area (e.g. colour-mode toggle).
   *  The fullscreen toggle is appended automatically. */
  controls?: ReactNode
  /** Overlay inside the graph area — typically <GraphLegend/>, self-positioned. */
  legend?: ReactNode
  /** Left column (e.g. cluster list). Stacks above the graph on mobile. */
  sidebar?: ReactNode
  /** Right column node-detail panel. Rendered only when truthy. */
  sidePanel?: ReactNode
  /** Spinner overlay. */
  loading?: boolean
  /** When true (and not loading) the emptyState replaces the graph area. */
  empty?: boolean
  emptyState?: ReactNode
  /** Default true. */
  enableFullscreen?: boolean
  /** Tailwind min-height class for the graph area (panel mode). */
  minHeightClassName?: string
  /**
   * Fill the parent's height instead of stacking as a panel. Use for graph
   * canvases embedded in a page that already sets the height (the mission DAG,
   * the codegraph viz) rather than standalone panels (the Knowledge Graph).
   */
  fillHeight?: boolean
  /**
   * Drop the glass-card framing on the graph area (transparent, borderless).
   * Use for canvases already inside a framed page region (the mission DAG,
   * the codegraph viz) so they don't double-border.
   */
  bareArea?: boolean
  className?: string
}

export function GraphView({
  children,
  toolbar,
  controls,
  legend,
  sidebar,
  sidePanel,
  loading = false,
  empty = false,
  emptyState,
  enableFullscreen = true,
  minHeightClassName = 'min-h-[500px]',
  fillHeight = false,
  bareArea = false,
  className,
}: GraphViewProps) {
  // Panel mode stacks vertically (KG); fill mode fills the parent's height
  // for canvases embedded in a page that owns the height (DAGs).
  const areaSizeClass = fillHeight ? 'h-full min-h-0' : minHeightClassName
  const areaChrome = bareArea
    ? ''
    : 'glass-card bg-white/5 backdrop-blur-sm border border-white/10'
  const graphContainerRef = useRef<HTMLDivElement>(null)
  const [isFullscreen, setIsFullscreen] = useState(false)

  const handleFullscreen = useCallback(() => {
    const el = graphContainerRef.current
    if (!el) return
    if (!document.fullscreenElement) {
      el.requestFullscreen?.().then(() => setIsFullscreen(true)).catch(() => {})
    } else {
      document.exitFullscreen?.().then(() => setIsFullscreen(false)).catch(() => {})
    }
  }, [])

  // Keep state in sync if the user ESC-exits without clicking the toggle.
  useEffect(() => {
    const onChange = () => setIsFullscreen(!!document.fullscreenElement)
    document.addEventListener('fullscreenchange', onChange)
    return () => document.removeEventListener('fullscreenchange', onChange)
  }, [])

  const graphArea = (
    <div
      ref={graphContainerRef}
      className={`flex-1 ${areaChrome} rounded-lg overflow-hidden ${areaSizeClass} relative`}
    >
      {/* Top-right controls — caller's controls + fullscreen */}
      {(controls || enableFullscreen) && (
        <div className="absolute top-3 right-3 z-10 flex items-center gap-2">
          {controls}
          {enableFullscreen && (
            <button
              type="button"
              onClick={handleFullscreen}
              className="p-1.5 rounded-md bg-black/50 backdrop-blur-sm border border-white/10 text-muted-foreground hover:text-foreground hover:bg-white/10 transition"
              title={isFullscreen ? 'Exit fullscreen' : 'Fullscreen'}
              aria-label={isFullscreen ? 'Exit fullscreen' : 'Enter fullscreen'}
            >
              {isFullscreen ? <Minimize2 className="w-3.5 h-3.5" /> : <Maximize2 className="w-3.5 h-3.5" />}
            </button>
          )}
        </div>
      )}

      {/* Legend overlay (self-positioned) */}
      {legend}

      {/* Renderer */}
      <GraphErrorBoundary>{children}</GraphErrorBoundary>

      {/* Loading overlay */}
      {loading && (
        <div className="absolute inset-0 z-20 flex items-center justify-center bg-background/40 backdrop-blur-sm">
          <Loader2 className="w-6 h-6 animate-spin text-primary" />
          <span className="ml-2 text-sm text-muted-foreground">Loading graph…</span>
        </div>
      )}
    </div>
  )

  // Empty state replaces the whole graph area (but keeps the toolbar).
  const body =
    empty && !loading ? (
      <div
        className={`flex-1 ${areaChrome} rounded-lg ${areaSizeClass} flex items-center justify-center`}
      >
        {emptyState}
      </div>
    ) : (
      graphArea
    )

  return (
    <div className={`${fillHeight ? 'h-full flex flex-col' : 'space-y-4'} ${className ?? ''}`}>
      {toolbar}
      <div className={`flex flex-col md:flex-row gap-4 ${fillHeight ? 'flex-1 min-h-0' : ''}`}>
        {sidebar}
        {body}
        {sidePanel && (
          <div className="w-full md:w-72 shrink-0">{sidePanel}</div>
        )}
      </div>
    </div>
  )
}
