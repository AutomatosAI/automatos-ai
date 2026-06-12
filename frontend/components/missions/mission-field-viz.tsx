'use client'

import { useEffect, useMemo, useRef, useState } from 'react'
import dynamic from 'next/dynamic'
import { cn } from '@/lib/utils'
import type { FieldPattern } from '@/hooks/use-missions-api'
import { patternsToFieldGraph, type FieldGraphNode } from './mission-field-viz-utils'
import { FieldVizErrorBoundary } from './field-viz-error-boundary'

interface MissionFieldVizProps {
  patterns: FieldPattern[]
  className?: string
}

// react-force-graph is WebGL/canvas + window-bound, so it must never run on
// the server. The 3D build is the primary renderer; the 2D build is the
// graceful fallback when WebGL is unavailable or the 3D renderer throws.
const ForceGraph3D = dynamic(() => import('react-force-graph-3d'), {
  ssr: false,
  loading: () => (
    <div className="w-full h-full flex items-center justify-center text-xs text-muted-foreground">
      Loading field…
    </div>
  ),
}) as any

const ForceGraph2D = dynamic(() => import('react-force-graph-2d'), {
  ssr: false,
  loading: () => (
    <div className="w-full h-full flex items-center justify-center text-xs text-muted-foreground">
      Loading field…
    </div>
  ),
}) as any

const BACKGROUND = '#0a0a12'
const LINK_COLOR = 'rgba(148, 163, 184, 0.25)'

function nodeColor(node: FieldGraphNode): string {
  return node.color
}

function nodeLabel(node: FieldGraphNode): string {
  if (node.kind === 'pattern') {
    return `${node.label}${node.archived ? ' (archived)' : ''}`
  }
  return node.label
}

export function MissionFieldViz({ patterns, className }: MissionFieldVizProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const [size, setSize] = useState({ w: 800, h: 400 })

  useEffect(() => {
    if (!containerRef.current) return
    const el = containerRef.current
    const update = () => setSize({ w: el.clientWidth, h: el.clientHeight || 400 })
    update()
    const ro = new ResizeObserver(update)
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  const graphData = useMemo(() => patternsToFieldGraph(patterns), [patterns])

  const common = {
    graphData,
    width: size.w,
    height: size.h,
    backgroundColor: BACKGROUND,
    nodeLabel,
    nodeColor,
    nodeVal: (n: FieldGraphNode) => n.val,
    nodeOpacity: 0.9,
    linkColor: () => LINK_COLOR,
    linkWidth: 1,
    enableNodeDrag: false,
  }

  return (
    <div
      ref={containerRef}
      className={cn('w-full h-full min-h-[400px]', className)}
      style={{ background: BACKGROUND }}
    >
      {graphData.nodes.length <= 1 ? (
        <div className="w-full h-full flex items-center justify-center text-xs text-muted-foreground">
          No field patterns yet.
        </div>
      ) : (
        <FieldVizErrorBoundary
          fallback={
            <ForceGraph2D
              {...common}
              linkDirectionalParticles={1}
              linkDirectionalParticleWidth={1.5}
            />
          }
        >
          <ForceGraph3D
            {...common}
            showNavInfo={false}
            linkDirectionalParticles={1}
            linkDirectionalParticleWidth={1.5}
          />
        </FieldVizErrorBoundary>
      )}
    </div>
  )
}
