'use client'

/**
 * LayoutPresets Toolbar (US-010)
 *
 * Horizontal toolbar with 4 layout preset buttons: Grid, Split, Focus, Stack.
 * Each preset calculates positions/sizes on a 12-column grid and applies them
 * to every widget via the workspace store.
 */

import { useCallback } from 'react'
import { LayoutGrid, Columns, Maximize2, Rows } from 'lucide-react'
import { useWorkspaceStore } from '@/stores/workspace-store'
import type { LayoutMode, WidgetPosition, WidgetSize } from '@/components/widgets/types'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'

// ── Preset definitions ──────────────────────────────────────────────────

interface LayoutPreset {
  mode: LayoutMode
  label: string
  icon: React.ComponentType<{ className?: string }>
  layout: (widgetIds: string[]) => { positions: Record<string, WidgetPosition>; sizes: Record<string, WidgetSize> }
}

/**
 * Grid: equal-sized cells, cols = ceil(sqrt(n)), each widget width = 12/cols, height = 4
 */
function gridLayout(ids: string[]) {
  const n = ids.length
  if (n === 0) return { positions: {} as Record<string, WidgetPosition>, sizes: {} as Record<string, WidgetSize> }

  const cols = Math.ceil(Math.sqrt(n))
  const cellWidth = Math.floor(12 / cols)
  const positions: Record<string, WidgetPosition> = {}
  const sizes: Record<string, WidgetSize> = {}

  ids.forEach((id, i) => {
    const col = i % cols
    const row = Math.floor(i / cols)
    positions[id] = { x: col * cellWidth, y: row * 4 }
    sizes[id] = { width: cellWidth, height: 4 }
  })

  return { positions, sizes }
}

/**
 * Split: first 2 widgets 50/50 side-by-side (width=6, height=4),
 * rest below at width=6, height=3
 */
function splitLayout(ids: string[]) {
  const positions: Record<string, WidgetPosition> = {}
  const sizes: Record<string, WidgetSize> = {}

  ids.forEach((id, i) => {
    if (i < 2) {
      // First two: side by side
      positions[id] = { x: i * 6, y: 0 }
      sizes[id] = { width: 6, height: 4 }
    } else {
      // Rest below, alternating columns
      const belowIndex = i - 2
      const col = belowIndex % 2
      const row = Math.floor(belowIndex / 2)
      positions[id] = { x: col * 6, y: 4 + row * 3 }
      sizes[id] = { width: 6, height: 3 }
    }
  })

  return { positions, sizes }
}

/**
 * Focus: first widget at width=9, height=6 on the left,
 * rest stacked in a sidebar at x=9, width=3, height=2
 */
function focusLayout(ids: string[]) {
  const positions: Record<string, WidgetPosition> = {}
  const sizes: Record<string, WidgetSize> = {}

  ids.forEach((id, i) => {
    if (i === 0) {
      positions[id] = { x: 0, y: 0 }
      sizes[id] = { width: 9, height: 6 }
    } else {
      const sidebarIndex = i - 1
      positions[id] = { x: 9, y: sidebarIndex * 2 }
      sizes[id] = { width: 3, height: 2 }
    }
  })

  return { positions, sizes }
}

/**
 * Stack: all widgets full width (12), stacked vertically, height=4
 */
function stackLayout(ids: string[]) {
  const positions: Record<string, WidgetPosition> = {}
  const sizes: Record<string, WidgetSize> = {}

  ids.forEach((id, i) => {
    positions[id] = { x: 0, y: i * 4 }
    sizes[id] = { width: 12, height: 4 }
  })

  return { positions, sizes }
}

const PRESETS: LayoutPreset[] = [
  { mode: 'grid', label: 'Grid', icon: LayoutGrid, layout: gridLayout },
  { mode: 'split', label: 'Split', icon: Columns, layout: splitLayout },
  { mode: 'focus', label: 'Focus', icon: Maximize2, layout: focusLayout },
  { mode: 'stack', label: 'Stack', icon: Rows, layout: stackLayout },
]

// ── Component ───────────────────────────────────────────────────────────

interface LayoutPresetsProps {
  className?: string
}

export function LayoutPresets({ className }: LayoutPresetsProps) {
  const layoutMode = useWorkspaceStore((s) => s.layoutMode)
  const widgetIds = useWorkspaceStore((s) => s.widgetIds)
  const updatePosition = useWorkspaceStore((s) => s.updatePosition)
  const updateSize = useWorkspaceStore((s) => s.updateSize)
  const setLayoutMode = useWorkspaceStore((s) => s.setLayoutMode)

  const applyPreset = useCallback(
    (preset: LayoutPreset) => {
      const { positions, sizes } = preset.layout(widgetIds)

      // Update each widget's position and size
      for (const id of widgetIds) {
        if (positions[id]) updatePosition(id, positions[id])
        if (sizes[id]) updateSize(id, sizes[id])
      }

      setLayoutMode(preset.mode)
    },
    [widgetIds, updatePosition, updateSize, setLayoutMode]
  )

  return (
    <div className={cn('flex items-center gap-1', className)}>
      {PRESETS.map((preset) => {
        const Icon = preset.icon
        const isActive = layoutMode === preset.mode

        return (
          <Button
            key={preset.mode}
            variant="ghost"
            size="sm"
            className={cn(
              'h-8 gap-1.5 px-2.5 text-xs font-medium',
              isActive && 'bg-accent text-accent-foreground'
            )}
            onClick={() => applyPreset(preset)}
            title={`${preset.label} layout`}
          >
            <Icon className="h-3.5 w-3.5" />
            <span className="hidden sm:inline">{preset.label}</span>
          </Button>
        )
      })}
    </div>
  )
}
