/**
 * WorkspaceViewToggle (PRD-129: Workspace Outputs Hub)
 * ====================================================
 *
 * Three-way toggle between Gallery (consumer-facing outputs), Explorer
 * (VS Code-style file tree + editor), and Activity (placeholder for a
 * later PRD). Rendered in the workspace page header.
 */

'use client'

import { Activity, FolderTree, LayoutGrid } from 'lucide-react'

import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'

export type WorkspaceView = 'gallery' | 'explorer' | 'activity'

export interface WorkspaceViewToggleProps {
  view: WorkspaceView
  onViewChange: (view: WorkspaceView) => void
}

const VIEW_ITEMS: ReadonlyArray<{
  value: WorkspaceView
  label: string
  icon: React.ComponentType<{ className?: string }>
}> = [
  { value: 'gallery', label: 'Outputs', icon: LayoutGrid },
  { value: 'explorer', label: 'Explorer', icon: FolderTree },
  { value: 'activity', label: 'Activity', icon: Activity },
]

export function WorkspaceViewToggle({
  view,
  onViewChange,
}: WorkspaceViewToggleProps) {
  return (
    <ToggleGroup
      type="single"
      value={view}
      onValueChange={(next) => {
        // Radix fires '' when the active item is re-clicked. Ignore that so
        // we never end up in an undefined view state.
        if (next && next !== view) {
          onViewChange(next as WorkspaceView)
        }
      }}
      variant="outline"
      size="sm"
      aria-label="Workspace view"
    >
      {VIEW_ITEMS.map(({ value, label, icon: Icon }) => (
        <ToggleGroupItem
          key={value}
          value={value}
          aria-label={label}
          className="gap-1.5 px-3"
        >
          <Icon className="h-4 w-4" />
          <span className="text-xs font-medium">{label}</span>
        </ToggleGroupItem>
      ))}
    </ToggleGroup>
  )
}

export default WorkspaceViewToggle
