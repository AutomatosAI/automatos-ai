'use client'

/**
 * EditorTabs — Tab bar for open files in the Coding Canvas
 * PRD-66 Phase 1: Code Viewer Widget
 *
 * Extended: per-tab Preview/Source toggle so previewable files (html, md,
 * pdf, docx, xlsx, image, etc.) can be rendered via FilePreview while
 * source remains in Monaco.
 */

import { Code2, Eye, X } from 'lucide-react'
import { cn } from '@/lib/utils'
import type { OpenFileTab } from '../types'

interface EditorTabsProps {
  tabs: OpenFileTab[]
  activeTabPath: string | null
  onSelectTab: (path: string) => void
  onCloseTab: (path: string) => void
  /** Whether the active tab supports preview (hides toggle otherwise) */
  activeSupportsPreview?: boolean
  /** Current view mode for the active tab */
  activeViewMode?: 'source' | 'preview'
  /** Change view mode on the active tab */
  onSetViewMode?: (mode: 'source' | 'preview') => void
}

export function EditorTabs({
  tabs,
  activeTabPath,
  onSelectTab,
  onCloseTab,
  activeSupportsPreview,
  activeViewMode = 'source',
  onSetViewMode,
}: EditorTabsProps) {
  if (tabs.length === 0) return null

  return (
    <div className="flex items-center border-b border-border/40 bg-muted/20 min-h-[34px]">
      <div className="flex-1 flex items-center overflow-x-auto">
        {tabs.map((tab) => {
          const isActive = tab.path === activeTabPath
          return (
            <div
              key={tab.path}
              className={cn(
                'group flex items-center gap-1.5 px-3 py-1.5 text-xs cursor-pointer',
                'border-r border-border/30 hover:bg-muted/40 transition-colors',
                'max-w-[180px] min-w-[80px]',
                isActive
                  ? 'bg-background text-foreground border-b-2 border-b-primary'
                  : 'text-muted-foreground'
              )}
              onClick={() => onSelectTab(tab.path)}
            >
              {/* Dirty indicator */}
              {tab.isDirty && (
                <span className="w-1.5 h-1.5 rounded-full bg-amber-500 flex-shrink-0" />
              )}

              {/* File name */}
              <span className="truncate flex-1">{tab.name}</span>

              {/* Close button */}
              <button
                className={cn(
                  'flex-shrink-0 p-0.5 rounded hover:bg-muted/60',
                  'opacity-0 group-hover:opacity-100 transition-opacity',
                  isActive && 'opacity-60'
                )}
                onClick={(e) => {
                  e.stopPropagation()
                  onCloseTab(tab.path)
                }}
              >
                <X className="h-3 w-3" />
              </button>
            </div>
          )
        })}
      </div>

      {/* Preview/Source toggle — only when the active tab supports preview */}
      {activeSupportsPreview && onSetViewMode && (
        <div className="flex items-center gap-0.5 mr-1 rounded-md border border-border/40 bg-background/60 px-0.5 py-0.5">
          <button
            type="button"
            onClick={() => onSetViewMode('source')}
            className={cn(
              'flex items-center gap-1 rounded px-1.5 py-0.5 text-[11px] font-medium transition-colors',
              activeViewMode === 'source'
                ? 'bg-muted text-foreground'
                : 'text-muted-foreground hover:text-foreground'
            )}
            title="Show source"
          >
            <Code2 className="h-3 w-3" />
            Source
          </button>
          <button
            type="button"
            onClick={() => onSetViewMode('preview')}
            className={cn(
              'flex items-center gap-1 rounded px-1.5 py-0.5 text-[11px] font-medium transition-colors',
              activeViewMode === 'preview'
                ? 'bg-muted text-foreground'
                : 'text-muted-foreground hover:text-foreground'
            )}
            title="Show preview"
          >
            <Eye className="h-3 w-3" />
            Preview
          </button>
        </div>
      )}
    </div>
  )
}
