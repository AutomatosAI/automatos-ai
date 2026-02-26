'use client'

/**
 * EditorTabs — Tab bar for open files in the Coding Canvas
 * PRD-66 Phase 1: Code Viewer Widget
 */

import { X } from 'lucide-react'
import { cn } from '@/lib/utils'
import type { OpenFileTab } from '../types'

interface EditorTabsProps {
  tabs: OpenFileTab[]
  activeTabPath: string | null
  onSelectTab: (path: string) => void
  onCloseTab: (path: string) => void
}

export function EditorTabs({ tabs, activeTabPath, onSelectTab, onCloseTab }: EditorTabsProps) {
  if (tabs.length === 0) return null

  return (
    <div className="flex items-center border-b border-border/40 bg-muted/20 overflow-x-auto min-h-[34px]">
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
  )
}
