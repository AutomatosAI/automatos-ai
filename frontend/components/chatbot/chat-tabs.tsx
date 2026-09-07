'use client'

/**
 * PRD-237 S3 — the conversation tab strip.
 *
 * One tab per open conversation plus the "New chat" draft; the strip scrolls
 * sideways on narrow screens. Pure presentation: the session store owns which
 * tabs exist, the page owns what a click means.
 */
import { Plus, X } from 'lucide-react'
import { cn } from '@/lib/utils'

export interface ChatTab {
  /** null = the draft tab. */
  id: string | null
  isDraft: boolean
  title: string
  active: boolean
  unread: boolean
}

export interface ChatTabsProps {
  tabs: ChatTab[]
  onSelect: (tab: ChatTab) => void
  onClose: (tab: ChatTab) => void
  onNew: () => void
}

export function ChatTabs({ tabs, onSelect, onClose, onNew }: ChatTabsProps) {
  // The strip is never empty — the last tab has no close control.
  const canClose = tabs.length > 1

  return (
    <div
      role="tablist"
      aria-label="Open conversations"
      className="flex shrink-0 items-end gap-1 overflow-x-auto border-b border-border/40 px-2 pt-1 [scrollbar-width:thin]"
    >
      {tabs.map((tab) => (
        <div
          key={tab.id ?? 'draft'}
          role="tab"
          aria-selected={tab.active}
          className={cn(
            'group flex max-w-[220px] shrink-0 items-center gap-1 rounded-t-xl border border-b-0 px-3 py-1.5 text-sm transition-colors',
            tab.active
              ? 'border-primary/40 bg-primary/10 text-foreground'
              : 'border-transparent text-muted-foreground hover:bg-secondary/40 hover:text-foreground',
          )}
        >
          <button
            type="button"
            onClick={() => onSelect(tab)}
            className="flex min-w-0 items-center gap-1.5"
            title={tab.title}
          >
            {tab.unread && (
              <span aria-label="Unread" className="h-1.5 w-1.5 shrink-0 rounded-full bg-primary" />
            )}
            <span className={cn('truncate', tab.isDraft && 'italic')}>{tab.title}</span>
          </button>
          {canClose && (
            <button
              type="button"
              onClick={(event) => {
                event.stopPropagation()
                onClose(tab)
              }}
              aria-label={`Close ${tab.title}`}
              className="rounded-full p-0.5 text-muted-foreground opacity-60 transition-opacity hover:bg-secondary/60 hover:text-foreground group-hover:opacity-100"
            >
              <X className="h-3.5 w-3.5" />
            </button>
          )}
        </div>
      ))}
      <button
        type="button"
        onClick={onNew}
        aria-label="New chat"
        title="New chat"
        className="mb-0.5 ml-1 flex h-7 w-7 shrink-0 items-center justify-center rounded-full text-muted-foreground transition-colors hover:bg-secondary/40 hover:text-foreground"
      >
        <Plus className="h-4 w-4" />
      </button>
    </div>
  )
}
