'use client'

/**
 * PRD-239 S7b — the terminal strip: the session (or a terminal) plus as many
 * plain shells as the host allows, each its own PTY on the operator's machine.
 * Inactive tabs stay mounted so their shells keep running.
 */

import { useState } from 'react'
import { Plus, X } from 'lucide-react'
import dynamic from 'next/dynamic'

import type { CanvasTerminalProps } from './CanvasTerminal'
import { addShellTab, canAddTab, closeTab, initialTabs, nextActive } from './terminal-tabs'

const CanvasTerminal = dynamic<CanvasTerminalProps>(
  () => import('./CanvasTerminal').then((m) => m.CanvasTerminal),
  { ssr: false, loading: () => <div className="p-3 text-xs text-muted-foreground">Loading the terminal…</div> },
)

export interface CanvasTerminalsProps {
  taskId?: string | number | null
  runtime: boolean
  /** The host's limit on simultaneous terminals (its `max_terminals` capability). */
  maxTerminals?: number | null
}

export function CanvasTerminals({ taskId, runtime, maxTerminals }: CanvasTerminalsProps) {
  const [tabs, setTabs] = useState(() => initialTabs(runtime))
  const [active, setActive] = useState(tabs[0].id)
  const addable = canAddTab(tabs, maxTerminals)

  const add = () => {
    const next = addShellTab(tabs, maxTerminals)
    if (next === tabs) return
    setTabs(next)
    setActive(next[next.length - 1].id)
  }
  const close = (id: string) => {
    const next = closeTab(tabs, id)
    if (next === tabs) return
    setActive(nextActive(tabs, active, id))
    setTabs(next)
  }

  return (
    <div className="flex h-full min-h-0 flex-col" data-testid="canvas-terminals">
      <div className="flex items-center gap-1 border-b border-border px-2 py-1" role="tablist" aria-label="Terminals">
        {tabs.map((t) => (
          <span key={t.id} className={`inline-flex items-center rounded text-xs ${active === t.id ? 'bg-secondary text-foreground' : 'text-muted-foreground hover:text-foreground'}`}>
            <button type="button" role="tab" aria-selected={active === t.id} onClick={() => setActive(t.id)} className="px-2 py-1">
              {t.label}
            </button>
            {t.kind === 'shell' && (
              <button type="button" aria-label={`Close ${t.label}`} onClick={() => close(t.id)} className="pr-1.5 text-muted-foreground hover:text-foreground">
                <X className="h-3 w-3" />
              </button>
            )}
          </span>
        ))}
        <button
          type="button"
          onClick={add}
          disabled={!addable}
          title={addable ? 'New shell in this folder, on your machine' : `Your host serves at most ${maxTerminals ?? 4} terminals at once`}
          className="ml-1 rounded p-1 text-muted-foreground hover:text-foreground disabled:opacity-40"
          aria-label="New shell"
          data-testid="canvas-terminals-add"
        >
          <Plus className="h-3.5 w-3.5" />
        </button>
      </div>
      <div className="relative min-h-0 flex-1">
        {tabs.map((t) => (
          <div key={t.id} className={`absolute inset-0 ${active === t.id ? '' : 'hidden'}`} data-testid={`canvas-terminal-tab-${t.kind}`}>
            <CanvasTerminal taskId={taskId} autoOpen={t.kind === 'session' || t.kind === 'shell'} runtime={t.kind === 'session'} shell={t.kind === 'shell'} />
          </div>
        ))}
      </div>
    </div>
  )
}
