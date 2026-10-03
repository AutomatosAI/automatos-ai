'use client'

/**
 * The Studio board's toolbar: column or by-agent mode, density, the priority and
 * agent filters, and search. Split out of BoardTab (PRD-252), which had grown
 * past the component limit; the controls and their styles are unchanged.
 */

import { Columns, LayoutList, Rows, AlignJustify, Search } from 'lucide-react'

export type BoardMode = 'column' | 'lane'
export type BoardDensity = 'comfortable' | 'compact'

interface BoardToolbarProps {
  mode: BoardMode
  onMode: (mode: BoardMode) => void
  density: BoardDensity
  onDensity: (density: BoardDensity) => void
  priority: string | null
  onPriority: (priority: string | null) => void
  agentId: number | null
  onAgentId: (agentId: number | null) => void
  agents: unknown
  search: string
  onSearch: (search: string) => void
}

const SELECT_STYLE = { height: 28, fontSize: 11.5, paddingRight: 24 } as const

export function BoardToolbar(props: BoardToolbarProps) {
  const { mode, onMode, density, onDensity } = props
  return (
    <div className="cc-toolbar">
      <div className="cc-seg" role="group" aria-label="Board mode">
        <button type="button" className={mode === 'column' ? 'on' : ''} onClick={() => onMode('column')}>
          <Columns style={{ width: 12, height: 12 }} /> Columns
        </button>
        <button type="button" className={mode === 'lane' ? 'on' : ''} onClick={() => onMode('lane')}>
          <LayoutList style={{ width: 12, height: 12 }} /> By agent
        </button>
      </div>

      <div className="cc-seg" role="group" aria-label="Density">
        <button type="button" className={density === 'comfortable' ? 'on' : ''} onClick={() => onDensity('comfortable')}>
          <Rows style={{ width: 11, height: 11 }} /> Comfortable
        </button>
        <button type="button" className={density === 'compact' ? 'on' : ''} onClick={() => onDensity('compact')}>
          <AlignJustify style={{ width: 11, height: 11 }} /> Compact
        </button>
      </div>

      <ToolbarFilters {...props} />

      <div style={{ marginLeft: 'auto' }}>
        <div
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: 6,
            padding: '0 10px',
            border: '1px solid hsl(var(--border))',
            borderRadius: 6,
            background: 'hsl(var(--card))',
            minWidth: 220,
          }}
        >
          <Search style={{ width: 12, height: 12, color: 'hsl(var(--muted-foreground))' }} />
          <input
            type="search"
            value={props.search}
            onChange={(e) => props.onSearch(e.target.value)}
            placeholder="Search tasks…"
            style={{
              background: 'transparent',
              border: 0,
              outline: 'none',
              fontSize: 12,
              height: 28,
              flex: 1,
              color: 'hsl(var(--foreground))',
            }}
          />
        </div>
      </div>
    </div>
  )
}

function ToolbarFilters({ priority, onPriority, agentId, onAgentId, agents }: BoardToolbarProps) {
  return (
    <div style={{ display: 'inline-flex', gap: 6 }}>
      <select
        className="cc-btn"
        value={priority ?? ''}
        onChange={(e) => onPriority(e.target.value || null)}
        aria-label="Filter by priority"
        style={SELECT_STYLE}
      >
        <option value="">All priorities</option>
        <option value="urgent">Urgent</option>
        <option value="high">High</option>
        <option value="medium">Medium</option>
        <option value="low">Low</option>
      </select>
      <select
        className="cc-btn"
        value={agentId ?? ''}
        onChange={(e) => onAgentId(e.target.value ? Number(e.target.value) : null)}
        aria-label="Filter by agent"
        style={SELECT_STYLE}
      >
        <option value="">All agents</option>
        {Array.isArray(agents) &&
          agents.map((a: any) => (
            <option key={a.id} value={a.id}>
              {a.name}
            </option>
          ))}
      </select>
    </div>
  )
}
