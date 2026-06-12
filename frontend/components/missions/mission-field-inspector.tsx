'use client'

/**
 * MissionFieldInspector (PRD-166 S4)
 *
 * The retrieval-trace side panel for the field graph (Letta-style): inspect a
 * selected pattern's provenance + scores, and run a trace query to see WHICH
 * patterns fire for a given prompt and WHY (resonance = similarity × strength).
 * Firing pattern ids are reported up so the graph can highlight them.
 */

import { useState } from 'react'
import { Search, Loader2, X } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { useFieldQuery, type FieldPattern, type FieldTraceHit } from '@/hooks/use-missions-api'

interface MissionFieldInspectorProps {
  missionId: string
  selected: FieldPattern | null
  onClearSelection: () => void
  onFiringChange: (firingIds: Set<string>) => void
}

function ScoreRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between text-[11px]">
      <span className="text-muted-foreground">{label}</span>
      <span className="tabular-nums">{value}</span>
    </div>
  )
}

export function MissionFieldInspector({
  missionId,
  selected,
  onClearSelection,
  onFiringChange,
}: MissionFieldInspectorProps) {
  const [query, setQuery] = useState('')
  const trace = useFieldQuery(missionId)
  const hits: FieldTraceHit[] = trace.data?.results ?? []

  const runTrace = async () => {
    const q = query.trim()
    if (!q) return
    try {
      const res = await trace.mutateAsync({ query: q })
      onFiringChange(new Set((res.results ?? []).map((h) => `pattern:${h.id}`)))
    } catch {
      onFiringChange(new Set())
    }
  }

  return (
    <div className="flex flex-col gap-4 rounded-lg border border-border bg-card/40 p-3 h-full overflow-y-auto">
      {/* Selected pattern detail */}
      {selected ? (
        <div className="space-y-2">
          <div className="flex items-start justify-between gap-2">
            <h4 className="text-xs font-semibold">{selected.key}</h4>
            <button onClick={onClearSelection} className="text-muted-foreground hover:text-foreground">
              <X className="h-3.5 w-3.5" />
            </button>
          </div>
          <p className="text-[11px] text-muted-foreground leading-relaxed">{selected.value}</p>
          <div className="space-y-1 pt-1">
            <ScoreRow label="Strength" value={`${(selected.decayed_strength * 100).toFixed(0)}%`} />
            <ScoreRow label="Accessed" value={`${selected.access_count}×`} />
            <ScoreRow label="From" value={selected.agent_id === 0 ? 'System' : `Agent ${selected.agent_id}`} />
            {selected.mission_id && <ScoreRow label="Mission" value={selected.mission_id.slice(0, 8)} />}
            {selected.is_archived && <ScoreRow label="State" value="archived" />}
          </div>
        </div>
      ) : (
        <p className="text-[11px] text-muted-foreground">Select a pattern to inspect its provenance.</p>
      )}

      {/* Retrieval-trace query */}
      <div className="space-y-2 border-t border-border pt-3">
        <label className="text-[11px] font-medium text-muted-foreground">Retrieval trace</label>
        <div className="flex gap-1.5">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter') runTrace() }}
            placeholder="query the field…"
            className="flex-1 rounded border border-border bg-background px-2 py-1 text-[11px] outline-none focus:border-primary"
          />
          <Button size="sm" variant="outline" disabled={trace.isLoading || !query.trim()} onClick={runTrace}>
            {trace.isLoading ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Search className="h-3.5 w-3.5" />}
          </Button>
        </div>

        {hits.length > 0 && (
          <ol className="space-y-1.5 pt-1">
            {hits.map((h, i) => (
              <li key={h.id} className="rounded border border-border/60 p-1.5 text-[11px]">
                <div className="flex items-center justify-between gap-2">
                  <span className="font-medium truncate">{i + 1}. {h.key}</span>
                  <span className="tabular-nums text-muted-foreground shrink-0">{(h.score * 100).toFixed(0)}%</span>
                </div>
                <div className="mt-0.5 flex gap-3 text-[10px] text-muted-foreground tabular-nums">
                  <span>sim {(h.cosine_similarity * 100).toFixed(0)}%</span>
                  <span>str {(h.decayed_strength * 100).toFixed(0)}%</span>
                </div>
              </li>
            ))}
          </ol>
        )}
        {trace.data && hits.length === 0 && !trace.isLoading && (
          <p className="text-[11px] text-muted-foreground">No patterns fired for that query.</p>
        )}
      </div>
    </div>
  )
}
