'use client'

import * as React from 'react'
import { apiClient } from '@/lib/api'
import { Card } from '@/components/ui/card'

export function AgentRunsPanel({ id }: { id: string }) {
  const [rows, setRows] = React.useState<any[]>([])

  React.useEffect(() => {
    let alive = true
    apiClient.getAgentRuns(id, 50).then((r) => { if (alive) setRows(r.items || []) })
    return () => { alive = false }
  }, [id])

  return (
    <Card className="rounded-2xl p-0 overflow-hidden">
      <table className="w-full text-sm">
        <thead className="bg-muted/40">
          <tr className="[&>th]:px-3 [&>th]:py-2 text-left">
            <th>ID</th><th>Status</th><th>Latency</th><th>Cost</th><th>Started</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r:any)=> (
            <tr key={r.id} className="border-t [&>td]:px-3 [&>td]:py-2">
              <td className="font-mono">{r.id}</td>
              <td>{r.status || '—'}</td>
              <td>{r.latency_ms != null ? `${r.latency_ms} ms` : '—'}</td>
              <td>{r.cost_usd != null ? `$${r.cost_usd}` : '—'}</td>
              <td className="text-muted-foreground">{r.started_at || '—'}</td>
            </tr>
          ))}
          {!rows.length && (
            <tr><td colSpan={5} className="text-center text-sm text-muted-foreground py-8">No runs.</td></tr>
          )}
        </tbody>
      </table>
    </Card>
  )
}


