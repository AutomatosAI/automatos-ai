// @ts-nocheck
'use client'

import * as React from 'react'
import { apiClient } from '@/lib/api'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'

export function AgentsTable2({ onSelect }: { onSelect?: (id: string)=>void }) {
  const [q, setQ] = React.useState('')
  const [items, setItems] = React.useState<any[]>([])
  const [loading, setLoading] = React.useState(false)

  async function fetchAgents() {
    setLoading(true)
    try {
      const data = await apiClient.getAgents({ q, limit: 50 })
      setItems(Array.isArray(data) ? data : [])
    } catch {
      setItems([])
    } finally {
      setLoading(false)
    }
  }

  React.useEffect(() => { fetchAgents() }, [])
  React.useEffect(() => {
    const h = () => fetchAgents()
    window.addEventListener('agents:refresh', h)
    return () => window.removeEventListener('agents:refresh', h)
  }, [])

  return (
    <div className="space-y-3">
      <div className="flex items-end gap-3">
        <Input placeholder="Search agents" value={q} onChange={e => setQ(e.target.value)} className="max-w-sm" />
        <Button onClick={fetchAgents} disabled={loading}>Search</Button>
      </div>
      <div className="rounded-2xl border overflow-hidden">
        <table className="w-full text-sm">
          <thead className="bg-muted/40">
            <tr className="[&>th]:px-3 [&>th]:py-2 text-left">
              <th>Name</th><th>Type</th><th>Status</th><th>Updated</th><th></th>
            </tr>
          </thead>
          <tbody>
            {items.map((a: any) => (
              <tr key={a.id} className="border-t [&>td]:px-3 [&>td]:py-2">
                <td className="font-medium">{a.name || a.id}</td>
                <td className="text-muted-foreground">{a.agent_type || '—'}</td>
                <td className="text-muted-foreground">{a.status || '—'}</td>
                <td className="text-muted-foreground">{a.updated_at?.slice?.(0,19)?.replace?.('T',' ') || '—'}</td>
                <td className="text-right"><Button size="sm" onClick={()=>onSelect?.(String(a.id))}>Select</Button></td>
              </tr>
            ))}
            {!items.length && (
              <tr><td colSpan={5} className="text-center text-sm text-muted-foreground py-8">No agents found.</td></tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  )
}
