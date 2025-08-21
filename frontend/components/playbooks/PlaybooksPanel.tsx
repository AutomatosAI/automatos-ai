'use client'

import { useEffect, useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { apiClient } from '@/lib/api'

export function PlaybooksPanel() {
  const [tenantId, setTenantId] = useState<string>('')
  const [items, setItems] = useState<any[]>([])
  const [loading, setLoading] = useState(false)
  const [minSupport, setMinSupport] = useState<number>(5)
  const [topK, setTopK] = useState<number>(20)
  const [prefix, setPrefix] = useState<string>('auto')
  const [error, setError] = useState<string | null>(null)

  const load = async () => {
    try {
      setLoading(true)
      setError(null)
      const data = await apiClient.listPlaybooks(tenantId ? { tenant_id: tenantId } : undefined)
      setItems(data?.items || [])
    } catch (e: any) {
      setError(e?.message || 'Failed to load playbooks')
    } finally {
      setLoading(false)
    }
  }

  const mine = async () => {
    try {
      setLoading(true)
      setError(null)
      await apiClient.minePlaybooks({ tenant_id: tenantId || undefined, min_support: minSupport, top_k: topK, name_prefix: prefix })
      await load()
    } catch (e: any) {
      setError(e?.message || 'Mining failed')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { load() }, [])

  return (
    <Card className="glass-card">
      <CardHeader>
        <CardTitle>Playbooks</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-2">
            <Input type="number" placeholder="Min support" value={minSupport} onChange={(e)=>setMinSupport(Number(e.target.value||0))} />
            <Input type="number" placeholder="Top K" value={topK} onChange={(e)=>setTopK(Number(e.target.value||0))} />
            <Input placeholder="Name prefix" value={prefix} onChange={(e)=>setPrefix(e.target.value)} />
          </div>
          <div className="flex gap-2">
            <Button onClick={load} variant="outline" disabled={loading}>{loading ? 'Loading…' : 'Refresh'}</Button>
            <Button onClick={mine} disabled={loading}>Mine</Button>
          </div>
          {error && <div className="text-sm text-red-500">{error}</div>}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
            {items.map((p) => (
              <div key={p.id} className="p-4 rounded border border-border/50 bg-secondary/20">
                <div className="font-medium">{p.name}</div>
                <div className="text-xs text-muted-foreground">Support: {p.support}</div>
                <div className="text-xs text-muted-foreground">Tenant: {p.tenant_id || '—'}</div>
                <div className="text-xs text-muted-foreground">Created: {p.created_at}</div>
              </div>
            ))}
          </div>
        </div>
      </CardContent>
    </Card>
  )
}


