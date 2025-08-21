'use client'

import * as React from 'react'
import { apiClient } from '@/lib/api'
import { Card } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'

export function AgentDetailPanel({ id }: { id: string }) {
  const [agent, setAgent] = React.useState<any | null>(null)
  const [name, setName] = React.useState('')
  const [desc, setDesc] = React.useState('')
  const [saving, setSaving] = React.useState(false)

  React.useEffect(() => {
    let alive = true
    apiClient.getAgent(id).then(a => {
      if (!alive) return
      if (a) {
        setAgent(a)
        setName(a.name || '')
        setDesc(a.description || '')
      }
    })
    return () => { alive = false }
  }, [id])

  async function save() {
    setSaving(true)
    try {
      await apiClient.updateAgent(id, { name, description: desc })
      const fresh = await apiClient.getAgent(id)
      setAgent(fresh)
    } finally { setSaving(false) }
  }

  if (!agent) return <div className="text-sm text-muted-foreground">Loading agent…</div>

  return (
    <Card className="rounded-2xl p-4 grid gap-3">
      <div>
        <div className="text-xs text-muted-foreground mb-1">Name</div>
        <Input value={name} onChange={e=>setName(e.target.value)} />
      </div>
      <div>
        <div className="text-xs text-muted-foreground mb-1">Description</div>
        <Textarea rows={3} value={desc} onChange={e=>setDesc(e.target.value)} />
      </div>
      <div className="flex gap-2">
        <Button onClick={save} disabled={saving}>Save</Button>
      </div>
    </Card>
  )
}


