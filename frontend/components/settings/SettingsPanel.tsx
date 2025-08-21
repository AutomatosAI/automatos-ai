'use client'

import { useEffect, useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'

interface Overrides {
  base?: string
  prefix?: string
  apiKey?: string
  authToken?: string
  tenantId?: string
}

export function SettingsPanel() {
  const [overrides, setOverrides] = useState<Overrides>({})
  const [saved, setSaved] = useState(false)

  useEffect(() => {
    if (typeof window === 'undefined') return
    try {
      const current = JSON.parse(localStorage.getItem('automatos.api.overrides') || '{}')
      setOverrides(current || {})
    } catch {}
  }, [])

  const save = () => {
    if (typeof window === 'undefined') return
    localStorage.setItem('automatos.api.overrides', JSON.stringify(overrides || {}))
    setSaved(true)
    setTimeout(()=>setSaved(false), 1500)
  }

  const clear = () => {
    if (typeof window === 'undefined') return
    localStorage.removeItem('automatos.api.overrides')
    setOverrides({})
  }

  return (
    <Card className="glass-card">
      <CardHeader>
        <CardTitle>Settings</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <Input placeholder="API Base URL" value={overrides.base || ''} onChange={(e)=>setOverrides({ ...overrides, base: e.target.value })} />
            <Input placeholder="API Prefix (e.g. /api)" value={overrides.prefix || ''} onChange={(e)=>setOverrides({ ...overrides, prefix: e.target.value })} />
            <Input placeholder="API Key (optional)" value={overrides.apiKey || ''} onChange={(e)=>setOverrides({ ...overrides, apiKey: e.target.value })} />
            <Input placeholder="Auth Bearer Token (optional)" value={overrides.authToken || ''} onChange={(e)=>setOverrides({ ...overrides, authToken: e.target.value })} />
            <Input placeholder="Tenant ID (optional)" value={overrides.tenantId || ''} onChange={(e)=>setOverrides({ ...overrides, tenantId: e.target.value })} />
          </div>
          <div className="flex gap-2">
            <Button onClick={save}>Save</Button>
            <Button variant="outline" onClick={clear}>Clear</Button>
            {saved && <span className="text-sm text-green-500">Saved</span>}
          </div>
        </div>
      </CardContent>
    </Card>
  )
}


