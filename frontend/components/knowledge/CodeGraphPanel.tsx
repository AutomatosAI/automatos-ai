'use client'

import { useEffect, useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { apiClient } from '@/lib/api'

export function CodeGraphPanel() {
  const [project, setProject] = useState('automatos-ai')
  const [rootDir, setRootDir] = useState('/app/automatos-ai/orchestrator')
  const [query, setQuery] = useState('assemble_context CodeGraphBuilder')
  const [indexing, setIndexing] = useState(false)
  const [result, setResult] = useState<string>('')
  const [count, setCount] = useState<number>(0)
  const [error, setError] = useState<string | null>(null)

  const runIndex = async () => {
    try {
      setError(null)
      setIndexing(true)
      await apiClient.codegraphIndex({ project, root_dir: rootDir })
    } catch (e: any) {
      setError(e?.message || 'Indexing failed')
    } finally {
      setIndexing(false)
    }
  }

  const runSearch = async () => {
    try {
      setError(null)
      const data = await apiClient.codegraphSearch({ project, q: query, limit: 12 })
      setResult(data?.prompt_block || '')
      setCount(data?.count || 0)
    } catch (e: any) {
      setError(e?.message || 'Search failed')
    }
  }

  return (
    <Card className="glass-card">
      <CardHeader>
        <CardTitle>CodeGraph</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-3">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-2">
            <Input placeholder="Project" value={project} onChange={(e)=>setProject(e.target.value)} />
            <Input placeholder="Root directory" value={rootDir} onChange={(e)=>setRootDir(e.target.value)} />
            <Button onClick={runIndex} disabled={indexing}>{indexing ? 'Indexing…' : 'Index Repo'}</Button>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-2">
            <Input className="md:col-span-2" placeholder="Search symbols…" value={query} onChange={(e)=>setQuery(e.target.value)} />
            <Button onClick={runSearch}>Search</Button>
          </div>
          {error && (
            <div className="text-sm text-red-500">{error}</div>
          )}
          <div className="text-xs text-muted-foreground">Matches: {count}</div>
          <pre className="p-3 rounded bg-secondary/30 whitespace-pre-wrap text-sm max-h-[420px] overflow-auto">{result || 'Run a search to see prompt block output'}</pre>
        </div>
      </CardContent>
    </Card>
  )
}


