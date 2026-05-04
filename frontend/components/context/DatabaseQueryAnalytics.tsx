'use client'

import { useState, useEffect } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Database, Clock, CheckCircle, TrendingUp, AlertCircle, Target, Play } from 'lucide-react'
import { toast } from 'sonner'

export function DatabaseQueryAnalytics() {
  const [stats, setStats] = useState<any>(null)
  const [performance, setPerformance] = useState<any[]>([])
  const [topQueries, setTopQueries] = useState<any[]>([])
  const [benchmarks, setBenchmarks] = useState<any[]>([])
  const [runningBenchmark, setRunningBenchmark] = useState(false)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetchAnalytics()
  }, [])

  const fetchAnalytics = async () => {
    try {
      const [statsRes, perfRes, queriesRes] = await Promise.all([
        fetch('/api/database/analytics/stats'),
        fetch('/api/database/analytics/performance'),
        fetch('/api/database/analytics/top-queries?limit=5')
      ])

      if (statsRes.ok) setStats(await statsRes.json())
      if (perfRes.ok) setPerformance(await perfRes.json())
      if (queriesRes.ok) setTopQueries(await queriesRes.json())

      // Fetch benchmark history (best effort - may not have sources)
      try {
        const sourcesRes = await fetch('/api/knowledge/sources/database')
        if (sourcesRes.ok) {
          const sources = await sourcesRes.json()
          if (sources.length > 0) {
            const benchRes = await fetch(`/api/knowledge/sources/database/${sources[0].id}/benchmark/history?limit=10`)
            if (benchRes.ok) setBenchmarks(await benchRes.json())
          }
        }
      } catch { /* no benchmarks available */ }
    } catch (error) {
      console.error('Failed to fetch database analytics:', error)
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return (
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {[1, 2, 3, 4].map(i => (
          <Card key={i} className="glass-card animate-pulse">
            <CardHeader>
              <div className="h-6 bg-muted rounded w-1/2" />
            </CardHeader>
            <CardContent>
              <div className="h-20 bg-muted rounded" />
            </CardContent>
          </Card>
        ))}
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Overview Stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card className="glass-card">
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Total Queries (24h)</p>
                <p className="text-2xl font-bold text-info">{stats?.total_queries || 0}</p>
              </div>
              <Database className="h-8 w-8 text-info opacity-50" />
            </div>
          </CardContent>
        </Card>

        <Card className="glass-card">
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Success Rate</p>
                <p className="text-2xl font-bold text-success">{stats?.success_rate?.toFixed(1) || '100'}%</p>
              </div>
              <CheckCircle className="h-8 w-8 text-success opacity-50" />
            </div>
          </CardContent>
        </Card>

        <Card className="glass-card">
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Avg Query Time</p>
                <p className="text-2xl font-bold text-agent">{stats?.avg_execution_time_ms?.toFixed(0) || 0}ms</p>
              </div>
              <Clock className="h-8 w-8 text-agent opacity-50" />
            </div>
          </CardContent>
        </Card>

        <Card className="glass-card">
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Total Rows Retrieved</p>
                <p className="text-2xl font-bold text-orange-400">{(stats?.total_rows || 0).toLocaleString()}</p>
              </div>
              <TrendingUp className="h-8 w-8 text-orange-400 opacity-50" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Top Queries */}
      <Card className="glass-card">
        <CardHeader>
          <CardTitle>Most Frequent Queries (Last 7 Days)</CardTitle>
        </CardHeader>
        <CardContent>
          {topQueries && topQueries.length > 0 ? (
            <div className="space-y-3">
              {topQueries.map((q, idx) => (
                <div key={idx} className="p-3 bg-secondary/30 rounded-lg border border-border/50">
                  <div className="flex items-start justify-between mb-2">
                    <p className="text-sm font-medium">{q.query}</p>
                    <Badge variant="outline" className="bg-info/10 border-info/20 text-info">
                      {q.count}x
                    </Badge>
                  </div>
                  <div className="flex items-center justify-between text-xs text-muted-foreground">
                    <code className="bg-muted px-2 py-1 rounded text-xs">
                      {q.sql?.substring(0, 80)}...
                    </code>
                    <span>{q.avg_time_ms?.toFixed(0)}ms avg</span>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-8 text-muted-foreground">
              <Database className="h-12 w-12 mx-auto mb-2 opacity-50" />
              <p>No database queries yet</p>
              <p className="text-xs mt-1">Query analytics will appear here after you run database queries</p>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Performance Trends */}
      {performance && performance.length > 0 && (
        <Card className="glass-card">
          <CardHeader>
            <CardTitle>Query Performance (Last 24 Hours)</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {performance.map((p, idx) => (
                <div key={idx} className="flex items-center justify-between text-sm">
                  <span className="text-muted-foreground">{p.time}</span>
                  <div className="flex items-center gap-4">
                    <span>{p.queries} queries</span>
                    <span className="text-agent">{p.avg_time_ms?.toFixed(0)}ms avg</span>
                    <span className="text-success">{p.success_rate?.toFixed(1)}% success</span>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* NL2SQL Benchmark Accuracy */}
      <Card className="glass-card">
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Target className="h-5 w-5 text-agent" />
              NL2SQL Benchmark Accuracy
            </div>
            <Button
              variant="outline"
              size="sm"
              disabled={runningBenchmark}
              onClick={async () => {
                setRunningBenchmark(true)
                try {
                  const sourcesRes = await fetch('/api/knowledge/sources/database')
                  if (!sourcesRes.ok) throw new Error('No sources')
                  const sources = await sourcesRes.json()
                  if (sources.length === 0) { toast.error('No database sources to benchmark'); return }
                  const res = await fetch(`/api/knowledge/sources/database/${sources[0].id}/benchmark/run`, { method: 'POST' })
                  if (res.ok) {
                    const result = await res.json()
                    toast.success(`Benchmark complete: ${(result.exact_match_rate * 100).toFixed(1)}% exact match`)
                    setBenchmarks(prev => [result, ...prev].slice(0, 10))
                  } else {
                    const err = await res.json()
                    toast.error(err.detail || 'Benchmark failed')
                  }
                } catch { toast.error('Could not run benchmark') }
                finally { setRunningBenchmark(false) }
              }}
            >
              <Play className="h-4 w-4 mr-1" />
              {runningBenchmark ? 'Running...' : 'Run Benchmark'}
            </Button>
          </CardTitle>
        </CardHeader>
        <CardContent>
          {benchmarks.length > 0 ? (
            <div className="space-y-3">
              {benchmarks.map((b, idx) => (
                <div key={b.id || idx} className="flex items-center justify-between p-3 bg-secondary/30 rounded-lg border border-border/50">
                  <div className="flex items-center gap-3">
                    <div className="text-center">
                      <div className="text-lg font-bold text-agent">
                        {((b.exact_match_rate || 0) * 100).toFixed(1)}%
                      </div>
                      <div className="text-[10px] text-muted-foreground">Exact</div>
                    </div>
                    <div className="text-center">
                      <div className="text-lg font-bold text-success">
                        {((b.execution_match_rate || 0) * 100).toFixed(1)}%
                      </div>
                      <div className="text-[10px] text-muted-foreground">Execution</div>
                    </div>
                  </div>
                  <div className="text-right text-xs text-muted-foreground">
                    <div>{b.total_examples || 0} examples tested</div>
                    <div>{b.created_at ? new Date(b.created_at).toLocaleDateString() : 'Unknown'}</div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-8 text-muted-foreground">
              <Target className="h-12 w-12 mx-auto mb-2 opacity-50" />
              <p>No benchmark runs yet</p>
              <p className="text-xs mt-1">Run a benchmark to measure NL2SQL accuracy against verified training examples</p>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Info Message */}
      <Card className="glass-card border-info/30 bg-info/5">
        <CardContent className="pt-6">
          <div className="flex items-start gap-3">
            <AlertCircle className="h-5 w-5 text-info flex-shrink-0 mt-0.5" />
            <div className="space-y-1">
              <p className="text-sm font-medium text-info">Real-Time Analytics</p>
              <p className="text-xs text-muted-foreground">
                All metrics are computed from the <code className="bg-muted px-1 rounded">database_query_audit</code> table. 
                Statistics update automatically as users query connected databases via natural language.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

