'use client'

import { useState, useMemo } from 'react'
import { motion } from 'framer-motion'
import {
  GitBranch,
  CheckCircle,
  Clock,
  Activity,
  ArrowUpDown,
  ChevronDown,
  ChevronUp,
  XCircle,
  Zap,
} from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Skeleton } from '@/components/ui/skeleton'
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, Legend } from 'recharts'
import { useWorkflowAnalytics } from '@/hooks/use-unified-analytics'

interface Props {
  days: number
}

type SortField = 'name' | 'totalRuns' | 'successRate' | 'avgDuration' | 'cost'

export function AnalyticsWorkflows({ days }: Props) {
  const { data, isLoading } = useWorkflowAnalytics(days)
  const [sortField, setSortField] = useState<SortField>('totalRuns')
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('desc')
  const [expandedWf, setExpandedWf] = useState<string | null>(null)

  const toggleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDir(prev => prev === 'asc' ? 'desc' : 'asc')
    } else {
      setSortField(field)
      setSortDir('desc')
    }
  }

  const parseDuration = (value: string): number => {
    const match = value.match(/^([\d.]+)\s*(ms|s|m|h)$/)
    if (!match) return 0
    const n = parseFloat(match[1])
    switch (match[2]) {
      case 'ms': return n
      case 's': return n * 1000
      case 'm': return n * 60000
      case 'h': return n * 3600000
      default: return 0
    }
  }

  const sortedWorkflows = useMemo(() => {
    if (!data?.workflows) return []
    return [...data.workflows].sort((a: any, b: any) => {
      const aVal = a[sortField] ?? 0
      const bVal = b[sortField] ?? 0
      if (sortField === 'avgDuration') {
        const aMs = parseDuration(String(aVal))
        const bMs = parseDuration(String(bVal))
        return sortDir === 'asc' ? aMs - bMs : bMs - aMs
      }
      if (typeof aVal === 'string' && typeof bVal === 'string') {
        return sortDir === 'asc' ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal)
      }
      return sortDir === 'asc' ? (aVal as number) - (bVal as number) : (bVal as number) - (aVal as number)
    })
  }, [data?.workflows, sortField, sortDir])

  const SortHeader = ({ field, children }: { field: SortField; children: React.ReactNode }) => (
    <button
      onClick={() => toggleSort(field)}
      className="flex items-center gap-1 text-xs font-medium text-muted-foreground hover:text-foreground transition-colors"
    >
      {children}
      <ArrowUpDown className="w-3 h-3" />
    </button>
  )

  if (isLoading) {
    return (
      <div className="space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          {Array.from({ length: 4 }).map((_, i) => (
            <Card key={i} className="glass-card"><CardContent className="p-6"><Skeleton className="h-8 w-16 mb-2" /><Skeleton className="h-4 w-24" /></CardContent></Card>
          ))}
        </div>
        <Skeleton className="h-64 w-full" />
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        {[
          { label: 'Total Workflows', value: data?.summary?.totalWorkflows || 0, icon: GitBranch, color: 'text-purple-400' },
          { label: 'Executions', value: data?.summary?.totalExecutions || 0, icon: Activity, color: 'text-blue-400' },
          { label: 'Success Rate', value: `${(data?.summary?.successRate || 0).toFixed(0)}%`, icon: CheckCircle, color: 'text-green-400' },
          { label: 'Avg Duration', value: data?.summary?.avgDuration || '0s', icon: Clock, color: 'text-orange-400' },
        ].map((card, i) => (
          <motion.div key={card.label} initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.1 }}>
            <Card className="glass-card">
              <CardContent className="p-6">
                <card.icon className={`w-5 h-5 ${card.color} mb-2`} />
                <h3 className="text-2xl font-bold">{card.value}</h3>
                <p className="text-sm text-muted-foreground">{card.label}</p>
              </CardContent>
            </Card>
          </motion.div>
        ))}
      </div>

      {/* Execution Trend Chart */}
      {data?.stats && (data.stats as any)?.recent_executions?.length > 0 && (
        <Card className="glass-card">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Activity className="w-5 h-5 text-blue-400" />
              Execution Trend
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={(data.stats as any).recent_executions || []}>
                  <XAxis dataKey="date" axisLine={false} tickLine={false} tick={{ fontSize: 11, fill: 'hsl(var(--muted-foreground))' }} />
                  <YAxis axisLine={false} tickLine={false} tick={{ fontSize: 11, fill: 'hsl(var(--muted-foreground))' }} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'hsl(var(--card))',
                      border: '1px solid hsl(var(--border))',
                      borderRadius: '8px',
                      fontSize: '12px',
                    }}
                  />
                  <Legend />
                  <Bar dataKey="success" fill="#72BF78" name="Success" stackId="a" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="failed" fill="#ef4444" name="Failed" stackId="a" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Workflow Table */}
      <Card className="glass-card overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-border/50">
                <th className="text-left p-4"><SortHeader field="name">Name</SortHeader></th>
                <th className="text-left p-4"><SortHeader field="totalRuns">Runs</SortHeader></th>
                <th className="text-left p-4"><SortHeader field="successRate">Success Rate</SortHeader></th>
                <th className="text-left p-4 hidden md:table-cell"><SortHeader field="avgDuration">Avg Duration</SortHeader></th>
                <th className="text-left p-4 hidden lg:table-cell"><SortHeader field="cost">Cost</SortHeader></th>
                <th className="text-left p-4 hidden lg:table-cell"><span className="text-xs font-medium text-muted-foreground">Last Run</span></th>
                <th className="text-left p-4 hidden md:table-cell"><span className="text-xs font-medium text-muted-foreground">Status</span></th>
              </tr>
            </thead>
            <tbody>
              {sortedWorkflows.length === 0 ? (
                <tr>
                  <td colSpan={7} className="p-12 text-center text-muted-foreground">
                    <GitBranch className="w-10 h-10 mx-auto mb-3 opacity-50" />
                    <p>No workflows found</p>
                  </td>
                </tr>
              ) : (
                sortedWorkflows.map((wf) => (
                  <tr
                    key={wf.id}
                    className="border-b border-border/30 hover:bg-secondary/20 cursor-pointer transition-colors"
                    onClick={() => setExpandedWf(expandedWf === wf.id.toString() ? null : wf.id.toString())}
                  >
                    <td className="p-4">
                      <div className="flex items-center gap-2">
                        {expandedWf === wf.id.toString() ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                        <span className="font-medium">{wf.name}</span>
                      </div>
                    </td>
                    <td className="p-4">{wf.totalRuns}</td>
                    <td className="p-4">
                      <span className={wf.successRate < 80 && wf.successRate > 0 ? 'text-red-400' : ''}>
                        {wf.successRate > 0 ? `${wf.successRate.toFixed(0)}%` : '-'}
                      </span>
                    </td>
                    <td className="p-4 hidden md:table-cell">{wf.avgDuration || '-'}</td>
                    <td className="p-4 hidden lg:table-cell">{wf.cost > 0 ? `$${wf.cost.toFixed(2)}` : '-'}</td>
                    <td className="p-4 hidden lg:table-cell text-sm text-muted-foreground">
                      {wf.lastRun ? new Date(wf.lastRun).toLocaleDateString() : '-'}
                    </td>
                    <td className="p-4 hidden md:table-cell">
                      <Badge variant="outline" className={
                        wf.status === 'active' ? 'text-green-400 border-green-400/30' :
                        wf.status === 'failed' ? 'text-red-400 border-red-400/30' : ''
                      }>
                        {wf.status || 'idle'}
                      </Badge>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}
