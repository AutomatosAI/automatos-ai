'use client'

import { useState, useMemo } from 'react'
import { motion } from 'framer-motion'
import {
  Bot,
  CheckCircle,
  Clock,
  DollarSign,
  TrendingUp,
  AlertTriangle,
  ChevronDown,
  ChevronUp,
  ArrowUpDown,
  Zap,
} from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Skeleton } from '@/components/ui/skeleton'
import { StatsBar, type StatItem } from '@/components/shared/stats-bar'
import { useAgentAnalytics } from '@/hooks/use-unified-analytics'

interface Props {
  days: number
}

type SortField = 'name' | 'successRate' | 'avgRunTime' | 'tokensUsed' | 'cost'
type SortDir = 'asc' | 'desc'

function formatNumber(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`
  return n.toFixed(0)
}

export function AnalyticsAgents({ days }: Props) {
  const { data, isLoading } = useAgentAnalytics(days)
  const [sortField, setSortField] = useState<SortField>('cost')
  const [sortDir, setSortDir] = useState<SortDir>('desc')
  const [expandedAgent, setExpandedAgent] = useState<string | null>(null)
  const [filterStatus, setFilterStatus] = useState<string>('all')

  const toggleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDir(prev => (prev === 'asc' ? 'desc' : 'asc'))
    } else {
      setSortField(field)
      setSortDir('desc')
    }
  }

  const sortedAgents = useMemo(() => {
    if (!data?.agents) return []
    let filtered = data.agents
    if (filterStatus !== 'all') {
      filtered = filtered.filter(a => a.status === filterStatus)
    }
    return [...filtered].sort((a, b) => {
      const aVal = a[sortField] ?? 0
      const bVal = b[sortField] ?? 0
      if (typeof aVal === 'string' && typeof bVal === 'string') {
        return sortDir === 'asc' ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal)
      }
      return sortDir === 'asc' ? (aVal as number) - (bVal as number) : (bVal as number) - (aVal as number)
    })
  }, [data?.agents, sortField, sortDir, filterStatus])

  const SortHeader = ({ field, children }: { field: SortField; children: React.ReactNode }) => (
    <button
      onClick={() => toggleSort(field)}
      className="flex items-center gap-1 text-xs font-medium text-muted-foreground hover:text-foreground transition-colors"
    >
      {children}
      <ArrowUpDown className="w-3 h-3" />
    </button>
  )

  const getHealthBadge = (agent: any) => {
    if (agent.successRate > 0 && agent.successRate < 80)
      return <Badge variant="destructive" className="text-xs">Low success</Badge>
    if (agent.tokensUsed > 0 && agent.cost > (data?.summary?.totalCost || 0) * 0.3)
      return <Badge className="text-xs bg-yellow-500/20 text-yellow-400 border-yellow-500/30">High cost</Badge>
    return null
  }

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
      <StatsBar stats={[
        { label: 'Total Agents', value: data?.summary?.totalAgents || 0, change: `${data?.summary?.activeAgents || 0} active`, icon: Bot, iconColor: 'text-primary' },
        { label: 'Avg Success Rate', value: `${(data?.summary?.avgSuccessRate || 0).toFixed(0)}%`, change: 'Across all agents', icon: CheckCircle, iconColor: 'text-[hsl(var(--success))]' },
        { label: 'Total Tokens', value: formatNumber(data?.summary?.totalTokens || 0), change: 'This period', icon: Zap, iconColor: 'text-[hsl(var(--info))]' },
        { label: 'Total Cost', value: `$${(data?.summary?.totalCost || 0).toFixed(2)}`, change: 'This period', icon: DollarSign, iconColor: 'text-[hsl(var(--agent))]' },
      ]} loading={isLoading} />

      {/* Filter */}
      <div className="flex items-center gap-3">
        <Select value={filterStatus} onValueChange={setFilterStatus}>
          <SelectTrigger className="w-40 bg-secondary/50">
            <SelectValue placeholder="Status" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Agents</SelectItem>
            <SelectItem value="active">Active</SelectItem>
            <SelectItem value="inactive">Inactive</SelectItem>
          </SelectContent>
        </Select>
        <span className="text-sm text-muted-foreground">{sortedAgents.length} agents</span>
      </div>

      {/* Agent Table */}
      <Card className="glass-card overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-border/50">
                <th className="text-left p-4"><SortHeader field="name">Name</SortHeader></th>
                <th className="text-left p-4 hidden md:table-cell"><span className="text-xs font-medium text-muted-foreground">Status</span></th>
                <th className="text-left p-4"><SortHeader field="successRate">Success Rate</SortHeader></th>
                <th className="text-left p-4 hidden lg:table-cell"><SortHeader field="avgRunTime">Avg Run Time</SortHeader></th>
                <th className="text-left p-4"><SortHeader field="tokensUsed">Tokens</SortHeader></th>
                <th className="text-left p-4"><SortHeader field="cost">Cost</SortHeader></th>
                <th className="text-left p-4 hidden lg:table-cell"><span className="text-xs font-medium text-muted-foreground">LLM Model</span></th>
                <th className="text-left p-4 hidden xl:table-cell"><span className="text-xs font-medium text-muted-foreground">Health</span></th>
              </tr>
            </thead>
            <tbody>
              {sortedAgents.length === 0 ? (
                <tr>
                  <td colSpan={8} className="p-12 text-center text-muted-foreground">
                    <Bot className="w-10 h-10 mx-auto mb-3 opacity-50" />
                    <p>No agents found</p>
                  </td>
                </tr>
              ) : (
                sortedAgents.map((agent) => (
                  <motion.tr
                    key={agent.id}
                    className="border-b border-border/30 hover:bg-secondary/20 cursor-pointer transition-colors"
                    onClick={() => setExpandedAgent(expandedAgent === agent.id.toString() ? null : agent.id.toString())}
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                  >
                    <td className="p-4">
                      <div className="flex items-center gap-2">
                        {expandedAgent === agent.id.toString() ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                        <span className="font-medium">{agent.name}</span>
                      </div>
                    </td>
                    <td className="p-4 hidden md:table-cell">
                      <Badge variant="outline" className={agent.status === 'active' ? 'text-green-400 border-green-400/30' : 'text-gray-400'}>
                        {agent.status}
                      </Badge>
                    </td>
                    <td className="p-4">
                      <span className={agent.successRate < 80 && agent.successRate > 0 ? 'text-red-400' : ''}>
                        {agent.successRate > 0 ? `${agent.successRate.toFixed(0)}%` : '-'}
                      </span>
                    </td>
                    <td className="p-4 hidden lg:table-cell">
                      {agent.avgRunTime > 0 ? `${agent.avgRunTime.toFixed(1)}s` : '-'}
                    </td>
                    <td className="p-4">{agent.tokensUsed > 0 ? formatNumber(agent.tokensUsed) : '-'}</td>
                    <td className="p-4">{agent.cost > 0 ? `$${agent.cost.toFixed(2)}` : '-'}</td>
                    <td className="p-4 hidden lg:table-cell">
                      <Badge variant="secondary" className="text-xs font-mono">{agent.llmModel}</Badge>
                    </td>
                    <td className="p-4 hidden xl:table-cell">{getHealthBadge(agent)}</td>
                  </motion.tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}
