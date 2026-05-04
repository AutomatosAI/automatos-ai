'use client'

import { useState, useMemo, Fragment } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Bot,
  Brain,
  CheckCircle,
  Clock,
  DollarSign,
  TrendingUp,
  AlertTriangle,
  ChevronDown,
  ChevronUp,
  ArrowUpDown,
  Zap,
  Activity,
} from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Skeleton } from '@/components/ui/skeleton'
import { Progress } from '@/components/ui/progress'
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

function formatDate(dateStr: string | null): string {
  if (!dateStr) return '-'
  const d = new Date(dateStr)
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })
}

function AgentExpandedPanel({ agent }: { agent: any }) {
  const memoryTypes = agent.memoryTypes || {}
  const memoryLevels = agent.memoryLevels || {}
  const hasMemory = agent.memoryCount > 0

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
      {/* Memory Column */}
      <div className="space-y-3">
        <div className="flex items-center gap-2 text-sm font-medium">
          <Brain className="w-4 h-4 text-[hsl(var(--agent))]" />
          Memory
        </div>
        {hasMemory ? (
          <>
            <div className="flex items-center justify-between text-sm">
              <span className="text-muted-foreground">Total memories</span>
              <span className="font-medium">{agent.memoryCount}</span>
            </div>
            <div className="space-y-1">
              <div className="flex items-center justify-between text-sm">
                <span className="text-muted-foreground">Avg importance</span>
                <span className="font-medium">{agent.avgImportance.toFixed(2)}</span>
              </div>
              <Progress value={agent.avgImportance * 100} className="h-1.5" />
            </div>
            {Object.keys(memoryTypes).length > 0 && (
              <div className="space-y-2">
                <span className="text-xs text-muted-foreground uppercase tracking-wider">Types</span>
                <div className="flex flex-wrap gap-1.5">
                  {Object.entries(memoryTypes).map(([type, count]) => (
                    <Badge key={type} variant="secondary" className="text-xs">
                      {type.replace(/_/g, ' ').toLowerCase()} ({count as number})
                    </Badge>
                  ))}
                </div>
              </div>
            )}
            {Object.keys(memoryLevels).length > 0 && (
              <div className="space-y-2">
                <span className="text-xs text-muted-foreground uppercase tracking-wider">Levels</span>
                <div className="flex flex-wrap gap-1.5">
                  {Object.entries(memoryLevels).map(([level, count]) => (
                    <Badge key={level} variant="outline" className="text-xs">
                      {level.replace(/_/g, ' ').toLowerCase()} ({count as number})
                    </Badge>
                  ))}
                </div>
              </div>
            )}
          </>
        ) : (
          <p className="text-sm text-muted-foreground italic">No memories</p>
        )}
      </div>

      {/* Usage Details Column */}
      <div className="space-y-3">
        <div className="flex items-center gap-2 text-sm font-medium">
          <Zap className="w-4 h-4 text-[hsl(var(--info))]" />
          Usage Details
        </div>
        <div className="flex items-center justify-between text-sm">
          <span className="text-muted-foreground">Total requests</span>
          <span className="font-medium">{agent.totalRequests > 0 ? formatNumber(agent.totalRequests) : '-'}</span>
        </div>
        <div className="flex items-center justify-between text-sm">
          <span className="text-muted-foreground">Avg tokens/request</span>
          <span className="font-medium">
            {agent.totalRequests > 0 ? formatNumber(Math.round(agent.tokensUsed / agent.totalRequests)) : '-'}
          </span>
        </div>
        <div className="flex items-center justify-between text-sm">
          <span className="text-muted-foreground">Total tokens</span>
          <span className="font-medium">{agent.tokensUsed > 0 ? formatNumber(agent.tokensUsed) : '-'}</span>
        </div>
        <div className="flex items-center justify-between text-sm">
          <span className="text-muted-foreground">Total cost</span>
          <span className="font-medium">{agent.cost > 0 ? `$${agent.cost.toFixed(2)}` : '-'}</span>
        </div>
        {hasMemory && (
          <div className="flex items-center justify-between text-sm">
            <span className="text-muted-foreground">Memory accesses</span>
            <span className="font-medium">{agent.totalAccesses}</span>
          </div>
        )}
      </div>

      {/* Activity Column */}
      <div className="space-y-3">
        <div className="flex items-center gap-2 text-sm font-medium">
          <Activity className="w-4 h-4 text-[hsl(var(--success))]" />
          Activity
        </div>
        <div className="flex items-center justify-between text-sm">
          <span className="text-muted-foreground">Last used</span>
          <span className="font-medium">{formatDate(agent.lastUsed)}</span>
        </div>
        <div className="flex items-center justify-between text-sm">
          <span className="text-muted-foreground">Last memory</span>
          <span className="font-medium">{formatDate(agent.lastMemoryAt)}</span>
        </div>
        <div className="flex items-center justify-between text-sm">
          <span className="text-muted-foreground">Agent type</span>
          <Badge variant="secondary" className="text-xs capitalize">{agent.agentType || 'general'}</Badge>
        </div>
        <div className="flex items-center justify-between text-sm">
          <span className="text-muted-foreground">Model</span>
          <Badge variant="secondary" className="text-xs font-mono">{agent.llmModel}</Badge>
        </div>
      </div>
    </div>
  )
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
      return <Badge className="text-xs bg-[hsl(var(--warning))]/20 text-[hsl(var(--warning))] border-[hsl(var(--warning))]/30">High cost</Badge>
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
        { label: 'Total Agents', value: data?.summary?.totalAgents || 0, change: `${data?.summary?.activeAgents || 0} active`, icon: Bot, iconColor: 'text-primary', globalIconKey: 'global_agent' },
        { label: 'Avg Success Rate', value: `${(data?.summary?.avgSuccessRate || 0).toFixed(0)}%`, change: 'Across all agents', icon: CheckCircle, iconColor: 'text-[hsl(var(--success))]', globalIconKey: 'global_performance' },
        { label: 'Total Tokens', value: formatNumber(data?.summary?.totalTokens || 0), change: 'This period', icon: Zap, iconColor: 'text-[hsl(var(--info))]' },
        { label: 'Total Cost', value: `$${(data?.summary?.totalCost || 0).toFixed(2)}`, change: 'This period', icon: DollarSign, iconColor: 'text-[hsl(var(--agent))]', globalIconKey: 'global_cost' },
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
        <div className="overflow-x-auto max-h-[500px] overflow-y-auto">
          <table className="w-full">
            <thead className="sticky top-0 bg-card z-10">
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
                  <td colSpan={8}>
                    <div className="flex flex-col items-center justify-center py-12 gap-2 text-muted-foreground">
                      <Bot className="w-8 h-8 opacity-50" />
                      <p className="text-sm">No agents found</p>
                    </div>
                  </td>
                </tr>
              ) : (
                sortedAgents.map((agent) => {
                  const isExpanded = expandedAgent === agent.id.toString()
                  return (
                    <Fragment key={agent.id}>
                      <motion.tr
                        className="border-b border-border/30 hover:bg-secondary/20 cursor-pointer transition-colors"
                        onClick={() => setExpandedAgent(isExpanded ? null : agent.id.toString())}
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                      >
                        <td className="p-4">
                          <div className="flex items-center gap-2">
                            {isExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                            <span className="font-medium">{agent.name}</span>
                          </div>
                        </td>
                        <td className="p-4 hidden md:table-cell">
                          <Badge variant="outline" className={agent.status === 'active' ? 'text-[hsl(var(--success))] border-[hsl(var(--success))]/30' : 'text-muted-foreground'}>
                            {agent.status}
                          </Badge>
                        </td>
                        <td className="p-4">
                          <span className={agent.successRate < 80 && agent.successRate > 0 ? 'text-[hsl(var(--destructive))]' : ''}>
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
                      <AnimatePresence>
                        {isExpanded && (
                          <motion.tr
                            initial={{ opacity: 0, height: 0 }}
                            animate={{ opacity: 1, height: 'auto' }}
                            exit={{ opacity: 0, height: 0 }}
                            transition={{ duration: 0.2 }}
                            className="border-b border-border/30"
                          >
                            <td colSpan={8} className="p-0">
                              <motion.div
                                initial={{ opacity: 0 }}
                                animate={{ opacity: 1 }}
                                exit={{ opacity: 0 }}
                                className="p-5 bg-secondary/10"
                              >
                                <AgentExpandedPanel agent={agent} />
                              </motion.div>
                            </td>
                          </motion.tr>
                        )}
                      </AnimatePresence>
                    </Fragment>
                  )
                })
              )}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}
