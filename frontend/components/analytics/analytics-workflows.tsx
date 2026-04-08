'use client'

import { useState, useMemo, Fragment } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
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
  ChefHat,
  Star,
  Shield,
} from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Skeleton } from '@/components/ui/skeleton'
import { Progress } from '@/components/ui/progress'
import { StatsBar } from '@/components/shared/stats-bar'
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, Legend } from 'recharts'
import { useWorkflowAnalytics } from '@/hooks/use-unified-analytics'

interface Props {
  days: number
}

type SortField = 'name' | 'totalRuns' | 'successRate' | 'avgDuration' | 'cost'
type RecipeSortField = 'name' | 'useCount' | 'successRate' | 'qualityScore'

export function AnalyticsWorkflows({ days }: Props) {
  const { data, isLoading } = useWorkflowAnalytics(days)
  const [sortField, setSortField] = useState<SortField>('totalRuns')
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('desc')
  const [expandedWf, setExpandedWf] = useState<string | null>(null)

  // Recipe sorting state
  const [recipeSortField, setRecipeSortField] = useState<RecipeSortField>('useCount')
  const [recipeSortDir, setRecipeSortDir] = useState<'asc' | 'desc'>('desc')
  const [expandedRecipe, setExpandedRecipe] = useState<number | null>(null)

  const toggleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDir(prev => prev === 'asc' ? 'desc' : 'asc')
    } else {
      setSortField(field)
      setSortDir('desc')
    }
  }

  const toggleRecipeSort = (field: RecipeSortField) => {
    if (recipeSortField === field) {
      setRecipeSortDir(prev => prev === 'asc' ? 'desc' : 'asc')
    } else {
      setRecipeSortField(field)
      setRecipeSortDir('desc')
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

  const sortedRecipes = useMemo(() => {
    if (!data?.recipes) return []
    return [...data.recipes].sort((a: any, b: any) => {
      const aVal = a[recipeSortField] ?? 0
      const bVal = b[recipeSortField] ?? 0
      if (typeof aVal === 'string' && typeof bVal === 'string') {
        return recipeSortDir === 'asc' ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal)
      }
      return recipeSortDir === 'asc'
        ? (Number(aVal) || 0) - (Number(bVal) || 0)
        : (Number(bVal) || 0) - (Number(aVal) || 0)
    })
  }, [data?.recipes, recipeSortField, recipeSortDir])

  const SortHeader = ({ field, children }: { field: SortField; children: React.ReactNode }) => (
    <button
      onClick={() => toggleSort(field)}
      className="flex items-center gap-1 text-xs font-medium text-muted-foreground hover:text-foreground transition-colors"
    >
      {children}
      <ArrowUpDown className="w-3 h-3" />
    </button>
  )

  const RecipeSortHeader = ({ field, children }: { field: RecipeSortField; children: React.ReactNode }) => (
    <button
      onClick={() => toggleRecipeSort(field)}
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
      {/* ===== WORKFLOWS SECTION ===== */}
      <StatsBar stats={[
        { label: 'Total Missions', value: data?.summary?.totalWorkflows || 0, icon: GitBranch, iconColor: 'text-primary', globalIconKey: 'global_workflow' },
        { label: 'Executions', value: data?.summary?.totalExecutions || 0, icon: Activity, iconColor: 'text-[hsl(var(--info))]', globalIconKey: 'global_activity' },
        { label: 'Success Rate', value: `${(data?.summary?.successRate || 0).toFixed(0)}%`, icon: CheckCircle, iconColor: 'text-[hsl(var(--success))]', globalIconKey: 'global_performance' },
        { label: 'Avg Duration', value: data?.summary?.avgDuration || '0s', icon: Clock, iconColor: 'text-[hsl(var(--agent))]' },
      ]} loading={isLoading} />

      {/* Execution Trend Chart */}
      {data?.stats && (data.stats as any)?.recent_executions?.length > 0 && (
        <Card className="glass-card">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Activity className="w-5 h-5 text-[hsl(var(--info))]" />
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
                    <p>No missions found</p>
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
                      <span className={wf.successRate < 80 && wf.successRate > 0 ? 'text-[hsl(var(--destructive))]' : ''}>
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
                        wf.status === 'active' || wf.status === 'completed' ? 'text-[hsl(var(--success))] border-[hsl(var(--success))]/30' :
                        wf.status === 'failed' ? 'text-[hsl(var(--destructive))] border-[hsl(var(--destructive))]/30' :
                        wf.status === 'running' || wf.status === 'executing' ? 'text-[hsl(var(--info))] border-[hsl(var(--info))]/30' : ''
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

      {/* ===== RECIPES SECTION ===== */}
      <div className="pt-2">
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <ChefHat className="w-5 h-5 text-primary" />
          Recipes
        </h3>

        <StatsBar stats={[
          { label: 'Total Recipes', value: data?.recipeSummary?.totalRecipes || 0, icon: ChefHat, iconColor: 'text-primary', globalIconKey: 'global_recipe' },
          { label: 'Executions', value: data?.recipeSummary?.totalExecutions || 0, icon: Activity, iconColor: 'text-[hsl(var(--info))]', globalIconKey: 'global_activity' },
          { label: 'Avg Quality', value: `${((data?.recipeSummary?.avgQualityScore || 0) * 100).toFixed(0)}%`, icon: CheckCircle, iconColor: 'text-[hsl(var(--success))]', globalIconKey: 'global_performance' },
          { label: 'Avg Success Rate', value: `${(data?.recipeSummary?.avgSuccessRate || 0).toFixed(0)}%`, icon: Zap, iconColor: 'text-[hsl(var(--agent))]', globalIconKey: 'global_performance' },
        ]} loading={isLoading} />
      </div>

      {/* Recipe Table */}
      <Card className="glass-card overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-border/50">
                <th className="text-left p-4"><RecipeSortHeader field="name">Name</RecipeSortHeader></th>
                <th className="text-left p-4"><span className="text-xs font-medium text-muted-foreground">Steps</span></th>
                <th className="text-left p-4"><RecipeSortHeader field="useCount">Runs</RecipeSortHeader></th>
                <th className="text-left p-4"><RecipeSortHeader field="successRate">Success Rate</RecipeSortHeader></th>
                <th className="text-left p-4 hidden md:table-cell"><RecipeSortHeader field="qualityScore">Quality</RecipeSortHeader></th>
                <th className="text-left p-4 hidden lg:table-cell"><span className="text-xs font-medium text-muted-foreground">Tags</span></th>
                <th className="text-left p-4 hidden lg:table-cell"><span className="text-xs font-medium text-muted-foreground">Last Used</span></th>
              </tr>
            </thead>
            <tbody>
              {sortedRecipes.length === 0 ? (
                <tr>
                  <td colSpan={7} className="p-12 text-center text-muted-foreground">
                    <ChefHat className="w-10 h-10 mx-auto mb-3 opacity-50" />
                    <p>No recipes found</p>
                  </td>
                </tr>
              ) : (
                sortedRecipes.map((recipe: any) => (
                  <Fragment key={recipe.id}>
                    <tr
                      className="border-b border-border/30 hover:bg-secondary/20 cursor-pointer transition-colors"
                      onClick={() => setExpandedRecipe(expandedRecipe === recipe.id ? null : recipe.id)}
                    >
                      <td className="p-4">
                        <div className="flex items-center gap-2">
                          {expandedRecipe === recipe.id ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                          <span className="font-medium">{recipe.name}</span>
                          {recipe.isSystem && <Shield className="w-3.5 h-3.5 text-[hsl(var(--info))]" />}
                          {recipe.isFeatured && <Star className="w-3.5 h-3.5 text-[hsl(var(--warning))] fill-[hsl(var(--warning))]" />}
                        </div>
                      </td>
                      <td className="p-4 text-muted-foreground">{recipe.stepsCount}</td>
                      <td className="p-4">{recipe.useCount}</td>
                      <td className="p-4">
                        <span className={recipe.successRate < 80 && recipe.successRate > 0 ? 'text-[hsl(var(--destructive))]' : ''}>
                          {recipe.successRate > 0 ? `${recipe.successRate.toFixed(0)}%` : '-'}
                        </span>
                      </td>
                      <td className="p-4 hidden md:table-cell">
                        {recipe.qualityScore != null ? `${(recipe.qualityScore * 100).toFixed(0)}%` : '-'}
                      </td>
                      <td className="p-4 hidden lg:table-cell">
                        <div className="flex gap-1 flex-wrap">
                          {(recipe.tags || []).slice(0, 2).map((tag: string) => (
                            <Badge key={tag} variant="secondary" className="text-[10px] px-1.5 py-0">
                              {tag}
                            </Badge>
                          ))}
                          {(recipe.tags || []).length > 2 && (
                            <span className="text-[10px] text-muted-foreground">+{recipe.tags.length - 2}</span>
                          )}
                        </div>
                      </td>
                      <td className="p-4 hidden lg:table-cell text-sm text-muted-foreground">
                        {recipe.lastUsedAt ? new Date(recipe.lastUsedAt).toLocaleDateString() : '-'}
                      </td>
                    </tr>
                    <AnimatePresence>
                      {expandedRecipe === recipe.id && (
                        <tr>
                          <td colSpan={7} className="p-0">
                            <motion.div
                              initial={{ height: 0, opacity: 0 }}
                              animate={{ height: 'auto', opacity: 1 }}
                              exit={{ height: 0, opacity: 0 }}
                              transition={{ duration: 0.2 }}
                              className="overflow-hidden"
                            >
                              <div className="px-6 py-4 bg-secondary/10 border-b border-border/30 space-y-3">
                                {recipe.description && (
                                  <p className="text-sm text-muted-foreground">{recipe.description}</p>
                                )}
                                {recipe.tags?.length > 0 && (
                                  <div className="flex gap-1.5 flex-wrap">
                                    {recipe.tags.map((tag: string) => (
                                      <Badge key={tag} variant="outline" className="text-xs">
                                        {tag}
                                      </Badge>
                                    ))}
                                  </div>
                                )}
                                {recipe.qualityScore != null && (
                                  <div className="flex items-center gap-3 max-w-xs">
                                    <span className="text-xs text-muted-foreground whitespace-nowrap">Quality</span>
                                    <Progress value={recipe.qualityScore * 100} className="h-2" />
                                    <span className="text-xs font-medium">{(recipe.qualityScore * 100).toFixed(0)}%</span>
                                  </div>
                                )}
                                <div className="flex gap-2">
                                  {recipe.isSystem && (
                                    <Badge variant="outline" className="text-[hsl(var(--info))] border-[hsl(var(--info))]/30 text-xs">
                                      <Shield className="w-3 h-3 mr-1" /> System
                                    </Badge>
                                  )}
                                  {recipe.isFeatured && (
                                    <Badge variant="outline" className="text-[hsl(var(--warning))] border-[hsl(var(--warning))]/30 text-xs">
                                      <Star className="w-3 h-3 mr-1" /> Featured
                                    </Badge>
                                  )}
                                </div>
                              </div>
                            </motion.div>
                          </td>
                        </tr>
                      )}
                    </AnimatePresence>
                  </Fragment>
                ))
              )}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}
