'use client'

import { motion } from 'framer-motion'
import {
  Shield,
  Users,
  DollarSign,
  Zap,
  Activity,
  Download,
  AlertTriangle,
  Building,
} from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Skeleton } from '@/components/ui/skeleton'
import { useAdminWorkspaceAnalytics } from '@/hooks/use-unified-analytics'

interface Props {
  days: number
}

function formatNumber(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`
  return n.toFixed(0)
}

export function AnalyticsAdmin({ days }: Props) {
  const { data, isLoading } = useAdminWorkspaceAnalytics(days)

  const escapeCsvField = (value: string): string => {
    // Neutralize formula-characters to prevent CSV injection
    let escaped = value
    if (/^[=+\-@]/.test(escaped)) {
      escaped = "'" + escaped
    }
    // Double internal quotes and wrap in quotes
    return '"' + escaped.replace(/"/g, '""') + '"'
  }

  const handleExport = () => {
    if (!data?.workspaces) return

    const headers = ['Workspace,Plan,Users,Agents,Workflows,API Calls,Tokens,Cost,Status']
    const rows = data.workspaces.map(ws =>
      [
        escapeCsvField(ws.name),
        escapeCsvField(ws.plan),
        ws.users,
        ws.agents,
        ws.workflows,
        ws.apiCalls,
        ws.tokens,
        `$${ws.cost.toFixed(2)}`,
        escapeCsvField(ws.status),
      ].join(',')
    )
    const csv = [...headers, ...rows].join('\n')
    const blob = new Blob([csv], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `workspace-analytics-${new Date().toISOString().split('T')[0]}.csv`
    a.click()
    URL.revokeObjectURL(url)
  }

  if (isLoading) {
    return (
      <div className="space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-5 gap-6">
          {Array.from({ length: 5 }).map((_, i) => (
            <Card key={i} className="glass-card"><CardContent className="p-6"><Skeleton className="h-8 w-16 mb-2" /><Skeleton className="h-4 w-24" /></CardContent></Card>
          ))}
        </div>
        <Skeleton className="h-64 w-full" />
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Admin Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Shield className="w-5 h-5 text-red-400" />
          <span className="text-sm font-medium text-muted-foreground">Super Admin View</span>
        </div>
        <Button variant="outline" size="sm" onClick={handleExport}>
          <Download className="w-4 h-4 mr-2" />
          Export CSV
        </Button>
      </div>

      {/* Platform Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-6">
        {[
          { label: 'Workspaces', value: data?.platformSummary?.totalWorkspaces || 0, icon: Building, color: 'text-blue-400' },
          { label: 'Total Users', value: data?.platformSummary?.totalUsers || 0, icon: Users, color: 'text-green-400' },
          { label: 'API Calls', value: formatNumber(data?.platformSummary?.totalApiCalls || 0), icon: Activity, color: 'text-purple-400' },
          { label: 'Total Tokens', value: formatNumber(data?.platformSummary?.totalTokens || 0), icon: Zap, color: 'text-orange-400' },
          { label: 'Platform Cost', value: `$${(data?.platformSummary?.totalCost || 0).toFixed(2)}`, icon: DollarSign, color: 'text-red-400' },
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

      {/* Workspace Table */}
      <Card className="glass-card overflow-hidden">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Building className="w-5 h-5 text-blue-400" />
            Workspace Usage
          </CardTitle>
        </CardHeader>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-border/50">
                <th className="text-left p-4 text-xs font-medium text-muted-foreground">Workspace</th>
                <th className="text-left p-4 text-xs font-medium text-muted-foreground">Plan</th>
                <th className="text-left p-4 text-xs font-medium text-muted-foreground hidden md:table-cell">Users</th>
                <th className="text-left p-4 text-xs font-medium text-muted-foreground">Agents</th>
                <th className="text-left p-4 text-xs font-medium text-muted-foreground hidden md:table-cell">Workflows</th>
                <th className="text-left p-4 text-xs font-medium text-muted-foreground">API Calls</th>
                <th className="text-left p-4 text-xs font-medium text-muted-foreground hidden lg:table-cell">Tokens</th>
                <th className="text-left p-4 text-xs font-medium text-muted-foreground">Cost</th>
                <th className="text-left p-4 text-xs font-medium text-muted-foreground hidden lg:table-cell">Status</th>
              </tr>
            </thead>
            <tbody>
              {(!data?.workspaces || data.workspaces.length === 0) ? (
                <tr>
                  <td colSpan={9} className="p-12 text-center text-muted-foreground">
                    <Building className="w-10 h-10 mx-auto mb-3 opacity-50" />
                    <p>No workspace data available</p>
                  </td>
                </tr>
              ) : (
                data.workspaces.map((ws) => (
                  <tr key={ws.id} className="border-b border-border/30 hover:bg-secondary/20 transition-colors">
                    <td className="p-4 font-medium">{ws.name}</td>
                    <td className="p-4">
                      <Badge variant="outline" className="text-xs">{ws.plan}</Badge>
                    </td>
                    <td className="p-4 text-sm hidden md:table-cell">{ws.users}</td>
                    <td className="p-4 text-sm">{ws.agents}</td>
                    <td className="p-4 text-sm hidden md:table-cell">{ws.workflows}</td>
                    <td className="p-4 text-sm">{formatNumber(ws.apiCalls)}</td>
                    <td className="p-4 text-sm hidden lg:table-cell">{formatNumber(ws.tokens)}</td>
                    <td className="p-4 text-sm font-medium">${ws.cost.toFixed(2)}</td>
                    <td className="p-4 hidden lg:table-cell">
                      <Badge variant="outline" className={ws.status === 'active' ? 'text-green-400 border-green-400/30' : ''}>
                        {ws.status}
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
