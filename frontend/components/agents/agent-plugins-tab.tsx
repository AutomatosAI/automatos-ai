'use client'

import { useState, useEffect, useCallback } from 'react'
import { Sparkles, Terminal, Zap, Shield, ExternalLink, AlertCircle, RefreshCw } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { apiClient } from '@/lib/api-client'

interface AgentPluginsProps {
  agents: any[]
  selectedAgentId: string | null
  onAgentSelect: (agentId: string | null) => void
}

export function AgentPlugins({ agents, selectedAgentId, onAgentSelect }: AgentPluginsProps) {
  const [plugins, setPlugins] = useState<any[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [fetchKey, setFetchKey] = useState(0)

  const fetchPlugins = useCallback(() => {
    if (!selectedAgentId) {
      setPlugins([])
      setError(null)
      return
    }
    let mounted = true
    setLoading(true)
    setError(null)
    apiClient.request<any>(`/api/agents/${selectedAgentId}/plugins`, { method: 'GET' })
      .then((data) => {
        if (!mounted) return
        const items = Array.isArray(data) ? data : (data?.items || data?.plugins || [])
        setPlugins(items)
      })
      .catch((err) => {
        if (mounted) {
          setError(err?.message || 'Failed to load plugins')
        }
      })
      .finally(() => { if (mounted) setLoading(false) })
    return () => { mounted = false }
  }, [selectedAgentId, fetchKey])

  useEffect(() => {
    fetchPlugins()
  }, [fetchPlugins])

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">Capability Management</h2>
          <p className="text-muted-foreground">
            View and manage capabilities assigned to your agents
          </p>
        </div>
        <Select
          value={selectedAgentId || ''}
          onValueChange={(value) => onAgentSelect(value)}
        >
          <SelectTrigger className="w-[220px]">
            <SelectValue placeholder="Select an agent" />
          </SelectTrigger>
          <SelectContent>
            {agents.map((agent) => (
              <SelectItem key={agent.id} value={agent.id.toString()}>
                {agent.name}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      {!selectedAgentId ? (
        <Card className="glass-card">
          <CardContent className="p-12 text-center">
            <Sparkles className="w-16 h-16 mx-auto text-muted-foreground mb-4" />
            <h3 className="text-lg font-semibold mb-2">Select an Agent</h3>
            <p className="text-muted-foreground">
              Choose an agent above to view its assigned capabilities
            </p>
          </CardContent>
        </Card>
      ) : loading ? (
        <div className="flex items-center justify-center py-12">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
        </div>
      ) : error ? (
        <Card className="glass-card border-destructive/20">
          <CardContent className="p-12 text-center">
            <AlertCircle className="w-16 h-16 mx-auto text-destructive mb-4" />
            <h3 className="text-lg font-semibold mb-2">Failed to Load Capabilities</h3>
            <p className="text-muted-foreground mb-4">{error}</p>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setFetchKey((k) => k + 1)}
            >
              <RefreshCw className="w-4 h-4 mr-2" />
              Retry
            </Button>
          </CardContent>
        </Card>
      ) : plugins.length > 0 ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {plugins.map((plugin: any) => (
            <Card key={plugin.plugin_id} className="glass-card hover:border-primary/20 transition-all">
              <CardHeader className="pb-3">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-base flex items-center gap-2">
                    <Sparkles className="w-5 h-5 text-primary" />
                    {plugin.name}
                  </CardTitle>
                  <Badge variant="outline" className="text-xs">v{plugin.version}</Badge>
                </div>
              </CardHeader>
              <CardContent className="space-y-3">
                <p className="text-sm text-muted-foreground line-clamp-2">
                  {plugin.description || 'No description available'}
                </p>
                <div className="flex items-center gap-4 text-xs text-muted-foreground">
                  <span className="flex items-center gap-1">
                    <Terminal className="w-3 h-3" />
                    {plugin.skills_count || 0} skills
                  </span>
                  <span className="flex items-center gap-1">
                    <Zap className="w-3 h-3" />
                    {plugin.commands_count || 0} commands
                  </span>
                </div>
                {plugin.security_status === 'safe' && (
                  <Badge variant="secondary" className="text-xs text-success border-success/30">
                    <Shield className="w-3 h-3 mr-1" />
                    Verified
                  </Badge>
                )}
              </CardContent>
            </Card>
          ))}
        </div>
      ) : (
        <Card className="glass-card">
          <CardContent className="p-12 text-center">
            <Sparkles className="w-16 h-16 mx-auto text-muted-foreground mb-4" />
            <h3 className="text-lg font-semibold mb-2">No Capabilities Assigned</h3>
            <p className="text-muted-foreground mb-4">
              This agent doesn't have any capabilities yet. Assign capabilities from the agent's configuration.
            </p>
            <Button
              variant="outline"
              size="sm"
              onClick={() => window.location.href = '/marketplace'}
            >
              <ExternalLink className="w-4 h-4 mr-2" />
              Browse Marketplace
            </Button>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
