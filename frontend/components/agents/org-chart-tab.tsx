'use client'

import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Loader2, Network, Rocket, Users } from 'lucide-react'
import { apiClient } from '@/lib/api-client'
import { OrgChartCanvas, type OrgChartAgent } from './org-chart-canvas'

interface OrgChartResponse {
  success: boolean
  nodes: OrgChartAgent[]
  edges: Array<{ from: number; to: number }>
  teams: string[]
}

export function OrgChartTab() {
  const [selectedTeam, setSelectedTeam] = useState<string | null>(null)

  const { data, isLoading, error } = useQuery<OrgChartResponse>({
    queryKey: ['org-chart'],
    queryFn: async () => {
      return apiClient.request('/api/agents/org-chart')
    },
  })

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-96">
        <Loader2 className="w-6 h-6 animate-spin text-[hsl(var(--info))]" />
        <span className="ml-2 text-sm text-muted-foreground">Loading org chart...</span>
      </div>
    )
  }

  if (error || !data?.success) {
    return (
      <div className="flex flex-col items-center justify-center h-96 text-muted-foreground">
        <Network className="w-8 h-8 mb-2 text-muted-foreground" />
        <p className="text-sm">Failed to load org chart</p>
      </div>
    )
  }

  const hasOrgData = data.nodes.some(n => n.team || n.job_title)

  // Filter by team if selected
  const filteredNodes = selectedTeam
    ? data.nodes.filter(n => n.team === selectedTeam)
    : data.nodes
  const filteredNodeIds = new Set(filteredNodes.map(n => n.id))
  const filteredEdges = data.edges.filter(
    e => filteredNodeIds.has(e.from) && filteredNodeIds.has(e.to),
  )

  return (
    <div className="space-y-4">
      {/* Header bar */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Users className="w-4 h-4" />
            <span>{data.nodes.length} agents</span>
            {data.teams.length > 0 && (
              <>
                <span className="text-muted-foreground">|</span>
                <span>{data.teams.length} teams</span>
              </>
            )}
          </div>
        </div>

        {/* Team filter chips */}
        {data.teams.length > 0 && (
          <div className="flex items-center gap-1.5">
            <button
              onClick={() => setSelectedTeam(null)}
              className={`px-2.5 py-1 rounded-full text-xs font-medium transition-colors ${
                selectedTeam === null
                  ? 'bg-primary/20 text-primary border border-primary/30'
                  : 'bg-card text-muted-foreground border border-border hover:border-border'
              }`}
            >
              All
            </button>
            {data.teams.map(team => (
              <button
                key={team}
                onClick={() => setSelectedTeam(selectedTeam === team ? null : team)}
                className={`px-2.5 py-1 rounded-full text-xs font-medium transition-colors ${
                  selectedTeam === team
                    ? 'bg-primary/20 text-primary border border-primary/30'
                    : 'bg-card text-muted-foreground border border-border hover:border-border'
                }`}
              >
                {team}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Empty state — ask Auto to build the team */}
      {!hasOrgData && (
        <div className="rounded-xl border border-dashed border-border bg-card p-8 text-center">
          <Rocket className="w-10 h-10 mx-auto mb-3 text-primary/60" />
          <h3 className="text-lg font-semibold text-foreground mb-1">
            No org structure yet
          </h3>
          <p className="text-sm text-muted-foreground mb-4 max-w-md mx-auto">
            Ask <strong className="text-primary">Auto</strong> to design your AI company structure.
            He&apos;ll audit your roster, browse the marketplace, and build your team.
          </p>
          <a
            href="/chat"
            className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-primary/20 text-primary text-sm font-medium hover:bg-primary/30 transition-colors border border-primary/30"
          >
            <Rocket className="w-4 h-4" />
            Ask Auto to build your team
          </a>
        </div>
      )}

      {/* Org Chart Canvas */}
      <div className="rounded-xl border border-border bg-card overflow-hidden" style={{ height: 600 }}>
        <OrgChartCanvas
          agents={filteredNodes}
          edges={filteredEdges}
          onAgentSelect={(id) => {
            // Could open agent details modal
          }}
          className="w-full h-full"
        />
      </div>
    </div>
  )
}
