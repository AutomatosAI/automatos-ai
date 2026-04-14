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
        <Loader2 className="w-6 h-6 animate-spin text-zinc-400" />
        <span className="ml-2 text-sm text-zinc-400">Loading org chart...</span>
      </div>
    )
  }

  if (error || !data?.success) {
    return (
      <div className="flex flex-col items-center justify-center h-96 text-zinc-400">
        <Network className="w-8 h-8 mb-2 text-zinc-600" />
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
          <div className="flex items-center gap-2 text-sm text-zinc-400">
            <Users className="w-4 h-4" />
            <span>{data.nodes.length} agents</span>
            {data.teams.length > 0 && (
              <>
                <span className="text-zinc-600">|</span>
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
                  ? 'bg-orange-500/20 text-orange-400 border border-orange-500/30'
                  : 'bg-zinc-800 text-zinc-400 border border-zinc-700/50 hover:border-zinc-600'
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
                    ? 'bg-orange-500/20 text-orange-400 border border-orange-500/30'
                    : 'bg-zinc-800 text-zinc-400 border border-zinc-700/50 hover:border-zinc-600'
                }`}
              >
                {team}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Empty state — prompt Mission Zero */}
      {!hasOrgData && (
        <div className="rounded-xl border border-dashed border-zinc-700/50 bg-zinc-900/50 p-8 text-center">
          <Rocket className="w-10 h-10 mx-auto mb-3 text-orange-500/60" />
          <h3 className="text-lg font-semibold text-zinc-200 mb-1">
            No org structure yet
          </h3>
          <p className="text-sm text-zinc-400 mb-4 max-w-md mx-auto">
            Run <strong className="text-orange-400">Mission Zero</strong> to design your AI company structure.
            The CTO agent will audit your roster, browse the marketplace, and build your team.
          </p>
          <a
            href="/activity?tab=missions"
            className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-orange-500/20 text-orange-400 text-sm font-medium hover:bg-orange-500/30 transition-colors border border-orange-500/30"
          >
            <Rocket className="w-4 h-4" />
            Launch Mission Zero
          </a>
        </div>
      )}

      {/* Org Chart Canvas */}
      <div className="rounded-xl border border-zinc-800/50 bg-zinc-950/50 overflow-hidden" style={{ height: 600 }}>
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
