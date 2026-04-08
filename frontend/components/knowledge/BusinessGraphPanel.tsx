'use client'

import { useState, useMemo, useCallback } from 'react'
import { useQuery } from '@tanstack/react-query'
import { apiClient } from '@/lib/api-client'
import { useWorkspace } from '@/hooks/use-workspace'
import { Input } from '@/components/ui/input'
import { Slider } from '@/components/ui/slider'
import { Badge } from '@/components/ui/badge'
import { BusinessGraphVisualization } from './BusinessGraphVisualization'
import {
  Network, Loader2, Search, X, ChevronRight,
  FileText, Clock, Layers,
} from 'lucide-react'
import { formatDistanceToNow } from 'date-fns'

// ─── Types ──────────────────────────────────────────────

interface GraphNode {
  id: string
  label: string
  file_type?: string
  source_file?: string
  community?: number
  x?: number
  y?: number
}

interface GraphEdge {
  source: string
  target: string
  relation: string
  confidence: string
  confidence_score: number
}

interface GraphData {
  nodes: GraphNode[]
  links: GraphEdge[]
}

interface GraphMeta {
  node_count: number
  edge_count: number
  community_count: number
  last_built: number | null
}

// ─── Query Keys ─────────────────────────────────────────

const graphQueryKeys = {
  data: (wsId: string) => ['business-graph', 'data', wsId] as const,
  meta: (wsId: string) => ['business-graph', 'meta', wsId] as const,
}

// ─── Component ──────────────────────────────────────────

export function BusinessGraphPanel() {
  const { workspaceId } = useWorkspace()

  // UI state
  const [searchTerm, setSearchTerm] = useState('')
  const [confidenceMin, setConfidenceMin] = useState(0)
  const [selectedCommunity, setSelectedCommunity] = useState<number | null>(null)
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null)

  // ── Data fetching ──

  const {
    data: graphData,
    isLoading: graphLoading,
    error: graphError,
  } = useQuery<GraphData>({
    queryKey: graphQueryKeys.data(workspaceId ?? ''),
    queryFn: () => apiClient.getWorkspaceFileContent(workspaceId!, 'graph/graph.json'),
    enabled: !!workspaceId,
    staleTime: 5 * 60 * 1000,
  })

  const {
    data: meta,
    isLoading: metaLoading,
  } = useQuery<GraphMeta>({
    queryKey: graphQueryKeys.meta(workspaceId ?? ''),
    queryFn: () => apiClient.getWorkspaceFileContent(workspaceId!, 'graph/meta.json'),
    enabled: !!workspaceId,
    staleTime: 5 * 60 * 1000,
  })

  // ── Derived data ──

  const communities = useMemo(() => {
    if (!graphData?.nodes) return []
    const map = new Map<number, number>()
    for (const n of graphData.nodes) {
      if (n.community != null) {
        map.set(n.community, (map.get(n.community) ?? 0) + 1)
      }
    }
    return Array.from(map.entries())
      .sort((a, b) => b[1] - a[1])
      .map(([id, count]) => ({ id, count }))
  }, [graphData])

  const filteredData = useMemo((): GraphData | null => {
    if (!graphData) return null

    const lowerSearch = searchTerm.toLowerCase()

    const filteredNodes = graphData.nodes.filter((n) => {
      if (selectedCommunity != null && n.community !== selectedCommunity) return false
      if (lowerSearch && !n.label.toLowerCase().includes(lowerSearch)) return false
      return true
    })

    const nodeIds = new Set(filteredNodes.map((n) => n.id))

    const filteredEdges = graphData.links.filter(
      (e) =>
        e.confidence_score >= confidenceMin &&
        nodeIds.has(e.source) &&
        nodeIds.has(e.target)
    )

    return { nodes: filteredNodes, links: filteredEdges }
  }, [graphData, searchTerm, confidenceMin, selectedCommunity])

  const connectedEdges = useMemo(() => {
    if (!selectedNode || !graphData?.links) return []
    return graphData.links.filter(
      (e) => e.source === selectedNode.id || e.target === selectedNode.id
    )
  }, [selectedNode, graphData])

  const getNodeLabel = useCallback(
    (id: string) => graphData?.nodes.find((n) => n.id === id)?.label ?? id,
    [graphData]
  )

  const handleNodeSelect = useCallback((node: GraphNode | null) => {
    setSelectedNode(node)
  }, [])

  // ── Loading state ──

  const isLoading = graphLoading || metaLoading

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-24">
        <Loader2 className="w-6 h-6 animate-spin text-primary" />
        <span className="ml-2 text-sm text-muted-foreground">Loading business graph...</span>
      </div>
    )
  }

  // ── Empty state ──

  if (graphError || !graphData?.nodes?.length) {
    return (
      <div className="text-center py-24">
        <Network className="w-12 h-12 text-muted-foreground mx-auto mb-4 opacity-50" />
        <h3 className="text-lg font-semibold mb-2">No business graph yet</h3>
        <p className="text-sm text-muted-foreground">
          Upload documents to get started.
        </p>
      </div>
    )
  }

  // ── Stats ──

  const nodeCount = meta?.node_count ?? graphData.nodes.length
  const edgeCount = meta?.edge_count ?? graphData.links.length
  const communityCount = meta?.community_count ?? communities.length
  const lastBuilt = meta?.last_built
    ? formatDistanceToNow(new Date(meta.last_built * 1000), { addSuffix: true })
    : null

  return (
    <div className="space-y-4">
      {/* Stats Bar */}
      <div className="glass-card bg-white/5 backdrop-blur-sm border border-white/10 rounded-lg px-4 py-3 flex flex-wrap items-center gap-x-6 gap-y-1 text-sm">
        <span className="flex items-center gap-1.5">
          <span className="font-semibold text-blue-400">{nodeCount.toLocaleString()}</span>
          <span className="text-muted-foreground">nodes</span>
        </span>
        <span className="flex items-center gap-1.5">
          <span className="font-semibold text-green-400">{edgeCount.toLocaleString()}</span>
          <span className="text-muted-foreground">edges</span>
        </span>
        <span className="flex items-center gap-1.5">
          <span className="font-semibold text-purple-400">{communityCount}</span>
          <span className="text-muted-foreground">clusters</span>
        </span>
        {lastBuilt && (
          <span className="flex items-center gap-1.5 text-muted-foreground ml-auto">
            <Clock className="w-3.5 h-3.5" />
            Last built {lastBuilt}
          </span>
        )}
      </div>

      {/* Main Layout: Sidebar + Graph */}
      <div className="flex flex-col md:flex-row gap-4 min-h-[500px]">
        {/* Community Sidebar */}
        <div className="w-full md:w-48 shrink-0 glass-card bg-white/5 backdrop-blur-sm border border-white/10 rounded-lg p-3 space-y-1 max-h-[500px] overflow-y-auto">
          <div className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-2 flex items-center gap-1.5">
            <Layers className="w-3.5 h-3.5" />
            Communities
          </div>

          <button
            onClick={() => setSelectedCommunity(null)}
            className={`w-full text-left text-sm px-2 py-1.5 rounded transition-colors ${
              selectedCommunity === null
                ? 'bg-primary/20 text-primary'
                : 'text-muted-foreground hover:bg-white/5'
            }`}
          >
            All clusters
          </button>

          {communities.map(({ id, count }) => (
            <button
              key={id}
              onClick={() =>
                setSelectedCommunity(selectedCommunity === id ? null : id)
              }
              className={`w-full text-left text-sm px-2 py-1.5 rounded flex items-center justify-between transition-colors ${
                selectedCommunity === id
                  ? 'bg-primary/20 text-primary'
                  : 'text-muted-foreground hover:bg-white/5'
              }`}
            >
              <span>Cluster {id}</span>
              <Badge variant="secondary" className="text-[10px] h-5">
                {count}
              </Badge>
            </button>
          ))}
        </div>

        {/* Graph Visualization */}
        <div className="flex-1 glass-card bg-white/5 backdrop-blur-sm border border-white/10 rounded-lg overflow-hidden min-h-[400px]">
          <BusinessGraphVisualization
            graphData={filteredData ?? { nodes: [], links: [] }}
            onNodeSelect={handleNodeSelect}
            selectedCommunity={selectedCommunity}
            minConfidence={confidenceMin}
          />
        </div>
      </div>

      {/* Search + Confidence Controls */}
      <div className="glass-card bg-white/5 backdrop-blur-sm border border-white/10 rounded-lg px-4 py-3 flex flex-col sm:flex-row items-start sm:items-center gap-4">
        <div className="relative flex-1 w-full sm:max-w-xs">
          <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
          <Input
            placeholder="Search nodes..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="pl-9 bg-transparent border-white/10"
          />
          {searchTerm && (
            <button
              onClick={() => setSearchTerm('')}
              className="absolute right-2.5 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          )}
        </div>

        <div className="flex items-center gap-3 flex-1 w-full sm:w-auto">
          <span className="text-xs text-muted-foreground whitespace-nowrap">
            Confidence:
          </span>
          <Slider
            min={0}
            max={1}
            step={0.05}
            value={[confidenceMin]}
            onValueChange={([val]) => setConfidenceMin(val)}
            className="flex-1 max-w-[200px]"
          />
          <span className="text-xs font-mono w-8 text-right">
            {confidenceMin.toFixed(2)}
          </span>
        </div>
      </div>

      {/* Node Detail Panel */}
      {selectedNode && (
        <div className="glass-card bg-white/5 backdrop-blur-sm border border-white/10 rounded-lg p-4 space-y-3">
          <div className="flex items-start justify-between">
            <div className="space-y-1">
              <h4 className="font-semibold flex items-center gap-2">
                {selectedNode.label}
                {selectedNode.file_type && (
                  <Badge variant="outline" className="text-[10px]">
                    <FileText className="w-3 h-3 mr-1" />
                    {selectedNode.file_type}
                  </Badge>
                )}
              </h4>
              {selectedNode.source_file && (
                <p className="text-xs text-muted-foreground font-mono">
                  {selectedNode.source_file}
                </p>
              )}
            </div>
            <button
              onClick={() => setSelectedNode(null)}
              className="text-muted-foreground hover:text-foreground"
            >
              <X className="w-4 h-4" />
            </button>
          </div>

          {connectedEdges.length > 0 && (
            <div className="space-y-1">
              <div className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
                Connections ({connectedEdges.length})
              </div>
              <div className="max-h-32 overflow-y-auto space-y-1">
                {connectedEdges.map((edge, i) => {
                  const isOutgoing = edge.source === selectedNode.id
                  const otherLabel = getNodeLabel(
                    isOutgoing ? edge.target : edge.source
                  )
                  return (
                    <div
                      key={i}
                      className="text-sm flex items-center gap-2 text-muted-foreground"
                    >
                      <span className="font-mono text-xs">
                        {edge.relation}
                      </span>
                      <ChevronRight
                        className={`w-3 h-3 ${isOutgoing ? '' : 'rotate-180'}`}
                      />
                      <span className="text-foreground">{otherLabel}</span>
                      <span className="text-xs ml-auto font-mono opacity-60">
                        {edge.confidence_score.toFixed(2)}
                      </span>
                    </div>
                  )
                })}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
