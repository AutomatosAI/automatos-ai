'use client'

import { useState, useRef, useMemo, useCallback } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { apiClient } from '@/lib/api-client'
import { useWorkspace } from '@/hooks/use-workspace'
import { Input } from '@/components/ui/input'
import { Slider } from '@/components/ui/slider'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import BusinessGraphVisualization, {
  type BusinessGraphHandle,
} from './BusinessGraphVisualization'
import { KnowledgeGraphExplorer } from './KnowledgeGraphExplorer'
import { GraphView, GraphLegend, useGraphPrefs, colorForType, type LegendSection } from '../graph'
import {
  Network, Loader2, Search, X, ChevronRight,
  FileText, Clock, Layers, Upload, RefreshCw, Plus, Palette,
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

/** Max legend chips rendered per section before the title switches to a
 *  "top N/total" summary — keeps a free-text-heavy graph from flooding the
 *  legend over the canvas (paired with the height-bounded GraphLegend). */
const MAX_LEGEND_CHIPS = 12

// ─── Component ──────────────────────────────────────────

export function BusinessGraphPanel() {
  const { workspaceId } = useWorkspace()
  const queryClient = useQueryClient()
  const fileInputRef = useRef<HTMLInputElement>(null)

  // Ephemeral UI state (not worth persisting across reloads)
  const [searchTerm, setSearchTerm] = useState('')
  const [confidenceMin, setConfidenceMin] = useState(0)
  const [selectedCommunity, setSelectedCommunity] = useState<number | null>(null)
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null)
  const [importing, setImporting] = useState(false)
  const [importError, setImportError] = useState<string | null>(null)
  const [building, setBuilding] = useState(false)
  const [dragActive, setDragActive] = useState(false)

  // Persisted per user+workspace (PRD-165 S1 / Q25): colour mode, legend
  // collapse, and hidden type/relation filters survive a reload.
  const [prefs, updatePrefs] = useGraphPrefs(workspaceId, 'knowledge')
  const vizRef = useRef<BusinessGraphHandle>(null)

  // ── File handling (matches document-management.tsx pattern) ──

  const handleImport = useCallback(async (file: File, merge: boolean = false) => {
    if (!workspaceId) {
      setImportError('No workspace selected')
      return
    }
    setImporting(true)
    setImportError(null)
    try {
      // Inline fetch — bypass apiClient entirely, copy uploadDocument exactly
      const formData = new FormData()
      formData.append('file', file)
      formData.append('merge', String(merge))

      const headers: Record<string, string> = {}

      // Workspace ID from localStorage (same as uploadDocument)
      const wsId = localStorage.getItem('last_active_workspace') || localStorage.getItem('last_active_org')
      if (wsId) headers['X-Workspace-ID'] = wsId

      // Clerk auth (same as uploadDocument)
      try {
        const token = await (apiClient as any).getClerkToken?.()
        if (token) headers['Authorization'] = `Bearer ${token}`
      } catch (_) {}

      const BACKEND = process.env.NEXT_PUBLIC_API_URL || ''
      const url = `${BACKEND}/api/knowledge/graph/import`

      const response = await fetch(url, {
        method: 'POST',
        headers,
        body: formData,
      })

      if (!response.ok) {
        const text = await response.text()
        throw new Error(text || `HTTP ${response.status}`)
      }

      await response.json()
      queryClient.invalidateQueries({ queryKey: ['business-graph'] })
    } catch (err: any) {
      const msg = err?.message || String(err) || 'Import failed'
      setImportError(msg)
    } finally {
      setImporting(false)
    }
  }, [workspaceId, queryClient])

  const handleChooseFile = useCallback(() => {
    const input = document.createElement('input')
    input.type = 'file'
    input.accept = '.json'
    input.onchange = () => {
      const file = input.files?.[0]
      if (file) handleImport(file)
    }
    input.click()
  }, [handleImport])

  const handleFileChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      handleImport(file)
    }
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }, [handleImport])

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true)
    } else if (e.type === 'dragleave') {
      setDragActive(false)
    }
  }, [])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)
    const file = e.dataTransfer.files?.[0]
    if (file && file.name.endsWith('.json')) {
      handleImport(file)
    } else if (file) {
      setImportError('Only .json files are supported')
    }
  }, [handleImport])

  const handleBuild = useCallback(async () => {
    if (!workspaceId) return
    setBuilding(true)
    setImportError(null)
    try {
      await apiClient.buildBusinessGraph()
      queryClient.invalidateQueries({ queryKey: ['business-graph'] })
    } catch (err: any) {
      setImportError(err.message || 'Build failed')
    } finally {
      setBuilding(false)
    }
  }, [workspaceId, queryClient])

  // ── Data fetching ──

  const {
    data: meta,
    isLoading: metaLoading,
    error: metaError,
  } = useQuery<GraphMeta>({
    queryKey: graphQueryKeys.meta(workspaceId ?? ''),
    queryFn: async () => {
      const result: any = await apiClient.getWorkspaceFileContent(workspaceId!, 'graph/meta.json')
      const content = result?.content ?? result
      return typeof content === 'string' ? JSON.parse(content) : content
    },
    enabled: !!workspaceId,
    staleTime: 5 * 60 * 1000,
    retry: false,
  })

  // The Canvas-based ForceGraph2D renderer handles tens of thousands of nodes
  // smoothly. Cap at 50,000 — high enough for InbuildUK (~24k) but low enough
  // that the graph.json fetch + parse stays under ~30MB. Higher catalogs
  // (~100k+) should switch to the cluster-first drill-in pattern (PRD-165 S2).
  const vizSafe = !!meta && (meta.node_count ?? 0) <= 50000

  const {
    data: graphData,
    isLoading: graphLoading,
  } = useQuery<GraphData>({
    queryKey: graphQueryKeys.data(workspaceId ?? ''),
    queryFn: async () => {
      const result: any = await apiClient.getWorkspaceFileContent(workspaceId!, 'graph/graph.json')
      const content = result?.content ?? result
      return typeof content === 'string' ? JSON.parse(content) : content
    },
    enabled: !!workspaceId && vizSafe,
    staleTime: 5 * 60 * 1000,
    retry: false,
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

  const typeCounts = useMemo(() => {
    if (!graphData?.nodes) return new Map<string, number>()
    const m = new Map<string, number>()
    for (const n of graphData.nodes) {
      const t = n.file_type ?? 'unknown'
      m.set(t, (m.get(t) ?? 0) + 1)
    }
    return m
  }, [graphData])

  // Pretty label + chip ordering for the known node types. Colour is derived
  // from the shared deterministic palette (colorForType) — no per-type hex.
  const TYPE_DISPLAY: Record<string, { label: string; order: number }> = {
    shopify_product:    { label: 'Products',    order: 1 },
    shopify_variant:    { label: 'Variants',    order: 2 },
    shopify_vendor:     { label: 'Vendors',     order: 3 },
    shopify_collection: { label: 'Collections', order: 4 },
    shopify_metafield:  { label: 'Metafields',  order: 5 },
  }

  const typeChips = useMemo(() => {
    return Array.from(typeCounts.entries())
      .map(([t, n]) => ({
        type: t,
        count: n,
        label: TYPE_DISPLAY[t]?.label ?? t.replace(/_/g, ' '),
        color: colorForType(t),
        order: TYPE_DISPLAY[t]?.order ?? 99,
      }))
      .sort((a, b) => a.order - b.order || b.count - a.count)
  }, [typeCounts])

  const relationCounts = useMemo(() => {
    if (!graphData?.links) return new Map<string, number>()
    const m = new Map<string, number>()
    for (const l of graphData.links) {
      const r = l.relation ?? 'related_to'
      m.set(r, (m.get(r) ?? 0) + 1)
    }
    return m
  }, [graphData])

  const RELATION_DISPLAY: Record<string, { label: string; color: string; order: number }> = {
    frequently_bought_with: { label: 'Order pairs',  color: '#ff3d8c', order: 1 },
    variant_of:             { label: 'Variants',     color: '#ffb347', order: 2 },
    in_collection:          { label: 'Collections',  color: '#10e89e', order: 3 },
    by_vendor:              { label: 'Vendors',      color: '#c084fc', order: 4 },
    has_metafield:          { label: 'Metafields',   color: '#38bdf8', order: 5 },
  }

  const relationChips = useMemo(() => {
    return Array.from(relationCounts.entries())
      .map(([r, n]) => ({
        relation: r,
        count: n,
        label: RELATION_DISPLAY[r]?.label ?? r.replace(/_/g, ' '),
        color: RELATION_DISPLAY[r]?.color ?? '#94a3b8',
        order: RELATION_DISPLAY[r]?.order ?? 99,
      }))
      .sort((a, b) => a.order - b.order || b.count - a.count)
  }, [relationCounts])

  // ── Filter prefs (persisted hidden sets; empty = show all) ──

  const hiddenTypeSet = useMemo(() => new Set(prefs.hiddenTypes), [prefs.hiddenTypes])
  const hiddenRelSet = useMemo(() => new Set(prefs.hiddenRelations), [prefs.hiddenRelations])

  const toggleType = useCallback((t: string) => {
    const next = new Set(prefs.hiddenTypes)
    next.has(t) ? next.delete(t) : next.add(t)
    updatePrefs({ hiddenTypes: Array.from(next) })
  }, [prefs.hiddenTypes, updatePrefs])

  const toggleRelation = useCallback((r: string) => {
    const next = new Set(prefs.hiddenRelations)
    next.has(r) ? next.delete(r) : next.add(r)
    updatePrefs({ hiddenRelations: Array.from(next) })
  }, [prefs.hiddenRelations, updatePrefs])

  // Convert hidden → visible sets for the renderer (undefined = show all).
  const visibleTypes = useMemo(() => {
    if (hiddenTypeSet.size === 0) return undefined
    return new Set(typeChips.filter((c) => !hiddenTypeSet.has(c.type)).map((c) => c.type))
  }, [hiddenTypeSet, typeChips])

  const visibleRelations = useMemo(() => {
    if (hiddenRelSet.size === 0) return undefined
    return new Set(relationChips.filter((c) => !hiddenRelSet.has(c.relation)).map((c) => c.relation))
  }, [hiddenRelSet, relationChips])

  // Cap chips per section: a general knowledge graph carries hundreds of
  // free-text edge relations (each count 1) that flood the legend and bury the
  // canvas. Chips are pre-sorted (curated order, then count desc), so the slice
  // keeps the most meaningful ones and the title shows "top N/total" when
  // truncated. The legend itself is height-bounded + scrollable for any residual.
  const legendSections: LegendSection[] = useMemo(() => {
    const cap = (title: string, total: number) =>
      total > MAX_LEGEND_CHIPS ? `${title} · top ${MAX_LEGEND_CHIPS}/${total}` : title
    return [
      {
        title: cap('Nodes', typeChips.length),
        onToggle: toggleType,
        chips: typeChips.slice(0, MAX_LEGEND_CHIPS).map((c) => ({
          key: c.type, label: c.label, color: c.color, count: c.count,
          hidden: hiddenTypeSet.has(c.type),
        })),
      },
      {
        title: cap('Edges', relationChips.length),
        onToggle: toggleRelation,
        chips: relationChips.slice(0, MAX_LEGEND_CHIPS).map((c) => ({
          key: c.relation, label: c.label, color: c.color, count: c.count,
          hidden: hiddenRelSet.has(c.relation),
        })),
      },
    ]
  }, [typeChips, relationChips, hiddenTypeSet, hiddenRelSet, toggleType, toggleRelation])

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
      (e) => e.confidence_score >= confidenceMin && nodeIds.has(e.source) && nodeIds.has(e.target),
    )
    return { nodes: filteredNodes, links: filteredEdges }
  }, [graphData, searchTerm, confidenceMin, selectedCommunity])

  const connectedEdges = useMemo(() => {
    if (!selectedNode || !graphData?.links) return []
    return graphData.links.filter(
      (e) => e.source === selectedNode.id || e.target === selectedNode.id,
    )
  }, [selectedNode, graphData])

  const getNodeLabel = useCallback(
    (id: string) => graphData?.nodes.find((n) => n.id === id)?.label ?? id,
    [graphData],
  )

  const handleNodeSelect = useCallback((node: GraphNode | null) => {
    setSelectedNode(node)
  }, [])

  // ── Empty state — drag-and-drop import zone (bespoke, kept off the shell) ──

  if (!meta && (metaError || !metaLoading)) {
    return (
      <div className="space-y-6">
        <input ref={fileInputRef} type="file" accept=".json" onChange={handleFileChange} className="hidden" />
        <div
          className={`border-2 border-dashed rounded-lg p-8 text-center transition-all duration-220 ${
            dragActive ? 'border-primary bg-primary/5' : 'border-border/50 hover:border-primary/50'
          }`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
        >
          {importing ? (
            <div className="space-y-4">
              <Upload className="w-12 h-12 mx-auto text-primary animate-bounce" />
              <h3 className="text-lg font-semibold">Importing graph...</h3>
              <div className="w-full max-w-xs mx-auto bg-secondary rounded-full h-2">
                <div className="bg-gradient-to-r from-blue-500 to-purple-500 h-2 rounded-full transition-all duration-300 animate-pulse" />
              </div>
              <p className="text-sm text-muted-foreground">Processing nodes and edges...</p>
            </div>
          ) : (
            <>
              <Network className={`w-12 h-12 mx-auto mb-4 ${dragActive ? 'text-primary' : 'text-muted-foreground opacity-50'}`} />
              <h3 className="text-lg font-semibold mb-2">
                {dragActive ? 'Drop graph.json here' : 'No knowledge graph yet'}
              </h3>
              <p className="text-muted-foreground mb-4">
                {dragActive ? 'Release to import your knowledge graph' : 'Drag and drop a graph.json file, or choose one below'}
              </p>
              <div className="flex items-center justify-center gap-3">
                <Button className="gradient-accent hover:opacity-90" onClick={handleChooseFile}>
                  <Plus className="w-4 h-4 mr-2" />
                  Import graph.json
                </Button>
                <Button variant="outline" onClick={handleBuild} disabled={building}>
                  {building ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : <RefreshCw className="w-4 h-4 mr-2" />}
                  Build from Documents
                </Button>
              </div>
            </>
          )}
        </div>
        {importError && <p className="text-sm text-destructive text-center">{importError}</p>}
      </div>
    )
  }

  // ── Stats ──

  const nodeCount = meta?.node_count ?? graphData?.nodes?.length ?? 0
  const edgeCount = meta?.edge_count ?? graphData?.links?.length ?? 0
  const communityCount = meta?.community_count ?? communities.length
  const lastBuilt = meta?.last_built
    ? formatDistanceToNow(new Date(meta.last_built * 1000), { addSuffix: true })
    : null

  const statsBar = (
    <div className="glass-card bg-white/5 backdrop-blur-sm border border-white/10 rounded-lg px-4 py-3 flex flex-wrap items-center gap-x-6 gap-y-1 text-sm">
      <span className="flex items-center gap-1.5">
        <span className="font-semibold text-info">{nodeCount.toLocaleString()}</span>
        <span className="text-muted-foreground">nodes</span>
      </span>
      <span className="flex items-center gap-1.5">
        <span className="font-semibold text-success">{edgeCount.toLocaleString()}</span>
        <span className="text-muted-foreground">edges</span>
      </span>
      <span className="flex items-center gap-1.5">
        <span className="font-semibold text-agent">{communityCount}</span>
        <span className="text-muted-foreground">clusters</span>
      </span>
      <span className="flex items-center gap-2 ml-auto">
        {lastBuilt && (
          <span className="flex items-center gap-1.5 text-muted-foreground">
            <Clock className="w-3.5 h-3.5" />
            Last built {lastBuilt}
          </span>
        )}
        <Button size="sm" variant="ghost" className="h-7 text-xs" onClick={handleChooseFile} disabled={importing}>
          {importing ? <Loader2 className="w-3 h-3 mr-1 animate-spin" /> : <Upload className="w-3 h-3 mr-1" />}
          Import
        </Button>
        <Button size="sm" variant="ghost" className="h-7 text-xs" onClick={handleBuild} disabled={building}>
          {building ? <Loader2 className="w-3 h-3 mr-1 animate-spin" /> : <RefreshCw className="w-3 h-3 mr-1" />}
          Rebuild
        </Button>
      </span>
      {importError && <span className="text-xs text-destructive w-full">{importError}</span>}
    </div>
  )

  // ── Large-graph notice (browser viz disabled, agents still query) ──

  // Large graphs (>50k) can't ship the full graph.json to the browser — drill
  // in server-side, cluster-first, instead of the old dead "viz disabled"
  // notice (PRD-165 S2 / Q28).
  if (!vizSafe && meta) {
    return (
      <div className="space-y-4">
        {statsBar}
        <KnowledgeGraphExplorer />
      </div>
    )
  }

  // ── Controls + side-panel slots ──

  const colorToggle = (
    <div className="flex items-center bg-black/50 backdrop-blur-sm rounded-md border border-white/10 overflow-hidden">
      <button
        type="button"
        onClick={() => updatePrefs({ colorMode: 'community' })}
        className={`px-2.5 py-1 text-xs flex items-center gap-1 transition ${
          prefs.colorMode === 'community' ? 'bg-white/15 text-foreground' : 'text-muted-foreground hover:text-foreground hover:bg-white/5'
        }`}
        title="Color by cluster"
      >
        <Layers className="w-3 h-3" /> Cluster
      </button>
      <button
        type="button"
        onClick={() => updatePrefs({ colorMode: 'type' })}
        className={`px-2.5 py-1 text-xs flex items-center gap-1 transition border-l border-white/10 ${
          prefs.colorMode === 'type' ? 'bg-white/15 text-foreground' : 'text-muted-foreground hover:text-foreground hover:bg-white/5'
        }`}
        title="Color by node type"
      >
        <Palette className="w-3 h-3" /> Type
      </button>
    </div>
  )

  const nodeDetailPanel = selectedNode ? (
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
            <p className="text-xs text-muted-foreground font-mono break-all">{selectedNode.source_file}</p>
          )}
        </div>
        <button type="button" onClick={() => setSelectedNode(null)} className="text-muted-foreground hover:text-foreground">
          <X className="w-4 h-4" />
        </button>
      </div>
      {connectedEdges.length > 0 && (
        <div className="space-y-1">
          <div className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
            Connections ({connectedEdges.length})
          </div>
          <div className="max-h-64 overflow-y-auto space-y-1">
            {connectedEdges.map((edge, i) => {
              const isOutgoing = edge.source === selectedNode.id
              const otherLabel = getNodeLabel(isOutgoing ? edge.target : edge.source)
              return (
                <div key={i} className="text-sm flex items-center gap-2 text-muted-foreground">
                  <span className="font-mono text-xs">{edge.relation}</span>
                  <ChevronRight className={`w-3 h-3 shrink-0 ${isOutgoing ? '' : 'rotate-180'}`} />
                  <span className="text-foreground truncate">{otherLabel}</span>
                  <span className="text-xs ml-auto font-mono opacity-60">{edge.confidence_score.toFixed(2)}</span>
                </div>
              )
            })}
          </div>
        </div>
      )}
    </div>
  ) : undefined

  const searchConfidenceBar = (
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
          <button type="button" onClick={() => setSearchTerm('')} className="absolute right-2.5 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground">
            <X className="w-3.5 h-3.5" />
          </button>
        )}
      </div>
      <div className="flex items-center gap-3 flex-1 w-full sm:w-auto">
        <span className="text-xs text-muted-foreground whitespace-nowrap">Confidence:</span>
        <Slider min={0} max={1} step={0.05} value={[confidenceMin]} onValueChange={([val]) => setConfidenceMin(val)} className="flex-1 max-w-[200px]" />
        <span className="text-xs font-mono w-8 text-right">{confidenceMin.toFixed(2)}</span>
      </div>
    </div>
  )

  return (
    <GraphView
      loading={graphLoading || metaLoading}
      toolbar={<div className="space-y-4">{statsBar}{searchConfidenceBar}</div>}
      controls={colorToggle}
      legend={
        <GraphLegend
          sections={legendSections}
          collapsed={prefs.legendCollapsed}
          onCollapsedChange={(collapsed) => updatePrefs({ legendCollapsed: collapsed })}
        />
      }
      sidebar={
        graphData ? (
          <ClusterSidebar communities={communities} selectedCommunity={selectedCommunity} onSelect={setSelectedCommunity} />
        ) : undefined
      }
      sidePanel={nodeDetailPanel}
    >
      <BusinessGraphVisualization
        ref={vizRef}
        graphData={(filteredData ?? { nodes: [], links: [] }) as any}
        onNodeSelect={handleNodeSelect}
        selectedCommunity={selectedCommunity}
        minConfidence={confidenceMin}
        visibleTypes={visibleTypes}
        visibleRelations={visibleRelations}
        colorMode={prefs.colorMode}
      />
    </GraphView>
  )
}

// ─── Cluster Sidebar ───────────────────────────────────────────────────
// At 800+ clusters a flat list is unusable. Filters tiny clusters, sorts by
// size, paginates, and offers a search box.

interface ClusterSidebarProps {
  communities: Array<{ id: number; count: number }>
  selectedCommunity: number | null
  onSelect: (id: number | null) => void
}

function ClusterSidebar({ communities, selectedCommunity, onSelect }: ClusterSidebarProps) {
  const [query, setQuery] = useState('')
  const [showAll, setShowAll] = useState(false)
  const MIN_SIZE = 5  // hide singleton/pair clusters — pure long-tail noise
  const PAGE = 30

  const filtered = useMemo(() => {
    const min = communities.filter((c) => c.count >= MIN_SIZE)
    const q = query.trim()
    if (!q) return min
    if (/^>\s*\d+$/.test(q)) {
      const threshold = parseInt(q.replace(/^>\s*/, ''), 10)
      return min.filter((c) => c.count > threshold)
    }
    return min.filter((c) => String(c.id).includes(q))
  }, [communities, query])

  const visible = showAll ? filtered : filtered.slice(0, PAGE)
  const totalNodes = communities.reduce((a, b) => a + b.count, 0)
  const hiddenLongTail = communities.length - filtered.length

  return (
    <div className="w-full md:w-56 shrink-0 glass-card bg-white/5 backdrop-blur-sm border border-white/10 rounded-lg p-3 space-y-2 max-h-[500px] overflow-y-auto">
      <div className="text-xs font-medium text-muted-foreground uppercase tracking-wider flex items-center gap-1.5">
        <Layers className="w-3.5 h-3.5" />
        Clusters
        <span className="ml-auto text-[10px] text-muted-foreground/70">{communities.length} total</span>
      </div>

      <div className="relative">
        <Search className="w-3.5 h-3.5 absolute left-2 top-1/2 -translate-y-1/2 text-muted-foreground" />
        <Input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Search… or >100" className="h-8 pl-7 text-xs bg-black/30 border-white/10" />
      </div>

      <button
        type="button"
        onClick={() => onSelect(null)}
        className={`w-full text-left text-xs px-2 py-1.5 rounded transition-colors ${
          selectedCommunity === null ? 'bg-primary/25 text-primary' : 'text-muted-foreground hover:bg-white/5'
        }`}
      >
        All clusters
        <span className="ml-1.5 text-[10px] text-muted-foreground/70">{totalNodes.toLocaleString()} nodes</span>
      </button>

      {visible.map(({ id, count }) => (
        <button
          key={id}
          type="button"
          onClick={() => onSelect(selectedCommunity === id ? null : id)}
          className={`w-full text-left text-xs px-2 py-1.5 rounded flex items-center justify-between transition-colors ${
            selectedCommunity === id ? 'bg-primary/25 text-primary' : 'text-muted-foreground hover:bg-white/5'
          }`}
        >
          <span>Cluster {id}</span>
          <Badge variant="secondary" className="text-[10px] h-5">{count}</Badge>
        </button>
      ))}

      {!showAll && filtered.length > PAGE && (
        <button type="button" onClick={() => setShowAll(true)} className="w-full text-xs px-2 py-1.5 rounded text-muted-foreground hover:bg-white/5 transition-colors">
          + {filtered.length - PAGE} more clusters
        </button>
      )}

      {hiddenLongTail > 0 && (
        <div className="text-[10px] text-muted-foreground/60 px-2 pt-1 border-t border-white/5">
          {hiddenLongTail} tiny clusters hidden (&lt;{MIN_SIZE} nodes each)
        </div>
      )}
    </div>
  )
}
