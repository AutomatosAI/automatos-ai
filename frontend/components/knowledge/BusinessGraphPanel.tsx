'use client'

import { useState, useRef, useMemo, useCallback, useEffect } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { apiClient } from '@/lib/api-client'
import { useWorkspace } from '@/hooks/use-workspace'
import { Input } from '@/components/ui/input'
import { Slider } from '@/components/ui/slider'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import BusinessGraphVisualization, {
  type BusinessGraphHandle,
  type ColorMode,
} from './BusinessGraphVisualization'
import { GraphErrorBoundary } from './GraphErrorBoundary'
import {
  Network, Loader2, Search, X, ChevronRight,
  FileText, Clock, Layers, Upload, RefreshCw, Plus,
  Maximize2, Minimize2, Palette,
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
  const queryClient = useQueryClient()
  const fileInputRef = useRef<HTMLInputElement>(null)

  // UI state
  const [searchTerm, setSearchTerm] = useState('')
  const [confidenceMin, setConfidenceMin] = useState(0)
  const [selectedCommunity, setSelectedCommunity] = useState<number | null>(null)
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null)
  const [importing, setImporting] = useState(false)
  const [importError, setImportError] = useState<string | null>(null)
  const [building, setBuilding] = useState(false)
  const [dragActive, setDragActive] = useState(false)

  // PRD-009 Phase-2 polish state
  const [visibleTypes, setVisibleTypes] = useState<Set<string>>(new Set())  // empty = all
  const [colorMode, setColorMode] = useState<ColorMode>('community')
  const [isFullscreen, setIsFullscreen] = useState(false)
  const vizRef = useRef<BusinessGraphHandle>(null)
  const graphContainerRef = useRef<HTMLDivElement>(null)

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

      console.log('[GraphImport] fetch', url, file.name, file.size, 'bytes')

      const response = await fetch(url, {
        method: 'POST',
        headers,
        body: formData,
      })

      console.log('[GraphImport] status:', response.status)

      if (!response.ok) {
        const text = await response.text()
        throw new Error(text || `HTTP ${response.status}`)
      }

      const result = await response.json()
      console.log('[GraphImport] Success:', result)
      queryClient.invalidateQueries({ queryKey: ['business-graph'] })
    } catch (err: any) {
      const msg = err?.message || String(err) || 'Import failed'
      console.error('[GraphImport] Error:', msg, err)
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
  // smoothly (Obsidian-tier perf). Cap raised from 5,000 (old SVG/d3 limit)
  // to 50,000 — high enough for InbuildUK (~24k) but low enough that the
  // graph.json fetch + parse stays safely under ~30MB. Higher catalogs
  // (~100k+) should switch to the cluster-first drill-in pattern instead.
  const vizSafe = !!meta && (meta.node_count ?? 0) <= 50000

  const {
    data: graphData,
    isLoading: graphLoading,
    error: graphError,
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

  // Per-type counts for the filter chips. We pre-compute once so the chip
  // row stays static even while the user toggles things off.
  const typeCounts = useMemo(() => {
    if (!graphData?.nodes) return new Map<string, number>()
    const m = new Map<string, number>()
    for (const n of graphData.nodes) {
      const t = n.file_type ?? 'unknown'
      m.set(t, (m.get(t) ?? 0) + 1)
    }
    return m
  }, [graphData])

  // Pretty label for each node type — sortable for chip ordering.
  const TYPE_DISPLAY: Record<string, { label: string; color: string; order: number }> = {
    shopify_product:    { label: 'Products',    color: '#ff5e3a', order: 1 },
    shopify_variant:    { label: 'Variants',    color: '#ffb347', order: 2 },
    shopify_vendor:     { label: 'Vendors',     color: '#c084fc', order: 3 },
    shopify_collection: { label: 'Collections', color: '#10e89e', order: 4 },
    shopify_metafield:  { label: 'Metafields',  color: '#38bdf8', order: 5 },
  }

  const typeChips = useMemo(() => {
    return Array.from(typeCounts.entries())
      .map(([t, n]) => ({
        type: t,
        count: n,
        label: TYPE_DISPLAY[t]?.label ?? t.replace(/_/g, ' '),
        color: TYPE_DISPLAY[t]?.color ?? '#94a3b8',
        order: TYPE_DISPLAY[t]?.order ?? 99,
      }))
      .sort((a, b) => a.order - b.order || b.count - a.count)
  }, [typeCounts])

  const toggleType = useCallback((t: string) => {
    setVisibleTypes((prev) => {
      const next = new Set(prev)
      // Empty set = "show all". First click on any chip switches into
      // explicit mode with ONLY the others removed.
      if (next.size === 0) {
        for (const ct of typeChips) {
          if (ct.type !== t) next.add(ct.type)
        }
      } else if (next.has(t)) {
        next.delete(t)
      } else {
        next.add(t)
      }
      // If user re-enables everything, fall back to "show all" sentinel.
      if (next.size === typeChips.length) next.clear()
      return next
    })
  }, [typeChips])

  const isTypeVisible = useCallback(
    (t: string) => visibleTypes.size === 0 || visibleTypes.has(t),
    [visibleTypes],
  )

  // Native HTML5 fullscreen on the graph container.
  const handleFullscreen = useCallback(() => {
    const el = graphContainerRef.current
    if (!el) return
    if (!document.fullscreenElement) {
      el.requestFullscreen?.().then(() => setIsFullscreen(true)).catch(() => {})
    } else {
      document.exitFullscreen?.().then(() => setIsFullscreen(false)).catch(() => {})
    }
  }, [])

  // Keep isFullscreen in sync if the user ESC-exits without clicking the
  // toggle — otherwise the icon shows the wrong state.
  useEffect(() => {
    const onChange = () => setIsFullscreen(!!document.fullscreenElement)
    document.addEventListener('fullscreenchange', onChange)
    return () => document.removeEventListener('fullscreenchange', onChange)
  }, [])

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

  // ── Empty state — drag-and-drop zone (matches document upload pattern) ──
  // Only show when there's genuinely no graph (meta missing/errored)

  if (!meta && (metaError || !metaLoading)) {
    return (
      <div className="space-y-6">
        {/* Hidden file input — same pattern as document-management.tsx */}
        <input
          ref={fileInputRef}
          type="file"
          accept=".json"
          onChange={handleFileChange}
          className="hidden"
        />

        <div
          className={`border-2 border-dashed rounded-lg p-8 text-center transition-all duration-220 ${
            dragActive
              ? 'border-primary bg-primary/5'
              : 'border-border/50 hover:border-primary/50'
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
              <Network className={`w-12 h-12 mx-auto mb-4 ${
                dragActive ? 'text-primary' : 'text-muted-foreground opacity-50'
              }`} />
              <h3 className="text-lg font-semibold mb-2">
                {dragActive ? 'Drop graph.json here' : 'No business graph yet'}
              </h3>
              <p className="text-muted-foreground mb-4">
                {dragActive
                  ? 'Release to import your knowledge graph'
                  : 'Drag and drop a graph.json file, or choose one below'}
              </p>
              <div className="flex items-center justify-center gap-3">
                <Button
                  className="gradient-accent hover:opacity-90"
                  onClick={handleChooseFile}
                >
                  <Plus className="w-4 h-4 mr-2" />
                  Import graph.json
                </Button>
                <Button
                  variant="outline"
                  onClick={handleBuild}
                  disabled={building}
                >
                  {building ? (
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  ) : (
                    <RefreshCw className="w-4 h-4 mr-2" />
                  )}
                  Build from Documents
                </Button>
              </div>
            </>
          )}
        </div>

        {importError && (
          <p className="text-sm text-destructive text-center">{importError}</p>
        )}
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

  return (
    <div className="space-y-4">
      {/* Hidden file input for stats bar import */}
      <input
        ref={fileInputRef}
        type="file"
        accept=".json"
        onChange={handleFileChange}
        className="hidden"
      />

      {/* Stats Bar */}
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
          <Button
            size="sm"
            variant="ghost"
            className="h-7 text-xs"
            onClick={handleChooseFile}
            disabled={importing}
          >
            {importing ? (
              <Loader2 className="w-3 h-3 mr-1 animate-spin" />
            ) : (
              <Upload className="w-3 h-3 mr-1" />
            )}
            Import
          </Button>
          <Button
            size="sm"
            variant="ghost"
            className="h-7 text-xs"
            onClick={handleBuild}
            disabled={building}
          >
            {building ? (
              <Loader2 className="w-3 h-3 mr-1 animate-spin" />
            ) : (
              <RefreshCw className="w-3 h-3 mr-1" />
            )}
            Rebuild
          </Button>
        </span>
        {importError && (
          <span className="text-xs text-destructive w-full">{importError}</span>
        )}
      </div>

      {/* Large graph info */}
      {!vizSafe && meta && (
        <div className="glass-card bg-white/5 backdrop-blur-sm border border-white/10 rounded-lg p-6 text-center">
          <Network className="w-8 h-8 mx-auto mb-2 text-success" />
          <h3 className="text-base font-semibold mb-1">Knowledge Graph Active</h3>
          <p className="text-sm text-muted-foreground">
            {nodeCount.toLocaleString()} nodes · {edgeCount.toLocaleString()} edges · {communityCount} communities.
            Browser visualization is disabled for graphs over 5,000 nodes.
            Agents can query this graph via platform tools.
          </p>
        </div>
      )}

      {/* Main Layout: Sidebar + Graph */}
      {vizSafe && <div className="flex flex-col md:flex-row gap-4 min-h-[500px]">
        {/* Community Sidebar — at 800+ clusters a flat list is useless. We
            hide tiny long-tail clusters (<5 nodes) and cap the visible
            list at 30. 'Show all' expands. Sorted by size descending. */}
        <ClusterSidebar
          communities={communities}
          selectedCommunity={selectedCommunity}
          onSelect={setSelectedCommunity}
        />

        {/* Graph Visualization + controls — wrapped in a localised error
            boundary so a renderer-level exception doesn't take down the
            whole dashboard. Container is the fullscreen target. */}
        <div
          ref={graphContainerRef}
          className="flex-1 glass-card bg-white/5 backdrop-blur-sm border border-white/10 rounded-lg overflow-hidden min-h-[400px] relative"
        >
          {/* Top-right controls — color-mode toggle + fullscreen */}
          <div className="absolute top-3 right-3 z-10 flex items-center gap-2">
            <div className="flex items-center bg-black/50 backdrop-blur-sm rounded-md border border-white/10 overflow-hidden">
              <button
                onClick={() => setColorMode('community')}
                className={`px-2.5 py-1 text-xs flex items-center gap-1 transition ${
                  colorMode === 'community'
                    ? 'bg-white/15 text-foreground'
                    : 'text-muted-foreground hover:text-foreground hover:bg-white/5'
                }`}
                title="Color by cluster"
              >
                <Layers className="w-3 h-3" /> Cluster
              </button>
              <button
                onClick={() => setColorMode('type')}
                className={`px-2.5 py-1 text-xs flex items-center gap-1 transition border-l border-white/10 ${
                  colorMode === 'type'
                    ? 'bg-white/15 text-foreground'
                    : 'text-muted-foreground hover:text-foreground hover:bg-white/5'
                }`}
                title="Color by node type"
              >
                <Palette className="w-3 h-3" /> Type
              </button>
            </div>
            <button
              onClick={handleFullscreen}
              className="p-1.5 rounded-md bg-black/50 backdrop-blur-sm border border-white/10 text-muted-foreground hover:text-foreground hover:bg-white/10 transition"
              title={isFullscreen ? 'Exit fullscreen' : 'Fullscreen'}
            >
              {isFullscreen ? <Minimize2 className="w-3.5 h-3.5" /> : <Maximize2 className="w-3.5 h-3.5" />}
            </button>
          </div>

          {/* Type filter chips — bottom-left, doesn't compete with hover tooltip */}
          {typeChips.length > 0 && (
            <div className="absolute bottom-3 right-3 z-10 flex flex-wrap gap-1.5 max-w-[60%] justify-end">
              {typeChips.map(({ type, label, color, count }) => {
                const visible = isTypeVisible(type)
                return (
                  <button
                    key={type}
                    onClick={() => toggleType(type)}
                    className={`flex items-center gap-1.5 px-2 py-1 rounded-md text-xs border transition ${
                      visible
                        ? 'bg-white/10 border-white/20 text-foreground'
                        : 'bg-black/40 border-white/10 text-muted-foreground/60 line-through'
                    }`}
                    title={`${count.toLocaleString()} ${label.toLowerCase()}`}
                  >
                    <span
                      className="w-2 h-2 rounded-full"
                      style={{ backgroundColor: visible ? color : '#52525b' }}
                    />
                    {label}
                    <span className="text-[10px] text-muted-foreground">
                      {count.toLocaleString()}
                    </span>
                  </button>
                )
              })}
            </div>
          )}

          <GraphErrorBoundary>
            <BusinessGraphVisualization
              ref={vizRef}
              graphData={(filteredData ?? { nodes: [], links: [] }) as any}
              onNodeSelect={handleNodeSelect}
              selectedCommunity={selectedCommunity}
              minConfidence={confidenceMin}
              visibleTypes={visibleTypes.size > 0 ? visibleTypes : undefined}
              colorMode={colorMode}
            />
          </GraphErrorBoundary>
        </div>
      </div>}

      {/* Search + Confidence Controls */}
      {vizSafe && <div className="glass-card bg-white/5 backdrop-blur-sm border border-white/10 rounded-lg px-4 py-3 flex flex-col sm:flex-row items-start sm:items-center gap-4">
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
      </div>}

      {/* Node Detail Panel */}
      {vizSafe && selectedNode && (
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

// ─── Cluster Sidebar ───────────────────────────────────────────────────
// Lives in this file because it's panel-specific and shares the same
// types. At 800+ clusters a flat list is unusable. Filters tiny clusters,
// sorts by size, paginates, and offers a search box.

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
    // Allow searching by cluster id (e.g. "47") or size threshold ">100".
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
        <span className="ml-auto text-[10px] text-muted-foreground/70">
          {communities.length} total
        </span>
      </div>

      <div className="relative">
        <Search className="w-3.5 h-3.5 absolute left-2 top-1/2 -translate-y-1/2 text-muted-foreground" />
        <Input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search… or >100"
          className="h-8 pl-7 text-xs bg-black/30 border-white/10"
        />
      </div>

      <button
        onClick={() => onSelect(null)}
        className={`w-full text-left text-xs px-2 py-1.5 rounded transition-colors ${
          selectedCommunity === null
            ? 'bg-primary/25 text-primary'
            : 'text-muted-foreground hover:bg-white/5'
        }`}
      >
        All clusters
        <span className="ml-1.5 text-[10px] text-muted-foreground/70">
          {totalNodes.toLocaleString()} nodes
        </span>
      </button>

      {visible.map(({ id, count }) => (
        <button
          key={id}
          onClick={() => onSelect(selectedCommunity === id ? null : id)}
          className={`w-full text-left text-xs px-2 py-1.5 rounded flex items-center justify-between transition-colors ${
            selectedCommunity === id
              ? 'bg-primary/25 text-primary'
              : 'text-muted-foreground hover:bg-white/5'
          }`}
        >
          <span>Cluster {id}</span>
          <Badge variant="secondary" className="text-[10px] h-5">
            {count}
          </Badge>
        </button>
      ))}

      {!showAll && filtered.length > PAGE && (
        <button
          onClick={() => setShowAll(true)}
          className="w-full text-xs px-2 py-1.5 rounded text-muted-foreground hover:bg-white/5 transition-colors"
        >
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
