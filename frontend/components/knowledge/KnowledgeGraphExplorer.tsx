'use client'

import { useCallback, useMemo, useRef, useState } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { apiClient } from '@/lib/api-client'
import { useWorkspace } from '@/hooks/use-workspace'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import BusinessGraphVisualization, {
  type BusinessGraphHandle,
} from './BusinessGraphVisualization'
import { GraphView, useGraphPrefs } from '../graph'
import {
  Layers, Search, X, FileText, Loader2,
  GitBranch, Crosshair, Route, Pencil, Check,
} from 'lucide-react'

/**
 * Cluster-first knowledge-graph explorer (PRD-165 S2, BINDING Q28).
 *
 * For graphs too large to ship whole, drill in *server-side*: list communities,
 * load one community's subgraph, expand a node's neighbourhood, or trace the
 * shortest path between two nodes — the full graph.json never reaches the
 * browser. Reuses the GraphView shell + the force renderer.
 */

interface ExplorerNode {
  id: string
  label: string
  file_type?: string
  community?: number
  source_file?: string
  confidence?: number
}
interface ExplorerLink {
  source: string
  target: string
  relation: string
  confidence?: number
  confidence_score: number
}
interface SubgraphData {
  nodes: ExplorerNode[]
  links: ExplorerLink[]
  truncated?: boolean
}
interface CommunityOverview {
  community_id: number
  member_count: number
  title?: string | null
  summary?: string | null
}

const EMPTY: SubgraphData = { nodes: [], links: [] }

export function KnowledgeGraphExplorer() {
  const { workspaceId } = useWorkspace()
  const queryClient = useQueryClient()
  const [prefs] = useGraphPrefs(workspaceId, 'knowledge')
  const vizRef = useRef<BusinessGraphHandle>(null)

  const [activeCommunity, setActiveCommunity] = useState<number | null>(null)
  const [subgraph, setSubgraph] = useState<SubgraphData | null>(null)
  const [selectedNode, setSelectedNode] = useState<ExplorerNode | null>(null)
  const [loadingSub, setLoadingSub] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [pathStart, setPathStart] = useState<ExplorerNode | null>(null)
  const [editingLabel, setEditingLabel] = useState(false)
  const [labelDraft, setLabelDraft] = useState('')

  const [search, setSearch] = useState('')
  const [matches, setMatches] = useState<ExplorerNode[] | null>(null)

  // ── Community overview (the cluster-first entry point) ──
  const { data: commData, isLoading: commsLoading } = useQuery<{ communities: CommunityOverview[] }>({
    queryKey: ['kg-communities', workspaceId],
    queryFn: () => apiClient.graphCommunitiesOverview() as any,
    enabled: !!workspaceId,
    staleTime: 5 * 60 * 1000,
    retry: false,
  })
  const communities = commData?.communities ?? []

  // ── Drill-in actions ──

  const loadCommunity = useCallback(async (cid: number) => {
    setLoadingSub(true); setError(null); setSelectedNode(null)
    // Same drill-mode rule as the in-browser panel: picking a cluster clears
    // any stale neighbourhood focus so it can't AND the new subgraph to nothing.
    vizRef.current?.resetFocus()
    try {
      const res: any = await apiClient.graphCommunitySubgraph(cid)
      setSubgraph({ nodes: res.nodes ?? [], links: res.links ?? [], truncated: res.truncated })
      setActiveCommunity(cid)
    } catch (e: any) {
      setError(e?.message || 'Failed to load community')
    } finally {
      setLoadingSub(false)
    }
  }, [])

  const expandNode = useCallback(async (node: ExplorerNode) => {
    setLoadingSub(true); setError(null)
    try {
      const res: any = await apiClient.graphExpandNode(node.id)
      // Merge the neighbourhood into the current view (dedup by id / endpoints).
      setSubgraph((prev) => mergeSubgraph(prev ?? EMPTY, { nodes: res.nodes ?? [], links: res.links ?? [] }))
    } catch (e: any) {
      setError(e?.message || 'Failed to expand node')
    } finally {
      setLoadingSub(false)
    }
  }, [])

  const findPath = useCallback(async (from: ExplorerNode, to: ExplorerNode) => {
    setLoadingSub(true); setError(null)
    try {
      const res: any = await apiClient.graphPath(from.id, to.id)
      if (!res.found) {
        setError(res.error || 'No path between those nodes')
      } else {
        setSubgraph({ nodes: res.nodes ?? [], links: res.links ?? [] })
        setActiveCommunity(null)
      }
    } catch (e: any) {
      setError(e?.message || 'Path lookup failed')
    } finally {
      setLoadingSub(false)
      setPathStart(null)
    }
  }, [])

  // ── Search-to-focus ──
  const runSearch = useCallback(async (q: string) => {
    if (!q.trim()) { setMatches(null); return }
    try {
      const res: any = await apiClient.graphSearchNodes(q.trim())
      setMatches(res.matches ?? [])
    } catch {
      setMatches([])
    }
  }, [])

  const focusMatch = useCallback(async (node: ExplorerNode) => {
    setSearch(''); setMatches(null)
    setLoadingSub(true); setError(null)
    try {
      const res: any = await apiClient.graphExpandNode(node.id)
      setSubgraph({ nodes: res.nodes ?? [], links: res.links ?? [] })
      setActiveCommunity(null)
      setSelectedNode(node)
    } catch (e: any) {
      setError(e?.message || 'Failed to focus node')
    } finally {
      setLoadingSub(false)
    }
  }, [])

  const handleNodeSelect = useCallback((node: ExplorerNode | null) => {
    if (node && pathStart && node.id !== pathStart.id) {
      void findPath(pathStart, node)
      return
    }
    setSelectedNode(node)
  }, [pathStart, findPath])

  // ── Editable cluster label (PRD-165 S3) ──
  const activeCommunityTitle = useMemo(() => {
    if (activeCommunity == null) return ''
    const c = communities.find((x) => x.community_id === activeCommunity)
    return c?.title || `Cluster ${activeCommunity}`
  }, [activeCommunity, communities])

  const saveLabel = useCallback(async () => {
    if (activeCommunity == null || !labelDraft.trim()) { setEditingLabel(false); return }
    try {
      await apiClient.graphSetCommunityLabel(activeCommunity, labelDraft.trim())
      queryClient.invalidateQueries({ queryKey: ['kg-communities', workspaceId] })
    } catch (e: any) {
      setError(e?.message || 'Rename failed')
    } finally {
      setEditingLabel(false)
    }
  }, [activeCommunity, labelDraft, queryClient, workspaceId])

  // ── Slots ──

  const sidebar = (
    <div className="w-full md:w-60 shrink-0 glass-card bg-white/5 backdrop-blur-sm border border-white/10 rounded-lg p-3 space-y-2 max-h-[560px] overflow-y-auto">
      <div className="text-xs font-medium text-muted-foreground uppercase tracking-wider flex items-center gap-1.5">
        <Layers className="w-3.5 h-3.5" /> Clusters
        <span className="ml-auto text-[10px] text-muted-foreground/70">{communities.length}</span>
      </div>

      <div className="relative">
        <Search className="w-3.5 h-3.5 absolute left-2 top-1/2 -translate-y-1/2 text-muted-foreground" />
        <Input
          value={search}
          onChange={(e) => { setSearch(e.target.value); void runSearch(e.target.value) }}
          placeholder="Search nodes…"
          className="h-8 pl-7 text-xs bg-black/30 border-white/10"
        />
        {search && (
          <button type="button" onClick={() => { setSearch(''); setMatches(null) }} className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground">
            <X className="w-3 h-3" />
          </button>
        )}
      </div>

      {matches != null ? (
        <div className="space-y-1">
          <div className="text-[10px] text-muted-foreground/70 uppercase tracking-wider">Matches</div>
          {matches.length === 0 && <div className="text-xs text-muted-foreground/60 px-1">No matches</div>}
          {matches.map((m) => (
            <button key={m.id} type="button" onClick={() => void focusMatch(m)} className="w-full text-left text-xs px-2 py-1.5 rounded text-muted-foreground hover:bg-white/5 flex items-center gap-1.5">
              <Crosshair className="w-3 h-3 shrink-0" />
              <span className="truncate">{m.label}</span>
            </button>
          ))}
        </div>
      ) : commsLoading ? (
        <div className="flex items-center gap-2 text-xs text-muted-foreground px-1 py-2">
          <Loader2 className="w-3.5 h-3.5 animate-spin" /> Loading clusters…
        </div>
      ) : (
        communities.map((c) => (
          <button
            key={c.community_id}
            type="button"
            onClick={() => void loadCommunity(c.community_id)}
            className={`w-full text-left text-xs px-2 py-1.5 rounded flex items-center justify-between transition-colors ${
              activeCommunity === c.community_id ? 'bg-primary/25 text-primary' : 'text-muted-foreground hover:bg-white/5'
            }`}
          >
            <span className="truncate">{c.title || `Cluster ${c.community_id}`}</span>
            <Badge variant="secondary" className="text-[10px] h-5 ml-1.5 shrink-0">{c.member_count}</Badge>
          </button>
        ))
      )}
    </div>
  )

  const connectionCount = useMemo(() => {
    if (!selectedNode || !subgraph) return 0
    return subgraph.links.filter((l) => l.source === selectedNode.id || l.target === selectedNode.id).length
  }, [selectedNode, subgraph])

  const nodeDetail = selectedNode ? (
    <div className="glass-card bg-white/5 backdrop-blur-sm border border-white/10 rounded-lg p-4 space-y-3">
      <div className="flex items-start justify-between">
        <div className="space-y-1 min-w-0">
          <h4 className="font-semibold flex items-center gap-2 truncate">
            {selectedNode.label}
            {selectedNode.file_type && (
              <Badge variant="outline" className="text-[10px] shrink-0">{selectedNode.file_type}</Badge>
            )}
          </h4>
          {selectedNode.source_file && (
            <p className="text-xs text-muted-foreground font-mono flex items-center gap-1 break-all">
              <FileText className="w-3 h-3 shrink-0" /> {selectedNode.source_file}
            </p>
          )}
          <p className="text-[11px] text-muted-foreground">{connectionCount} connection{connectionCount === 1 ? '' : 's'} in view</p>
        </div>
        <button type="button" onClick={() => setSelectedNode(null)} className="text-muted-foreground hover:text-foreground shrink-0">
          <X className="w-4 h-4" />
        </button>
      </div>

      <div className="flex flex-col gap-1.5">
        <Button size="sm" variant="outline" className="justify-start h-8 text-xs" onClick={() => void expandNode(selectedNode)}>
          <GitBranch className="w-3.5 h-3.5 mr-2" /> Expand from here
        </Button>
        {pathStart?.id === selectedNode.id ? (
          <div className="text-[11px] text-primary flex items-center gap-1.5 px-1">
            <Route className="w-3 h-3" /> Path start set — click another node
            <button type="button" onClick={() => setPathStart(null)} className="ml-auto text-muted-foreground hover:text-foreground"><X className="w-3 h-3" /></button>
          </div>
        ) : (
          <Button size="sm" variant="outline" className="justify-start h-8 text-xs" onClick={() => setPathStart(selectedNode)}>
            <Route className="w-3.5 h-3.5 mr-2" /> Find path from here…
          </Button>
        )}
      </div>
    </div>
  ) : undefined

  const active = subgraph ?? EMPTY
  const emptyState = (
    <div className="text-center text-sm text-muted-foreground p-8 max-w-sm">
      <Layers className="w-8 h-8 mx-auto mb-2 opacity-50" />
      <p className="font-medium text-foreground mb-1">Pick a cluster to start</p>
      <p>This graph is too large to draw all at once. Choose a cluster on the left, then expand nodes or trace paths — everything loads on demand.</p>
    </div>
  )

  return (
    <div className="space-y-3">
      {error && <div className="text-xs text-destructive px-1">{error}</div>}
      {active.truncated && (
        <div className="text-[11px] text-warning px-1">Large cluster — showing the {active.nodes.length} most-connected nodes.</div>
      )}
      {activeCommunity != null && (
        <div className="flex items-center gap-2 px-1">
          {editingLabel ? (
            <>
              <Input
                value={labelDraft}
                onChange={(e) => setLabelDraft(e.target.value)}
                onKeyDown={(e) => { if (e.key === 'Enter') void saveLabel(); if (e.key === 'Escape') setEditingLabel(false) }}
                className="h-7 text-sm max-w-xs bg-black/30 border-white/10"
                autoFocus
              />
              <Button size="sm" className="h-7 px-2" onClick={() => void saveLabel()}><Check className="w-3.5 h-3.5" /></Button>
              <Button size="sm" variant="ghost" className="h-7 px-2" onClick={() => setEditingLabel(false)}><X className="w-3.5 h-3.5" /></Button>
            </>
          ) : (
            <>
              <span className="text-sm font-medium text-foreground">{activeCommunityTitle}</span>
              <button
                type="button"
                onClick={() => { setLabelDraft(activeCommunityTitle); setEditingLabel(true) }}
                className="text-muted-foreground hover:text-foreground"
                title="Rename cluster"
              >
                <Pencil className="w-3.5 h-3.5" />
              </button>
            </>
          )}
        </div>
      )}
      <GraphView
        loading={loadingSub}
        empty={active.nodes.length === 0}
        emptyState={emptyState}
        sidebar={sidebar}
        sidePanel={nodeDetail}
        minHeightClassName="min-h-[520px]"
      >
        <BusinessGraphVisualization
          ref={vizRef}
          graphData={active as any}
          onNodeSelect={handleNodeSelect as any}
          colorMode={prefs.colorMode}
        />
      </GraphView>
    </div>
  )
}

// Merge two subgraphs, de-duplicating nodes by id and links by endpoint pair.
function mergeSubgraph(a: SubgraphData, b: SubgraphData): SubgraphData {
  const nodes = new Map<string, ExplorerNode>()
  for (const n of a.nodes) nodes.set(n.id, n)
  for (const n of b.nodes) if (!nodes.has(n.id)) nodes.set(n.id, n)
  const linkKey = (l: ExplorerLink) => [l.source, l.target].sort().join('::')
  const links = new Map<string, ExplorerLink>()
  for (const l of [...a.links, ...b.links]) links.set(linkKey(l), l)
  return { nodes: Array.from(nodes.values()), links: Array.from(links.values()) }
}
