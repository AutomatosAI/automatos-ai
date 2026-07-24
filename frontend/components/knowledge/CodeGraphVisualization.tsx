'use client'

import React, { useState, useEffect, useCallback } from 'react'
import { apiClient } from '@/lib/api-client'
import { GraphErrorBoundary } from '../graph'
import ReactFlow, {
  Node,
  Edge,
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
  MarkerType,
  Panel,
} from 'reactflow'
import 'reactflow/dist/style.css'
import { Search, Download, ZoomIn, ZoomOut, Maximize2, Minimize2, Flame, Layers, Code, ChevronRight, ChevronLeft, AlertTriangle } from 'lucide-react'

interface CodeGraphVisualizationProps {
  project: string
  projectId?: number
}

interface CallGraphNode {
  id: string
  symbol: string
  type: string
  file: string
  calls: string[]
  calledBy: string[]
}

interface ArchitectureData {
  communities: Array<{
    cluster_id: number
    size: number
    files: string[]
    members: Array<{ id: number; name: string; type: string; file: string }>
  }>
  hotspots: Array<{
    id: number
    name: string
    file: string
    fan_out: number
    betweenness: number
  }>
  cycles: string[][]
  metrics: Record<string, { mean: number; max: number }>
  modularity_score: number
}

interface SelectedNodeData {
  name: string
  type: string
  file: string
  line?: number
  signature?: string
  docstring?: string
  code_snippet?: string
}

// Automatos brand palette — orange primary with warm supporting tones
const nodeColors = {
  function: '#e8590c',   // brand orange
  class: '#d97706',      // amber
  method: '#c2410c',     // deep orange
  import: '#b45309',     // warm brown
  interface: '#ea580c',  // bright orange
}

// Cluster palette — brand-consistent warm tones
const clusterColors = [
  '#e8590c', '#d97706', '#c2410c', '#dc2626', '#ea580c',
  '#b45309', '#f59e0b', '#f97316', '#ef4444', '#ca8a04',
  '#a16207', '#e11d48', '#fb923c', '#fbbf24', '#78350f',
]

// Heatmap: green (low risk) → yellow → red (high risk)
function heatmapColor(value: number, max: number): string {
  const ratio = max > 0 ? Math.min(value / max, 1) : 0
  if (ratio < 0.33) return `rgb(${Math.round(ratio * 3 * 255)}, 200, 80)`
  if (ratio < 0.66) return `rgb(255, ${Math.round((1 - (ratio - 0.33) * 3) * 200)}, 60)`
  return `rgb(255, ${Math.round((1 - ratio) * 100)}, 60)`
}

export function CodeGraphVisualization({ project, projectId }: CodeGraphVisualizationProps) {
  const [nodes, setNodes, onNodesChange] = useNodesState([])
  const [edges, setEdges, onEdgesChange] = useEdgesState([])
  const [searchQuery, setSearchQuery] = useState('')
  const [selectedSymbol, setSelectedSymbol] = useState<string | null>(null)
  const [graphType, setGraphType] = useState<'calls' | 'imports' | 'extends'>('calls')
  const [depth, setDepth] = useState(2)
  const [direction, setDirection] = useState<'outgoing' | 'incoming' | 'both'>('outgoing')
  const [loading, setLoading] = useState(false)

  // PRD-62: Architecture overlay state
  const [viewMode, setViewMode] = useState<'default' | 'clusters' | 'heatmap'>('default')
  const [archData, setArchData] = useState<ArchitectureData | null>(null)
  const [archLoading, setArchLoading] = useState(false)

  // PRD-62: Code snippet panel state
  const [selectedNode, setSelectedNode] = useState<SelectedNodeData | null>(null)
  const [codePanelOpen, setCodePanelOpen] = useState(false)

  // Fullscreen state
  const [isFullscreen, setIsFullscreen] = useState(false)

  // PRD-62: File tree sidebar state (US-018)
  const [fileTreeOpen, setFileTreeOpen] = useState(false)
  const [fileTreeData, setFileTreeData] = useState<Array<{ file_path: string; language: string; lines_of_code: number; symbol_count?: number }>>([])
  const [fileFilter, setFileFilter] = useState('')
  const [highlightedFile, setHighlightedFile] = useState<string | null>(null)

  // Escape key exits fullscreen
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isFullscreen) setIsFullscreen(false)
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [isFullscreen])

  // PRD-62: Fetch architecture data
  const fetchArchitecture = useCallback(async () => {
    if (!projectId) return
    setArchLoading(true)
    try {
      const data = await apiClient.codegraphGetArchitecture(projectId)
      setArchData(data)
    } catch (err) {
      console.error('Architecture fetch failed:', err)
    } finally {
      setArchLoading(false)
    }
  }, [projectId])

  // PRD-62: Fetch file tree data
  const fetchFileTree = useCallback(async () => {
    if (!projectId) return
    try {
      const data = await apiClient.codegraphGetArchitecture(projectId)
      if (data) {
        // Extract file info from communities
        const fileMap = new Map<string, { language: string; symbol_count: number; lines_of_code: number }>()
        if (data.communities) {
          data.communities.forEach((c: any) => {
            c.members?.forEach((m: any) => {
              if (m.file && !fileMap.has(m.file)) {
                fileMap.set(m.file, { language: '', symbol_count: 0, lines_of_code: 0 })
              }
              if (m.file) {
                const entry = fileMap.get(m.file)!
                entry.symbol_count = (entry.symbol_count || 0) + 1
              }
            })
          })
        }
        setFileTreeData(Array.from(fileMap.entries()).map(([fp, data]) => ({
          file_path: fp, ...data
        })).sort((a, b) => a.file_path.localeCompare(b.file_path)))
      }
    } catch (err) {
      // Non-critical, silently fail
    }
  }, [projectId])

  useEffect(() => {
    if (projectId) {
      fetchArchitecture()
      fetchFileTree()
    }
  }, [projectId, fetchArchitecture, fetchFileTree])

  const fetchCallGraph = useCallback(async (symbol: string) => {
    if (!symbol || !project) return

    setLoading(true)
    try {
      const data = await apiClient.codegraphGetCallGraph({
        project,
        symbol,
        depth,
        direction,
      })

      const flowNodes: Node[] = []
      const flowEdges: Edge[] = []

      // Build cluster lookup from architecture data
      const nodeClusterMap: Record<string, number> = {}
      const hotspotNames = new Set<string>()
      const nodeMetrics: Record<string, number> = {}

      if (archData) {
        archData.communities.forEach((c) => {
          c.members.forEach((m) => {
            nodeClusterMap[m.name] = c.cluster_id
          })
        })
        archData.hotspots.forEach((h) => hotspotNames.add(h.name))
      }

      if (data.nodes && Array.isArray(data.nodes)) {
        data.nodes.forEach((node: any, index: number) => {
          const symbolType = node.type || 'function'
          const isRoot = node.symbol === symbol
          const isHotspot = hotspotNames.has(node.name)

          // Determine node color based on view mode
          let bgColor = nodeColors[symbolType as keyof typeof nodeColors] || nodeColors.function
          if (viewMode === 'clusters' && nodeClusterMap[node.name] !== undefined) {
            bgColor = clusterColors[nodeClusterMap[node.name] % clusterColors.length]
          } else if (viewMode === 'heatmap') {
            const metric = nodeMetrics[node.name] || 0
            const maxMetric = archData?.metrics?.fan_out?.max || 1
            bgColor = heatmapColor(metric, maxMetric)
          }

          flowNodes.push({
            id: node.symbol,
            type: 'default',
            data: {
              label: (
                <div className="text-center">
                  {isHotspot && <span title="Hotspot">⚠️ </span>}
                  {node.name}
                  {viewMode === 'clusters' && nodeClusterMap[node.name] !== undefined && (
                    <div className="text-[10px] opacity-70">
                      Cluster {nodeClusterMap[node.name]}
                    </div>
                  )}
                </div>
              ),
            },
            position: {
              x: isRoot ? 400 : (index % 3) * 300 + 100,
              y: isRoot ? 300 : Math.floor(index / 3) * 150 + 50
            },
            style: {
              background: bgColor,
              color: 'white',
              border: isRoot ? '3px solid #fff' : isHotspot ? '2px dashed #fbbf24' : '1px solid rgba(255,255,255,0.15)',
              borderRadius: '8px',
              padding: '12px',
              minWidth: '150px',
              fontSize: '12px',
            },
          })
        })
      }

      // Create edges — highlight cycles in red
      const cycleEdges = new Set<string>()
      if (archData?.cycles) {
        archData.cycles.forEach(cycle => {
          for (let i = 0; i < cycle.length; i++) {
            const next = cycle[(i + 1) % cycle.length]
            cycleEdges.add(`${cycle[i]}->${next}`)
          }
        })
      }

      if (data.edges && Array.isArray(data.edges)) {
        data.edges.forEach((edge: any) => {
          const isCycle = cycleEdges.has(`${edge.from}->${edge.to}`)
          const edgeColor = isCycle ? '#ef4444' :
                           edge.type === 'extends' ? '#d97706' :
                           edge.type === 'calls' ? '#e8590c' : '#f59e0b'

          flowEdges.push({
            id: `${edge.from}->${edge.to}`,
            source: edge.from,
            target: edge.to,
            type: 'smoothstep',
            animated: isCycle || edge.type === 'calls',
            label: isCycle ? `${edge.type} (cycle)` : edge.type,
            style: { stroke: edgeColor, strokeWidth: isCycle ? 3 : 2 },
            markerEnd: {
              type: MarkerType.ArrowClosed,
              color: edgeColor,
            },
          })
        })
      }

      setNodes(flowNodes)
      setEdges(flowEdges)
    } catch (error) {
      console.error('Error fetching call graph:', error)
    } finally {
      setLoading(false)
    }
  }, [project, depth, direction, setNodes, setEdges, archData, viewMode])

  const handleSearch = () => {
    if (searchQuery.trim()) {
      setSelectedSymbol(searchQuery.trim())
      fetchCallGraph(searchQuery.trim())
    }
  }

  // Re-render graph when viewMode or archData changes (to recolor nodes)
  useEffect(() => {
    if (selectedSymbol && nodes.length > 0) {
      fetchCallGraph(selectedSymbol)
    }
  }, [viewMode, archData])

  // PRD-62: Handle node click for code panel
  const handleNodeClick = useCallback((_: any, node: Node) => {
    const nodeData = node.data as any
    setSelectedNode({
      name: typeof nodeData.label === 'string' ? nodeData.label : node.id,
      type: 'symbol',
      file: '',
      signature: node.id,
    })
    setCodePanelOpen(true)
  }, [])

  return (
    <div className={`flex flex-col bg-background/50 backdrop-blur-sm rounded-lg border border-border/50 ${
      isFullscreen ? 'fixed inset-0 z-50 rounded-none h-screen bg-background' : 'h-[600px]'
    }`}>
      {/* Controls */}
      <div className="p-4 border-b border-border/50 space-y-3">
        {/* Search */}
        <div className="flex gap-2">
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
            <input
              type="text"
              placeholder="Enter symbol name (e.g., AgentFactory, execute_task)..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
              className="w-full pl-10 pr-4 py-2 bg-secondary/80 border border-border/40 rounded-lg text-white placeholder-muted-foreground focus:outline-none focus:border-warning"
            />
          </div>
          <button
            onClick={() => setIsFullscreen(!isFullscreen)}
            className="px-3 py-2 bg-secondary/80 hover:bg-secondary text-foreground/80 rounded-lg transition-colors border border-border/40"
            title={isFullscreen ? 'Exit fullscreen' : 'Fullscreen'}
          >
            {isFullscreen ? <Minimize2 className="w-4 h-4" /> : <Maximize2 className="w-4 h-4" />}
          </button>
          <button
            onClick={handleSearch}
            disabled={loading || !searchQuery.trim()}
            className="px-4 py-2 bg-warning hover:bg-warning disabled:bg-secondary/50 disabled:cursor-not-allowed text-white rounded-lg font-medium transition-colors"
          >
            {loading ? 'Loading...' : 'Visualize'}
          </button>
        </div>

        {/* Options */}
        <div className="flex gap-4 text-sm flex-wrap">
          {/* Graph Type */}
          <div className="flex items-center gap-2">
            <label className="text-foreground/80">Type:</label>
            <select
              value={graphType}
              onChange={(e) => setGraphType(e.target.value as any)}
              className="bg-secondary/80 border border-border/40 rounded px-2 py-1 text-white"
            >
              <option value="calls">Call Graph</option>
              <option value="imports">Dependencies</option>
              <option value="extends">Inheritance</option>
            </select>
          </div>

          {/* Direction */}
          <div className="flex items-center gap-2">
            <label className="text-foreground/80">Direction:</label>
            <select
              value={direction}
              onChange={(e) => setDirection(e.target.value as any)}
              className="bg-secondary/80 border border-border/40 rounded px-2 py-1 text-white"
            >
              <option value="outgoing">Outgoing</option>
              <option value="incoming">Incoming</option>
              <option value="both">Both</option>
            </select>
          </div>

          {/* Depth */}
          <div className="flex items-center gap-2">
            <label className="text-foreground/80">Depth:</label>
            <input
              type="number"
              min="1"
              max="5"
              value={depth}
              onChange={(e) => setDepth(parseInt(e.target.value))}
              className="w-16 bg-secondary/80 border border-border/40 rounded px-2 py-1 text-white"
            />
          </div>

          {/* PRD-62: View Mode Toggle */}
          <div className="flex items-center gap-1 ml-auto">
            <button
              onClick={() => setViewMode('default')}
              className={`px-2 py-1 rounded text-xs transition-colors ${viewMode === 'default' ? 'bg-warning text-white' : 'bg-secondary/80 text-muted-foreground hover:text-white'}`}
              title="Default colors by type"
            >
              Default
            </button>
            <button
              onClick={() => {
                if (!archData && !archLoading) fetchArchitecture()
                setViewMode('clusters')
              }}
              className={`px-2 py-1 rounded text-xs transition-colors flex items-center gap-1 ${viewMode === 'clusters' ? 'bg-warning text-white' : 'bg-secondary/80 text-muted-foreground hover:text-white'} ${archLoading ? 'opacity-50' : ''}`}
              title="Color by Louvain module cluster"
            >
              <Layers className="w-3 h-3" /> Clusters {archLoading && '...'}
            </button>
            <button
              onClick={() => {
                if (!archData && !archLoading) fetchArchitecture()
                setViewMode('heatmap')
              }}
              className={`px-2 py-1 rounded text-xs transition-colors flex items-center gap-1 ${viewMode === 'heatmap' ? 'bg-destructive text-white' : 'bg-secondary/80 text-muted-foreground hover:text-white'} ${archLoading ? 'opacity-50' : ''}`}
              title="Heatmap by complexity (green=low, red=high)"
            >
              <Flame className="w-3 h-3" /> Heatmap {archLoading && '...'}
            </button>
            {archData && archData.hotspots.length > 0 && (
              <span className="ml-2 px-2 py-0.5 bg-yellow-900/50 text-warning text-xs rounded flex items-center gap-1">
                <AlertTriangle className="w-3 h-3" /> {archData.hotspots.length} hotspots
              </span>
            )}
          </div>
        </div>
      </div>

      {/* Graph + Code Panel + File Tree */}
      <div className="flex-1 flex relative">
        {/* PRD-62: File Tree Sidebar (US-018) */}
        {fileTreeOpen && (
          <div className="w-56 border-r border-border/50 bg-background/80 flex flex-col overflow-hidden">
            <div className="flex items-center justify-between p-2 border-b border-border/50">
              <span className="text-xs font-medium text-foreground/80">Files</span>
              <button onClick={() => setFileTreeOpen(false)} className="text-muted-foreground hover:text-white text-xs">
                Close
              </button>
            </div>
            <div className="p-2">
              <input
                type="text"
                placeholder="Filter files..."
                value={fileFilter}
                onChange={(e) => setFileFilter(e.target.value)}
                className="w-full px-2 py-1 bg-secondary/80 border border-border/40 rounded text-xs text-white placeholder-muted-foreground focus:outline-none focus:border-warning"
              />
            </div>
            <div className="flex-1 overflow-y-auto">
              {fileTreeData
                .filter(f => !fileFilter || f.file_path.toLowerCase().includes(fileFilter.toLowerCase()))
                .map((file, idx) => (
                <button
                  key={idx}
                  onClick={() => {
                    setHighlightedFile(file.file_path === highlightedFile ? null : file.file_path)
                    // Highlight nodes from this file
                    setNodes((nds) => nds.map(n => ({
                      ...n,
                      style: {
                        ...n.style,
                        boxShadow: file.file_path === highlightedFile ? 'none' :
                          (n.data?.file === file.file_path ? '0 0 12px rgba(232, 89, 12, 0.8)' : 'none'),
                      }
                    })))
                  }}
                  className={`w-full text-left px-2 py-1.5 text-xs hover:bg-secondary/80 transition-colors ${
                    highlightedFile === file.file_path ? 'bg-warning/30 border-l-2 border-warning' : ''
                  }`}
                >
                  <div className="text-foreground/80 truncate font-mono">{file.file_path.split('/').pop()}</div>
                  <div className="text-muted-foreground/70 text-[10px] truncate">{file.file_path}</div>
                  <div className="flex gap-2 text-[10px] text-muted-foreground/50 mt-0.5">
                    {file.symbol_count ? <span>{file.symbol_count} symbols</span> : null}
                    {file.lines_of_code ? <span>{file.lines_of_code} LOC</span> : null}
                  </div>
                </button>
              ))}
              {fileTreeData.length === 0 && (
                <div className="text-muted-foreground/70 text-xs p-2 text-center">No files indexed</div>
              )}
            </div>
          </div>
        )}

        {/* File tree toggle button */}
        {!fileTreeOpen && nodes.length > 0 && fileTreeData.length > 0 && (
          <button
            onClick={() => setFileTreeOpen(true)}
            className="absolute left-2 top-2 bg-secondary/80 border border-border/50 p-1.5 rounded hover:bg-secondary transition-colors z-10"
            title="Show file tree"
          >
            <ChevronRight className="w-4 h-4 text-muted-foreground" />
          </button>
        )}
        {/* Graph Area */}
        <div className={`flex-1 relative ${codePanelOpen ? 'w-2/3' : 'w-full'}`}>
          {nodes.length === 0 ? (
            <div className="absolute inset-0 flex items-center justify-center text-muted-foreground">
              <div className="text-center">
                <Maximize2 className="w-12 h-12 mx-auto mb-2 opacity-50" />
                <p>Search for a symbol to visualize its relationships</p>
                <p className="text-sm mt-1">Try: AgentFactory, execute_task, WorkflowOrchestrator</p>
              </div>
            </div>
          ) : (
            <GraphErrorBoundary>
            <ReactFlow
              nodes={nodes}
              edges={edges}
              onNodesChange={onNodesChange}
              onEdgesChange={onEdgesChange}
              onNodeClick={handleNodeClick}
              fitView
              attributionPosition="bottom-right"
            >
              <Background color="hsl(16 30% 20%)" gap={16} />
              <Controls />
              <MiniMap
                nodeColor={(node) => {
                  const style = node.style as any
                  return style?.background || '#e8590c'
                }}
                maskColor="rgba(0, 0, 0, 0.7)"
              />
              <Panel position="top-right" className="bg-secondary/80/90 backdrop-blur-sm p-3 rounded-lg border border-border/50">
                <div className="text-xs space-y-1">
                  {viewMode === 'default' ? (
                    <>
                      <div className="flex items-center gap-2">
                        <div className="w-3 h-3 rounded" style={{ background: nodeColors.function }}></div>
                        <span className="text-foreground/80">Function</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <div className="w-3 h-3 rounded" style={{ background: nodeColors.class }}></div>
                        <span className="text-foreground/80">Class</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <div className="w-3 h-3 rounded" style={{ background: nodeColors.method }}></div>
                        <span className="text-foreground/80">Method</span>
                      </div>
                    </>
                  ) : viewMode === 'clusters' ? (
                    <>
                      <div className="text-foreground/80 font-medium mb-1">Module Clusters</div>
                      {archData?.communities.slice(0, 5).map((c, i) => (
                        <div key={c.cluster_id} className="flex items-center gap-2">
                          <div className="w-3 h-3 rounded" style={{ background: clusterColors[c.cluster_id % clusterColors.length] }}></div>
                          <span className="text-foreground/80">Cluster {c.cluster_id} ({c.size})</span>
                        </div>
                      ))}
                      {archData && archData.modularity_score > 0 && (
                        <div className="text-muted-foreground/70 mt-1">
                          Modularity: {archData.modularity_score.toFixed(3)}
                        </div>
                      )}
                    </>
                  ) : (
                    <>
                      <div className="text-foreground/80 font-medium mb-1">Complexity Heatmap</div>
                      <div className="flex items-center gap-1">
                        <div className="w-3 h-3 rounded" style={{ background: 'rgb(0, 200, 80)' }}></div>
                        <span className="text-muted-foreground">Low</span>
                        <div className="w-3 h-3 rounded ml-1" style={{ background: 'rgb(255, 200, 60)' }}></div>
                        <span className="text-muted-foreground">Med</span>
                        <div className="w-3 h-3 rounded ml-1" style={{ background: 'rgb(255, 60, 60)' }}></div>
                        <span className="text-muted-foreground">High</span>
                      </div>
                    </>
                  )}
                </div>
              </Panel>
            </ReactFlow>
            </GraphErrorBoundary>
          )}
        </div>

        {/* PRD-62: Code Snippet Panel (US-017) */}
        {codePanelOpen && (
          <div className="w-1/3 border-l border-border/50 bg-background/80 flex flex-col">
            <div className="flex items-center justify-between p-3 border-b border-border/50">
              <div className="flex items-center gap-2 text-sm text-foreground/80">
                <Code className="w-4 h-4" />
                <span className="font-medium">Symbol Details</span>
              </div>
              <button
                onClick={() => setCodePanelOpen(false)}
                className="text-muted-foreground hover:text-white p-1"
              >
                <ChevronRight className="w-4 h-4" />
              </button>
            </div>

            {selectedNode ? (
              <div className="flex-1 overflow-auto p-3 space-y-3">
                <div>
                  <div className="text-white font-mono text-sm">{selectedNode.name}</div>
                  <div className="text-muted-foreground text-xs mt-1">{selectedNode.file}</div>
                  {selectedNode.line && (
                    <div className="text-muted-foreground/70 text-xs">Line {selectedNode.line}</div>
                  )}
                </div>

                {selectedNode.signature && (
                  <div>
                    <div className="text-muted-foreground text-xs mb-1">Signature</div>
                    <pre className="bg-secondary/80 p-2 rounded text-xs text-success overflow-x-auto">
                      {selectedNode.signature}
                    </pre>
                  </div>
                )}

                {selectedNode.docstring && (
                  <div>
                    <div className="text-muted-foreground text-xs mb-1">Documentation</div>
                    <div className="bg-secondary/80 p-2 rounded text-xs text-foreground/80">
                      {selectedNode.docstring}
                    </div>
                  </div>
                )}

                {selectedNode.code_snippet && (
                  <div>
                    <div className="text-muted-foreground text-xs mb-1">Source Code</div>
                    <pre className="bg-secondary/80 p-2 rounded text-xs text-foreground/90 overflow-x-auto whitespace-pre-wrap max-h-[300px] overflow-y-auto">
                      {selectedNode.code_snippet}
                    </pre>
                  </div>
                )}
              </div>
            ) : (
              <div className="flex-1 flex items-center justify-center text-muted-foreground/70 text-sm">
                Click a node to view details
              </div>
            )}
          </div>
        )}

        {/* Toggle code panel button */}
        {!codePanelOpen && nodes.length > 0 && (
          <button
            onClick={() => setCodePanelOpen(true)}
            className="absolute right-2 top-2 bg-secondary/80 border border-border/50 p-1.5 rounded hover:bg-secondary transition-colors z-10"
            title="Show code panel"
          >
            <ChevronLeft className="w-4 h-4 text-muted-foreground" />
          </button>
        )}
      </div>
    </div>
  )
}
