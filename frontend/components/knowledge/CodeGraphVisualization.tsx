'use client'

import React, { useState, useEffect, useCallback } from 'react'
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
import { Search, Download, ZoomIn, ZoomOut, Maximize2, Flame, Layers, Code, ChevronRight, ChevronLeft, AlertTriangle } from 'lucide-react'

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

const nodeColors = {
  function: '#3b82f6',
  class: '#10b981',
  method: '#8b5cf6',
  import: '#f59e0b',
  interface: '#ec4899',
}

// Cluster palette for Louvain communities
const clusterColors = [
  '#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6',
  '#ec4899', '#06b6d4', '#f97316', '#84cc16', '#a855f7',
  '#14b8a6', '#e11d48', '#0ea5e9', '#d946ef', '#65a30d',
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

  // PRD-62: Fetch architecture data
  const fetchArchitecture = useCallback(async () => {
    if (!projectId) return
    setArchLoading(true)
    try {
      const res = await fetch(`/api/code-graph/projects/${projectId}/architecture`)
      if (res.ok) {
        const data = await res.json()
        setArchData(data)
      }
    } catch (err) {
      console.error('Architecture fetch failed:', err)
    } finally {
      setArchLoading(false)
    }
  }, [projectId])

  useEffect(() => {
    if (projectId) fetchArchitecture()
  }, [projectId, fetchArchitecture])

  const fetchCallGraph = useCallback(async (symbol: string) => {
    if (!symbol || !project) return

    setLoading(true)
    try {
      const response = await fetch(
        `/api/code-graph/call-graph?` +
        `project=${encodeURIComponent(project)}&` +
        `symbol=${encodeURIComponent(symbol)}&` +
        `depth=${depth}&` +
        `direction=${direction}`
      )

      if (!response.ok) throw new Error('Failed to fetch call graph')

      const data = await response.json()

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
              border: isRoot ? '3px solid #ef4444' : isHotspot ? '2px dashed #f59e0b' : '1px solid #94a3b8',
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
                           edge.type === 'extends' ? '#10b981' :
                           edge.type === 'calls' ? '#3b82f6' : '#f59e0b'

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
    <div className="flex flex-col h-[600px] bg-slate-900/50 backdrop-blur-sm rounded-lg border border-slate-700">
      {/* Controls */}
      <div className="p-4 border-b border-slate-700 space-y-3">
        {/* Search */}
        <div className="flex gap-2">
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" />
            <input
              type="text"
              placeholder="Enter symbol name (e.g., AgentFactory, execute_task)..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
              className="w-full pl-10 pr-4 py-2 bg-slate-800 border border-slate-600 rounded-lg text-white placeholder-slate-400 focus:outline-none focus:border-blue-500"
            />
          </div>
          <button
            onClick={handleSearch}
            disabled={loading || !searchQuery.trim()}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-slate-700 disabled:cursor-not-allowed text-white rounded-lg font-medium transition-colors"
          >
            {loading ? 'Loading...' : 'Visualize'}
          </button>
        </div>

        {/* Options */}
        <div className="flex gap-4 text-sm flex-wrap">
          {/* Graph Type */}
          <div className="flex items-center gap-2">
            <label className="text-slate-300">Type:</label>
            <select
              value={graphType}
              onChange={(e) => setGraphType(e.target.value as any)}
              className="bg-slate-800 border border-slate-600 rounded px-2 py-1 text-white"
            >
              <option value="calls">Call Graph</option>
              <option value="imports">Dependencies</option>
              <option value="extends">Inheritance</option>
            </select>
          </div>

          {/* Direction */}
          <div className="flex items-center gap-2">
            <label className="text-slate-300">Direction:</label>
            <select
              value={direction}
              onChange={(e) => setDirection(e.target.value as any)}
              className="bg-slate-800 border border-slate-600 rounded px-2 py-1 text-white"
            >
              <option value="outgoing">Outgoing</option>
              <option value="incoming">Incoming</option>
              <option value="both">Both</option>
            </select>
          </div>

          {/* Depth */}
          <div className="flex items-center gap-2">
            <label className="text-slate-300">Depth:</label>
            <input
              type="number"
              min="1"
              max="5"
              value={depth}
              onChange={(e) => setDepth(parseInt(e.target.value))}
              className="w-16 bg-slate-800 border border-slate-600 rounded px-2 py-1 text-white"
            />
          </div>

          {/* PRD-62: View Mode Toggle */}
          <div className="flex items-center gap-1 ml-auto">
            <button
              onClick={() => setViewMode('default')}
              className={`px-2 py-1 rounded text-xs transition-colors ${viewMode === 'default' ? 'bg-blue-600 text-white' : 'bg-slate-800 text-slate-400 hover:text-white'}`}
              title="Default colors by type"
            >
              Default
            </button>
            <button
              onClick={() => setViewMode('clusters')}
              disabled={!archData}
              className={`px-2 py-1 rounded text-xs transition-colors flex items-center gap-1 ${viewMode === 'clusters' ? 'bg-purple-600 text-white' : 'bg-slate-800 text-slate-400 hover:text-white'} disabled:opacity-30`}
              title="Color by Louvain module cluster"
            >
              <Layers className="w-3 h-3" /> Clusters
            </button>
            <button
              onClick={() => setViewMode('heatmap')}
              disabled={!archData}
              className={`px-2 py-1 rounded text-xs transition-colors flex items-center gap-1 ${viewMode === 'heatmap' ? 'bg-orange-600 text-white' : 'bg-slate-800 text-slate-400 hover:text-white'} disabled:opacity-30`}
              title="Heatmap by complexity (green=low, red=high)"
            >
              <Flame className="w-3 h-3" /> Heatmap
            </button>
            {archData && archData.hotspots.length > 0 && (
              <span className="ml-2 px-2 py-0.5 bg-yellow-900/50 text-yellow-400 text-xs rounded flex items-center gap-1">
                <AlertTriangle className="w-3 h-3" /> {archData.hotspots.length} hotspots
              </span>
            )}
          </div>
        </div>
      </div>

      {/* Graph + Code Panel */}
      <div className="flex-1 flex relative">
        {/* Graph Area */}
        <div className={`flex-1 relative ${codePanelOpen ? 'w-2/3' : 'w-full'}`}>
          {nodes.length === 0 ? (
            <div className="absolute inset-0 flex items-center justify-center text-slate-400">
              <div className="text-center">
                <Maximize2 className="w-12 h-12 mx-auto mb-2 opacity-50" />
                <p>Search for a symbol to visualize its relationships</p>
                <p className="text-sm mt-1">Try: AgentFactory, execute_task, WorkflowOrchestrator</p>
              </div>
            </div>
          ) : (
            <ReactFlow
              nodes={nodes}
              edges={edges}
              onNodesChange={onNodesChange}
              onEdgesChange={onEdgesChange}
              onNodeClick={handleNodeClick}
              fitView
              attributionPosition="bottom-right"
            >
              <Background color="#475569" gap={16} />
              <Controls className="bg-slate-800 border-slate-600" />
              <MiniMap
                nodeColor={(node) => {
                  const style = node.style as any
                  return style?.background || '#3b82f6'
                }}
                className="bg-slate-800 border-slate-600"
              />
              <Panel position="top-right" className="bg-slate-800/90 backdrop-blur-sm p-3 rounded-lg border border-slate-700">
                <div className="text-xs space-y-1">
                  {viewMode === 'default' ? (
                    <>
                      <div className="flex items-center gap-2">
                        <div className="w-3 h-3 rounded" style={{ background: nodeColors.function }}></div>
                        <span className="text-slate-300">Function</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <div className="w-3 h-3 rounded" style={{ background: nodeColors.class }}></div>
                        <span className="text-slate-300">Class</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <div className="w-3 h-3 rounded" style={{ background: nodeColors.method }}></div>
                        <span className="text-slate-300">Method</span>
                      </div>
                    </>
                  ) : viewMode === 'clusters' ? (
                    <>
                      <div className="text-slate-300 font-medium mb-1">Module Clusters</div>
                      {archData?.communities.slice(0, 5).map((c, i) => (
                        <div key={c.cluster_id} className="flex items-center gap-2">
                          <div className="w-3 h-3 rounded" style={{ background: clusterColors[c.cluster_id % clusterColors.length] }}></div>
                          <span className="text-slate-300">Cluster {c.cluster_id} ({c.size})</span>
                        </div>
                      ))}
                      {archData && archData.modularity_score > 0 && (
                        <div className="text-slate-500 mt-1">
                          Modularity: {archData.modularity_score.toFixed(3)}
                        </div>
                      )}
                    </>
                  ) : (
                    <>
                      <div className="text-slate-300 font-medium mb-1">Complexity Heatmap</div>
                      <div className="flex items-center gap-1">
                        <div className="w-3 h-3 rounded" style={{ background: 'rgb(0, 200, 80)' }}></div>
                        <span className="text-slate-400">Low</span>
                        <div className="w-3 h-3 rounded ml-1" style={{ background: 'rgb(255, 200, 60)' }}></div>
                        <span className="text-slate-400">Med</span>
                        <div className="w-3 h-3 rounded ml-1" style={{ background: 'rgb(255, 60, 60)' }}></div>
                        <span className="text-slate-400">High</span>
                      </div>
                    </>
                  )}
                </div>
              </Panel>
            </ReactFlow>
          )}
        </div>

        {/* PRD-62: Code Snippet Panel (US-017) */}
        {codePanelOpen && (
          <div className="w-1/3 border-l border-slate-700 bg-slate-900/80 flex flex-col">
            <div className="flex items-center justify-between p-3 border-b border-slate-700">
              <div className="flex items-center gap-2 text-sm text-slate-300">
                <Code className="w-4 h-4" />
                <span className="font-medium">Symbol Details</span>
              </div>
              <button
                onClick={() => setCodePanelOpen(false)}
                className="text-slate-400 hover:text-white p-1"
              >
                <ChevronRight className="w-4 h-4" />
              </button>
            </div>

            {selectedNode ? (
              <div className="flex-1 overflow-auto p-3 space-y-3">
                <div>
                  <div className="text-white font-mono text-sm">{selectedNode.name}</div>
                  <div className="text-slate-400 text-xs mt-1">{selectedNode.file}</div>
                  {selectedNode.line && (
                    <div className="text-slate-500 text-xs">Line {selectedNode.line}</div>
                  )}
                </div>

                {selectedNode.signature && (
                  <div>
                    <div className="text-slate-400 text-xs mb-1">Signature</div>
                    <pre className="bg-slate-800 p-2 rounded text-xs text-green-400 overflow-x-auto">
                      {selectedNode.signature}
                    </pre>
                  </div>
                )}

                {selectedNode.docstring && (
                  <div>
                    <div className="text-slate-400 text-xs mb-1">Documentation</div>
                    <div className="bg-slate-800 p-2 rounded text-xs text-slate-300">
                      {selectedNode.docstring}
                    </div>
                  </div>
                )}

                {selectedNode.code_snippet && (
                  <div>
                    <div className="text-slate-400 text-xs mb-1">Source Code</div>
                    <pre className="bg-slate-800 p-2 rounded text-xs text-slate-200 overflow-x-auto whitespace-pre-wrap max-h-[300px] overflow-y-auto">
                      {selectedNode.code_snippet}
                    </pre>
                  </div>
                )}
              </div>
            ) : (
              <div className="flex-1 flex items-center justify-center text-slate-500 text-sm">
                Click a node to view details
              </div>
            )}
          </div>
        )}

        {/* Toggle code panel button */}
        {!codePanelOpen && nodes.length > 0 && (
          <button
            onClick={() => setCodePanelOpen(true)}
            className="absolute right-2 top-2 bg-slate-800 border border-slate-700 p-1.5 rounded hover:bg-slate-700 transition-colors z-10"
            title="Show code panel"
          >
            <ChevronLeft className="w-4 h-4 text-slate-400" />
          </button>
        )}
      </div>
    </div>
  )
}
