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
import { Search, Download, ZoomIn, ZoomOut, Maximize2 } from 'lucide-react'

interface CodeGraphVisualizationProps {
  project: string
}

interface CallGraphNode {
  id: string
  symbol: string
  type: string
  file: string
  calls: string[]
  calledBy: string[]
}

const nodeColors = {
  function: '#3b82f6',
  class: '#10b981',
  method: '#8b5cf6',
  import: '#f59e0b',
}

export function CodeGraphVisualization({ project }: CodeGraphVisualizationProps) {
  const [nodes, setNodes, onNodesChange] = useNodesState([])
  const [edges, setEdges, onEdgesChange] = useEdgesState([])
  const [searchQuery, setSearchQuery] = useState('')
  const [selectedSymbol, setSelectedSymbol] = useState<string | null>(null)
  const [graphType, setGraphType] = useState<'calls' | 'imports' | 'extends'>('calls')
  const [depth, setDepth] = useState(2)
  const [direction, setDirection] = useState<'outgoing' | 'incoming' | 'both'>('outgoing')
  const [loading, setLoading] = useState(false)

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
      
      console.log('📊 API Response:', data)
      console.log('📊 Nodes count:', data.nodes?.length)
      console.log('📊 Edges count:', data.edges?.length)
      
      // Transform API response to ReactFlow nodes and edges
      const flowNodes: Node[] = []
      const flowEdges: Edge[] = []

      // Create nodes from API data
      if (data.nodes && Array.isArray(data.nodes)) {
        data.nodes.forEach((node: any, index: number) => {
          const symbolType = node.type || 'function'
          const isRoot = node.symbol === symbol
          
          flowNodes.push({
            id: node.symbol,
            type: 'default',
            data: {
              label: node.name,
            },
            position: { 
              x: isRoot ? 400 : (index % 3) * 300 + 100, 
              y: isRoot ? 300 : Math.floor(index / 3) * 150 + 50 
            },
            style: {
              background: nodeColors[symbolType as keyof typeof nodeColors] || nodeColors.function,
              color: 'white',
              border: isRoot ? '3px solid #ef4444' : '1px solid #94a3b8',
              borderRadius: '8px',
              padding: '12px',
              minWidth: '150px',
              fontSize: '12px',
            },
          })
        })
      }

      // Create edges from API data
      if (data.edges && Array.isArray(data.edges)) {
        data.edges.forEach((edge: any) => {
          const edgeColor = edge.type === 'extends' ? '#10b981' : 
                           edge.type === 'calls' ? '#3b82f6' : '#f59e0b'
          
          flowEdges.push({
            id: `${edge.from}->${edge.to}`,
            source: edge.from,
            target: edge.to,
            type: 'smoothstep',
            animated: edge.type === 'calls',
            label: edge.type,
            style: { stroke: edgeColor, strokeWidth: 2 },
            markerEnd: {
              type: MarkerType.ArrowClosed,
              color: edgeColor,
            },
          })
        })
      }

      console.log('📊 Created flowNodes:', flowNodes.length)
      console.log('📊 Created flowEdges:', flowEdges.length)
      console.log('📊 Sample node:', flowNodes[0])
      
      setNodes(flowNodes)
      setEdges(flowEdges)
    } catch (error) {
      console.error('Error fetching call graph:', error)
    } finally {
      setLoading(false)
    }
  }, [project, depth, direction, setNodes, setEdges])

  const handleSearch = () => {
    if (searchQuery.trim()) {
      setSelectedSymbol(searchQuery.trim())
      fetchCallGraph(searchQuery.trim())
    }
  }

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
        <div className="flex gap-4 text-sm">
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
        </div>
      </div>

      {/* Graph */}
      <div className="flex-1 relative">
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
              </div>
            </Panel>
          </ReactFlow>
        )}
      </div>
    </div>
  )
}

