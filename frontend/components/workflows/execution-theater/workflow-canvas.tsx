'use client'

import { useCallback, useEffect, useState, useMemo } from 'react'
import ReactFlow, {
  Node,
  Edge,
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
  MarkerType,
  Position
} from 'reactflow'
import 'reactflow/dist/style.css'
import { Bot, Database, FileText, Zap, Brain } from 'lucide-react'
import { Badge } from '@/components/ui/badge'

interface WorkflowCanvasProps {
  workflow: any
  isExecuting: boolean
  onAgentSelect: (agentId: number) => void
  selectedAgent: number | null
}

// Custom Agent Node Component
function AgentNode({ data }: { data: any }) {
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'executing': return 'border-info bg-info/10'
      case 'completed': return 'border-success bg-success/10'
      case 'waiting': return 'border-warning bg-warning/10'
      case 'error': return 'border-destructive bg-destructive/10'
      default: return 'border-border bg-secondary/50'
    }
  }

  const getStatusIcon = () => {
    switch (data.status) {
      case 'executing': return <div className="w-2 h-2 rounded-full bg-info animate-pulse" />
      case 'completed': return <div className="w-2 h-2 rounded-full bg-success" />
      case 'waiting': return <div className="w-2 h-2 rounded-full bg-warning animate-pulse" />
      case 'error': return <div className="w-2 h-2 rounded-full bg-destructive" />
      default: return <div className="w-2 h-2 rounded-full bg-muted-foreground" />
    }
  }

  return (
    <div className={`px-4 py-3 rounded-lg border-2 ${getStatusColor(data.status)} ${data.selected ? 'ring-2 ring-primary' : ''} min-w-[200px]`}>
      <div className="flex items-center space-x-2 mb-2">
        <Bot className="w-5 h-5 text-primary" />
        <span className="font-semibold">{data.label}</span>
        {getStatusIcon()}
      </div>
      <div className="text-xs text-muted-foreground mb-2">{data.agent_type}</div>
      {data.currentTask && (
        <div className="text-xs text-primary mt-2 font-medium">{data.currentTask}</div>
      )}
      {data.progress !== undefined && (
        <div className="mt-2">
          <div className="flex justify-between text-xs mb-1">
            <span>Progress</span>
            <span>{Math.round(data.progress)}%</span>
          </div>
          <div className="h-1 bg-secondary rounded-full overflow-hidden">
            <div 
              className="h-full bg-primary transition-all duration-500"
              style={{ width: `${data.progress}%` }}
            />
          </div>
        </div>
      )}
    </div>
  )
}

// Custom Resource Node Component
function ResourceNode({ data }: { data: any }) {
  const getIcon = () => {
    switch (data.resourceType) {
      case 'memory': return <Brain className="w-5 h-5 text-agent" />
      case 'database': return <Database className="w-5 h-5 text-info" />
      case 'document': return <FileText className="w-5 h-5 text-success" />
      case 'tool': return <Zap className="w-5 h-5 text-warning" />
      default: return <Database className="w-5 h-5 text-muted-foreground" />
    }
  }

  const getColor = () => {
    switch (data.resourceType) {
      case 'memory': return 'border-agent/50 bg-agent/10'
      case 'document': return 'border-success/50 bg-success/10'
      case 'tool': return 'border-warning/50 bg-warning/10'
      default: return 'border-border/50 bg-background/50'
    }
  }

  return (
    <div className={`px-4 py-3 rounded-lg border-2 ${getColor()} backdrop-blur min-w-[180px] cursor-pointer hover:scale-105 transition-transform`}>
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center space-x-2">
          {getIcon()}
          <span className="text-sm font-semibold">{data.label}</span>
        </div>
        {data.activeConnections > 0 && (
          <div className="w-2 h-2 rounded-full bg-info animate-pulse" />
        )}
      </div>
      <div className="space-y-1">
        {data.count !== undefined && (
          <div className="text-xs text-muted-foreground">
            <span className="font-mono">{data.count}</span> items stored
          </div>
        )}
        {data.activeConnections > 0 && (
          <div className="text-xs text-info font-medium">
            {data.activeConnections} active {data.activeConnections === 1 ? 'connection' : 'connections'}
          </div>
        )}
        {data.lastOperation && (
          <div className="text-xs text-muted-foreground italic">
            {data.lastOperation}
          </div>
        )}
      </div>
    </div>
  )
}

const nodeTypes = {
  agent: AgentNode,
  resource: ResourceNode
}

export function WorkflowCanvas({
  workflow,
  isExecuting,
  onAgentSelect,
  selectedAgent
}: WorkflowCanvasProps) {
  const [nodes, setNodes, onNodesChange] = useNodesState([])
  const [edges, setEdges, onEdgesChange] = useEdgesState([])
  const [selectedResource, setSelectedResource] = useState<any>(null)

  // Generate nodes and edges from workflow data
  useEffect(() => {
    // Mock data for visualization
    const mockAgents = [
      { 
        id: 1, 
        name: 'Context Optimizer',
        agent_type: 'Analyzer',
        status: 'executing',
        currentTask: 'Analyzing context...',
        progress: 65
      },
      { 
        id: 2, 
        name: 'Code Reviewer',
        agent_type: 'Validator',
        status: 'completed',
        currentTask: 'Review complete',
        progress: 100
      },
      { 
        id: 3, 
        name: 'Report Writer',
        agent_type: 'Generator',
        status: 'pending',
        currentTask: 'Waiting...',
        progress: 0
      }
    ]
    
    const mockProgress = {
      memory_operations: 12,
      rag_operations: 5,
      tool_calls: 3
    }
    
    const agents = workflow?.agents?.length > 0 ? workflow.agents : mockAgents
    const execution = workflow?.execution
    const liveProgress = workflow?.liveProgress || mockProgress
    
    // Create agent nodes - stacked vertically
    const agentNodes: Node[] = agents.map((agent: any, index: number) => {
      const agentProgress = liveProgress.agents?.[agent.id]
      
      return {
        id: `agent-${agent.id}`,
        type: 'agent',
        position: { 
          x: 100, 
          y: 50 + (index * 200) // Stack vertically with 200px spacing
        },
        data: {
          label: agent.name,
          agent_type: agent.agent_type,
          status: agentProgress?.status || agent.status || 'idle',
          currentTask: agentProgress?.current_task || agent.currentTask,
          progress: agentProgress?.progress !== undefined ? agentProgress.progress : agent.progress || 0,
          selected: selectedAgent === agent.id
        },
        draggable: true
      }
    })

    // Create resource nodes - stacked vertically beside agents
    const resourceNodes: Node[] = [
      {
        id: 'memory-pool',
        type: 'resource',
        position: { x: 450, y: 50 },
        data: {
          label: 'Shared Memory',
          resourceType: 'memory',
          count: liveProgress.memory_operations || 12,
          activeConnections: workflow ? 0 : 2, // Mock: Context Optimizer + Code Reviewer writing
          lastOperation: workflow ? '' : 'Context Optimizer: Writing analysis results...'
        }
      },
      {
        id: 'document-rag',
        type: 'resource',
        position: { x: 450, y: 250 },
        data: {
          label: 'Document RAG',
          resourceType: 'document',
          count: workflow?.documents?.length || 5,
          activeConnections: workflow ? 0 : 1, // Mock: Context Optimizer querying
          lastOperation: workflow ? '' : 'Context Optimizer: Querying "API documentation"'
        }
      },
      {
        id: 'tools',
        type: 'resource',
        position: { x: 450, y: 450 },
        data: {
          label: 'Tools',
          resourceType: 'tool',
          count: workflow?.tools?.length || 3,
          activeConnections: workflow ? 0 : 0,
          lastOperation: workflow ? '' : 'Last used: Code Reviewer called "lint_checker"'
        }
      }
    ]

    // Create edges between agents (workflow flow)
    const agentEdges: Edge[] = []
    agents.forEach((agent: any, index: number) => {
      if (index < agents.length - 1) {
        agentEdges.push({
          id: `edge-${agent.id}-${agents[index + 1].id}`,
          source: `agent-${agent.id}`,
          target: `agent-${agents[index + 1].id}`,
          type: 'smoothstep',
          animated: isExecuting && liveProgress.agents?.[agent.id]?.status === 'executing',
          markerEnd: {
            type: MarkerType.ArrowClosed,
            color: '#6366f1'
          },
          style: { stroke: '#6366f1' }
        })
      }
    })

    // Create edges from agents to resources with detailed connections
    const resourceEdges: Edge[] = []
    
    // Mock detailed connections showing different operations
    if (!workflow || agents.length === mockAgents.length) {
      // Context Optimizer → Memory (writing)
      resourceEdges.push({
        id: 'edge-1-memory-write',
        source: 'agent-1',
        target: 'memory-pool',
        type: 'smoothstep',
        animated: true,
        label: 'Writing',
        labelStyle: { fill: '#a855f7', fontSize: 10, fontWeight: 600 },
        labelBgStyle: { fill: '#1a1a1a', fillOpacity: 0.8 },
        style: { 
          stroke: '#a855f7',
          strokeWidth: 2
        },
        markerEnd: {
          type: MarkerType.ArrowClosed,
          color: '#a855f7'
        }
      })
      
      // Context Optimizer → RAG (querying)
      resourceEdges.push({
        id: 'edge-1-rag-query',
        source: 'agent-1',
        target: 'document-rag',
        type: 'smoothstep',
        animated: true,
        label: 'RAG Query',
        labelStyle: { fill: '#22c55e', fontSize: 10, fontWeight: 600 },
        labelBgStyle: { fill: '#1a1a1a', fillOpacity: 0.8 },
        style: { 
          stroke: '#22c55e',
          strokeWidth: 2
        },
        markerEnd: {
          type: MarkerType.ArrowClosed,
          color: '#22c55e'
        }
      })
      
      // Code Reviewer → Memory (reading)
      resourceEdges.push({
        id: 'edge-2-memory-read',
        source: 'agent-2',
        target: 'memory-pool',
        type: 'smoothstep',
        animated: false,
        label: 'Reading',
        labelStyle: { fill: '#a855f7', fontSize: 10 },
        labelBgStyle: { fill: '#1a1a1a', fillOpacity: 0.8 },
        style: { 
          stroke: '#a855f7',
          strokeWidth: 1,
          strokeDasharray: '5,5'
        }
      })
      
      // Code Reviewer → Tools (completed)
      resourceEdges.push({
        id: 'edge-2-tools',
        source: 'agent-2',
        target: 'tools',
        type: 'smoothstep',
        animated: false,
        label: 'Used Tool',
        labelStyle: { fill: '#eab308', fontSize: 10 },
        labelBgStyle: { fill: '#1a1a1a', fillOpacity: 0.8 },
        style: { 
          stroke: '#eab308',
          strokeWidth: 1,
          strokeDasharray: '5,5'
        }
      })
      
      // Report Writer → Memory (pending, dashed)
      resourceEdges.push({
        id: 'edge-3-memory-pending',
        source: 'agent-3',
        target: 'memory-pool',
        type: 'smoothstep',
        animated: false,
        label: 'Will read',
        labelStyle: { fill: '#6b7280', fontSize: 10 },
        labelBgStyle: { fill: '#1a1a1a', fillOpacity: 0.8 },
        style: { 
          stroke: '#6b7280',
          strokeWidth: 1,
          strokeDasharray: '5,5',
          opacity: 0.5
        }
      })
    } else {
      // Real workflow: simple dashed connections
      agents.forEach((agent: any) => {
        resourceEdges.push({
          id: `edge-${agent.id}-memory`,
          source: `agent-${agent.id}`,
          target: 'memory-pool',
          type: 'step',
          animated: false,
          style: { 
            stroke: '#a855f7',
            strokeDasharray: '5,5'
          }
        })
      })
    }

    setNodes([...agentNodes, ...resourceNodes])
    setEdges([...agentEdges, ...resourceEdges])
  }, [workflow, isExecuting, selectedAgent])

  const onNodeClick = useCallback((event: React.MouseEvent, node: Node) => {
    if (node.type === 'agent') {
      const agentId = parseInt(node.id.replace('agent-', ''))
      onAgentSelect(agentId)
    } else if (node.type === 'resource') {
      // Mock resource data for drill-down
      const mockResourceData: any = {
        'memory-pool': {
          items: [
            { id: 1, agent: 'Context Optimizer', action: 'WRITE', content: 'Extracted 15 key concepts from API documentation', timestamp: '10:23:45' },
            { id: 2, agent: 'Context Optimizer', action: 'WRITE', content: 'Found 3 high-priority issues in context analysis', timestamp: '10:23:47' },
            { id: 3, agent: 'Code Reviewer', action: 'READ', content: 'Retrieved context analysis results', timestamp: '10:23:50' },
            { id: 4, agent: 'Code Reviewer', action: 'WRITE', content: 'Added 3 code quality findings to shared memory', timestamp: '10:23:52' }
          ]
        },
        'document-rag': {
          items: [
            { id: 1, agent: 'Context Optimizer', query: 'API authentication methods', results: 12, timestamp: '10:23:44' },
            { id: 2, agent: 'Context Optimizer', query: 'Error handling patterns', results: 8, timestamp: '10:23:46' },
            { id: 3, agent: 'Report Writer', query: 'Best practices documentation', results: 15, timestamp: '10:23:55' }
          ]
        },
        'tools': {
          items: [
            { id: 1, agent: 'Code Reviewer', tool: 'lint_checker', result: 'Found 3 issues', timestamp: '10:23:51' },
            { id: 2, agent: 'Code Reviewer', tool: 'complexity_analyzer', result: 'Complexity score: 7.2', timestamp: '10:23:53' }
          ]
        }
      }
      
      setSelectedResource({
        ...node.data,
        id: node.id,
        details: mockResourceData[node.id] || { items: [] }
      })
    }
  }, [onAgentSelect])

  return (
    <div className="w-full h-full relative">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onNodeClick={onNodeClick}
        nodeTypes={nodeTypes}
        fitView
        minZoom={0.3}
        maxZoom={1.2}
        defaultEdgeOptions={{
          animated: false
        }}
      >
        <Background />
        <Controls />
        <MiniMap 
          nodeColor={(node) => {
            if (node.type === 'agent') return '#6366f1'
            return '#666'
          }}
          maskColor="rgba(0, 0, 0, 0.8)"
          style={{
            width: 100,
            height: 80
          }}
        />
      </ReactFlow>

      {/* Resource Details Panel */}
      {selectedResource && (
        <div className="absolute top-4 right-4 w-96 bg-background/95 backdrop-blur border border-border rounded-lg shadow-2xl z-50 max-h-[80%] overflow-hidden flex flex-col">
          <div className="p-4 border-b border-border flex items-center justify-between">
            <div className="flex items-center gap-2">
              {selectedResource.resourceType === 'memory' && <Brain className="w-5 h-5 text-agent" />}
              {selectedResource.resourceType === 'document' && <FileText className="w-5 h-5 text-success" />}
              {selectedResource.resourceType === 'tool' && <Zap className="w-5 h-5 text-warning" />}
              <h3 className="font-semibold text-lg">{selectedResource.label}</h3>
            </div>
            <button
              onClick={() => setSelectedResource(null)}
              className="text-muted-foreground hover:text-foreground transition-colors"
            >
              ✕
            </button>
          </div>
          
          <div className="p-4 overflow-y-auto flex-1">
            <div className="space-y-3">
              {selectedResource.id === 'memory-pool' && selectedResource.details.items.map((item: any) => (
                <div key={item.id} className="p-3 bg-agent/10 border border-agent/30 rounded-lg">
                  <div className="flex items-center justify-between mb-2">
                    <Badge variant="outline" className="text-xs">
                      {item.agent}
                    </Badge>
                    <span className={`text-xs font-bold ${item.action === 'WRITE' ? 'text-success' : 'text-info'}`}>
                      {item.action}
                    </span>
                  </div>
                  <p className="text-sm">{item.content}</p>
                  <p className="text-xs text-muted-foreground mt-2 font-mono">{item.timestamp}</p>
                </div>
              ))}
              
              {selectedResource.id === 'document-rag' && selectedResource.details.items.map((item: any) => (
                <div key={item.id} className="p-3 bg-success/10 border border-success/30 rounded-lg">
                  <div className="flex items-center justify-between mb-2">
                    <Badge variant="outline" className="text-xs">
                      {item.agent}
                    </Badge>
                    <span className="text-xs text-success font-mono">{item.results} results</span>
                  </div>
                  <p className="text-sm font-medium mb-1">Query:</p>
                  <p className="text-sm text-muted-foreground italic">"{item.query}"</p>
                  <p className="text-xs text-muted-foreground mt-2 font-mono">{item.timestamp}</p>
                </div>
              ))}
              
              {selectedResource.id === 'tools' && selectedResource.details.items.map((item: any) => (
                <div key={item.id} className="p-3 bg-warning/10 border border-warning/30 rounded-lg">
                  <div className="flex items-center justify-between mb-2">
                    <Badge variant="outline" className="text-xs">
                      {item.agent}
                    </Badge>
                    <span className="text-xs text-warning font-mono">{item.tool}</span>
                  </div>
                  <p className="text-sm">{item.result}</p>
                  <p className="text-xs text-muted-foreground mt-2 font-mono">{item.timestamp}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

