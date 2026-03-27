'use client'

import React, { useCallback, useEffect, useMemo } from 'react'
import ReactFlow, {
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
  MarkerType,
  type Node,
  type Edge,
} from 'reactflow'
import 'reactflow/dist/style.css'
import { OrgChartNode, type OrgChartNodeData } from './org-chart-node'

// ── Types ─────────────────────────────────────────────────────

export interface OrgChartAgent {
  id: number
  name: string
  job_title: string | null
  team: string | null
  status: string
  model: string | null
  skills: string[]
  tools_count: number
  reports_to_id: number | null
  direct_reports_count: number
  is_system_agent?: boolean
}

interface OrgChartCanvasProps {
  agents: OrgChartAgent[]
  edges: Array<{ from: number; to: number }>
  onAgentSelect?: (agentId: number) => void
  className?: string
}

const nodeTypes = { orgAgent: OrgChartNode }

// ── Constants ────────────────────────────────────────────────

const NODE_WIDTH = 220
const NODE_HEIGHT = 120
const GAP_X = 40
const GAP_Y = 60

// ── Hierarchical tree layout ────────────────────────────────

interface TreeNode {
  agent: OrgChartAgent
  children: TreeNode[]
}

function buildTree(agents: OrgChartAgent[], edgeList: Array<{ from: number; to: number }>): TreeNode[] {
  const agentMap = new Map(agents.map(a => [a.id, a]))
  const childrenMap = new Map<number, number[]>()

  for (const e of edgeList) {
    const children = childrenMap.get(e.from) ?? []
    children.push(e.to)
    childrenMap.set(e.from, children)
  }

  const childIds = new Set(edgeList.map(e => e.to))
  const rootIds = agents
    .filter(a => !childIds.has(a.id))
    .map(a => a.id)

  function buildSubtree(id: number): TreeNode | null {
    const agent = agentMap.get(id)
    if (!agent) return null
    const kids = (childrenMap.get(id) ?? [])
      .map(buildSubtree)
      .filter((n): n is TreeNode => n !== null)
    // Sort children by team then name for consistent layout
    kids.sort((a, b) => {
      const teamA = a.agent.team ?? ''
      const teamB = b.agent.team ?? ''
      if (teamA !== teamB) return teamA.localeCompare(teamB)
      return a.agent.name.localeCompare(b.agent.name)
    })
    return { agent, children: kids }
  }

  return rootIds
    .map(buildSubtree)
    .filter((n): n is TreeNode => n !== null)
}

function layoutTree(roots: TreeNode[]): { nodes: Node<OrgChartNodeData>[]; edges: Edge[] } {
  const nodes: Node<OrgChartNodeData>[] = []
  const edges: Edge[] = []

  // First pass: compute subtree widths
  function subtreeWidth(node: TreeNode): number {
    if (node.children.length === 0) return NODE_WIDTH
    const childrenWidth = node.children.reduce(
      (sum, child) => sum + subtreeWidth(child) + GAP_X,
      -GAP_X,
    )
    return Math.max(NODE_WIDTH, childrenWidth)
  }

  // Second pass: position nodes
  function positionNode(node: TreeNode, x: number, y: number, parentId?: number): void {
    const nodeId = String(node.agent.id)

    nodes.push({
      id: nodeId,
      type: 'orgAgent',
      position: { x, y },
      data: {
        id: node.agent.id,
        name: node.agent.name,
        jobTitle: node.agent.job_title,
        team: node.agent.team,
        status: node.agent.status,
        model: node.agent.model,
        skills: node.agent.skills,
        toolsCount: node.agent.tools_count,
        directReportsCount: node.agent.direct_reports_count,
        isSystemAgent: node.agent.is_system_agent ?? false,
      },
    })

    if (parentId !== undefined) {
      edges.push({
        id: `e-${parentId}-${node.agent.id}`,
        source: String(parentId),
        target: nodeId,
        type: 'smoothstep',
        animated: false,
        style: {
          stroke: 'hsl(0, 0%, 35%)',
          strokeWidth: 1.5,
        },
        markerEnd: {
          type: MarkerType.ArrowClosed,
          color: 'hsl(0, 0%, 35%)',
          width: 12,
          height: 12,
        },
      })
    }

    if (node.children.length === 0) return

    const totalWidth = subtreeWidth(node)
    let childX = x + NODE_WIDTH / 2 - totalWidth / 2
    const childY = y + NODE_HEIGHT + GAP_Y

    for (const child of node.children) {
      const childWidth = subtreeWidth(child)
      const childCenterX = childX + childWidth / 2 - NODE_WIDTH / 2
      positionNode(child, childCenterX, childY, node.agent.id)
      childX += childWidth + GAP_X
    }
  }

  // Position all root trees side by side
  let rootX = 0
  for (const root of roots) {
    const width = subtreeWidth(root)
    const centerX = rootX + width / 2 - NODE_WIDTH / 2
    positionNode(root, centerX, 0)
    rootX += width + GAP_X * 2
  }

  return { nodes, edges }
}

// ── Component ───────────────────────────────────────────────

export function OrgChartCanvas({ agents, edges: edgeList, onAgentSelect, className }: OrgChartCanvasProps) {
  const { layoutedNodes, layoutedEdges } = useMemo(() => {
    const trees = buildTree(agents, edgeList)
    const { nodes: ln, edges: le } = layoutTree(trees)
    return { layoutedNodes: ln, layoutedEdges: le }
  }, [agents, edgeList])

  const [nodes, setNodes, onNodesChange] = useNodesState(layoutedNodes)
  const [flowEdges, setEdges, onEdgesChange] = useEdgesState(layoutedEdges)

  useEffect(() => {
    setNodes(layoutedNodes)
  }, [layoutedNodes, setNodes])

  useEffect(() => {
    setEdges(layoutedEdges)
  }, [layoutedEdges, setEdges])

  const handleNodeClick = useCallback(
    (_: React.MouseEvent, node: Node<OrgChartNodeData>) => {
      onAgentSelect?.(node.data.id)
    },
    [onAgentSelect],
  )

  return (
    <div className={className} style={{ width: '100%', height: '100%', minHeight: 500 }}>
      <ReactFlow
        nodes={nodes}
        edges={flowEdges}
        nodeTypes={nodeTypes}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onNodeClick={handleNodeClick}
        fitView
        fitViewOptions={{ padding: 0.15 }}
        minZoom={0.2}
        maxZoom={1.5}
        proOptions={{ hideAttribution: true }}
      >
        <Background gap={20} size={1} color="hsl(0, 0%, 12%)" />
        <Controls showInteractive={false} />
        <MiniMap
          nodeColor={(node) => {
            const data = node.data as OrgChartNodeData
            if (data.isSystemAgent) return 'hsl(16, 100%, 60%)'
            if (data.status === 'active') return 'hsl(142, 71%, 45%)'
            return 'hsl(0, 0%, 40%)'
          }}
          maskColor="rgba(0,0,0,0.7)"
        />
      </ReactFlow>
    </div>
  )
}
