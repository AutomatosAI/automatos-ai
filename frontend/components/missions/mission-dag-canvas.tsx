'use client'

import { useCallback, useEffect, useMemo, useState } from 'react'
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
import { MissionTaskNode, type MissionTaskNodeData } from './mission-task-node'
import type { TaskResponse, TaskState } from '@/types/missions'

// ── Types ─────────────────────────────────────────────────────

interface MissionDAGCanvasProps {
  tasks: TaskResponse[]
  mode: 'plan' | 'execution' | 'review'
  selectedTaskId: string | null
  onTaskSelect: (taskId: string) => void
  className?: string
}

const nodeTypes = { missionTask: MissionTaskNode }

// ── Auto-layout (sequential chain based on sequence_number) ───

function layoutTasks(tasks: TaskResponse[]): { nodes: Node<MissionTaskNodeData>[]; edges: Edge[] } {
  const sorted = [...tasks].sort((a, b) => a.sequence_number - b.sequence_number)

  const NODE_WIDTH = 220
  const NODE_HEIGHT = 100
  const GAP_Y = 40

  const nodes: Node<MissionTaskNodeData>[] = sorted.map((task, index) => ({
    id: task.id,
    type: 'missionTask',
    position: {
      x: 50,
      y: index * (NODE_HEIGHT + GAP_Y),
    },
    data: {
      id: task.id,
      title: task.title,
      agentName: null,  // Populated from agent data if available
      agentRole: task.agent_role,
      sequenceNumber: task.sequence_number,
      state: task.state,
      isSelected: false,
      mode: 'execution',
      outputExcerpt: task.output_excerpt,
      attemptNumber: task.attempt_number,
    },
    draggable: false,
  }))

  // Sequential edges: task N → task N+1
  const edges: Edge[] = sorted.slice(0, -1).map((task, index) => {
    const nextTask = sorted[index + 1]
    const isActive = isTaskActive(task.state) || isTaskActive(nextTask.state)

    return {
      id: `edge-${task.id}-${nextTask.id}`,
      source: task.id,
      target: nextTask.id,
      type: 'smoothstep',
      animated: isActive,
      markerEnd: {
        type: MarkerType.ArrowClosed,
        color: isActive ? 'hsl(16, 100%, 60%)' : 'hsl(0, 0%, 40%)',
      },
      style: {
        stroke: isActive ? 'hsl(16, 100%, 60%)' : 'hsl(0, 0%, 25%)',
        strokeWidth: isActive ? 2 : 1,
      },
    }
  })

  return { nodes, edges }
}

function isTaskActive(state: TaskState): boolean {
  return ['assigned', 'running', 'completed', 'verifying', 'retrying'].includes(state)
}

// ── Component ─────────────────────────────────────────────────

export function MissionDAGCanvas({
  tasks,
  mode,
  selectedTaskId,
  onTaskSelect,
  className,
}: MissionDAGCanvasProps) {
  const { nodes: layoutedNodes, edges: layoutedEdges } = useMemo(
    () => layoutTasks(tasks),
    [tasks],
  )

  // Apply selection and mode to node data
  const nodesWithState = useMemo(
    () =>
      layoutedNodes.map((node) => ({
        ...node,
        data: {
          ...node.data,
          isSelected: node.id === selectedTaskId,
          mode,
        },
      })),
    [layoutedNodes, selectedTaskId, mode],
  )

  const [nodes, setNodes, onNodesChange] = useNodesState(nodesWithState)
  const [edges, setEdges, onEdgesChange] = useEdgesState(layoutedEdges)

  // Update nodes when props change
  useEffect(() => {
    setNodes(nodesWithState)
  }, [nodesWithState, setNodes])

  useEffect(() => {
    setEdges(layoutedEdges)
  }, [layoutedEdges, setEdges])

  const onNodeClick = useCallback(
    (_event: React.MouseEvent, node: Node) => {
      onTaskSelect(node.id)
    },
    [onTaskSelect],
  )

  return (
    <div className={className} style={{ width: '100%', height: '100%' }}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onNodeClick={onNodeClick}
        nodeTypes={nodeTypes}
        fitView
        fitViewOptions={{ padding: 0.2 }}
        minZoom={0.3}
        maxZoom={1.5}
        proOptions={{ hideAttribution: true }}
      >
        <Background gap={20} size={1} color="hsl(0, 0%, 15%)" />
        <Controls
          showInteractive={false}
          className="!bg-background/80 !border-border !rounded-lg !shadow-lg"
        />
        <MiniMap
          nodeColor={() => 'hsl(16, 100%, 60%)'}
          maskColor="rgba(0, 0, 0, 0.8)"
          style={{ width: 120, height: 80 }}
          className="!bg-background/80 !border-border !rounded-lg"
        />
      </ReactFlow>
    </div>
  )
}
