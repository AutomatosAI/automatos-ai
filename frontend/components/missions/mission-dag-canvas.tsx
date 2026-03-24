'use client'

import React, { useCallback, useEffect, useMemo, useState } from 'react'
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

// ── Constants ────────────────────────────────────────────────

const NODE_WIDTH = 220
const NODE_HEIGHT = 100
const GAP_X = 40
const GAP_Y = 40
const GROUP_PADDING = 12

// ── Auto-layout (parallel-aware: same sequence_number → side-by-side) ───

function layoutTasks(tasks: TaskResponse[]): {
  nodes: Node<MissionTaskNodeData>[]
  edges: Edge[]
  groupRects: GroupRect[]
} {
  const sorted = [...tasks].sort((a, b) => a.sequence_number - b.sequence_number)

  // Group tasks by sequence_number
  const seqGroups = new Map<number, TaskResponse[]>()
  for (const task of sorted) {
    const group = seqGroups.get(task.sequence_number) ?? []
    group.push(task)
    seqGroups.set(task.sequence_number, group)
  }

  const seqNumbers = [...seqGroups.keys()].sort((a, b) => a - b)

  const nodes: Node<MissionTaskNodeData>[] = []
  const groupRects: GroupRect[] = []
  let currentY = 0

  for (const seq of seqNumbers) {
    const tasksInRow = seqGroups.get(seq)!
    const rowWidth = tasksInRow.length * NODE_WIDTH + (tasksInRow.length - 1) * GAP_X
    const startX = Math.max(50, (500 - rowWidth) / 2) // center the row

    // Check if ALL tasks in the row share the same non-null parallel_group
    const firstGroup = tasksInRow[0].parallel_group
    const parallelGroup = tasksInRow.length > 1 && firstGroup != null
      && tasksInRow.every((t) => t.parallel_group === firstGroup)
      ? firstGroup
      : null

    for (let i = 0; i < tasksInRow.length; i++) {
      const task = tasksInRow[i]
      nodes.push({
        id: task.id,
        type: 'missionTask',
        position: {
          x: startX + i * (NODE_WIDTH + GAP_X),
          y: currentY,
        },
        data: {
          id: task.id,
          title: task.title,
          agentName: null,
          agentRole: task.agent_role,
          sequenceNumber: task.sequence_number,
          state: task.state,
          taskType: task.task_type,
          tokensUsed: task.tokens_used,
          estimatedTokens: task.estimated_tokens,
          isSelected: false,
          mode: 'execution',
          outputExcerpt: task.output_excerpt,
          attemptNumber: task.attempt_number,
        },
        draggable: false,
      })
    }

    // Add group rectangle for parallel tasks
    if (parallelGroup && tasksInRow.length > 1) {
      groupRects.push({
        label: parallelGroup,
        x: startX - GROUP_PADDING,
        y: currentY - GROUP_PADDING,
        width: rowWidth + GROUP_PADDING * 2,
        height: NODE_HEIGHT + GROUP_PADDING * 2,
      })
    }

    currentY += NODE_HEIGHT + GAP_Y
  }

  // Build edges based on sequence dependencies
  // Tasks at seq N connect to all tasks at seq N+1
  const edges: Edge[] = []
  for (let i = 0; i < seqNumbers.length - 1; i++) {
    const sourceTasks = seqGroups.get(seqNumbers[i])!
    const targetTasks = seqGroups.get(seqNumbers[i + 1])!

    for (const source of sourceTasks) {
      for (const target of targetTasks) {
        const isActive = isTaskActive(source.state) || isTaskActive(target.state)
        const isSynthesisTarget = target.task_type === 'synthesis'

        edges.push({
          id: `edge-${source.id}-${target.id}`,
          source: source.id,
          target: target.id,
          type: 'smoothstep',
          animated: isActive,
          markerEnd: {
            type: MarkerType.ArrowClosed,
            color: isActive ? 'hsl(16, 100%, 60%)' : 'hsl(0, 0%, 40%)',
          },
          style: {
            stroke: isSynthesisTarget
              ? 'hsl(280, 80%, 60%)'
              : isActive
                ? 'hsl(16, 100%, 60%)'
                : 'hsl(0, 0%, 25%)',
            strokeWidth: isActive ? 2 : 1,
            strokeDasharray: isSynthesisTarget ? '5 3' : undefined,
          },
        })
      }
    }
  }

  return { nodes, edges, groupRects }
}

function isTaskActive(state: TaskState): boolean {
  return ['assigned', 'running', 'completed', 'verifying', 'retrying'].includes(state)
}

// ── Group rectangle overlay type ─────────────────────────────

interface GroupRect {
  label: string
  x: number
  y: number
  width: number
  height: number
}

// ── Component ─────────────────────────────────────────────────

export function MissionDAGCanvas({
  tasks,
  mode,
  selectedTaskId,
  onTaskSelect,
  className,
}: MissionDAGCanvasProps) {
  const { nodes: layoutedNodes, edges: layoutedEdges, groupRects } = useMemo(
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
    <div className={className} style={{ width: '100%', height: '100%', position: 'relative' }}>
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
        {/* Parallel group visual containers rendered as SVG overlays */}
        <svg style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', pointerEvents: 'none' }}>
          {groupRects.map((rect) => (
            <g key={rect.label}>
              <rect
                x={rect.x}
                y={rect.y}
                width={rect.width}
                height={rect.height}
                rx={8}
                ry={8}
                fill="hsla(16, 100%, 60%, 0.03)"
                stroke="hsla(16, 100%, 60%, 0.15)"
                strokeWidth={1}
                strokeDasharray="6 3"
              />
              <text
                x={rect.x + 8}
                y={rect.y + 12}
                fill="hsla(16, 100%, 60%, 0.5)"
                fontSize={10}
                fontFamily="monospace"
              >
                {rect.label}
              </text>
            </g>
          ))}
        </svg>

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
