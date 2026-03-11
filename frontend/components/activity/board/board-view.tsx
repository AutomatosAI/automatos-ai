'use client'

import { useState, useCallback } from 'react'
import { DragDropContext, type DropResult } from '@hello-pangea/dnd'
import { Plus } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { BoardColumn } from './board-column'
import { BoardAgentSidebar } from './board-agent-sidebar'
import { BoardFiltersBar } from './board-filters'
import { BoardTaskViewer } from './board-task-viewer'
import { CreateTaskDialog } from './create-task-dialog'
import { useBoardTasks, useUpdateTaskStatus, type BoardFilters } from '@/hooks/use-board-tasks'
import type { BoardTask, BoardStatus } from '@/types/board'
import { cn } from '@/lib/utils'

interface BoardViewProps {
  period: string
  className?: string
}

export function BoardView({ period, className }: BoardViewProps) {
  const [filters, setFilters] = useState<BoardFilters>({ period })
  const [selectedAgentId, setSelectedAgentId] = useState<number | null>(null)
  const [openTask, setOpenTask] = useState<BoardTask | null>(null)
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [createOpen, setCreateOpen] = useState(false)

  // Merge agent filter from sidebar
  const activeFilters = { ...filters, agent_id: selectedAgentId }
  const { columns, isLoading, data } = useBoardTasks(activeFilters)
  const updateStatus = useUpdateTaskStatus()

  const totalTasks = data?.total ?? 0

  const handleDragEnd = useCallback((result: DropResult) => {
    if (!result.destination) return
    const sourceStatus = result.source.droppableId as BoardStatus
    const destStatus = result.destination.droppableId as BoardStatus
    if (sourceStatus === destStatus) return

    updateStatus.mutate({
      taskId: result.draggableId,
      status: destStatus,
    })
  }, [updateStatus])

  const handleOpenTask = useCallback((task: BoardTask) => {
    setOpenTask(task)
  }, [])

  const handleCloseTask = useCallback(() => {
    setOpenTask(null)
  }, [])

  const handleSelectAgent = useCallback((agentId: number | null) => {
    setSelectedAgentId(agentId)
  }, [])

  return (
    <div className={cn('space-y-3', className)}>
      {/* Filters + Create */}
      <div className="flex items-center gap-2">
        <BoardFiltersBar
          filters={filters}
          onFiltersChange={setFilters}
          agentCount={columns.reduce((acc, c) => {
            const agents = new Set(c.tasks.filter(t => t.assignee).map(t => t.assignee!.agent_id))
            return acc + agents.size
          }, 0)}
          taskCount={totalTasks}
          className="flex-1"
        />
        <Button size="sm" onClick={() => setCreateOpen(true)} className="shrink-0">
          <Plus className="w-3.5 h-3.5 mr-1.5" />
          Create Task
        </Button>
      </div>

      <CreateTaskDialog open={createOpen} onOpenChange={setCreateOpen} />

      {/* Board layout */}
      <div className="flex gap-0 overflow-hidden rounded-xl border border-border/30">
        {/* Agent sidebar — hidden on mobile */}
        {sidebarOpen && (
          <div className="hidden lg:block">
            <BoardAgentSidebar
              selectedAgentId={selectedAgentId}
              onSelectAgent={handleSelectAgent}
            />
          </div>
        )}

        {/* Kanban columns */}
        <DragDropContext onDragEnd={handleDragEnd}>
          <div className="flex-1 flex gap-1 overflow-x-auto p-2 min-h-[500px]">
            {isLoading ? (
              // Skeleton columns
              Array.from({ length: 5 }).map((_, i) => (
                <div key={i} className="flex-1 min-w-[240px] space-y-2 p-2">
                  <div className="h-6 w-24 bg-secondary/30 rounded animate-pulse" />
                  {Array.from({ length: 3 - i }).map((_, j) => (
                    <div key={j} className="h-32 bg-secondary/20 rounded-lg animate-pulse" />
                  ))}
                </div>
              ))
            ) : (
              columns.map((column) => (
                <BoardColumn
                  key={column.status}
                  column={column}
                  onOpenTask={handleOpenTask}
                />
              ))
            )}
          </div>
        </DragDropContext>
      </div>

      {/* Task viewer slide-over */}
      {openTask && (
        <>
          {/* Backdrop */}
          <div
            className="fixed inset-0 bg-background/60 backdrop-blur-sm z-40"
            onClick={handleCloseTask}
          />
          <BoardTaskViewer task={openTask} onClose={handleCloseTask} />
        </>
      )}
    </div>
  )
}
