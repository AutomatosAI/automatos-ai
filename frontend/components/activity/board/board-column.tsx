'use client'

import { Droppable } from '@hello-pangea/dnd'
import { BoardCard } from './board-card'
import type { BoardColumn as BoardColumnType, BoardTask } from '@/types/board'
import { STATUS_CONFIG } from '@/types/board'
import { cn } from '@/lib/utils'

interface BoardColumnProps {
  column: BoardColumnType
  onOpenTask: (task: BoardTask) => void
  onDeleteTask?: (taskId: string) => void
}

export function BoardColumn({ column, onOpenTask, onDeleteTask }: BoardColumnProps) {
  const statusConf = STATUS_CONFIG[column.status]

  return (
    <div className="flex flex-col min-w-[260px] max-w-[320px] flex-1">
      {/* Column header */}
      <div className="flex items-center gap-2 px-3 py-2 mb-2">
        <div className={cn('w-2 h-2 rounded-full', statusConf.dotColor)} />
        <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
          {column.label}
        </h3>
        <span className="text-[10px] font-medium text-muted-foreground bg-secondary/50 px-1.5 py-0.5 rounded-full">
          {column.count}
        </span>
      </div>

      {/* Droppable area */}
      <Droppable droppableId={column.status}>
        {(provided, snapshot) => (
          <div
            ref={provided.innerRef}
            {...provided.droppableProps}
            className={cn(
              'flex-1 px-1.5 pb-2 rounded-lg transition-colors overflow-y-auto min-h-[200px]',
              snapshot.isDraggingOver && 'bg-secondary/20 ring-1 ring-dashed ring-primary/20',
            )}
          >
            {column.tasks.map((task, index) => (
              <BoardCard
                key={task.id}
                task={task}
                index={index}
                onOpen={onOpenTask}
                onDelete={onDeleteTask}
              />
            ))}
            {provided.placeholder}

            {column.tasks.length === 0 && !snapshot.isDraggingOver && (
              <div className="flex items-center justify-center h-24 text-[10px] text-muted-foreground/30">
                No tasks
              </div>
            )}
          </div>
        )}
      </Droppable>
    </div>
  )
}
