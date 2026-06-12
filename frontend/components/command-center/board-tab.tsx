'use client'

/**
 * BoardTab — Studio kanban for the Command Centre.
 *
 * Real interactions, no dead UI:
 *  - Drag-and-drop between columns via `@hello-pangea/dnd` + the existing
 *    `useUpdateTaskStatus` mutation (same one the classic BoardView uses,
 *    so optimistic updates + invalidations behave identically).
 *  - Click a card → opens the existing `BoardTaskViewer` slideover with
 *    the full task detail / logs / actions.
 *  - Two modes: six-column kanban (default), by-agent lanes.
 *  - Density toggle: comfortable / compact.
 *  - Server-side filters: priority + agent dropdowns + search. All wired
 *    to `useBoardTasks` (the same hook the classic view uses).
 *
 * No speculative "New task" / "Filter" / "Export" buttons here — the
 * Command Centre is for monitoring + triage, not creation. Task creation
 * lives in /chat?mode=plan and routine creation in /agents.
 */

import { useMemo, useState } from 'react'
import {
  DragDropContext,
  Droppable,
  Draggable,
  type DropResult,
} from '@hello-pangea/dnd'
import {
  Columns,
  LayoutList,
  Rows,
  AlignJustify,
  Search,
  BookMarked,
  CheckSquare,
} from 'lucide-react'
import { useBoardTasks, useUpdateTaskStatus } from '@/hooks/use-board-tasks'
import { useAssignableAgents } from '@/hooks/use-agent-api'
import { BoardTaskViewer } from '@/components/activity/board/board-task-viewer'
import type { BoardTask, BoardStatus } from '@/types/board'
import { toneFor } from './agent-tones'

type Mode = 'column' | 'lane'
type Density = 'comfortable' | 'compact'

const COLUMN_META: Record<BoardStatus, { label: string; color: string }> = {
  inbox:       { label: 'Inbox',       color: 'hsl(30 14% 12%)' },
  assigned:    { label: 'Assigned',    color: 'hsl(213 51% 35%)' },
  in_progress: { label: 'In Progress', color: 'hsl(38 78% 27%)' },
  review:      { label: 'Review',      color: 'hsl(45 80% 60%)' },
  blocked:     { label: 'Blocked',     color: 'hsl(15 76% 44%)' },
  done:        { label: 'Done',        color: 'hsl(82 30% 33%)' },
  failed:      { label: 'Failed',      color: 'hsl(0 62% 38%)' },
}
const COLUMNS_ORDER: BoardStatus[] = [
  'inbox', 'assigned', 'in_progress', 'review', 'blocked', 'done', 'failed',
]
const LANE_COLUMNS = COLUMNS_ORDER.filter((c) => c !== 'done')

export function BoardTab() {
  const [mode, setMode] = useState<Mode>('column')
  const [density, setDensity] = useState<Density>('comfortable')
  const [search, setSearch] = useState('')
  const [agentFilter, setAgentFilter] = useState<number | null>(null)
  const [priorityFilter, setPriorityFilter] = useState<string | null>(null)
  const [openTask, setOpenTask] = useState<BoardTask | null>(null)
  const [viewerOpen, setViewerOpen] = useState(false)

  const { columns, isLoading } = useBoardTasks({
    agent_id: agentFilter,
    priority: priorityFilter,
    search: search || null,
  })
  const { data: agents } = useAssignableAgents()
  const updateStatus = useUpdateTaskStatus()

  const allTasks = useMemo(() => columns.flatMap((c) => c.tasks), [columns])

  const handleDragEnd = (result: DropResult) => {
    if (!result.destination) return
    const { source, destination, draggableId } = result
    if (source.droppableId === destination.droppableId) return
    // Lane mode droppableIds are `${agentName}::${status}`; column mode is
    // just `${status}`. Strip the agent prefix when present.
    const nextStatus = (destination.droppableId.includes('::')
      ? destination.droppableId.split('::')[1]
      : destination.droppableId) as BoardStatus
    updateStatus.mutate({ taskId: draggableId, status: nextStatus })
  }

  const handleCardClick = (task: BoardTask) => {
    setOpenTask(task)
    setViewerOpen(true)
  }

  return (
    <>
      <div className="cc-toolbar">
        <div className="cc-seg" role="group" aria-label="Board mode">
          <button
            type="button"
            className={mode === 'column' ? 'on' : ''}
            onClick={() => setMode('column')}
          >
            <Columns style={{ width: 12, height: 12 }} /> Columns
          </button>
          <button
            type="button"
            className={mode === 'lane' ? 'on' : ''}
            onClick={() => setMode('lane')}
          >
            <LayoutList style={{ width: 12, height: 12 }} /> By agent
          </button>
        </div>

        <div className="cc-seg" role="group" aria-label="Density">
          <button
            type="button"
            className={density === 'comfortable' ? 'on' : ''}
            onClick={() => setDensity('comfortable')}
          >
            <Rows style={{ width: 11, height: 11 }} /> Comfortable
          </button>
          <button
            type="button"
            className={density === 'compact' ? 'on' : ''}
            onClick={() => setDensity('compact')}
          >
            <AlignJustify style={{ width: 11, height: 11 }} /> Compact
          </button>
        </div>

        <div style={{ display: 'inline-flex', gap: 6 }}>
          <select
            className="cc-btn"
            value={priorityFilter ?? ''}
            onChange={(e) => setPriorityFilter(e.target.value || null)}
            aria-label="Filter by priority"
            style={{ height: 28, fontSize: 11.5, paddingRight: 24 }}
          >
            <option value="">All priorities</option>
            <option value="urgent">Urgent</option>
            <option value="high">High</option>
            <option value="medium">Medium</option>
            <option value="low">Low</option>
          </select>
          <select
            className="cc-btn"
            value={agentFilter ?? ''}
            onChange={(e) =>
              setAgentFilter(e.target.value ? Number(e.target.value) : null)
            }
            aria-label="Filter by agent"
            style={{ height: 28, fontSize: 11.5, paddingRight: 24 }}
          >
            <option value="">All agents</option>
            {Array.isArray(agents) &&
              agents.map((a: any) => (
                <option key={a.id} value={a.id}>
                  {a.name}
                </option>
              ))}
          </select>
        </div>

        <div style={{ marginLeft: 'auto' }}>
          <div
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: 6,
              padding: '0 10px',
              border: '1px solid hsl(var(--border))',
              borderRadius: 6,
              background: 'hsl(var(--card))',
              minWidth: 220,
            }}
          >
            <Search
              style={{
                width: 12,
                height: 12,
                color: 'hsl(var(--muted-foreground))',
              }}
            />
            <input
              type="search"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search tasks…"
              style={{
                background: 'transparent',
                border: 0,
                outline: 'none',
                fontSize: 12,
                height: 28,
                flex: 1,
                color: 'hsl(var(--foreground))',
              }}
            />
          </div>
        </div>
      </div>

      {isLoading ? (
        <div className="cc-panel-empty">Loading tasks…</div>
      ) : (
        <DragDropContext onDragEnd={handleDragEnd}>
          {mode === 'column' ? (
            <ColumnMode
              columns={columns}
              density={density}
              onOpenTask={handleCardClick}
            />
          ) : (
            <LaneMode
              tasks={allTasks}
              density={density}
              onOpenTask={handleCardClick}
            />
          )}
        </DragDropContext>
      )}

      <BoardTaskViewer
        task={openTask}
        open={viewerOpen}
        onOpenChange={(o) => {
          setViewerOpen(o)
          if (!o) setOpenTask(null)
        }}
      />
    </>
  )
}

function KanbanCard({
  task,
  density,
  index,
  onOpen,
}: {
  task: BoardTask
  density: Density
  index: number
  onOpen: () => void
}) {
  const tone = toneFor(task.assignee?.agent_name)
  const isCompact = density === 'compact'
  const isPlaybook = task.type === 'playbook'
  const Icon = isPlaybook ? BookMarked : CheckSquare
  return (
    <Draggable draggableId={task.id} index={index}>
      {(provided, snapshot) => (
        <div
          ref={provided.innerRef}
          {...provided.draggableProps}
          {...provided.dragHandleProps}
          className={`cc-kb-card${isCompact ? ' compact' : ''}${
            snapshot.isDragging ? ' dragging' : ''
          }`}
          onClick={onOpen}
          role="button"
          tabIndex={0}
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') {
              e.preventDefault()
              onOpen()
            }
          }}
          style={provided.draggableProps.style}
        >
          <div className="kind">
            <Icon style={{ width: 11, height: 11 }} />
            {(task.type ?? 'task').toUpperCase()}
            {(task.priority === 'urgent' || task.priority === 'high') && (
              <span className="high">· {task.priority.toUpperCase()}</span>
            )}
            {(task.attempts ?? 0) > 0 && task.status !== 'done' && (
              <span
                className="high"
                style={{ color: 'hsl(0 72% 60%)' }}
                title={`Agent missed its ack deadline — task requeued ${task.attempts}×`}
              >
                · UNRESPONSIVE
              </span>
            )}
            {task.sla_deadline &&
              task.status !== 'done' &&
              task.status !== 'failed' &&
              new Date(task.sla_deadline).getTime() < Date.now() && (
                <span
                  className="high"
                  style={{ color: 'hsl(0 72% 60%)' }}
                  title={`SLA breached — was due ${new Date(task.sla_deadline).toLocaleString()}`}
                >
                  · OVERDUE
                </span>
              )}
          </div>
          <div className="ttl">{task.name}</div>
          {!isCompact && task.description && (
            <div className="body">{task.description}</div>
          )}
          <div className="row">
            {(task.tags || []).slice(0, isCompact ? 1 : 3).map((tg) => (
              <span key={tg} className="tag">
                {tg}
              </span>
            ))}
            <span className="ag">
              <span className="swatch" style={{ background: tone.bg }} />
              {task.assignee?.agent_name ?? 'Unassigned'}
            </span>
          </div>
        </div>
      )}
    </Draggable>
  )
}

function ColumnMode({
  columns,
  density,
  onOpenTask,
}: {
  columns: { status: BoardStatus; tasks: BoardTask[] }[]
  density: Density
  onOpenTask: (t: BoardTask) => void
}) {
  return (
    <div className="cc-kb-grid">
      {COLUMNS_ORDER.map((status) => {
        const col = columns.find((c) => c.status === status)
        const tasks = col?.tasks ?? []
        const meta = COLUMN_META[status]
        return (
          <div key={status} className="cc-kb-col">
            <div className="cc-kb-head">
              <span className="dot" style={{ background: meta.color }} />
              <span className="l">{meta.label}</span>
              <span className="ct">{tasks.length}</span>
            </div>
            <Droppable droppableId={status}>
              {(provided, snapshot) => (
                <div
                  ref={provided.innerRef}
                  {...provided.droppableProps}
                  className={`cc-kb-body${snapshot.isDraggingOver ? ' drag-over' : ''}`}
                >
                  {tasks.length === 0 ? (
                    <div className="cc-kb-empty">NO TASKS</div>
                  ) : (
                    tasks.map((t, i) => (
                      <KanbanCard
                        key={t.id}
                        task={t}
                        density={density}
                        index={i}
                        onOpen={() => onOpenTask(t)}
                      />
                    ))
                  )}
                  {provided.placeholder}
                </div>
              )}
            </Droppable>
          </div>
        )
      })}
    </div>
  )
}

function LaneMode({
  tasks,
  density,
  onOpenTask,
}: {
  tasks: BoardTask[]
  density: Density
  onOpenTask: (t: BoardTask) => void
}) {
  // Group by agent name, exclude done, sort agents asc.
  const groups = useMemo(() => {
    const map: Record<string, BoardTask[]> = {}
    tasks
      .filter((t) => t.status !== 'done')
      .forEach((t) => {
        const k = t.assignee?.agent_name || 'Unassigned'
        if (!map[k]) map[k] = []
        map[k].push(t)
      })
    return Object.entries(map).sort(([a], [b]) => a.localeCompare(b))
  }, [tasks])

  if (groups.length === 0) {
    return (
      <div className="cc-panel-empty">
        No active tasks. Done items don&apos;t appear in the lane view.
      </div>
    )
  }

  return (
    <div style={{ flex: 1, minHeight: 0, overflow: 'auto' }}>
      <div
        className="cc-kb-lane"
        style={{ background: 'hsl(var(--secondary))', padding: '8px 14px' }}
      >
        <div
          style={{
            width: 160,
            fontFamily: 'var(--font-geist-mono, monospace)',
            fontSize: 10,
            color: 'hsl(var(--muted-foreground))',
            letterSpacing: '0.10em',
            textTransform: 'uppercase',
          }}
        >
          AGENT
        </div>
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(5, 1fr)',
            gap: 14,
          }}
        >
          {LANE_COLUMNS.map((c) => {
            const meta = COLUMN_META[c]
            return (
              <div
                key={c}
                style={{ display: 'flex', alignItems: 'center', gap: 6 }}
              >
                <span
                  style={{
                    width: 8,
                    height: 8,
                    borderRadius: '50%',
                    background: meta.color,
                  }}
                />
                <span
                  style={{
                    fontFamily: 'var(--font-geist-mono, monospace)',
                    fontSize: 10,
                    color: 'hsl(var(--muted-foreground))',
                    letterSpacing: '0.10em',
                    textTransform: 'uppercase',
                    fontWeight: 600,
                  }}
                >
                  {meta.label}
                </span>
              </div>
            )
          })}
        </div>
      </div>
      {groups.map(([agentName, list]) => {
        const tone = toneFor(agentName)
        return (
          <div key={agentName} className="cc-kb-lane">
            <div className="head">
              <div className="ag">
                <span className="swatch" style={{ background: tone.bg }} />
                {agentName}
              </div>
              <div className="meta">{list.length} active</div>
            </div>
            <div className="lane-body">
              {LANE_COLUMNS.map((status) => {
                const cards = list.filter((t) => t.status === status)
                return (
                  <Droppable
                    key={status}
                    droppableId={`${agentName}::${status}`}
                  >
                    {(provided) => (
                      <div
                        ref={provided.innerRef}
                        {...provided.droppableProps}
                        className="lane-cell"
                      >
                        {cards.length === 0 ? (
                          <div className="lane-empty" />
                        ) : (
                          cards.map((t, i) => (
                            <KanbanCard
                              key={t.id}
                              task={t}
                              density={density}
                              index={i}
                              onOpen={() => onOpenTask(t)}
                            />
                          ))
                        )}
                        {provided.placeholder}
                      </div>
                    )}
                  </Droppable>
                )
              })}
            </div>
          </div>
        )
      })}
    </div>
  )
}
