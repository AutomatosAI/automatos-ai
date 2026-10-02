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

import { useCallback, useMemo, useState } from 'react'
import {
  DragDropContext,
  Droppable,
  Draggable,
  type DraggableProvided,
  type DropResult,
} from '@hello-pangea/dnd'
import { BookMarked, CheckSquare, RefreshCw, Target } from 'lucide-react'
import { useBoardTasks, useUpdateTaskStatus } from '@/hooks/use-board-tasks'
import { useAssignableAgents } from '@/hooks/use-agent-api'
import { useTicketDeepLink } from '@/hooks/use-ticket-deep-link'
import { BoardTaskViewer } from '@/components/activity/board/board-task-viewer'
import { TicketActionsMenu } from '@/components/activity/board/ticket-actions-menu'
import { stageReason } from '@/components/activity/board/ticket-stage'
import { runsInSession, ticketKind } from '@/components/activity/board/ticket-kind'
import { HostOfflineBanner } from '@/components/activity/board/host-offline-banner'
import type { BoardTask, BoardStatus } from '@/types/board'
import { toneFor } from './agent-tones'
import { BoardToolbar, type BoardDensity as Density, type BoardMode as Mode } from './board-toolbar'

const COLUMN_META: Record<BoardStatus, { label: string; color: string }> = {
  inbox:       { label: 'Inbox',       color: 'hsl(30 14% 12%)' },
  assigned:    { label: 'Assigned',    color: 'hsl(213 51% 35%)' },
  in_progress: { label: 'In Progress', color: 'hsl(38 78% 27%)' },
  review:      { label: 'Review',      color: 'hsl(45 80% 60%)' },
  blocked:     { label: 'Blocked',     color: 'hsl(15 76% 44%)' },
  done:        { label: 'Done',        color: 'hsl(82 30% 33%)' },
  failed:      { label: 'Failed',      color: 'hsl(0 62% 38%)' },
  cancelled:   { label: 'Cancelled',   color: 'hsl(30 6% 40%)' }, // PRD-234 S1a
  // Tidied away without a claim about the work (night 1, 2026-09-18).
  closed:      { label: 'Closed',      color: 'hsl(30 8% 30%)' },
}
// PRD-252 R7 (D1): Closed tickets sit in the Cancelled column.
const COLUMNS_ORDER: BoardStatus[] = [
  'inbox', 'assigned', 'in_progress', 'review', 'blocked', 'done', 'failed', 'cancelled',
]
const LANE_COLUMNS = COLUMNS_ORDER.filter((c) => c !== 'done')

export function BoardTab() {
  const [mode, setMode] = useState<Mode>('column')
  const [density, setDensity] = useState<Density>('comfortable')
  const [search, setSearch] = useState('')
  const [agentFilter, setAgentFilter] = useState<number | null>(null)
  const [priorityFilter, setPriorityFilter] = useState<string | null>(null)
  const [openTask, setOpenTask] = useState<BoardTask | null>(null)
  const [focusQuestion, setFocusQuestion] = useState<number | null>(null)
  const [viewerOpen, setViewerOpen] = useState(false)

  const { columns, isLoading } = useBoardTasks({
    agent_id: agentFilter,
    priority: priorityFilter,
    search: search || null,
  })
  const { data: agents } = useAssignableAgents()
  const updateStatus = useUpdateTaskStatus()

  const allTasks = useMemo(() => columns.flatMap((c) => c.tasks), [columns])

  const openTicket = useCallback((task: BoardTask, questionId: number | null = null) => {
    setOpenTask(task)
    setFocusQuestion(questionId)
    setViewerOpen(true)
  }, [])
  // PRD-252 R1: ?task_id= (and &question=) open that ticket, fetched by id so
  // the filters below can't hide it from the link.
  const clearDeepLink = useTicketDeepLink(openTicket)

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

  const handleCardClick = (task: BoardTask) => openTicket(task)

  return (
    <>
      {/* PRD-235 W3: Claude Code agents need the paired host — say so once, at the top */}
      <HostOfflineBanner />
      <BoardToolbar
        mode={mode}
        onMode={setMode}
        density={density}
        onDensity={setDensity}
        priority={priorityFilter}
        onPriority={setPriorityFilter}
        agentId={agentFilter}
        onAgentId={setAgentFilter}
        agents={agents}
        search={search}
        onSearch={setSearch}
      />

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
        focusQuestionId={focusQuestion}
        onOpenChange={(o) => {
          setViewerOpen(o)
          if (!o) {
            setOpenTask(null)
            clearDeepLink()
          }
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
  return (
    <Draggable draggableId={task.id} index={index}>
      {(provided, snapshot) => (
        <KanbanCardFace task={task} density={density} onOpen={onOpen} provided={provided} isDragging={snapshot.isDragging} />
      )}
    </Draggable>
  )
}

/** The card itself; its Draggable hands it the drag props. */
function KanbanCardFace({
  task,
  density,
  onOpen,
  provided,
  isDragging,
}: {
  task: BoardTask
  density: Density
  onOpen: () => void
  provided: DraggableProvided
  isDragging: boolean
}) {
  const tone = toneFor(task.assignee?.agent_name)
  const isCompact = density === 'compact'
  const reason = stageReason(task)
  return (
    <div
      ref={provided.innerRef}
      {...provided.draggableProps}
      {...provided.dragHandleProps}
      className={`cc-kb-card${isCompact ? ' compact' : ''}${isDragging ? ' dragging' : ''}`}
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
      <KanbanKindRow task={task} />
      <div className="ttl">{task.name}</div>
      {/* PRD-252 R3: why it waits in Review or Blocked */}
      {reason && <div className="cc-kb-reason" title={reason.says}>{reason.chip}</div>}
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
  )
}

/** The card's top line: its number and type, priority, the flags that say it needs a look, and its actions. */
function KanbanKindRow({ task }: { task: BoardTask }) {
  const kind = ticketKind(task.source_type)  // PRD-252 R4: the type a person reads
  const Icon = kind === 'Playbook' ? BookMarked : kind === 'Mission' ? Target : kind === 'Routine' ? RefreshCw : CheckSquare
  return (
    <div className="kind">
      {task.number && <span className="num">{task.number}</span>}
      <Icon style={{ width: 11, height: 11 }} />
      {kind.toUpperCase()}
      {runsInSession(task) && <span className="session" title="A Claude Code session runs this ticket">· &gt;_ SESSION</span>}
      {(task.priority === 'urgent' || task.priority === 'high') && (
        <span className="high">· {task.priority.toUpperCase()}</span>
      )}
      {/* A session ticket's FIRST claim sets attempts=1, so `> 0` badged
          47 of 125 perfectly healthy tickets UNRESPONSIVE on night 1.
          A requeue is only evidence of a missed ack from the second
          attempt on. */}
      {(task.attempts ?? 0) > 1 && task.status !== 'done' && (
        <span
          className="high"
          style={{ color: 'hsl(0 72% 60%)' }}
          title={`Agent missed its ack deadline — task requeued ${(task.attempts ?? 1) - 1}×`}
        >
          · UNRESPONSIVE
        </span>
      )}
      {task.sla_deadline &&
        task.status !== 'done' &&
        task.status !== 'failed' &&
        task.status !== 'cancelled' &&
        new Date(task.sla_deadline).getTime() < Date.now() && (
          <span
            className="high"
            style={{ color: 'hsl(0 72% 60%)' }}
            title={`SLA breached — was due ${new Date(task.sla_deadline).toLocaleString()}`}
          >
            · OVERDUE
          </span>
        )}
      {/* PRD-252 R7: assign and cancel from the card */}
      <TicketActionsMenu task={task} />
    </div>
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
