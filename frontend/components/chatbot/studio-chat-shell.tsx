'use client'

/**
 * StudioChatShell — wraps the existing <Chat> in CD round-3's ledger layout.
 *
 * Three columns when an active mission is attached to the conversation:
 * threads (collapsible) | thread (1fr) | mission rail (280px, collapsible).
 *
 * Real-data wiring:
 * - Threads list: `getChatHistory()` (existing API)
 * - Active mission: `useMissionStore.activePlanningMissionId` → `useMission(id)`
 *   for live state. Stats come from `computeMissionStats`. Pipeline is the
 *   mission's task list sorted by sequence.
 * - If no mission is attached to the active thread, the rail shows a single
 *   editorial empty state — we never render fabricated data.
 *
 * Both side columns persist their collapsed state to localStorage so the
 * user's preference sticks across reloads.
 */

import { useEffect, useState } from 'react'
import {
  Plus,
  ChevronRight,
  PanelLeftClose,
  PanelLeftOpen,
  PanelRightClose,
  PanelRightOpen,
  X,
} from 'lucide-react'
import { toast } from 'sonner'
import { getChat, getChatHistory, getChatMessages } from '@/lib/chat/api'
import { useMission } from '@/hooks/use-missions-api'
import { useMissionStore } from '@/stores/mission-store'
import {
  computeMissionStats,
  RUN_STATE_CONFIG,
  type TaskResponse,
} from '@/types/missions'
import type { Chat as ChatType, ChatMessage } from '@/types'

export interface StudioChatShellProps {
  children: React.ReactNode
  selectedChatId: string
  selectedChat: ChatType | null
  onSelectChat: (chat: ChatType, messages: ChatMessage[]) => void
  onNewChat: () => void
  /** PRD-237: the open tabs, shown above the recent list. */
  openChatIds?: string[]
  unreadChatIds?: string[]
  titles?: Record<string, string>
  onCloseTab?: (chatId: string) => void
}

const THREADS_KEY = 'studioChatThreadsCollapsed'
const RAIL_KEY = 'studioChatRailCollapsed'

export function StudioChatShell({
  children,
  selectedChatId,
  selectedChat,
  onSelectChat,
  onNewChat,
  openChatIds = [],
  unreadChatIds = [],
  titles = {},
  onCloseTab,
}: StudioChatShellProps) {
  const [threads, setThreads] = useState<ChatType[]>([])
  const [loadingThreads, setLoadingThreads] = useState(true)
  const [openingThreadId, setOpeningThreadId] = useState<string | null>(null)
  const [threadsCollapsed, setThreadsCollapsed] = useState(false)
  const [railCollapsed, setRailCollapsed] = useState(false)

  const activeMissionId = useMissionStore((s) => s.activePlanningMissionId)
  const { data: mission, isLoading: missionLoading } = useMission(activeMissionId)

  // Load persisted collapse states
  useEffect(() => {
    try {
      if (localStorage.getItem(THREADS_KEY) === '1') setThreadsCollapsed(true)
      if (localStorage.getItem(RAIL_KEY) === '1') setRailCollapsed(true)
    } catch {}
  }, [])

  const toggleThreads = () => {
    setThreadsCollapsed((prev) => {
      const next = !prev
      try {
        localStorage.setItem(THREADS_KEY, next ? '1' : '0')
      } catch {}
      return next
    })
  }

  const toggleRail = () => {
    setRailCollapsed((prev) => {
      const next = !prev
      try {
        localStorage.setItem(RAIL_KEY, next ? '1' : '0')
      } catch {}
      return next
    })
  }

  useEffect(() => {
    let cancelled = false
    setLoadingThreads(true)
    getChatHistory(20)
      .then((rows) => {
        if (!cancelled) setThreads(rows)
      })
      .catch(() => {
        if (!cancelled) setThreads([])
      })
      .finally(() => {
        if (!cancelled) setLoadingThreads(false)
      })
    return () => {
      cancelled = true
    }
  }, [selectedChatId])

  const handleThreadClick = async (t: ChatType) => {
    if (t.id === selectedChatId || openingThreadId) return
    setOpeningThreadId(t.id)
    try {
      const messages = await getChatMessages(t.id)
      onSelectChat(t, messages)
    } catch (err) {
      console.error('Failed to load chat messages:', err)
      toast.error('Failed to load chat')
    } finally {
      setOpeningThreadId(null)
    }
  }

  // PRD-237: an open tab may not be in the recent rows — fetch it on click.
  const handleOpenTabClick = async (chatId: string) => {
    if (chatId === selectedChatId || openingThreadId) return
    const known = threads.find((t) => t.id === chatId)
    if (known) {
      await handleThreadClick(known)
      return
    }
    setOpeningThreadId(chatId)
    try {
      const [chat, messages] = await Promise.all([getChat(chatId), getChatMessages(chatId)])
      onSelectChat(chat, messages)
    } catch (err) {
      console.error('Failed to load chat:', err)
      toast.error('Failed to load chat')
    } finally {
      setOpeningThreadId(null)
    }
  }

  const activeTitle = selectedChat?.title ?? titles[selectedChatId] ?? 'New conversation'

  return (
    <div
      className={
        'sh-chat' +
        (threadsCollapsed ? ' threads-collapsed' : '') +
        (railCollapsed ? ' rail-collapsed' : '')
      }
    >
      {/* Breadcrumb bar */}
      <div className="sh-chat-bar">
        <button
          type="button"
          className="sh-chat-side-toggle"
          onClick={toggleThreads}
          aria-label={threadsCollapsed ? 'Show threads' : 'Hide threads'}
          title={threadsCollapsed ? 'Show threads' : 'Hide threads'}
        >
          {threadsCollapsed ? (
            <PanelLeftOpen style={{ width: 14, height: 14, strokeWidth: 1.6 }} />
          ) : (
            <PanelLeftClose style={{ width: 14, height: 14, strokeWidth: 1.6 }} />
          )}
        </button>
        <span className="sh-chat-eyebrow">Operations</span>
        <span className="sh-chat-sep">·</span>
        <span className="sh-chat-crumb">Conversations</span>
        <ChevronRight className="sh-chat-chev" />
        <span className="sh-chat-active" title={activeTitle}>
          {activeTitle}
        </span>
        {selectedChatId && (
          <span className="sh-chat-pill brand">
            {selectedChatId.slice(0, 8)}
          </span>
        )}
        <button
          type="button"
          className="sh-chat-side-toggle"
          onClick={toggleRail}
          aria-label={railCollapsed ? 'Show mission rail' : 'Hide mission rail'}
          title={railCollapsed ? 'Show mission rail' : 'Hide mission rail'}
          style={{ marginLeft: 'auto' }}
        >
          {railCollapsed ? (
            <PanelRightOpen style={{ width: 14, height: 14, strokeWidth: 1.6 }} />
          ) : (
            <PanelRightClose style={{ width: 14, height: 14, strokeWidth: 1.6 }} />
          )}
        </button>
      </div>

      {/* Grid body */}
      <div className="sh-chat-grid">
        {/* Threads list */}
        {!threadsCollapsed && (
          <aside className="sh-chat-threads" aria-label="Chat threads">
            <div className="sh-chat-threads-head">
              <span className="sh-chat-eyebrow-mono">Threads</span>
              <button type="button" className="sh-chat-act" onClick={onNewChat}>
                <Plus style={{ width: 11, height: 11 }} />
                <span>New</span>
              </button>
            </div>
            {openChatIds.length > 0 && (
              <div className="sh-chat-thread-list" aria-label="Open conversations">
                <div className="sh-chat-thread-empty" style={{ paddingBottom: 2 }}>Open</div>
                {openChatIds.map((chatId) => {
                  const isActive = chatId === selectedChatId
                  const isOpening = chatId === openingThreadId
                  const title = titles[chatId] ?? threads.find((t) => t.id === chatId)?.title ?? 'Conversation'
                  const unread = unreadChatIds.includes(chatId)
                  return (
                    <div
                      key={chatId}
                      role="button"
                      tabIndex={0}
                      className={'sh-chat-thread' + (isActive ? ' active' : '') + (isOpening ? ' opening' : '')}
                      onClick={() => handleOpenTabClick(chatId)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' || e.key === ' ') {
                          e.preventDefault()
                          handleOpenTabClick(chatId)
                        }
                      }}
                      title={title}
                    >
                      <span
                        className={`sh-chat-thread-dot${isActive || unread ? ' warn' : ' ok'}`}
                        aria-hidden
                      />
                      <span className="sh-chat-thread-title">{title}</span>
                      {unread && <span className="sh-chat-pill brand">new</span>}
                      {onCloseTab && (
                        <button
                          type="button"
                          className="sh-chat-thread-close"
                          aria-label={`Close ${title}`}
                          title="Close tab"
                          onClick={(e) => {
                            e.stopPropagation()
                            onCloseTab(chatId)
                          }}
                        >
                          <X style={{ width: 11, height: 11 }} />
                        </button>
                      )}
                    </div>
                  )
                })}
                <div className="sh-chat-thread-empty" style={{ paddingTop: 8, paddingBottom: 2 }}>Recent</div>
              </div>
            )}
            <div className="sh-chat-thread-list">
              {loadingThreads ? (
                <div className="sh-chat-thread-empty">Loading…</div>
              ) : threads.length === 0 ? (
                <div className="sh-chat-thread-empty">No conversations yet</div>
              ) : (
                threads.map((t) => {
                  const isActive = t.id === selectedChatId
                  const isOpening = t.id === openingThreadId
                  const rel = relativeTime(t.createdAt)
                  return (
                    <button
                      key={t.id}
                      type="button"
                      className={
                        'sh-chat-thread' +
                        (isActive ? ' active' : '') +
                        (isOpening ? ' opening' : '')
                      }
                      onClick={() => handleThreadClick(t)}
                      disabled={isOpening}
                      title={t.title}
                    >
                      <span
                        className={`sh-chat-thread-dot${isActive ? ' warn' : ' ok'}`}
                        aria-hidden
                      />
                      <span className="sh-chat-thread-title">{t.title}</span>
                      {/* PRD-205 S7: mark the thread where Auto speaks unprompted */}
                      {t.kind === 'auto' && (
                        <span className="sh-chat-pill brand">Auto</span>
                      )}
                      <span className="sh-chat-thread-ts">
                        {isOpening ? '…' : rel}
                      </span>
                    </button>
                  )
                })
              )}
            </div>
          </aside>
        )}

        {/* Main thread */}
        <div className="sh-chat-main">{children}</div>

        {/* Mission rail — real data or editorial empty state */}
        {!railCollapsed && (
          <MissionRail
            missionId={activeMissionId}
            mission={mission}
            loading={missionLoading}
          />
        )}
      </div>
    </div>
  )
}

interface MissionRailProps {
  missionId: string | null
  mission: ReturnType<typeof useMission>['data']
  loading: boolean
}

function MissionRail({ missionId, mission, loading }: MissionRailProps) {
  // No mission attached — editorial empty state
  if (!missionId) {
    return (
      <aside className="sh-chat-rail" aria-label="Mission rail">
        <div className="sh-chat-rail-empty">
          <p className="sh-chat-rail-eyebrow">Mission · this thread</p>
          <p className="sh-chat-rail-empty-body">
            No mission attached. Ask the agent to plan one, or open Mission
            Mode from the composer to start tracking work here.
          </p>
        </div>
      </aside>
    )
  }

  // Mission exists but data hasn't arrived yet
  if (loading || !mission) {
    return (
      <aside className="sh-chat-rail" aria-label="Mission rail">
        <div>
          <p className="sh-chat-rail-eyebrow">Mission · this thread</p>
          <div className="sh-chat-rail-title">Loading mission…</div>
          <div className="sh-chat-rail-id">{missionId.slice(0, 14)}</div>
        </div>
      </aside>
    )
  }

  const stats = computeMissionStats(mission)
  const stateMeta = RUN_STATE_CONFIG[mission.state]
  const tasksOrdered = [...mission.tasks].sort(
    (a, b) => a.sequence_number - b.sequence_number,
  )
  const taskSlice = tasksOrdered.slice(0, 6)

  return (
    <aside className="sh-chat-rail" aria-label="Mission rail">
      <div>
        <p className="sh-chat-rail-eyebrow">Mission · this thread</p>
        <div className="sh-chat-rail-title">{mission.goal}</div>
        <div className="sh-chat-rail-meta">
          <span className="sh-chat-rail-id">{mission.id.slice(0, 14)}</span>
          <span className="sh-chat-rail-state">{stateMeta?.label ?? mission.state}</span>
        </div>
      </div>

      <div className="sh-chat-rail-stats">
        <Stat label="TASKS" value={`${stats.tasksDone} / ${stats.taskCount}`} tone="accent" />
        <Stat label="ACTIVE" value={String(stats.tasksActive)} />
        <Stat label="TOKENS" value={formatTokens(stats.tokensUsed)} tone="olive" />
        <Stat
          label="ELAPSED"
          value={stats.elapsedMs > 0 ? formatElapsed(stats.elapsedMs) : '—'}
        />
      </div>

      <div>
        <p className="sh-chat-rail-eyebrow">Pipeline</p>
        {taskSlice.length === 0 ? (
          <p className="sh-chat-rail-empty-body" style={{ margin: '4px 0 0' }}>
            No tasks yet. The agent will list steps once the plan is approved.
          </p>
        ) : (
          <div className="sh-chat-dag">
            {taskSlice.map((task) => (
              <DagRow key={task.id} task={task} />
            ))}
            {tasksOrdered.length > taskSlice.length && (
              <div className="sh-chat-rail-more">
                +{tasksOrdered.length - taskSlice.length} more
              </div>
            )}
          </div>
        )}
      </div>

      <a
        href={`/missions/${mission.id}`}
        className="sh-chat-rail-cta"
      >
        Open mission detail →
      </a>
    </aside>
  )
}

function Stat({
  label,
  value,
  tone,
}: {
  label: string
  value: string
  tone?: 'accent' | 'olive'
}) {
  return (
    <div>
      <div className="sh-chat-stat-label">{label}</div>
      <div className={`sh-chat-stat-value${tone ? ' ' + tone : ''}`}>{value}</div>
    </div>
  )
}

const DONE_STATES = new Set(['verified', 'completed', 'skipped'])
const ERROR_STATES = new Set(['failed', 'stalled'])
const ACTIVE_STATES = new Set([
  'assigned',
  'running',
  'verifying',
  'retrying',
  'queued',
])

function DagRow({ task }: { task: TaskResponse }) {
  let tone: 'done' | 'queued' | 'err' | 'pending' = 'pending'
  let pip: string | number = task.sequence_number
  if (DONE_STATES.has(task.state)) {
    tone = 'done'
    pip = '✓'
  } else if (ERROR_STATES.has(task.state)) {
    tone = 'err'
    pip = '!'
  } else if (ACTIVE_STATES.has(task.state)) {
    tone = 'queued'
    pip = '↻'
  }

  return (
    <div className={`sh-chat-dag-step ${tone}`}>
      <span className="sh-chat-dag-pip">{pip}</span>
      <div style={{ flex: 1, minWidth: 0 }}>
        <div className="sh-chat-dag-agent">{task.title}</div>
        <div className="sh-chat-dag-tool">{task.agent_role ?? task.task_type ?? task.state}</div>
      </div>
    </div>
  )
}

function relativeTime(input: string | Date | undefined): string {
  if (!input) return ''
  const then = typeof input === 'string' ? new Date(input) : input
  const diffMs = Date.now() - then.getTime()
  const m = Math.floor(diffMs / 60000)
  if (m < 1) return 'now'
  if (m < 60) return `${m}m`
  const h = Math.floor(m / 60)
  if (h < 24) return `${h}h`
  const d = Math.floor(h / 24)
  if (d < 7) return `${d}d`
  return then.toLocaleDateString('en-GB', { month: 'short', day: 'numeric' })
}

function formatTokens(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}k`
  return String(n)
}

function formatElapsed(ms: number): string {
  const s = Math.floor(ms / 1000)
  if (s < 60) return `${s}s`
  const m = Math.floor(s / 60)
  if (m < 60) return `${m}m ${s % 60}s`
  const h = Math.floor(m / 60)
  return `${h}h ${m % 60}m`
}

