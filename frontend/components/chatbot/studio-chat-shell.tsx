'use client'

/**
 * StudioChatShell — wraps the existing <Chat> component in CD round-3's
 * three-column ledger layout: threads (220px) | thread (1fr) | mission rail
 * (280px). Reuses chat-history data from `getChatHistory` so the thread list
 * shows real conversations.
 *
 * Mission rail uses pilot snapshot data for now (no mission-state API yet);
 * marked with TODO so it's obvious where to plug live data in later.
 *
 * Mobile + classic theme still use the original /chat layout; this only
 * renders in Studio desktop mode.
 */

import { useEffect, useState } from 'react'
import {
  Plus,
  GitFork,
  Share2,
  ChevronRight,
  FileText,
  Bot,
} from 'lucide-react'
import { getChatHistory } from '@/lib/chat/api'
import type { Chat as ChatType, ChatMessage } from '@/types'

export interface StudioChatShellProps {
  children: React.ReactNode
  selectedChatId: string
  selectedChat: ChatType | null
  onSelectChat: (chat: ChatType, messages: ChatMessage[]) => void
  onNewChat: () => void
}

interface DagStep {
  n: number
  agent: string
  tool: string
  status: 'done' | 'queued' | 'error' | 'pending'
  dur?: string
}

const PILOT_DAG: DagStep[] = [
  { n: 1, agent: 'Halberd', tool: 'research.collect', status: 'done', dur: '1m 04s' },
  { n: 2, agent: 'Scribe', tool: 'drafts.create', status: 'done', dur: '2m 17s' },
  { n: 3, agent: 'Sentinel', tool: 'review.pass', status: 'done', dur: '0m 49s' },
  { n: 4, agent: 'Sentinel', tool: 'github.create_pr', status: 'error', dur: '0m 03s' },
  { n: 5, agent: 'Auto', tool: 'recovery.retry', status: 'queued' },
]

const PILOT_MISSION = {
  id: 'msn_8f3a',
  title: 'Launch · Q3 product update',
  steps: '3 / 5',
  elapsed: '4m 31s',
  spend: '$0.0166',
  retryIn: '00:23',
}

export function StudioChatShell({
  children,
  selectedChatId,
  selectedChat,
  onSelectChat,
  onNewChat,
}: StudioChatShellProps) {
  const [threads, setThreads] = useState<ChatType[]>([])
  const [loadingThreads, setLoadingThreads] = useState(true)

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

  const activeTitle = selectedChat?.title ?? 'New conversation'

  return (
    <div className="sh-chat">
      {/* Breadcrumb bar */}
      <div className="sh-chat-bar">
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
        <div style={{ marginLeft: 'auto', display: 'flex', gap: 8 }}>
          <button type="button" className="sh-chat-act" title="Fork thread (coming soon)">
            <GitFork style={{ width: 11, height: 11 }} />
            <span>Fork thread</span>
          </button>
          <button type="button" className="sh-chat-act" title="Share (coming soon)">
            <Share2 style={{ width: 11, height: 11 }} />
            <span>Share</span>
          </button>
        </div>
      </div>

      {/* Three-column body */}
      <div className="sh-chat-grid">
        {/* Threads list */}
        <aside className="sh-chat-threads" aria-label="Chat threads">
          <div className="sh-chat-threads-head">
            <span className="sh-chat-eyebrow-mono">Threads</span>
            <button type="button" className="sh-chat-act" onClick={onNewChat}>
              <Plus style={{ width: 11, height: 11 }} />
              <span>New</span>
            </button>
          </div>
          <div className="sh-chat-thread-list">
            {loadingThreads ? (
              <div className="sh-chat-thread-empty">Loading…</div>
            ) : threads.length === 0 ? (
              <div className="sh-chat-thread-empty">No conversations yet</div>
            ) : (
              threads.map((t) => {
                const isActive = t.id === selectedChatId
                const rel = relativeTime(t.createdAt)
                return (
                  <button
                    key={t.id}
                    type="button"
                    className={`sh-chat-thread${isActive ? ' active' : ''}`}
                    onClick={() => onSelectChat(t, [])}
                    title={t.title}
                  >
                    <span
                      className={`sh-chat-thread-dot${isActive ? ' warn' : ' ok'}`}
                      aria-hidden
                    />
                    <span className="sh-chat-thread-body">
                      <span className="sh-chat-thread-title">{t.title}</span>
                      <span className="sh-chat-thread-id">{t.id.slice(0, 8)}</span>
                    </span>
                    <span className="sh-chat-thread-ts">{rel}</span>
                  </button>
                )
              })
            )}
          </div>
        </aside>

        {/* Main thread — slot the existing <Chat> here */}
        <div className="sh-chat-main">{children}</div>

        {/* Mission rail — TODO: wire to mission-state API once available */}
        <aside className="sh-chat-rail" aria-label="Mission details">
          <div>
            <p className="sh-chat-rail-eyebrow">Mission · this thread</p>
            <div className="sh-chat-rail-title">{PILOT_MISSION.title}</div>
            <div className="sh-chat-rail-id">{PILOT_MISSION.id}</div>
          </div>

          <div className="sh-chat-rail-stats">
            <Stat label="STEPS" value={PILOT_MISSION.steps} tone="accent" />
            <Stat label="ELAPSED" value={PILOT_MISSION.elapsed} />
            <Stat label="SPEND" value={PILOT_MISSION.spend} tone="olive" />
            <Stat label="RETRY IN" value={PILOT_MISSION.retryIn} tone="accent" />
          </div>

          <div>
            <p className="sh-chat-rail-eyebrow">Pipeline</p>
            <div className="sh-chat-dag">
              {PILOT_DAG.map((s) => (
                <DagRow key={s.n} step={s} />
              ))}
            </div>
          </div>

          <div>
            <p className="sh-chat-rail-eyebrow">Deliverables</p>
            <div className="sh-chat-deliverable">
              <FileText style={{ width: 13, height: 13, color: 'hsl(var(--info))' }} />
              <span style={{ flex: 1, fontWeight: 500, fontSize: 12 }}>
                Q3-update.md
              </span>
              <span className="sh-chat-rail-mono" style={{ fontSize: 10.5 }}>
                2.1 KB
              </span>
            </div>
          </div>

          <button type="button" className="sh-chat-rail-cta">
            <Bot style={{ width: 12, height: 12 }} />
            Open mission detail →
          </button>
        </aside>
      </div>
    </div>
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

function DagRow({ step }: { step: DagStep }) {
  const tone =
    step.status === 'error'
      ? 'err'
      : step.status === 'queued'
      ? 'queued'
      : step.status === 'done'
      ? 'done'
      : 'pending'
  const mark =
    step.status === 'done'
      ? '✓'
      : step.status === 'error'
      ? '!'
      : step.status === 'queued'
      ? '↻'
      : step.n
  return (
    <div className={`sh-chat-dag-step ${tone}`}>
      <span className="sh-chat-dag-pip">{mark}</span>
      <div style={{ flex: 1, minWidth: 0 }}>
        <div className="sh-chat-dag-agent">{step.agent}</div>
        <div className="sh-chat-dag-tool">{step.tool}</div>
      </div>
      {step.dur && <span className="sh-chat-rail-mono sh-chat-dag-dur">{step.dur}</span>}
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
