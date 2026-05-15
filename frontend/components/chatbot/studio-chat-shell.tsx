'use client'

/**
 * StudioChatShell — wraps the existing <Chat> in CD round-3's ledger layout.
 *
 * Two columns by default: collapsible threads (left) + main thread (right).
 * The mission rail is hidden until a real mission-state API is wired up — we
 * removed the pilot-snapshot rail because showing fake stats next to a real
 * chat reads as broken, not as a placeholder.
 *
 * Threads column persists collapsed/expanded state to localStorage so the
 * user's preference sticks across reloads.
 */

import { useEffect, useState } from 'react'
import {
  Plus,
  GitFork,
  Share2,
  ChevronRight,
  PanelLeftClose,
  PanelLeftOpen,
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

const STORAGE_KEY = 'studioChatThreadsCollapsed'

export function StudioChatShell({
  children,
  selectedChatId,
  selectedChat,
  onSelectChat,
  onNewChat,
}: StudioChatShellProps) {
  const [threads, setThreads] = useState<ChatType[]>([])
  const [loadingThreads, setLoadingThreads] = useState(true)
  const [threadsCollapsed, setThreadsCollapsed] = useState(false)

  // Load persisted collapse state
  useEffect(() => {
    try {
      if (localStorage.getItem(STORAGE_KEY) === '1') {
        setThreadsCollapsed(true)
      }
    } catch {}
  }, [])

  const toggleThreads = () => {
    setThreadsCollapsed((prev) => {
      const next = !prev
      try {
        localStorage.setItem(STORAGE_KEY, next ? '1' : '0')
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

  const activeTitle = selectedChat?.title ?? 'New conversation'

  return (
    <div className={`sh-chat${threadsCollapsed ? ' threads-collapsed' : ''}`}>
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

      {/* Two-column body */}
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
                      <span className="sh-chat-thread-title">{t.title}</span>
                      <span className="sh-chat-thread-ts">{rel}</span>
                    </button>
                  )
                })
              )}
            </div>
          </aside>
        )}

        {/* Main thread */}
        <div className="sh-chat-main">{children}</div>
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
