
'use client'

import { useState, useEffect, useCallback, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  X,
  Send,
  Bot,
  CheckCircle2,
  AlertCircle,
  ImageIcon,
  Trash2,
  Bug,
  Loader2,
  ArrowUpRight,
  ChevronDown,
  Plus,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import { Label } from '@/components/ui/label'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { useUser } from '@/lib/auth-hooks'
import { useSubmitBugReport, type BugReportRequest } from '@/hooks/use-bug-report-api'
import { useChat } from '@/lib/chat/hooks'
import { usePageContext } from '@/lib/page-context'
import { getPageEntry } from '@/lib/generated/page-manifest'
import { useSystemRole } from '@/contexts/role-context'
import { getChat, getChatHistory, getChatMessages } from '@/lib/chat/api'
import { useChatSessionHydration } from '@/hooks/use-chat-session'
import { useChatSessionStore } from '@/stores/chat-session-store'
import { sessionTabs, threadTimeAgo } from '@/lib/chat/chat-session'
import type { Chat, ChatMessage } from '@/types'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { useRouter } from 'next/navigation'

// =============================================================================
// Types
// =============================================================================

interface AutoWidgetProps {
  position?: 'bottom-right' | 'bottom-left'
  currentPage?: string
  visible?: boolean
}

type FormState = 'form' | 'loading' | 'success' | 'error'

// Page labels for context hint
const PAGE_LABELS: Record<string, string> = {
  dashboard: 'Dashboard',
  agents: 'Agent Management',
  documents: 'Knowledge Base',
  tools: 'Tools & Integrations',
  marketplace: 'Marketplace',
  analytics: 'Analytics',
  workspace: 'Workspace',
  activity: 'Activity',
  settings: 'Settings',
  team: 'Team Management',
  chat: 'Chat',
}

// =============================================================================
// Mini Auto Chat Tab — the same conversation session as the chat page (PRD-237)
// =============================================================================

interface WidgetView {
  chatId: string | null
  messages: ChatMessage[]
  awaitingReply: boolean
}

const WIDGET_DRAFT_VIEW: WidgetView = { chatId: null, messages: [], awaitingReply: false }
/** Recent rows fetched for row metadata (kind badge, time, preview). */
const WIDGET_HISTORY_ROWS = 10

function AutoChatTab({ currentPage, onClose }: { currentPage: string; onClose?: () => void }) {
  const router = useRouter()

  // PRD-237: one session store for the page and the widget — same active
  // conversation, same open tabs. The widget remounts on every navigation
  // (MainLayout is per page), so the store is what makes it "keep going".
  const { hydrated } = useChatSessionHydration()
  const session = useChatSessionStore((s) => s.session)
  const openChat = useChatSessionStore((s) => s.openChat)
  const newDraft = useChatSessionStore((s) => s.newDraft)
  const closeTab = useChatSessionStore((s) => s.closeTab)
  const chatIdAssigned = useChatSessionStore((s) => s.chatIdAssigned)
  const setTitles = useChatSessionStore((s) => s.setTitles)

  const [view, setView] = useState<WidgetView | null>(null)
  const [threads, setThreads] = useState<Chat[]>([])
  const [threadsOpen, setThreadsOpen] = useState(false)
  const [draftInstance, setDraftInstance] = useState(0)
  // Monotonic token so a slow fetch can't stomp a newer thread change.
  const loadSeqRef = useRef(0)

  const refreshThreads = useCallback(async () => {
    try {
      const rows = await getChatHistory(WIDGET_HISTORY_ROWS)
      setThreads(rows)
      setTitles(Object.fromEntries(rows.map((row) => [row.id, row.title])))
    } catch {
      // Row metadata is best-effort — the active conversation still works.
    }
  }, [setTitles])

  useEffect(() => {
    if (hydrated) refreshThreads().catch(() => {})
  }, [hydrated, refreshThreads])

  // Reconnect to the active conversation whenever the pointer moves.
  useEffect(() => {
    if (!hydrated) return
    const activeId = session.activeChatId
    if (activeId === null) {
      setView((current) => (current && current.chatId === null ? current : WIDGET_DRAFT_VIEW))
      return
    }
    if (view?.chatId === activeId) return
    const seq = ++loadSeqRef.current
    let cancelled = false
    Promise.all([getChatMessages(activeId), getChat(activeId)])
      .then(([messages, chat]) => {
        if (cancelled || seq !== loadSeqRef.current) return
        setView({ chatId: activeId, messages, awaitingReply: Boolean(chat.turnInFlight) })
        setTitles({ [activeId]: chat.title })
      })
      .catch(() => {
        // Stale or inaccessible chat (deleted, workspace switch) — drop the tab.
        if (!cancelled && seq === loadSeqRef.current) closeTab(activeId)
      })
    return () => {
      cancelled = true
    }
  }, [hydrated, session.activeChatId, view?.chatId, closeTab, setTitles])

  const switchThread = useCallback(
    (chatId: string) => {
      setThreadsOpen(false)
      openChat(chatId)
    },
    [openChat]
  )

  const startNewThread = useCallback(() => {
    setThreadsOpen(false)
    loadSeqRef.current++ // invalidate any in-flight load
    newDraft()
    setView(WIDGET_DRAFT_VIEW)
    setDraftInstance((v) => v + 1)
  }, [newDraft])

  // Closes the tab everywhere — the conversation itself stays in history.
  const closeThread = useCallback(
    (chatId: string, e: React.MouseEvent) => {
      e.stopPropagation()
      closeTab(chatId)
    },
    [closeTab]
  )

  // S3: promote — the full chat page reads the same session, so it lands on this thread.
  const openFullChat = useCallback(() => {
    setThreadsOpen(false)
    onClose?.()
    router.push('/chat')
  }, [onClose, router])

  const openTabs = sessionTabs(session).filter((tab) => !tab.isDraft).map((tab) => tab.id as string)
  const threadById = new Map(threads.map((row) => [row.id, row]))
  const activeChatId = session.activeChatId
  const activeTitle = activeChatId ? session.titles[activeChatId] ?? 'Conversation' : 'New conversation'
  const hasUnread = session.unreadChatIds.length > 0
  const pageLabel = PAGE_LABELS[currentPage] || currentPage

  return (
    <div className="flex flex-col h-[400px]">
      {/* Thread switcher toolbar — the open tabs, shared with the chat page */}
      <div className="relative px-3 pb-1.5">
        <div className="flex items-center gap-1">
          <button
            onClick={() => {
              setThreadsOpen((v) => !v)
              if (!threadsOpen) refreshThreads().catch(() => {})
            }}
            className="flex min-w-0 flex-1 items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors"
            title="Conversations"
          >
            <ChevronDown
              className={`w-3 h-3 shrink-0 transition-transform ${threadsOpen ? 'rotate-180' : ''}`}
            />
            <span className="truncate">{activeTitle}</span>
            {hasUnread && (
              <span
                className="w-1.5 h-1.5 rounded-full bg-primary shrink-0"
                aria-label="Unread messages"
              />
            )}
          </button>
          <Button
            variant="ghost"
            size="sm"
            onClick={startNewThread}
            className="h-6 w-6 p-0 shrink-0 text-muted-foreground hover:text-foreground"
            title="New thread"
          >
            <Plus className="w-3.5 h-3.5" />
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={openFullChat}
            className="h-6 w-6 p-0 shrink-0 text-muted-foreground hover:text-foreground"
            title="Open in full chat"
          >
            <ArrowUpRight className="w-3.5 h-3.5" />
          </Button>
        </div>

        {threadsOpen && (
          <>
            {/* Click-outside catcher */}
            <div className="fixed inset-0 z-10" onClick={() => setThreadsOpen(false)} />
            <div className="absolute left-2 right-2 top-7 z-20 overflow-hidden rounded-xl border border-border/60 bg-background/95 backdrop-blur-xl shadow-xl">
              <button
                onClick={startNewThread}
                className="flex w-full items-center gap-2 px-3 py-2 text-xs text-foreground hover:bg-secondary/40 transition-colors"
              >
                <Plus className="w-3 h-3" />
                New thread
              </button>
              {openTabs.length > 0 && <div className="h-px bg-border/50" />}
              <div className="max-h-56 overflow-y-auto">
                {openTabs.map((chatId) => {
                  const row = threadById.get(chatId)
                  const title = session.titles[chatId] ?? row?.title ?? 'Conversation'
                  return (
                    <div
                      key={chatId}
                      role="button"
                      tabIndex={0}
                      onClick={() => switchThread(chatId)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' || e.key === ' ') {
                          e.preventDefault()
                          switchThread(chatId)
                        }
                      }}
                      className={`group flex w-full cursor-pointer flex-col gap-0.5 px-3 py-2 text-left transition-colors hover:bg-secondary/40 ${
                        chatId === activeChatId ? 'bg-secondary/30' : ''
                      }`}
                    >
                      <div className="flex items-center gap-1.5">
                        {session.unreadChatIds.includes(chatId) && (
                          <span className="w-1.5 h-1.5 rounded-full bg-primary shrink-0" aria-label="Unread" />
                        )}
                        <span className="min-w-0 flex-1 truncate text-xs font-medium text-foreground">
                          {title}
                        </span>
                        {/* PRD-205 S7: mark the thread where Auto speaks unprompted */}
                        {row?.kind === 'auto' && (
                          <span className="shrink-0 rounded-full border border-warning/20 bg-warning/10 px-1.5 text-[9px] leading-4 text-warning">
                            Auto
                          </span>
                        )}
                        {row && (
                          <span className="shrink-0 text-[10px] text-muted-foreground">
                            {threadTimeAgo(row.updatedAt)}
                          </span>
                        )}
                        <button
                          onClick={(e) => closeThread(chatId, e)}
                          className="shrink-0 text-muted-foreground opacity-0 transition-opacity hover:text-foreground group-hover:opacity-100"
                          title="Close thread"
                          aria-label={`Close thread ${title}`}
                        >
                          <X className="w-3 h-3" />
                        </button>
                      </div>
                      {row?.lastMessagePreview && (
                        <p className="truncate text-[11px] text-muted-foreground">
                          {row.lastMessagePreview}
                        </p>
                      )}
                    </div>
                  )
                })}
              </div>
              <div className="h-px bg-border/50" />
              <button
                onClick={openFullChat}
                className="flex w-full items-center gap-2 px-3 py-2 text-xs text-muted-foreground hover:bg-secondary/40 hover:text-foreground transition-colors"
              >
                <ArrowUpRight className="w-3 h-3" />
                Open in full chat
              </button>
            </div>
          </>
        )}
      </div>

      {!view ? (
        <div className="flex flex-1 flex-col items-center justify-center gap-2 text-muted-foreground">
          <Loader2 className="w-4 h-4 animate-spin" />
          <p className="text-xs">Continuing conversation…</p>
        </div>
      ) : (
        <WidgetConversation
          key={view.chatId ?? `draft-${draftInstance}`}
          initialChatId={view.chatId}
          initialMessages={view.messages}
          initialAwaitingReply={view.awaitingReply}
          pageLabel={pageLabel}
          onChatIdAssigned={chatIdAssigned}
        />
      )}
    </div>
  )
}

function WidgetConversation({
  initialChatId,
  initialMessages,
  initialAwaitingReply = false,
  pageLabel,
  onChatIdAssigned,
}: {
  initialChatId: string | null
  initialMessages: ChatMessage[]
  /** PRD-237 S7: the server is still producing a reply for this chat. */
  initialAwaitingReply?: boolean
  pageLabel: string
  onChatIdAssigned: (chatId: string) => void
}) {
  const inputRef = useRef<HTMLInputElement>(null)
  const scrollRef = useRef<HTMLDivElement>(null)
  const [inputValue, setInputValue] = useState('')

  // PRD-221 S5: send the structured page context (manifest key + route), not
  // the display label. The label stays UI-only.
  const pageContext = usePageContext()

  // PRD-221 S7: the current page's quick-prompt chips (from the manifest
  // mirror), with admin-only prompts hidden for non-admins. The tool layer
  // remains the real enforcer — this is a UX filter only.
  const { isAdmin } = useSystemRole()
  const quickPrompts = (getPageEntry(pageContext.page)?.quick_prompts ?? []).filter(
    (qp) => !qp.admin_only || isAdmin,
  )

  const {
    messages,
    isLoading,
    sendMessage,
    stop,
  } = useChat({
    // Empty id → the backend creates the chat and streams its id back.
    id: initialChatId ?? '',
    initialMessages,
    initialAwaitingReply,
    selectedAgentId: undefined, // Routes to Auto (default agent)
    pageContext,
    onChatIdUpdate: onChatIdAssigned,
  })

  // Auto-scroll on new messages
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [messages])

  const handleSend = () => {
    const text = inputValue.trim()
    if (!text || isLoading) return
    sendMessage(text)
    setInputValue('')
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      {/* Messages area */}
      <div
        ref={scrollRef}
        className="flex-1 overflow-y-auto px-3 py-2 space-y-3"
      >
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-center px-4">
            <div className="w-10 h-10 rounded-full bg-primary/10 border border-primary/20 flex items-center justify-center mb-3">
              <Bot className="w-5 h-5 text-primary" />
            </div>
            <p className="text-sm font-medium text-foreground mb-1">Hey, I'm Auto</p>
            <p className="text-xs text-muted-foreground">
              Ask me anything about {pageLabel || 'this page'}, or tell me what you need help with.
            </p>
            <div className="mt-3 flex flex-wrap justify-center gap-1.5">
              {/* PRD-206 S3: resume chip — always offered on an empty thread,
                  answered in-chat by the platform_resume_context tool */}
              <button
                type="button"
                onClick={() => !isLoading && sendMessage('Where did we leave off?')}
                disabled={isLoading}
                className="rounded-full border border-border bg-muted/50 px-2.5 py-1 text-xs text-foreground transition-colors hover:bg-muted disabled:opacity-50"
              >
                Where did we leave off?
              </button>
              {quickPrompts.map((qp) => (
                <button
                  key={qp.text}
                  type="button"
                  onClick={() => !isLoading && sendMessage(qp.text)}
                  disabled={isLoading}
                  className="rounded-full border border-border bg-muted/50 px-2.5 py-1 text-xs text-foreground transition-colors hover:bg-muted disabled:opacity-50"
                >
                  {qp.text}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((msg) => (
          <div
            key={msg.id}
            className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`max-w-[85%] rounded-xl px-3 py-2 text-sm ${
                msg.role === 'user'
                  ? 'bg-primary/15 text-foreground'
                  : 'bg-secondary/40 text-foreground'
              }`}
            >
              {msg.role === 'assistant' ? (
                <div className="prose prose-sm prose-invert max-w-none [&_p]:mb-1.5 [&_p]:last:mb-0 [&_ul]:mb-1.5 [&_li]:mb-0.5 [&_code]:text-xs [&_pre]:text-xs [&_h1]:text-sm [&_h2]:text-sm [&_h3]:text-xs">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>
                    {msg.content || ''}
                  </ReactMarkdown>
                </div>
              ) : (
                <p className="whitespace-pre-wrap">{msg.content}</p>
              )}
            </div>
          </div>
        ))}

        {isLoading && messages[messages.length - 1]?.role === 'user' && (
          <div className="flex justify-start">
            <div className="bg-secondary/40 rounded-xl px-3 py-2">
              <Loader2 className="w-4 h-4 animate-spin text-muted-foreground" />
            </div>
          </div>
        )}
      </div>

      {/* Input area */}
      <div className="px-3 pb-3 pt-1">
        <div className="flex items-center gap-2">
          <Input
            ref={inputRef}
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask Auto anything..."
            disabled={isLoading}
            className="flex-1 text-sm h-9 bg-secondary/30 border-border/50"
          />
          {isLoading ? (
            <Button
              size="sm"
              variant="ghost"
              onClick={stop}
              className="h-9 w-9 p-0 shrink-0"
            >
              <X className="w-4 h-4" />
            </Button>
          ) : (
            <Button
              size="sm"
              onClick={handleSend}
              disabled={!inputValue.trim()}
              className="h-9 w-9 p-0 shrink-0 bg-primary/80 hover:bg-primary"
            >
              <Send className="w-4 h-4" />
            </Button>
          )}
        </div>
      </div>
    </div>
  )
}

// =============================================================================
// Component
// =============================================================================

export function AutoWidget({
  position = 'bottom-left',
  currentPage = 'dashboard',
  visible = true,
}: AutoWidgetProps) {
  const { user } = useUser()
  const submitBugReport = useSubmitBugReport()

  // Widget state
  const [isOpen, setIsOpen] = useState(false)
  const [activeTab, setActiveTab] = useState('auto')

  // Bug report form state
  const [title, setTitle] = useState('')
  const [description, setDescription] = useState('')
  const [severity, setSeverity] = useState('Minor')
  const [category, setCategory] = useState('Other')
  const [screenshot, setScreenshot] = useState<string | null>(null)
  const [formState, setFormState] = useState<FormState>('form')
  const [errorMessage, setErrorMessage] = useState('')
  const [successKey, setSuccessKey] = useState('')
  const [successUrl, setSuccessUrl] = useState('')

  // Console error capture for bug reports
  const consoleErrorsRef = useRef<string[]>([])

  useEffect(() => {
    const originalError = console.error
    const capture = (...args: any[]) => {
      const msg = args.map((a) => (typeof a === 'string' ? a : JSON.stringify(a))).join(' ')
      consoleErrorsRef.current = [...consoleErrorsRef.current.slice(-19), msg]
      originalError.apply(console, args)
    }
    console.error = capture
    return () => {
      console.error = originalError
    }
  }, [])

  // Escape key closes widget
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setIsOpen(false)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  // Screenshot paste handler
  const handlePaste = useCallback((e: React.ClipboardEvent) => {
    const items = e.clipboardData?.items
    if (!items) return
    for (const item of Array.from(items)) {
      if (item.type.startsWith('image/')) {
        const file = item.getAsFile()
        if (!file) continue
        if (file.size > 5 * 1024 * 1024) {
          setErrorMessage('Screenshot must be under 5 MB')
          return
        }
        const reader = new FileReader()
        reader.onload = () => {
          setScreenshot(reader.result as string)
        }
        reader.readAsDataURL(file)
        break
      }
    }
  }, [])

  const resetForm = () => {
    setTitle('')
    setDescription('')
    setSeverity('Minor')
    setCategory('Other')
    setScreenshot(null)
    setFormState('form')
    setErrorMessage('')
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!title.trim() || !description.trim()) return

    setFormState('loading')
    setErrorMessage('')

    const payload: BugReportRequest = {
      title: title.trim(),
      description: description.trim(),
      severity,
      category,
      screenshot_base64: screenshot || undefined,
      context: {
        url: window.location.href,
        page: currentPage,
        user_agent: navigator.userAgent,
        viewport: `${window.innerWidth}x${window.innerHeight}`,
        user_email: user?.primaryEmailAddress?.emailAddress,
        user_name: user?.fullName || undefined,
        console_errors: consoleErrorsRef.current.slice(-20),
        timestamp: new Date().toISOString(),
      },
    }

    submitBugReport.mutate(payload, {
      onSuccess: (data) => {
        if (data.success) {
          setFormState('success')
          setSuccessKey(data.jira_key || '')
          setSuccessUrl(data.jira_url || '')
        } else {
          setFormState('error')
          setErrorMessage(data.message || 'Failed to create bug report')
        }
      },
      onError: (err: any) => {
        setFormState('error')
        setErrorMessage(err?.message || 'An unexpected error occurred')
      },
    })
  }

  if (!visible) return null

  const positionClasses = {
    'bottom-right': 'bottom-3 right-3 md:bottom-4 md:right-4',
    'bottom-left': 'bottom-3 left-3 md:bottom-4 md:left-20',
  }

  return (
    <div className={`fixed ${positionClasses[position]} z-[60]`}>
      <AnimatePresence mode="wait">
        {isOpen && (
          <motion.div
            key="popup"
            initial={{ scale: 0.9, opacity: 0, y: 20 }}
            animate={{ scale: 1, opacity: 1, y: 0 }}
            exit={{ scale: 0.9, opacity: 0, y: 20 }}
            transition={{ type: 'spring', stiffness: 400, damping: 25 }}
            className="mb-4 w-[calc(100vw-2rem)] sm:w-[380px] glass-card card-glow rounded-2xl shadow-2xl overflow-hidden"
          >
            {/* Header */}
            <div className="flex items-center justify-between px-4 pt-3 pb-2">
              <div className="flex items-center gap-2">
                <div className="w-8 h-8 rounded-full bg-primary/10 border border-primary/20 flex items-center justify-center">
                  <Bot className="w-4 h-4 text-primary" />
                </div>
                <div>
                  <h3 className="text-sm font-semibold text-foreground leading-none">Auto</h3>
                  <p className="text-[10px] text-muted-foreground">Your AI assistant</p>
                </div>
              </div>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setIsOpen(false)}
                className="text-muted-foreground hover:text-foreground h-7 w-7 p-0"
              >
                <X className="w-4 h-4" />
              </Button>
            </div>

            {/* Tabs */}
            <Tabs value={activeTab} onValueChange={setActiveTab}>
              <div className="px-4">
                <TabsList className="w-full mb-0 h-8">
                  <TabsTrigger value="auto" className="flex-1 text-xs h-7">
                    <Bot className="w-3 h-3 mr-1" />
                    Auto
                  </TabsTrigger>
                  <TabsTrigger value="bug" className="flex-1 text-xs h-7">
                    <Bug className="w-3 h-3 mr-1" />
                    Report Bug
                  </TabsTrigger>
                </TabsList>
              </div>

              {/* ---- Auto Chat Tab ---- */}
              <TabsContent value="auto" className="mt-0">
                <AutoChatTab currentPage={currentPage} onClose={() => setIsOpen(false)} />
              </TabsContent>

              {/* ---- Bug Report Tab ---- */}
              <TabsContent value="bug" className="mt-0 px-4 pb-4 max-h-[460px] overflow-y-auto">
                {formState === 'form' || formState === 'loading' ? (
                  <form onSubmit={handleSubmit} className="space-y-3" onPaste={handlePaste}>
                    <div className="space-y-1">
                      <Label htmlFor="bug-title" className="text-xs">Title</Label>
                      <Input
                        id="bug-title"
                        value={title}
                        onChange={(e) => setTitle(e.target.value)}
                        placeholder="Brief summary of the issue"
                        required
                        disabled={formState === 'loading'}
                        className="text-sm"
                      />
                    </div>

                    <div className="space-y-1">
                      <Label htmlFor="bug-desc" className="text-xs">Description</Label>
                      <Textarea
                        id="bug-desc"
                        value={description}
                        onChange={(e) => setDescription(e.target.value)}
                        placeholder="What happened? Steps to reproduce..."
                        required
                        disabled={formState === 'loading'}
                        className="text-sm min-h-[80px] resize-none"
                      />
                    </div>

                    <div className="grid grid-cols-2 gap-2">
                      <div className="space-y-1">
                        <Label className="text-xs">Severity</Label>
                        <Select value={severity} onValueChange={setSeverity} disabled={formState === 'loading'}>
                          <SelectTrigger className="text-sm">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="Critical">Critical</SelectItem>
                            <SelectItem value="Major">Major</SelectItem>
                            <SelectItem value="Minor">Minor</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                      <div className="space-y-1">
                        <Label className="text-xs">Category</Label>
                        <Select value={category} onValueChange={setCategory} disabled={formState === 'loading'}>
                          <SelectTrigger className="text-sm">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="UI Bug">UI Bug</SelectItem>
                            <SelectItem value="Data Issue">Data Issue</SelectItem>
                            <SelectItem value="Performance">Performance</SelectItem>
                            <SelectItem value="Other">Other</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                    </div>

                    <div className="space-y-1">
                      <Label className="text-xs">Screenshot</Label>
                      {screenshot ? (
                        <div className="relative rounded-lg overflow-hidden border border-border">
                          <img src={screenshot} alt="Screenshot preview" className="w-full max-h-32 object-contain bg-black/5" />
                          <Button
                            type="button"
                            variant="ghost"
                            size="sm"
                            onClick={() => setScreenshot(null)}
                            className="absolute top-1 right-1 h-6 w-6 p-0 bg-black/50 hover:bg-black/70 text-white rounded-full"
                          >
                            <Trash2 className="w-3 h-3" />
                          </Button>
                        </div>
                      ) : (
                        <div className="flex items-center gap-2 p-3 rounded-lg border border-dashed border-border text-muted-foreground text-xs">
                          <ImageIcon className="w-4 h-4 shrink-0" />
                          <span>Paste a screenshot (Ctrl+V / Cmd+V)</span>
                        </div>
                      )}
                    </div>

                    {errorMessage && formState === 'form' && (
                      <p className="text-xs text-destructive">{errorMessage}</p>
                    )}

                    <Button
                      type="submit"
                      disabled={formState === 'loading' || !title.trim() || !description.trim()}
                      className="w-full bg-primary hover:bg-primary/90 text-primary-foreground"
                    >
                      {formState === 'loading' ? (
                        <span className="flex items-center gap-2">
                          <motion.div
                            animate={{ rotate: 360 }}
                            transition={{ repeat: Infinity, duration: 1, ease: 'linear' }}
                            className="w-4 h-4 border-2 border-white border-t-transparent rounded-full"
                          />
                          Submitting...
                        </span>
                      ) : (
                        'Submit Bug Report'
                      )}
                    </Button>
                  </form>
                ) : formState === 'success' ? (
                  <div className="flex flex-col items-center py-6 space-y-3">
                    <CheckCircle2 className="w-12 h-12 text-success" />
                    <p className="font-medium text-foreground">Bug report created!</p>
                    {successKey && (
                      successUrl ? (
                        <a
                          href={successUrl}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-sm text-primary hover:text-primary/80 underline"
                        >
                          {successKey}
                        </a>
                      ) : (
                        <span className="text-sm text-muted-foreground">{successKey}</span>
                      )
                    )}
                    <Button variant="outline" size="sm" onClick={resetForm}>
                      Submit another
                    </Button>
                  </div>
                ) : (
                  <div className="flex flex-col items-center py-6 space-y-3">
                    <AlertCircle className="w-12 h-12 text-destructive" />
                    <p className="font-medium text-foreground">Something went wrong</p>
                    <p className="text-xs text-muted-foreground text-center">{errorMessage}</p>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setFormState('form')}
                    >
                      Try again
                    </Button>
                  </div>
                )}
              </TabsContent>
            </Tabs>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Floating Auto button */}
      <motion.div
        initial={{ scale: 0, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ type: 'spring', stiffness: 500, damping: 30 }}
      >
        <Button
          onClick={() => setIsOpen((v) => !v)}
          className="w-12 h-12 rounded-full bg-primary/90 hover:bg-primary shadow-lg transition-all duration-300 group p-0"
          size="lg"
          title="Ask Auto"
        >
          <Bot className="w-5 h-5 md:w-6 md:h-6 text-primary-foreground group-hover:scale-110 transition-transform" />
        </Button>
      </motion.div>
    </div>
  )
}

// Backward compatibility
export const ChatWidget = AutoWidget
export const PilotHelperWidget = AutoWidget
