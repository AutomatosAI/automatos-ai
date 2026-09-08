'use client'

import { useCallback, useEffect, useRef, useState } from 'react'
import { normalizeCodeRoot } from '@/components/widgets/CodingCanvasWidget/code-root'
import { AnimatePresence, motion } from 'framer-motion'
import { useSearchParams, useRouter } from 'next/navigation'
import { ArrowLeft } from 'lucide-react'
import { toast } from 'sonner'
import { MainLayout } from '@/components/layout/main-layout'
import { Chat } from '@/components/chatbot/chat'
import { ChatTabs, type ChatTab } from '@/components/chatbot/chat-tabs'
import { FirstRunNudge } from '@/components/local/first-run-nudge'
import { AppSidebar } from '@/components/chatbot/sidebar'
import { StudioChatShell } from '@/components/chatbot/studio-chat-shell'
import { Sheet, SheetContent } from '@/components/ui/sheet'
import { Button } from '@/components/ui/button'
import { usePageAPI } from '@/hooks/use-page-api'
import { useIsMobile } from '@/hooks/use-mobile'
import { useIsStudio } from '@/hooks/use-studio-theme'
import { useChatSessionHydration } from '@/hooks/use-chat-session'
import { useMissionStore } from '@/stores/mission-store'
import { useChatSessionStore } from '@/stores/chat-session-store'
import { getChat, getChatHistory, getChatMessages } from '@/lib/chat/api'
import { sessionTabs } from '@/lib/chat/chat-session'
import type { Chat as ChatType, ChatMessage } from '@/types'

// Force dynamic rendering
export const dynamic = 'force-dynamic'

const DRAFT_TITLE = 'New chat'
const UNTITLED = 'Conversation'
/** How many recent conversations seed the tab-title cache. */
const TITLE_CACHE_ROWS = 20

/** What the conversation surface is showing right now. */
interface ConversationView {
  chatId: string | null
  chat: ChatType | null
  messages: ChatMessage[]
  awaitingReply: boolean
}

const DRAFT_VIEW: ConversationView = { chatId: null, chat: null, messages: [], awaitingReply: false }

function withoutParam(params: URLSearchParams | null, name: string): string {
  const next = new URLSearchParams(params?.toString() ?? '')
  next.delete(name)
  const query = next.toString()
  return query ? `/chat?${query}` : '/chat'
}

export default function ChatPage() {
  usePageAPI('chat')
  const isMobile = useIsMobile()
  const isStudio = useIsStudio()
  const searchParams = useSearchParams()
  const router = useRouter()
  const setPlanMode = useMissionStore((s) => s.setPlanMode)

  // US-015: Deep-link params — ?mode=plan&from=assignments
  const modeParam = searchParams?.get('mode') ?? null
  const fromParam = searchParams?.get('from') ?? null
  // Deep-link: /chat?chatId=<id> opens that conversation as a tab (Activity links).
  const chatIdParam = searchParams?.get('chatId') ?? null
  // PRD-235 W2: /chat?repo=<folder> opens Code mode on that folder (a ticket's session or a repo under projects/).
  const repoParam = normalizeCodeRoot(searchParams?.get('repo')) ?? undefined
  const ticketParam = (searchParams?.get('ticket') || '').replace(/[^0-9]/g, '') || undefined
  // PRD-239 S7 v2: /chat?ticket=…&runtime=1 opens the Runtime Canvas (explorer + the session's terminal)
  const runtimeParam = searchParams?.get('runtime') === '1'

  // Activate plan mode when arriving via ?mode=plan
  useEffect(() => {
    if (modeParam === 'plan') {
      setPlanMode(true)
    }
  }, [modeParam, setPlanMode])

  // PRD-237: the conversation session — which chat is on screen and which are
  // open as tabs — persists per workspace+user and is shared with the widget.
  const { hydrated } = useChatSessionHydration()
  const session = useChatSessionStore((s) => s.session)
  const openChat = useChatSessionStore((s) => s.openChat)
  const newDraft = useChatSessionStore((s) => s.newDraft)
  const closeTab = useChatSessionStore((s) => s.closeTab)
  const closeDraft = useChatSessionStore((s) => s.closeDraft)
  const setTitles = useChatSessionStore((s) => s.setTitles)

  const [view, setView] = useState<ConversationView | null>(null)
  const [draftInstance, setDraftInstance] = useState(0)
  const [isHistoryOpen, setIsHistoryOpen] = useState(false)
  // Monotonic token so a slow fetch can't stomp a newer tab change.
  const loadSeqRef = useRef(0)
  const coldStartRef = useRef(false)
  const deepLinkRef = useRef<string | null>(null)

  // A deep link wins over the stored pointer, once per value. The param is then
  // dropped from the URL so New Chat + reload does not re-open it.
  useEffect(() => {
    if (!hydrated || !chatIdParam || deepLinkRef.current === chatIdParam) return
    deepLinkRef.current = chatIdParam
    openChat(chatIdParam)
    router.replace(withoutParam(searchParams, 'chatId'))
  }, [hydrated, chatIdParam, openChat, router, searchParams])

  // Cold start on this device (nothing stored, no deep link): land on the most
  // recent conversation — "stay on the last chat" — else start a draft.
  useEffect(() => {
    if (!hydrated || chatIdParam || coldStartRef.current) return
    if (session.openChatIds.length > 0 || session.draftOpen) return
    coldStartRef.current = true
    let cancelled = false
    getChatHistory(1)
      .then((rows) => {
        if (cancelled) return
        if (rows[0]) openChat(rows[0].id)
        else newDraft()
      })
      .catch(() => {
        if (!cancelled) newDraft()
      })
    return () => {
      cancelled = true
    }
  }, [hydrated, chatIdParam, session.openChatIds.length, session.draftOpen, openChat, newDraft])

  // Show the active conversation: fetch when the pointer moves to a chat we are
  // not already showing (the history panel hands over fetched messages itself).
  useEffect(() => {
    if (!hydrated) return
    const activeId = session.activeChatId
    if (activeId === null) {
      setView((current) => (current && current.chatId === null ? current : DRAFT_VIEW))
      return
    }
    if (view?.chatId === activeId) return
    const seq = ++loadSeqRef.current
    let cancelled = false
    Promise.all([getChat(activeId), getChatMessages(activeId)])
      .then(([chat, messages]) => {
        if (cancelled || seq !== loadSeqRef.current) return
        setView({ chatId: activeId, chat, messages, awaitingReply: Boolean(chat.turnInFlight) })
        setTitles({ [activeId]: chat.title })
      })
      .catch((err) => {
        if (cancelled || seq !== loadSeqRef.current) return
        console.error('Failed to load conversation:', err)
        toast.error('That conversation is no longer available')
        closeTab(activeId) // stale pointer (deleted, another workspace) — drop the tab
      })
    return () => {
      cancelled = true
    }
  }, [hydrated, session.activeChatId, view?.chatId, closeTab, setTitles])

  // Tab labels: seed from recent history, then fill any open tab still unnamed.
  useEffect(() => {
    if (!hydrated) return
    let cancelled = false
    getChatHistory(TITLE_CACHE_ROWS)
      .then((rows) => {
        if (!cancelled) setTitles(Object.fromEntries(rows.map((row) => [row.id, row.title])))
      })
      .catch(() => {})
    return () => {
      cancelled = true
    }
  }, [hydrated, setTitles])

  const unnamedKey = session.openChatIds.filter((id) => !session.titles[id]).join(',')
  useEffect(() => {
    if (!hydrated || !unnamedKey) return
    let cancelled = false
    Promise.all(
      unnamedKey.split(',').map((id) =>
        getChat(id)
          .then((chat) => [id, chat.title] as const)
          .catch(() => null),
      ),
    ).then((pairs) => {
      if (cancelled) return
      const named = pairs.filter((pair): pair is readonly [string, string] => pair !== null)
      if (named.length > 0) setTitles(Object.fromEntries(named))
    })
    return () => {
      cancelled = true
    }
  }, [hydrated, unnamedKey, setTitles])

  // Handle selecting a chat from the history panel (messages already fetched).
  const handleChatSelect = useCallback(
    (chat: ChatType, messages: ChatMessage[]) => {
      loadSeqRef.current++ // any in-flight load is stale now
      setView({ chatId: chat.id, chat, messages, awaitingReply: Boolean(chat.turnInFlight) })
      setTitles({ [chat.id]: chat.title })
      openChat(chat.id)
      setIsHistoryOpen(false)
    },
    [openChat, setTitles],
  )

  // New Chat: a draft tab — the backend names the conversation on the first message.
  const handleNewChat = useCallback(() => {
    loadSeqRef.current++
    newDraft()
    setView(DRAFT_VIEW)
    setDraftInstance((v) => v + 1) // a fresh composer even when a draft was already showing
    setIsHistoryOpen(false)
  }, [newDraft])

  const tabs: ChatTab[] = sessionTabs(session).map((tab) => ({
    id: tab.id,
    isDraft: tab.isDraft,
    title: tab.isDraft ? DRAFT_TITLE : session.titles[tab.id as string] ?? UNTITLED,
    active: tab.isDraft ? session.activeChatId === null : session.activeChatId === tab.id,
    unread: tab.isDraft ? false : session.unreadChatIds.includes(tab.id as string),
  }))

  const handleTabSelect = (tab: ChatTab) => {
    if (tab.isDraft) {
      if (session.activeChatId !== null) newDraft()
      return
    }
    openChat(tab.id as string)
  }

  const handleTabClose = (tab: ChatTab) => {
    if (tab.isDraft) closeDraft()
    else closeTab(tab.id as string)
  }

  // Toggle chat history from main Automatos sidebar (chat-only menu item)
  useEffect(() => {
    const handler = () => setIsHistoryOpen((v) => !v)
    window.addEventListener('automatos:chat-history-toggle', handler as any)
    return () => window.removeEventListener('automatos:chat-history-toggle', handler as any)
  }, [])

  const conversation = view ? (
    <Chat
      key={view.chatId ?? `draft-${draftInstance}`}
      id={view.chatId ?? ''}
      initialMessages={view.messages}
      initialVisibilityType={view.chat?.visibility || 'private'}
      isReadonly={false}
      autoResume={false}
      initialLastContext={view.chat?.lastContext}
      initialAwaitingReply={view.awaitingReply}
      initialCodeRoot={repoParam}
      initialCodeTicket={ticketParam}
      initialCodeRuntime={runtimeParam}
    />
  ) : (
    <div className="flex h-full items-center justify-center" role="status" aria-label="Loading conversation">
      <div className="h-6 w-6 animate-spin rounded-full border-2 border-primary border-t-transparent" />
    </div>
  )

  const chatBody = (
    <>
      {/* PRD-233 S3: local-edition first-run nudge — renders nothing in saas
          or once an LLM key exists; a zero-height overlay in both layouts. */}
      <FirstRunNudge />
      {conversation}
    </>
  )

  const historyPanel = (
    <AppSidebar
      onChatSelect={handleChatSelect}
      onNewChat={handleNewChat}
      activeChatId={session.activeChatId}
      onChatClosed={closeTab}
    />
  )

  // Studio desktop: CD's three-column ledger layout
  if (isStudio && !isMobile) {
    return (
      <MainLayout fullBleed>
        <StudioChatShell
          selectedChatId={session.activeChatId ?? ''}
          selectedChat={view?.chat ?? null}
          onSelectChat={handleChatSelect}
          onNewChat={handleNewChat}
          openChatIds={session.openChatIds}
          unreadChatIds={session.unreadChatIds}
          titles={session.titles}
          onCloseTab={closeTab}
        >
          {chatBody}
        </StudioChatShell>
      </MainLayout>
    )
  }

  // Classic layout (mobile + non-studio desktop)
  return (
    <MainLayout>
      <div className="relative flex h-[calc(100dvh-5rem)] flex-col md:h-[calc(100vh-8rem)]">
        <ChatTabs tabs={tabs} onSelect={handleTabSelect} onClose={handleTabClose} onNew={handleNewChat} />
        <div className="relative min-h-0 flex-1">
          {fromParam === 'assignments' && (
            <div className="absolute top-2 left-3 z-30">
              <Button
                variant="ghost"
                size="sm"
                className="gap-1.5 text-muted-foreground hover:text-foreground"
                onClick={() => router.push('/assignments')}
              >
                <ArrowLeft className="h-4 w-4" />
                Assignments
              </Button>
            </div>
          )}
          {isMobile ? (
            <Sheet open={isHistoryOpen} onOpenChange={setIsHistoryOpen}>
              <SheetContent side="left" className="w-[300px] p-0 bg-background/95 backdrop-blur-lg">
                {historyPanel}
              </SheetContent>
            </Sheet>
          ) : (
            <AnimatePresence>
              {isHistoryOpen && (
                <motion.aside
                  initial={{ x: -24, opacity: 0 }}
                  animate={{ x: 0, opacity: 1 }}
                  exit={{ x: -24, opacity: 0 }}
                  transition={{ type: 'spring', stiffness: 320, damping: 30 }}
                  className="absolute left-0 top-0 z-20 h-full w-[320px]"
                >
                  <div className="h-full rounded-r-3xl border-r border-warning/20 bg-background/35 backdrop-blur-xl shadow-[0_0_80px_rgba(0,0,0,0.55)]">
                    {historyPanel}
                  </div>
                </motion.aside>
              )}
            </AnimatePresence>
          )}

          <div className="h-full">{chatBody}</div>
        </div>
      </div>
    </MainLayout>
  )
}
