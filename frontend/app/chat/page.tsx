'use client'

import { useEffect, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { useSearchParams, useRouter } from 'next/navigation'
import { ArrowLeft } from 'lucide-react'
import { MainLayout } from '@/components/layout/main-layout'
import { Chat } from '@/components/chatbot/chat'
import { AppSidebar } from '@/components/chatbot/sidebar'
import { StudioChatShell } from '@/components/chatbot/studio-chat-shell'
import { Sheet, SheetContent } from '@/components/ui/sheet'
import { Button } from '@/components/ui/button'
import { usePageAPI } from '@/hooks/use-page-api'
import { useIsMobile } from '@/hooks/use-mobile'
import { useIsStudio } from '@/hooks/use-studio-theme'
import { useMissionStore } from '@/stores/mission-store'
import type { Chat as ChatType, ChatMessage } from '@/types'

// Force dynamic rendering
export const dynamic = 'force-dynamic'

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
  // Deep-link: /chat?chatId=<id> auto-loads that chat using client-side
  // auth (avoids the server-side /chat/[id] route's 404-on-fetch problem).
  const chatIdParam = searchParams?.get('chatId') ?? null

  // Activate plan mode when arriving via ?mode=plan
  useEffect(() => {
    if (modeParam === 'plan') {
      setPlanMode(true)
    }
  }, [modeParam, setPlanMode])

  // State for current chat
  const [currentChatId, setCurrentChatId] = useState('')
  const [currentMessages, setCurrentMessages] = useState<ChatMessage[]>([])
  const [selectedChat, setSelectedChat] = useState<ChatType | null>(null)
  const [isHistoryOpen, setIsHistoryOpen] = useState(false)
  const [chatInstance, setChatInstance] = useState(0)

  // Handle selecting a chat from sidebar
  const handleChatSelect = (chat: ChatType, messages: ChatMessage[]) => {
    setCurrentChatId(chat.id)
    setCurrentMessages(messages)
    setSelectedChat(chat)
    setIsHistoryOpen(false)
  }

  // Handle new chat from sidebar
  const handleNewChat = () => {
    setCurrentChatId('')  // Empty ID for new chats - backend will create one
    setCurrentMessages([])
    setSelectedChat(null)
    setChatInstance((v) => v + 1) // Force Chat remount even when id is empty
    setIsHistoryOpen(false)
  }

  // Pick up conversation context handed off from the Auto widget
  useEffect(() => {
    try {
      const raw = sessionStorage.getItem('auto-widget-handoff')
      if (raw) {
        sessionStorage.removeItem('auto-widget-handoff')
        const handoffMessages: ChatMessage[] = JSON.parse(raw)
        if (handoffMessages.length > 0) {
          setCurrentMessages(handoffMessages)
          setChatInstance((v) => v + 1)
        }
      }
    } catch { /* corrupt data — start fresh */ }
  }, [])

  // Toggle chat history from main Automatos sidebar (chat-only menu item)
  useEffect(() => {
    const handler = () => setIsHistoryOpen((v) => !v)
    window.addEventListener('automatos:chat-history-toggle', handler as any)
    return () => window.removeEventListener('automatos:chat-history-toggle', handler as any)
  }, [])

  // Auto-select chat when arriving via ?chatId=<id>
  const [chatLoadAttempted, setChatLoadAttempted] = useState<string | null>(null)
  useEffect(() => {
    if (!chatIdParam || chatIdParam === currentChatId || chatLoadAttempted === chatIdParam) {
      return
    }
    setChatLoadAttempted(chatIdParam)
    let cancelled = false
    ;(async () => {
      try {
        const { getChat, getChatMessages } = await import('@/lib/chat/api')
        const [chat, messages] = await Promise.all([
          getChat(chatIdParam),
          getChatMessages(chatIdParam),
        ])
        if (!cancelled) {
          handleChatSelect(chat, messages)
        }
      } catch (err) {
        console.error('Failed to load chat from chatId param:', err)
      }
    })()
    return () => {
      cancelled = true
    }
  }, [chatIdParam, currentChatId, chatLoadAttempted])

  const chatBody = (
    <Chat
      key={`${currentChatId || 'new'}-${chatInstance}`}
      id={currentChatId}
      initialMessages={currentMessages}
      initialChatModel="gpt-4"
      initialVisibilityType={selectedChat?.visibility || 'private'}
      isReadonly={false}
      autoResume={false}
      initialLastContext={selectedChat?.lastContext}
    />
  )

  // Studio desktop: CD's three-column ledger layout
  if (isStudio && !isMobile) {
    return (
      <MainLayout fullBleed>
        <StudioChatShell
          selectedChatId={currentChatId}
          selectedChat={selectedChat}
          onSelectChat={handleChatSelect}
          onNewChat={handleNewChat}
        >
          {chatBody}
        </StudioChatShell>
      </MainLayout>
    )
  }

  // Classic layout (mobile + non-studio desktop)
  return (
    <MainLayout>
      <div className="relative h-[calc(100dvh-5rem)] md:h-[calc(100vh-8rem)]">
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
              <AppSidebar onChatSelect={handleChatSelect} onNewChat={handleNewChat} />
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
                <div className="h-full rounded-r-3xl border-r border-orange-500/20 bg-background/35 backdrop-blur-xl shadow-[0_0_80px_rgba(0,0,0,0.55)]">
                  <AppSidebar onChatSelect={handleChatSelect} onNewChat={handleNewChat} />
                </div>
              </motion.aside>
            )}
          </AnimatePresence>
        )}

        <div className="h-full">{chatBody}</div>
      </div>
    </MainLayout>
  )
}
