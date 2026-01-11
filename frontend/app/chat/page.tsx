'use client'

import { useEffect, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { MainLayout } from '@/components/layout/main-layout'
import { Chat } from '@/components/chatbot/chat'
import { AppSidebar } from '@/components/chatbot/sidebar'
import { usePageAPI } from '@/hooks/use-page-api'
import type { Chat as ChatType, ChatMessage } from '@/types'

// Force dynamic rendering
export const dynamic = 'force-dynamic'

export default function ChatPage() {
  usePageAPI('chat')
  
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

  // Toggle chat history from main Automatos sidebar (chat-only menu item)
  useEffect(() => {
    const handler = () => setIsHistoryOpen((v) => !v)
    window.addEventListener('automatos:chat-history-toggle', handler as any)
    return () => window.removeEventListener('automatos:chat-history-toggle', handler as any)
  }, [])

  return (
    <MainLayout>
      <div className="relative h-[calc(100vh-8rem)]">
        {/* Docked chat history panel (slides from left, blends with Automatos) */}
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

        {/* Main chat area */}
        <div className="h-full">
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
        </div>
      </div>
    </MainLayout>
  )
}
