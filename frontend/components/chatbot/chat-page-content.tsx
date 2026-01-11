'use client'

import { useState } from 'react'
import { MainLayout } from '@/components/layout/main-layout'
import { Chat } from '@/components/chatbot/chat'
import { AppSidebar } from '@/components/chatbot/sidebar'
import { usePageAPI } from '@/hooks/use-page-api'
import type { Chat as ChatType, ChatMessage } from '@/types'

export function ChatPageContent() {
    usePageAPI('chat')

    // State for current chat
    const [currentChatId, setCurrentChatId] = useState(() => crypto.randomUUID())
    const [currentMessages, setCurrentMessages] = useState<ChatMessage[]>([])
    const [selectedChat, setSelectedChat] = useState<ChatType | null>(null)
    const [isSidebarVisible, setIsSidebarVisible] = useState(false)

    // Handle selecting a chat from sidebar
    const handleChatSelect = (chat: ChatType, messages: ChatMessage[]) => {
        setCurrentChatId(chat.id)
        setCurrentMessages(messages)
        setSelectedChat(chat)
    }

    // Handle new chat from sidebar
    const handleNewChat = () => {
        // Generate a new ID immediately so useChat has a stable key
        setCurrentChatId(crypto.randomUUID())
        setCurrentMessages([])
        setSelectedChat(null)
    }

    return (
        <MainLayout>
            <div className="flex h-[calc(100vh-8rem)] relative">
                {/* Sidebar - with slide animation */}
                <div className={`transition-all duration-300 ${isSidebarVisible ? 'w-64' : 'w-0 overflow-hidden'}`}>
                    {isSidebarVisible && <AppSidebar onChatSelect={handleChatSelect} onNewChat={handleNewChat} />}
                </div>

                {/* Main chat area */}
                <div className="flex-1 bg-transparent relative">
                    {/* Toggle button */}
                    <button
                        onClick={() => setIsSidebarVisible(!isSidebarVisible)}
                        className="absolute top-4 left-4 z-10 p-2 rounded-xl bg-secondary/50 hover:bg-secondary border border-border/60 transition-all text-foreground"
                        title={isSidebarVisible ? 'Hide sidebar' : 'Show sidebar'}
                    >
                        <svg
                            xmlns="http://www.w3.org/2000/svg"
                            width="20"
                            height="20"
                            viewBox="0 0 24 24"
                            fill="none"
                            stroke="currentColor"
                            strokeWidth="2"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            className={`transition-transform ${isSidebarVisible ? '' : 'rotate-180'}`}
                        >
                            <rect width="18" height="18" x="3" y="3" rx="2" ry="2" />
                            <line x1="9" x2="9" y1="3" y2="21" />
                        </svg>
                    </button>

                    <Chat
                        key={currentChatId}
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
