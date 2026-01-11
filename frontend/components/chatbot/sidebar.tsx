'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Plus, Search, Settings } from 'lucide-react'
import { useRouter } from 'next/navigation'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Separator } from '@/components/ui/separator'
import { SidebarHistoryItem } from './sidebar-history-item'
import { getChatHistory } from '@/lib/chat/api'
import type { Chat } from '@/types'

export interface AppSidebarProps {
  user?: {
    id: string
    name?: string
    email?: string
  }
  onChatSelect?: (chat: Chat, messages: any[]) => void
  onNewChat?: () => void
}

export function AppSidebar({ user, onChatSelect, onNewChat }: AppSidebarProps) {
  const router = useRouter()
  const [chats, setChats] = useState<Chat[]>([])
  const [searchQuery, setSearchQuery] = useState('')
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    loadChatHistory()
  }, [])

  const loadChatHistory = async () => {
    try {
      setIsLoading(true)
      const history = await getChatHistory(20)
      setChats(history)
    } catch (error) {
      console.error('Failed to load chat history:', error)
    } finally {
      setIsLoading(false)
    }
  }

  const handleNewChat = () => {
    if (onNewChat) {
      onNewChat()
    }
    // refresh list so the most recent chat appears after first message
    loadChatHistory().catch(() => {})
  }

  const handleChatDelete = (chatId: string) => {
    setChats(prev => prev.filter(c => c.id !== chatId))
  }

  const filteredChats = searchQuery
    ? chats.filter(chat => 
        chat.title.toLowerCase().includes(searchQuery.toLowerCase())
      )
    : chats

  // Group chats by date (date-only comparisons for mutual exclusivity)
  const today = new Date()
  const todayDateString = today.toDateString()
  
  const yesterday = new Date(today)
  yesterday.setDate(yesterday.getDate() - 1)
  const yesterdayDateString = yesterday.toDateString()
  
  // Normalize to start of yesterday (midnight) for olderChats comparison
  const startOfYesterday = new Date(yesterday)
  startOfYesterday.setHours(0, 0, 0, 0)

  const todayChats = filteredChats.filter(c => {
    const chatDate = new Date(c.createdAt)
    return chatDate.toDateString() === todayDateString
  })

  const yesterdayChats = filteredChats.filter(c => {
    const chatDate = new Date(c.createdAt)
    return chatDate.toDateString() === yesterdayDateString
  })

  const olderChats = filteredChats.filter(c => {
    const chatDate = new Date(c.createdAt)
    const chatDateString = chatDate.toDateString()
    // Exclude chats from today or yesterday, only include older ones
    return chatDateString !== todayDateString && 
           chatDateString !== yesterdayDateString &&
           chatDate < startOfYesterday
  })

  return (
    <div className="flex flex-col h-full w-full bg-transparent">
      {/* Header */}
      <div className="p-4 space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold text-foreground/80 dark:text-white/90 uppercase tracking-wider">Chats</h2>
        </div>

        <Button
          onClick={handleNewChat}
          className="w-full rounded-2xl bg-gradient-to-br from-orange-500 to-red-500 hover:from-orange-600 hover:to-red-600 shadow-[0_0_18px_rgba(249,115,22,0.25)]"
        >
          <Plus className="w-4 h-4 mr-2" />
          New Chat
        </Button>

        {/* Search */}
        <div className="relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground dark:text-gray-400" />
          <Input
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search chats..."
            className="pl-10 bg-secondary/40 border-border/60 focus-visible:border-primary/40 focus-visible:ring-primary/20 text-foreground placeholder:text-muted-foreground rounded-2xl dark:bg-gray-950/30 dark:border-orange-500/15 dark:focus-visible:ring-orange-500/20 dark:text-white dark:placeholder-gray-400"
          />
        </div>
      </div>

      <Separator className="bg-border/50 dark:bg-gray-800/40" />

      {/* Chat History */}
      <ScrollArea className="flex-1 px-2">
        <div className="space-y-4 py-4">
          {isLoading ? (
            <div className="flex items-center justify-center py-8">
              <div className="w-6 h-6 border-2 border-orange-500 border-t-transparent rounded-full animate-spin" />
            </div>
          ) : filteredChats.length === 0 ? (
            <div className="text-center py-8 text-muted-foreground text-sm">
              {searchQuery ? 'No chats found' : 'No chat history'}
            </div>
          ) : (
            <>
              {todayChats.length > 0 && (
                <div>
                  <div className="px-3 py-2 text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                    Today
                  </div>
                  <div className="space-y-1">
                    {todayChats.map((chat) => (
                      <SidebarHistoryItem
                        key={chat.id}
                        chat={chat}
                        onDelete={handleChatDelete}
                        onSelect={onChatSelect}
                      />
                    ))}
                  </div>
                </div>
              )}

              {yesterdayChats.length > 0 && (
                <div>
                  <div className="px-3 py-2 text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                    Yesterday
                  </div>
                  <div className="space-y-1">
                    {yesterdayChats.map((chat) => (
                      <SidebarHistoryItem
                        key={chat.id}
                        chat={chat}
                        onDelete={handleChatDelete}
                        onSelect={onChatSelect}
                      />
                    ))}
                  </div>
                </div>
              )}

              {olderChats.length > 0 && (
                <div>
                  <div className="px-3 py-2 text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                    Older
                  </div>
                  <div className="space-y-1">
                    {olderChats.map((chat) => (
                      <SidebarHistoryItem
                        key={chat.id}
                        chat={chat}
                        onDelete={handleChatDelete}
                        onSelect={onChatSelect}
                      />
                    ))}
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      </ScrollArea>

      <Separator className="bg-border/50 dark:bg-gray-800/40" />

      {/* User Section */}
      {user && (
        <div className="p-4">
          <div className="flex items-center space-x-3 p-2 rounded-lg bg-secondary/30 border border-border/40 dark:bg-gray-900/40 dark:border-gray-800/30">
            <div className="w-8 h-8 rounded-full bg-gradient-to-br from-orange-500 to-red-500 flex items-center justify-center">
              <span className="text-sm font-semibold text-white">
                {user.name?.[0] || user.email?.[0] || 'U'}
              </span>
            </div>
            <div className="flex-1 min-w-0">
              <div className="text-sm font-medium text-foreground dark:text-white truncate">
                {user.name || user.email || 'User'}
              </div>
            </div>
            <Button variant="ghost" size="sm" className="text-muted-foreground hover:text-foreground dark:text-gray-400 dark:hover:text-white p-1">
              <Settings className="w-4 h-4" />
            </Button>
          </div>
        </div>
      )}
    </div>
  )
}

