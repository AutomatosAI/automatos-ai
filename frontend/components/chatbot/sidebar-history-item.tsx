'use client'

import { useState } from 'react'
import { MessageSquare, Trash2 } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { deleteChat } from '@/lib/chat/api'
import { truncate } from '@/lib/utils'
import type { Chat } from '@/types'
import { toast } from 'sonner'

export interface SidebarHistoryItemProps {
  chat: Chat
  onDelete: (chatId: string) => void
  onSelect?: (chat: Chat, messages: any[]) => void
  isActive?: boolean
}

export function SidebarHistoryItem({ chat, onDelete, onSelect, isActive = false }: SidebarHistoryItemProps) {
  const [isDeleting, setIsDeleting] = useState(false)

  const handleClick = async () => {
    if (onSelect) {
      try {
        // Import here to avoid circular dependency
        const { getChatMessages } = await import('@/lib/chat/api')
        const messages = await getChatMessages(chat.id)
        onSelect(chat, messages)
      } catch (error) {
        console.error('Failed to load chat messages:', error)
        toast.error('Failed to load chat')
      }
    }
  }

  const handleDelete = async (e: React.MouseEvent) => {
    e.stopPropagation()
    
    if (!confirm('Are you sure you want to delete this chat?')) {
      return
    }

    try {
      setIsDeleting(true)
      await deleteChat(chat.id)
      onDelete(chat.id)
      toast.success('Chat deleted')
      
    } catch (error) {
      toast.error('Failed to delete chat')
    } finally {
      setIsDeleting(false)
    }
  }

  return (
    <button
      onClick={handleClick}
      disabled={isDeleting}
      className={`
        group w-full flex items-center space-x-3 px-3 py-2 rounded-lg text-left transition-all
        ${isActive
          ? 'bg-primary/10 border border-primary/30 text-foreground'
          : 'text-muted-foreground hover:bg-primary/5 hover:border hover:border-primary/20 hover:text-foreground'
        }
        ${isDeleting ? 'opacity-50 cursor-not-allowed' : ''}
      `}
    >
      <MessageSquare className="w-4 h-4 flex-shrink-0" />
      <span className="flex-1 text-sm truncate">{truncate(chat.title, 30)}</span>
      {/* PRD-205 S7: subtle marker for the thread where Auto speaks unprompted */}
      {chat.kind === 'auto' && (
        <Badge variant="outline" className="shrink-0 px-1.5 py-0 text-[10px] bg-warning/10 border-warning/20 text-warning">
          Auto
        </Badge>
      )}

      <Button
        variant="ghost"
        size="sm"
        className="opacity-0 group-hover:opacity-100 h-6 w-6 p-0 text-muted-foreground hover:text-[hsl(var(--destructive))]"
        onClick={handleDelete}
        title="Delete"
      >
        <Trash2 className="w-3.5 h-3.5" />
      </Button>
    </button>
  )
}

