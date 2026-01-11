'use client'

import { useState } from 'react'
import { MessageSquare, Trash2, MoreHorizontal } from 'lucide-react'
import { Button } from '@/components/ui/button'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
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
          ? 'bg-primary/10 border border-primary/30 text-foreground dark:text-white' 
          : 'text-muted-foreground hover:bg-primary/5 hover:border hover:border-primary/20 hover:text-foreground dark:text-gray-400 dark:hover:text-white'
        }
        ${isDeleting ? 'opacity-50 cursor-not-allowed' : ''}
      `}
    >
      <MessageSquare className="w-4 h-4 flex-shrink-0" />
      <span className="flex-1 text-sm truncate">{truncate(chat.title, 30)}</span>
      
      <DropdownMenu>
        <DropdownMenuTrigger asChild onClick={(e) => e.stopPropagation()}>
          <Button
            variant="ghost"
            size="sm"
            className="opacity-0 group-hover:opacity-100 h-6 w-6 p-0"
          >
            <MoreHorizontal className="w-4 h-4" />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end" className="bg-popover border-border">
          <DropdownMenuItem
            onClick={handleDelete}
            className="text-red-500 hover:text-red-600 hover:bg-red-500/10 cursor-pointer"
          >
            <Trash2 className="w-4 h-4 mr-2" />
            Delete
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
    </button>
  )
}

