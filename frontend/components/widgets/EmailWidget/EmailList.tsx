'use client'

/**
 * EmailList Component for PRD-38.2 Extended Widgets
 *
 * Modern list view for emails with search, filter, and selection
 * Inspired by clean, minimal email client designs
 */

import { useState, useMemo } from 'react'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Badge } from '@/components/ui/badge'
import {
  Search,
  Paperclip,
  Circle,
  RefreshCw,
  Inbox,
} from 'lucide-react'
import { cn } from '@/lib/utils'
import { isToday, isYesterday, format } from 'date-fns'
import type { EmailSummary } from '../types'

interface EmailListProps {
  emails: EmailSummary[]
  selectedId?: string
  onSelect: (email: EmailSummary) => void
  onRefresh?: () => void
  isLoading?: boolean
}

export function EmailList({
  emails,
  selectedId,
  onSelect,
  onRefresh,
  isLoading,
}: EmailListProps) {
  const [searchQuery, setSearchQuery] = useState('')
  const [filterUnread, setFilterUnread] = useState(false)

  // Filter emails based on search and filters
  const filteredEmails = useMemo(() => {
    return emails.filter((email) => {
      // Filter by unread status
      if (filterUnread && email.isRead) return false

      // Filter by search query
      if (searchQuery) {
        const query = searchQuery.toLowerCase()
        return (
          email.subject.toLowerCase().includes(query) ||
          email.from.email.toLowerCase().includes(query) ||
          email.from.name?.toLowerCase().includes(query) ||
          email.snippet.toLowerCase().includes(query)
        )
      }

      return true
    })
  }, [emails, searchQuery, filterUnread])

  // Stats
  const unreadCount = useMemo(() =>
    emails.filter(e => !e.isRead).length,
    [emails]
  )

  // Format date for display - relative for recent, smart for older
  const formatDate = (dateStr: string) => {
    try {
      const date = new Date(dateStr)
      const now = new Date()

      if (isToday(date)) {
        // Show relative time for today: "2h ago", "5m ago"
        const diffMs = now.getTime() - date.getTime()
        const diffMin = Math.floor(diffMs / 60_000)
        if (diffMin < 1) return 'Just now'
        if (diffMin < 60) return `${diffMin}m ago`
        const diffHr = Math.floor(diffMin / 60)
        return `${diffHr}h ago`
      } else if (isYesterday(date)) {
        return 'Yesterday'
      } else {
        const diffDays = Math.floor(
          (now.getTime() - date.getTime()) / (1000 * 60 * 60 * 24)
        )

        if (diffDays < 7) {
          return format(date, 'EEE') // Mon, Tue, etc.
        } else if (date.getFullYear() === now.getFullYear()) {
          return format(date, 'MMM d')
        } else {
          return format(date, 'MM/dd/yy')
        }
      }
    } catch {
      return dateStr
    }
  }

  // Get sender display name
  const getSenderName = (email: EmailSummary) => {
    return email.from.name || email.from.email.split('@')[0]
  }

  // Get sender initials for avatar
  const getSenderInitials = (email: EmailSummary) => {
    const name = email.from.name || email.from.email
    const parts = name.split(/[\s@]/)
    if (parts.length >= 2) {
      return (parts[0][0] + parts[1][0]).toUpperCase()
    }
    return name.substring(0, 2).toUpperCase()
  }

  // Generate consistent color for sender
  const getAvatarColor = (email: string) => {
    const colors = [
      'bg-blue-500', 'bg-green-500', 'bg-purple-500',
      'bg-orange-500', 'bg-pink-500', 'bg-teal-500',
      'bg-indigo-500', 'bg-rose-500', 'bg-cyan-500'
    ]
    let hash = 0
    for (let i = 0; i < email.length; i++) {
      hash = email.charCodeAt(i) + ((hash << 5) - hash)
    }
    return colors[Math.abs(hash) % colors.length]
  }

  return (
    <div className="flex flex-col h-full">
      {/* Header with search and filters */}
      <div className="flex flex-col gap-2 p-3 border-b border-border/40 bg-muted/30">
        {/* Stats row */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 text-sm">
            <Inbox className="h-4 w-4 text-muted-foreground" />
            <span className="font-medium">{emails.length} emails</span>
            {unreadCount > 0 && (
              <Badge variant="secondary" className="h-5 px-1.5 text-xs bg-primary/10 text-primary">
                {unreadCount} new
              </Badge>
            )}
          </div>
          <div className="flex items-center gap-1">
            <Button
              variant={filterUnread ? 'secondary' : 'ghost'}
              size="sm"
              className="h-7 px-2 text-xs"
              onClick={() => setFilterUnread(!filterUnread)}
            >
              <Circle className={cn(
                "h-2 w-2 mr-1.5",
                filterUnread ? "fill-primary text-primary" : "text-muted-foreground"
              )} />
              Unread
            </Button>
            {onRefresh && (
              <Button
                variant="ghost"
                size="icon"
                className="h-7 w-7"
                onClick={onRefresh}
                disabled={isLoading}
              >
                <RefreshCw className={cn('h-3.5 w-3.5', isLoading && 'animate-spin')} />
              </Button>
            )}
          </div>
        </div>

        {/* Search */}
        <div className="relative">
          <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground" />
          <Input
            type="text"
            placeholder="Search emails..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pl-8 h-8 text-sm bg-background/50"
          />
        </div>
      </div>

      {/* Email list */}
      <ScrollArea className="flex-1">
        {filteredEmails.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-40 text-center px-4">
            <Inbox className="h-10 w-10 text-muted-foreground/40 mb-3" />
            <p className="text-sm text-muted-foreground">
              {searchQuery || filterUnread
                ? 'No emails match your filters'
                : 'No emails found'}
            </p>
          </div>
        ) : (
          <div className="divide-y divide-border/30">
            {filteredEmails.map((email) => (
              <button
                key={email.id}
                className={cn(
                  'w-full text-left px-3 py-3 transition-all duration-150',
                  'hover:bg-muted/60 focus:outline-none focus:bg-muted/60',
                  'group relative',
                  selectedId === email.id && 'bg-primary/8 hover:bg-primary/10',
                  !email.isRead && 'bg-primary/5'
                )}
                onClick={() => onSelect(email)}
              >
                {/* Selection indicator */}
                {selectedId === email.id && (
                  <div className="absolute left-0 top-0 bottom-0 w-0.5 bg-primary rounded-r" />
                )}

                <div className="flex items-start gap-3">
                  {/* Avatar */}
                  <div className={cn(
                    "flex-shrink-0 w-9 h-9 rounded-full flex items-center justify-center text-white text-xs font-medium",
                    getAvatarColor(email.from.email)
                  )}>
                    {getSenderInitials(email)}
                  </div>

                  {/* Content */}
                  <div className="flex-1 min-w-0 space-y-0.5">
                    {/* Top row: sender + date */}
                    <div className="flex items-center justify-between gap-2">
                      <span className={cn(
                        'text-sm truncate',
                        !email.isRead && 'font-semibold text-foreground'
                      )}>
                        {getSenderName(email)}
                      </span>
                      <span className={cn(
                        'text-xs flex-shrink-0',
                        !email.isRead ? 'text-primary font-medium' : 'text-muted-foreground'
                      )}>
                        {formatDate(email.date)}
                      </span>
                    </div>

                    {/* Subject */}
                    <div className={cn(
                      'text-sm truncate',
                      email.isRead ? 'text-muted-foreground' : 'text-foreground font-medium'
                    )}>
                      {email.subject || '(No Subject)'}
                    </div>

                    {/* Preview + indicators */}
                    <div className="flex items-center gap-2">
                      <span className="text-xs text-muted-foreground truncate flex-1 line-clamp-1">
                        {email.snippet}
                      </span>
                      <div className="flex items-center gap-1 flex-shrink-0">
                        {email.hasAttachments && (
                          <Paperclip className="h-3 w-3 text-muted-foreground" />
                        )}
                        {!email.isRead && (
                          <Circle className="h-2 w-2 fill-primary text-primary" />
                        )}
                      </div>
                    </div>

                    {/* Labels */}
                    {email.labels && email.labels.length > 0 && (
                      <div className="flex flex-wrap gap-1 pt-1">
                        {email.labels.slice(0, 3).map((label) => (
                          <Badge
                            key={label}
                            variant="outline"
                            className="h-4 px-1.5 text-[10px] bg-muted/50"
                          >
                            {label}
                          </Badge>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              </button>
            ))}
          </div>
        )}
      </ScrollArea>
    </div>
  )
}
