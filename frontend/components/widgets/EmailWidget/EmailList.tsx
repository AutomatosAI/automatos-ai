'use client'

/**
 * EmailList Component for PRD-38.2 Extended Widgets
 *
 * List view for emails with search, filter, and selection
 */

import { useState, useMemo } from 'react'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { ScrollArea } from '@/components/ui/scroll-area'
import {
  Search,
  Paperclip,
  Circle,
  Filter,
  RefreshCw,
} from 'lucide-react'
import { cn } from '@/lib/utils'
import { formatDistanceToNow } from 'date-fns'
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

  // Format date for display
  const formatDate = (dateStr: string) => {
    try {
      const date = new Date(dateStr)
      const now = new Date()
      const diffDays = Math.floor(
        (now.getTime() - date.getTime()) / (1000 * 60 * 60 * 24)
      )

      if (diffDays === 0) {
        return date.toLocaleTimeString([], {
          hour: 'numeric',
          minute: '2-digit',
        })
      } else if (diffDays < 7) {
        return formatDistanceToNow(date, { addSuffix: false })
      } else {
        return date.toLocaleDateString([], {
          month: 'short',
          day: 'numeric',
        })
      }
    } catch {
      return dateStr
    }
  }

  // Get sender display name
  const getSenderName = (email: EmailSummary) => {
    return email.from.name || email.from.email.split('@')[0]
  }

  return (
    <div className="flex flex-col h-full">
      {/* Search and filter bar */}
      <div className="flex items-center gap-2 p-2 border-b border-border/30">
        <div className="relative flex-1">
          <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            type="text"
            placeholder="Search emails..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pl-9 h-8 text-sm"
          />
        </div>

        <Button
          variant={filterUnread ? 'secondary' : 'ghost'}
          size="sm"
          className="h-8"
          onClick={() => setFilterUnread(!filterUnread)}
          title="Filter unread"
        >
          <Filter className="h-4 w-4" />
        </Button>

        {onRefresh && (
          <Button
            variant="ghost"
            size="sm"
            className="h-8"
            onClick={onRefresh}
            disabled={isLoading}
            title="Refresh"
          >
            <RefreshCw
              className={cn('h-4 w-4', isLoading && 'animate-spin')}
            />
          </Button>
        )}
      </div>

      {/* Email list */}
      <ScrollArea className="flex-1">
        {filteredEmails.length === 0 ? (
          <div className="flex items-center justify-center h-32 text-sm text-muted-foreground">
            {searchQuery || filterUnread
              ? 'No emails match your filters'
              : 'No emails'}
          </div>
        ) : (
          <div className="divide-y divide-border/30">
            {filteredEmails.map((email) => (
              <button
                key={email.id}
                className={cn(
                  'w-full text-left px-3 py-2.5 hover:bg-muted/50 transition-colors',
                  'focus:outline-none focus:bg-muted/50',
                  selectedId === email.id && 'bg-primary/5',
                  !email.isRead && 'bg-primary/5'
                )}
                onClick={() => onSelect(email)}
              >
                <div className="flex items-start gap-2">
                  {/* Unread indicator */}
                  <div className="mt-1.5 flex-shrink-0">
                    {!email.isRead && (
                      <Circle className="h-2 w-2 fill-primary text-primary" />
                    )}
                    {email.isRead && <div className="w-2" />}
                  </div>

                  {/* Email content */}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between gap-2">
                      <span
                        className={cn(
                          'text-sm truncate',
                          !email.isRead && 'font-semibold'
                        )}
                      >
                        {getSenderName(email)}
                      </span>
                      <span className="text-xs text-muted-foreground flex-shrink-0">
                        {formatDate(email.date)}
                      </span>
                    </div>

                    <div
                      className={cn(
                        'text-sm truncate',
                        email.isRead
                          ? 'text-muted-foreground'
                          : 'text-foreground'
                      )}
                    >
                      {email.subject}
                    </div>

                    <div className="flex items-center gap-2 mt-0.5">
                      <span className="text-xs text-muted-foreground truncate flex-1">
                        {email.snippet}
                      </span>
                      {email.hasAttachments && (
                        <Paperclip className="h-3 w-3 text-muted-foreground flex-shrink-0" />
                      )}
                    </div>
                  </div>
                </div>
              </button>
            ))}
          </div>
        )}
      </ScrollArea>

      {/* Footer with count */}
      <div className="px-3 py-1.5 border-t border-border/30 text-xs text-muted-foreground">
        {filteredEmails.length} of {emails.length} emails
        {filterUnread && ' (unread filter)'}
      </div>
    </div>
  )
}
