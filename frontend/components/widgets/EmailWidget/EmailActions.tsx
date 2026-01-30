'use client'

/**
 * EmailActions Component for PRD-38.2 Extended Widgets
 *
 * Action buttons for email operations: reply, forward, archive, delete
 */

import { Button } from '@/components/ui/button'
import {
  Reply,
  Forward,
  Archive,
  Trash2,
  MailOpen,
  Mail,
  Star,
} from 'lucide-react'
import { cn } from '@/lib/utils'
import type { EmailFull } from '../types'

interface EmailActionsProps {
  email: EmailFull
  onReply?: (email: EmailFull) => void
  onForward?: (email: EmailFull) => void
  onArchive?: (emailId: string) => void
  onDelete?: (emailId: string) => void
  onToggleRead?: (emailId: string, isRead: boolean) => void
  onToggleStar?: (emailId: string) => void
  className?: string
  compact?: boolean
}

export function EmailActions({
  email,
  onReply,
  onForward,
  onArchive,
  onDelete,
  onToggleRead,
  onToggleStar,
  className,
  compact = false,
}: EmailActionsProps) {
  return (
    <div className={cn('flex items-center gap-1', className)}>
      {/* Reply */}
      {onReply && (
        <Button
          variant="ghost"
          size={compact ? 'icon' : 'sm'}
          className={compact ? 'h-7 w-7' : 'h-8'}
          onClick={() => onReply(email)}
          title="Reply"
        >
          <Reply className="h-4 w-4" />
          {!compact && <span className="ml-1">Reply</span>}
        </Button>
      )}

      {/* Forward */}
      {onForward && (
        <Button
          variant="ghost"
          size={compact ? 'icon' : 'sm'}
          className={compact ? 'h-7 w-7' : 'h-8'}
          onClick={() => onForward(email)}
          title="Forward"
        >
          <Forward className="h-4 w-4" />
          {!compact && <span className="ml-1">Forward</span>}
        </Button>
      )}

      {/* Toggle read/unread */}
      {onToggleRead && (
        <Button
          variant="ghost"
          size="icon"
          className="h-7 w-7"
          onClick={() => onToggleRead(email.id, !email.isRead)}
          title={email.isRead ? 'Mark as unread' : 'Mark as read'}
        >
          {email.isRead ? (
            <Mail className="h-4 w-4" />
          ) : (
            <MailOpen className="h-4 w-4" />
          )}
        </Button>
      )}

      {/* Star */}
      {onToggleStar && (
        <Button
          variant="ghost"
          size="icon"
          className="h-7 w-7"
          onClick={() => onToggleStar(email.id)}
          title="Star"
        >
          <Star className="h-4 w-4" />
        </Button>
      )}

      {/* Archive */}
      {onArchive && (
        <Button
          variant="ghost"
          size="icon"
          className="h-7 w-7"
          onClick={() => onArchive(email.id)}
          title="Archive"
        >
          <Archive className="h-4 w-4" />
        </Button>
      )}

      {/* Delete */}
      {onDelete && (
        <Button
          variant="ghost"
          size="icon"
          className="h-7 w-7 hover:text-destructive"
          onClick={() => onDelete(email.id)}
          title="Delete"
        >
          <Trash2 className="h-4 w-4" />
        </Button>
      )}
    </div>
  )
}
