'use client'

/**
 * NotificationBell (PRD-128 US-008)
 * -----------------------------------
 * Bell icon + popover dropdown wired to /api/notifications.
 * - Polls unread-count every 30s
 * - Fetches the 20 most recent notifications when the popover opens
 * - Clicking a row marks it read and navigates via next/navigation router.push
 * - Dismiss (x), Mark-all-read, and a footer link to /settings/notifications
 */

import { useState } from 'react'
import { useRouter } from 'next/navigation'
import { Bell, Check, X, Settings as SettingsIcon } from 'lucide-react'
import { formatDistanceToNow } from 'date-fns'

import { Button } from '@/components/ui/button'
import {
  Popover,
  PopoverTrigger,
  PopoverContent,
} from '@/components/ui/popover'
import {
  NotificationRow,
  useNotifications,
  useUnreadNotificationCount,
  useMarkNotificationRead,
  useMarkAllNotificationsRead,
  useDismissNotification,
} from '@/hooks/use-notifications-api'

// ---------------------------------------------------------------------------
// Route mapping
// ---------------------------------------------------------------------------

/**
 * Derive the in-app route for a notification based on its ``link_type`` /
 * ``link_id``. Returns ``null`` when there is no meaningful destination, in
 * which case the row still marks itself read but does not navigate.
 */
function linkFor(row: NotificationRow): string | null {
  const { link_type, link_id } = row
  if (!link_type) return null

  switch (link_type) {
    case 'task':
      return '/command-center?tab=board'
    case 'mission':
      return link_id ? `/assignments?tab=missions&mission=${link_id}` : '/assignments?tab=missions'
    case 'playbook':
      return link_id ? `/assignments?tab=playbooks&execution=${link_id}` : '/assignments?tab=playbooks'
    case 'heartbeat':
      return '/command-center?tab=feed'
    case 'report':
      return link_id ? `/command-center?tab=feed&report=${link_id}` : '/command-center?tab=feed'
    case 'trigger':
      return '/command-center?tab=feed'
    case 'agent':
      return link_id ? `/agents?agent=${link_id}` : '/agents'
    default:
      return null
  }
}

// ---------------------------------------------------------------------------
// Status icon
// ---------------------------------------------------------------------------

function statusDotClass(status: string): string {
  switch (status) {
    case 'error':
      return 'bg-destructive'
    case 'warning':
      return 'bg-warning'
    case 'ok':
    default:
      return 'bg-orange-500'
  }
}

// ---------------------------------------------------------------------------
// Row
// ---------------------------------------------------------------------------

interface NotificationItemProps {
  row: NotificationRow
  onClick: (row: NotificationRow) => void
  onDismiss: (row: NotificationRow) => void
}

function NotificationItem({ row, onClick, onDismiss }: NotificationItemProps) {
  const unread = !row.read_at
  const relative = row.created_at
    ? formatDistanceToNow(new Date(row.created_at), { addSuffix: true })
    : ''

  return (
    <div
      role="button"
      tabIndex={0}
      onClick={() => onClick(row)}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault()
          onClick(row)
        }
      }}
      className={`group relative flex gap-3 px-4 py-3 border-b border-border/30 cursor-pointer transition-colors hover:bg-orange-500/5 ${
        unread ? 'bg-orange-500/[0.03]' : ''
      }`}
    >
      <span
        className={`mt-1.5 h-2 w-2 flex-shrink-0 rounded-full ${statusDotClass(row.status)} ${
          unread ? '' : 'opacity-40'
        }`}
        aria-hidden
      />
      <div className="flex-1 min-w-0">
        <div className="flex items-start justify-between gap-2">
          <p className="text-sm font-medium text-foreground truncate">
            {row.title}
          </p>
          <span className="text-[11px] text-muted-foreground whitespace-nowrap">
            {relative}
          </span>
        </div>
        {row.message && (
          <p className="text-xs text-muted-foreground line-clamp-2 mt-0.5">
            {row.message}
          </p>
        )}
        {row.agent_name && (
          <p className="text-[11px] text-orange-400/80 mt-1">
            {row.agent_name}
          </p>
        )}
      </div>
      <button
        type="button"
        aria-label="Dismiss notification"
        onClick={(e) => {
          e.stopPropagation()
          onDismiss(row)
        }}
        className="opacity-0 group-hover:opacity-100 transition-opacity h-6 w-6 rounded hover:bg-muted flex items-center justify-center flex-shrink-0"
      >
        <X className="w-3.5 h-3.5" />
      </button>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Bell
// ---------------------------------------------------------------------------

export function NotificationBell() {
  const router = useRouter()
  const [open, setOpen] = useState(false)

  const { data: unreadData } = useUnreadNotificationCount()
  const unreadCount = unreadData?.count ?? 0

  const { data: listData, isLoading } = useNotifications({
    limit: 20,
    enabled: open,
  })
  const notifications = listData?.notifications ?? []

  const markRead = useMarkNotificationRead()
  const markAllRead = useMarkAllNotificationsRead()
  const dismiss = useDismissNotification()

  const handleClick = (row: NotificationRow) => {
    if (!row.read_at) {
      markRead.mutate(row.id)
    }
    const route = linkFor(row)
    if (route) {
      setOpen(false)
      router.push(route)
    }
  }

  const handleDismiss = (row: NotificationRow) => {
    dismiss.mutate(row.id)
  }

  const handleMarkAll = () => {
    markAllRead.mutate()
  }

  const handleOpenSettings = () => {
    setOpen(false)
    router.push('/settings/notifications')
  }

  const badgeLabel = unreadCount > 9 ? '9+' : String(unreadCount)

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button
          variant="ghost"
          size="icon"
          aria-label={`Notifications${unreadCount > 0 ? ` (${unreadCount} unread)` : ''}`}
          className="relative text-orange-400 hover:text-orange-300 hover:bg-orange-500/5"
        >
          <Bell className="w-5 h-5" />
          {unreadCount > 0 && (
            <span className="absolute -top-1 -right-1 min-w-[18px] h-[18px] px-1 bg-orange-500 rounded-full text-[10px] font-semibold flex items-center justify-center text-black">
              {badgeLabel}
            </span>
          )}
        </Button>
      </PopoverTrigger>
      <PopoverContent
        align="end"
        sideOffset={8}
        className="w-96 p-0 overflow-hidden"
      >
        <div className="flex items-center justify-between px-4 py-3 border-b border-border/40">
          <div className="flex items-center gap-2">
            <Bell className="w-4 h-4 text-orange-400" />
            <h3 className="text-sm font-semibold">Notifications</h3>
            {unreadCount > 0 && (
              <span className="text-[11px] text-muted-foreground">
                {unreadCount} unread
              </span>
            )}
          </div>
          {unreadCount > 0 && (
            <button
              type="button"
              onClick={handleMarkAll}
              disabled={markAllRead.isLoading}
              className="inline-flex items-center gap-1 text-[11px] text-orange-400 hover:text-orange-300 disabled:opacity-50"
            >
              <Check className="w-3 h-3" />
              Mark all read
            </button>
          )}
        </div>

        <div className="max-h-96 overflow-y-auto">
          {isLoading ? (
            <div className="px-4 py-10 text-center text-xs text-muted-foreground">
              Loading…
            </div>
          ) : notifications.length === 0 ? (
            <div className="px-4 py-10 text-center">
              <Bell className="w-8 h-8 mx-auto text-muted-foreground/40 mb-2" />
              <p className="text-sm text-muted-foreground">No notifications</p>
              <p className="text-[11px] text-muted-foreground/70 mt-1">
                You&apos;re all caught up.
              </p>
            </div>
          ) : (
            notifications.map((row) => (
              <NotificationItem
                key={row.id}
                row={row}
                onClick={handleClick}
                onDismiss={handleDismiss}
              />
            ))
          )}
        </div>

        <button
          type="button"
          onClick={handleOpenSettings}
          className="w-full flex items-center justify-center gap-2 px-4 py-2.5 border-t border-border/40 text-xs text-muted-foreground hover:text-foreground hover:bg-muted/30 transition-colors"
        >
          <SettingsIcon className="w-3 h-3" />
          Notification settings
        </button>
      </PopoverContent>
    </Popover>
  )
}
