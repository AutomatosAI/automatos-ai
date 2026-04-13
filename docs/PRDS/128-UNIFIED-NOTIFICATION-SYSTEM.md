# PRD-128: Unified Notification System

**Version:** 1.0
**Status:** Draft
**Priority:** P1
**Author:** Gar Kavanagh + Claude
**Created:** 2026-04-10
**Updated:** 2026-04-10
**Dependencies:** PRD-55 (Heartbeats — COMPLETE), PRD-76 (Agent Reports — COMPLETE), PRD-82A (Missions — COMPLETE)

---

## Executive Summary

Automatos has six input channels (chat, tasks, playbooks, missions, heartbeats, triggers) but no unified way to notify users when work completes. Heartbeats have channel routing to Telegram/Slack, but tasks, missions, and playbooks silently complete with no notification. The bell icon in the navbar is a static image — there's no in-app notification center.

Users finish giving instructions and then ask "where is my output?" — there's no link, no notification, no breadcrumb trail.

This PRD introduces a **Unified Notification System** that:
1. Captures completion events from ALL sources (heartbeats, tasks, missions, playbooks, triggers)
2. Routes notifications based on user preferences (in-app, Telegram, Slack, webhook)
3. Provides an in-app notification center with direct links to outputs
4. Reuses existing `channel_connections` and `notification_service.py` infrastructure

### What We're Building

1. **`notification_preferences` table** — per-workspace event → destination routing
2. **`notifications` table** — in-app notification store with read/unread state
3. **`NotificationDispatcher` service** — unified entry point for all event sources
4. **Bell dropdown component** — real notification center replacing static icon
5. **Notification settings page** — user-configurable routing per event type
6. **Event hooks** — wire tasks, missions, playbooks to call dispatcher on completion

### What We're NOT Building

- Email notifications (adds complexity, users prefer Slack/Telegram)
- Push notifications (mobile app scope)
- Digest/batching (v1 sends immediately, batching is v2)
- Rich notification actions (v1 is read/dismiss only)

### Reuse Strategy

| Component | Reuse | Notes |
|-----------|-------|-------|
| `notification_service.py` | **100%** | `send_workspace_notification()` already sends to Telegram/Slack/webhook |
| `channel_connections` table | **100%** | Already stores bot tokens, chat IDs, webhook URLs |
| `HeartbeatService._deliver_notification()` | **Migrate** | Move logic to `NotificationDispatcher`, call from heartbeat |
| `workspace.settings.integrations` | **100%** | Fallback for credentials if no `channel_connections` row |

---

## 1. Database Schema

### 1.1 `notification_preferences` Table

User-configurable routing rules: which events go where.

```sql
CREATE TABLE notification_preferences (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id    UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    user_id         INTEGER REFERENCES users(id) ON DELETE CASCADE,  -- NULL = workspace default
    
    -- Event type
    event_type      VARCHAR(50) NOT NULL,
        -- heartbeat_complete
        -- task_complete
        -- mission_step_complete
        -- mission_complete
        -- playbook_step_complete
        -- playbook_complete
        -- trigger_fired
        -- report_submitted
        -- agent_error
    
    -- Destination(s)
    destination     VARCHAR(30) NOT NULL DEFAULT 'in_app',
        -- in_app: write to notifications table (bell icon)
        -- telegram: send via Telegram bot
        -- slack: send via Slack bot
        -- webhook: POST to configured URL
        -- channel: route to a specific channel_connection
        -- silent: don't notify (event logged but no notification)
    
    channel_connection_id UUID REFERENCES channel_connections(id) ON DELETE SET NULL,
        -- For destination='channel', specifies which connection
    
    enabled         BOOLEAN NOT NULL DEFAULT true,
    
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    UNIQUE (workspace_id, user_id, event_type)
);

CREATE INDEX ix_notif_prefs_workspace ON notification_preferences(workspace_id);
CREATE INDEX ix_notif_prefs_user ON notification_preferences(user_id) WHERE user_id IS NOT NULL;
```

### 1.2 `notifications` Table

In-app notification store for the bell dropdown.

```sql
CREATE TABLE notifications (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id    UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    user_id         INTEGER REFERENCES users(id) ON DELETE CASCADE,  -- NULL = all workspace members
    
    -- Event info
    event_type      VARCHAR(50) NOT NULL,
    title           VARCHAR(255) NOT NULL,
    message         TEXT,
    
    -- Link to output (polymorphic)
    link_type       VARCHAR(30),
        -- report: links to agent_reports
        -- task: links to board_tasks
        -- mission: links to missions
        -- playbook: links to playbook_runs
        -- file: links to workspace file
        -- heartbeat: links to heartbeat_results
    link_id         VARCHAR(255),   -- UUID or path depending on link_type
    
    -- Agent attribution
    agent_id        INTEGER REFERENCES agents(id) ON DELETE SET NULL,
    agent_name      VARCHAR(100),
    
    -- Status
    status          VARCHAR(20) NOT NULL DEFAULT 'ok',
        -- ok, warning, critical, info
    
    -- Read state
    read_at         TIMESTAMPTZ,
    dismissed_at    TIMESTAMPTZ,
    
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX ix_notifications_workspace ON notifications(workspace_id);
CREATE INDEX ix_notifications_user ON notifications(user_id) WHERE user_id IS NOT NULL;
CREATE INDEX ix_notifications_unread ON notifications(workspace_id, user_id, created_at DESC) 
    WHERE read_at IS NULL AND dismissed_at IS NULL;
CREATE INDEX ix_notifications_recent ON notifications(workspace_id, created_at DESC);
```

### 1.3 Default Preferences (Seed Data)

When a workspace is created, seed these defaults:

```sql
INSERT INTO notification_preferences (workspace_id, event_type, destination, enabled) VALUES
    (:ws_id, 'heartbeat_complete', 'in_app', true),
    (:ws_id, 'task_complete', 'in_app', true),
    (:ws_id, 'mission_step_complete', 'silent', true),  -- Too noisy to show every step
    (:ws_id, 'mission_complete', 'in_app', true),
    (:ws_id, 'playbook_step_complete', 'silent', true),
    (:ws_id, 'playbook_complete', 'in_app', true),
    (:ws_id, 'trigger_fired', 'in_app', true),
    (:ws_id, 'report_submitted', 'in_app', true),
    (:ws_id, 'agent_error', 'in_app', true);
```

---

## 2. NotificationDispatcher Service

### 2.1 Core Service

Single entry point for all notification sources. Lives in `core/services/notification_dispatcher.py`.

```python
"""
Unified Notification Dispatcher (PRD-128)

Single entry point for all event → notification routing.
Replaces HeartbeatService._deliver_notification() and provides
consistent notification handling across all event sources.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.orm import Session

from core.services.notification_service import send_workspace_notification

logger = logging.getLogger(__name__)


class NotificationDispatcher:
    """Dispatches notifications based on workspace preferences."""

    def __init__(self, db: Session, workspace_id: UUID):
        self.db = db
        self.workspace_id = workspace_id

    async def dispatch(
        self,
        event_type: str,
        title: str,
        message: Optional[str] = None,
        link_type: Optional[str] = None,
        link_id: Optional[str] = None,
        agent_id: Optional[int] = None,
        agent_name: Optional[str] = None,
        status: str = "ok",
        user_id: Optional[int] = None,  # Target specific user, or None for all
    ) -> Dict[str, Any]:
        """
        Dispatch a notification based on workspace preferences.

        Args:
            event_type: Event type (heartbeat_complete, task_complete, etc.)
            title: Short notification title
            message: Optional longer message
            link_type: Type of linked resource (report, task, mission, file)
            link_id: ID or path of linked resource
            agent_id: Optional agent ID for attribution
            agent_name: Optional agent name for display
            status: Status level (ok, warning, critical, info)
            user_id: Target specific user, or None for workspace-wide

        Returns:
            Dict with success status and destinations notified
        """
        # 1. Get preferences for this event type
        prefs = self._get_preferences(event_type, user_id)
        
        if not prefs:
            logger.debug(
                "[NotificationDispatcher] No preferences for event_type=%s, using default (in_app)",
                event_type,
            )
            prefs = [{"destination": "in_app", "enabled": True}]

        results = {"success": True, "dispatched_to": []}

        for pref in prefs:
            if not pref.get("enabled", True):
                continue

            destination = pref.get("destination", "in_app")

            if destination == "silent":
                continue

            if destination == "in_app":
                await self._create_in_app_notification(
                    event_type=event_type,
                    title=title,
                    message=message,
                    link_type=link_type,
                    link_id=link_id,
                    agent_id=agent_id,
                    agent_name=agent_name,
                    status=status,
                    user_id=user_id,
                )
                results["dispatched_to"].append("in_app")

            elif destination in ("telegram", "slack", "webhook"):
                # Reuse existing notification_service.py
                formatted_msg = self._format_external_message(
                    title=title,
                    message=message,
                    status=status,
                    agent_name=agent_name,
                    link_type=link_type,
                    link_id=link_id,
                )
                ok = await send_workspace_notification(
                    workspace_id=str(self.workspace_id),
                    message=formatted_msg,
                    channel=destination,
                )
                if ok:
                    results["dispatched_to"].append(destination)

            elif destination == "channel":
                # Route to specific channel_connection
                conn_id = pref.get("channel_connection_id")
                if conn_id:
                    await self._send_via_channel_connection(conn_id, title, message, status)
                    results["dispatched_to"].append(f"channel:{conn_id}")

        logger.info(
            "[NotificationDispatcher] event=%s dispatched to %s",
            event_type,
            results["dispatched_to"],
        )
        return results

    def _get_preferences(
        self, event_type: str, user_id: Optional[int]
    ) -> List[Dict[str, Any]]:
        """Load preferences for event type, with user override fallback to workspace default."""
        
        # Try user-specific first, then workspace default
        query = text("""
            SELECT destination, channel_connection_id, enabled
            FROM notification_preferences
            WHERE workspace_id = :ws_id
              AND event_type = :event_type
              AND (user_id = :user_id OR user_id IS NULL)
            ORDER BY user_id NULLS LAST
            LIMIT 1
        """)
        
        row = self.db.execute(
            query,
            {"ws_id": str(self.workspace_id), "event_type": event_type, "user_id": user_id},
        ).fetchone()

        if row:
            return [{
                "destination": row.destination,
                "channel_connection_id": str(row.channel_connection_id) if row.channel_connection_id else None,
                "enabled": row.enabled,
            }]
        return []

    async def _create_in_app_notification(
        self,
        event_type: str,
        title: str,
        message: Optional[str],
        link_type: Optional[str],
        link_id: Optional[str],
        agent_id: Optional[int],
        agent_name: Optional[str],
        status: str,
        user_id: Optional[int],
    ):
        """Insert notification into notifications table."""
        self.db.execute(
            text("""
                INSERT INTO notifications
                    (workspace_id, user_id, event_type, title, message,
                     link_type, link_id, agent_id, agent_name, status, created_at)
                VALUES
                    (:ws_id, :user_id, :event_type, :title, :message,
                     :link_type, :link_id, :agent_id, :agent_name, :status, NOW())
            """),
            {
                "ws_id": str(self.workspace_id),
                "user_id": user_id,
                "event_type": event_type,
                "title": title,
                "message": message,
                "link_type": link_type,
                "link_id": link_id,
                "agent_id": agent_id,
                "agent_name": agent_name,
                "status": status,
            },
        )
        self.db.commit()

    def _format_external_message(
        self,
        title: str,
        message: Optional[str],
        status: str,
        agent_name: Optional[str],
        link_type: Optional[str],
        link_id: Optional[str],
    ) -> str:
        """Format notification for external channels (Telegram/Slack)."""
        status_icons = {
            "ok": "✓",
            "warning": "⚠️",
            "critical": "🚨",
            "info": "ℹ️",
        }
        icon = status_icons.get(status, "•")
        
        lines = [f"{icon} {title}"]
        
        if agent_name:
            lines.append(f"Agent: {agent_name}")
        
        if message:
            lines.append(message[:200] + "..." if len(message) > 200 else message)
        
        # TODO: Generate deep link URL when frontend routing is ready
        # if link_type and link_id:
        #     lines.append(f"View: {config.FRONTEND_URL}/workspace/{link_type}/{link_id}")
        
        return "\n".join(lines)

    async def _send_via_channel_connection(
        self,
        connection_id: str,
        title: str,
        message: Optional[str],
        status: str,
    ):
        """Send via a specific channel_connection row."""
        row = self.db.execute(
            text("""
                SELECT platform, config FROM channel_connections
                WHERE id = :conn_id AND workspace_id = :ws_id
            """),
            {"conn_id": connection_id, "ws_id": str(self.workspace_id)},
        ).fetchone()

        if not row:
            logger.warning("[NotificationDispatcher] Channel connection %s not found", connection_id)
            return

        formatted = self._format_external_message(title, message, status, None, None, None)
        
        # Reuse notification_service internals
        await send_workspace_notification(
            workspace_id=str(self.workspace_id),
            message=formatted,
            channel=row.platform,
        )
```

### 2.2 Integration Points

Wire the dispatcher into existing completion handlers:

| Source | File | Integration Point |
|--------|------|-------------------|
| Heartbeat | `services/heartbeat_service.py` | Replace `_deliver_notification()` with `dispatcher.dispatch(event_type="heartbeat_complete", ...)` |
| Task Complete | `api/tasks.py` | Add `dispatcher.dispatch()` call in task completion endpoint |
| Mission Step | `services/coordinator_service.py` | Call after each step completes |
| Mission Complete | `services/coordinator_service.py` | Call when mission reaches terminal state |
| Playbook Step | `services/playbook_executor.py` | Call after each step |
| Playbook Complete | `services/playbook_executor.py` | Call when playbook finishes |
| Trigger Fired | `api/webhooks.py` | Call when webhook trigger executes |
| Report Submitted | `services/report_service.py` | Call after `create_report()` succeeds |

---

## 3. API Endpoints

### 3.1 Notifications API

```
GET  /api/notifications                 — List notifications (paginated, filtered)
GET  /api/notifications/unread-count    — Get unread count for badge
POST /api/notifications/{id}/read       — Mark single notification as read
POST /api/notifications/read-all        — Mark all as read
POST /api/notifications/{id}/dismiss    — Dismiss (hide) notification
```

### 3.2 Notification Preferences API

```
GET  /api/notification-preferences      — Get all preferences for workspace
PUT  /api/notification-preferences      — Bulk update preferences
```

### 3.3 API Implementation

File: `orchestrator/api/notifications.py`

```python
"""
Notifications API (PRD-128)
"""

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import text

from core.auth.hybrid import get_request_context_hybrid
from core.auth.dependencies import RequestContext
from core.database.database import SessionLocal

router = APIRouter(prefix="/api/notifications", tags=["notifications"])


class NotificationResponse(BaseModel):
    id: str
    event_type: str
    title: str
    message: Optional[str]
    link_type: Optional[str]
    link_id: Optional[str]
    agent_name: Optional[str]
    status: str
    read_at: Optional[str]
    created_at: str


@router.get("")
async def list_notifications(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    unread_only: bool = Query(False),
    limit: int = Query(20, le=100),
    offset: int = Query(0),
):
    """List notifications for current user/workspace."""
    db = SessionLocal()
    try:
        conditions = ["workspace_id = :ws_id", "dismissed_at IS NULL"]
        params = {"ws_id": str(ctx.workspace_id), "limit": limit, "offset": offset}

        if ctx.user_id:
            conditions.append("(user_id = :user_id OR user_id IS NULL)")
            params["user_id"] = ctx.user_id

        if unread_only:
            conditions.append("read_at IS NULL")

        where = " AND ".join(conditions)

        rows = db.execute(
            text(f"""
                SELECT id, event_type, title, message, link_type, link_id,
                       agent_name, status, read_at, created_at
                FROM notifications
                WHERE {where}
                ORDER BY created_at DESC
                LIMIT :limit OFFSET :offset
            """),
            params,
        ).fetchall()

        count = db.execute(
            text(f"SELECT COUNT(*) FROM notifications WHERE {where}"),
            params,
        ).scalar()

        return {
            "success": True,
            "notifications": [
                {
                    "id": str(r.id),
                    "event_type": r.event_type,
                    "title": r.title,
                    "message": r.message,
                    "link_type": r.link_type,
                    "link_id": r.link_id,
                    "agent_name": r.agent_name,
                    "status": r.status,
                    "read_at": r.read_at.isoformat() if r.read_at else None,
                    "created_at": r.created_at.isoformat(),
                }
                for r in rows
            ],
            "total": count,
            "limit": limit,
            "offset": offset,
        }
    finally:
        db.close()


@router.get("/unread-count")
async def get_unread_count(ctx: RequestContext = Depends(get_request_context_hybrid)):
    """Get unread notification count for badge."""
    db = SessionLocal()
    try:
        conditions = ["workspace_id = :ws_id", "read_at IS NULL", "dismissed_at IS NULL"]
        params = {"ws_id": str(ctx.workspace_id)}

        if ctx.user_id:
            conditions.append("(user_id = :user_id OR user_id IS NULL)")
            params["user_id"] = ctx.user_id

        count = db.execute(
            text(f"SELECT COUNT(*) FROM notifications WHERE {' AND '.join(conditions)}"),
            params,
        ).scalar()

        return {"success": True, "count": count}
    finally:
        db.close()


@router.post("/{notification_id}/read")
async def mark_as_read(
    notification_id: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
):
    """Mark a single notification as read."""
    db = SessionLocal()
    try:
        result = db.execute(
            text("""
                UPDATE notifications SET read_at = NOW()
                WHERE id = :id AND workspace_id = :ws_id AND read_at IS NULL
                RETURNING id
            """),
            {"id": notification_id, "ws_id": str(ctx.workspace_id)},
        )
        db.commit()
        return {"success": result.rowcount > 0}
    finally:
        db.close()


@router.post("/read-all")
async def mark_all_as_read(ctx: RequestContext = Depends(get_request_context_hybrid)):
    """Mark all notifications as read."""
    db = SessionLocal()
    try:
        conditions = ["workspace_id = :ws_id", "read_at IS NULL"]
        params = {"ws_id": str(ctx.workspace_id)}

        if ctx.user_id:
            conditions.append("(user_id = :user_id OR user_id IS NULL)")
            params["user_id"] = ctx.user_id

        result = db.execute(
            text(f"UPDATE notifications SET read_at = NOW() WHERE {' AND '.join(conditions)}"),
            params,
        )
        db.commit()
        return {"success": True, "marked_count": result.rowcount}
    finally:
        db.close()


@router.post("/{notification_id}/dismiss")
async def dismiss_notification(
    notification_id: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
):
    """Dismiss (hide) a notification."""
    db = SessionLocal()
    try:
        result = db.execute(
            text("""
                UPDATE notifications SET dismissed_at = NOW()
                WHERE id = :id AND workspace_id = :ws_id
                RETURNING id
            """),
            {"id": notification_id, "ws_id": str(ctx.workspace_id)},
        )
        db.commit()
        return {"success": result.rowcount > 0}
    finally:
        db.close()
```

---

## 4. Frontend Components

### 4.1 NotificationBell Component

Replaces the static bell icon with a functional dropdown.

File: `frontend/components/notifications/notification-bell.tsx`

```tsx
'use client'

import { useState, useEffect } from 'react'
import { Bell, Check, X, FileText, Zap, ClipboardList, Calendar } from 'lucide-react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from '@/components/ui/popover'
import { Button } from '@/components/ui/button'
import { ScrollArea } from '@/components/ui/scroll-area'
import { cn } from '@/lib/utils'
import { apiClient } from '@/lib/api-client'
import { formatDistanceToNow } from 'date-fns'

interface Notification {
  id: string
  event_type: string
  title: string
  message: string | null
  link_type: string | null
  link_id: string | null
  agent_name: string | null
  status: 'ok' | 'warning' | 'critical' | 'info'
  read_at: string | null
  created_at: string
}

const EVENT_ICONS: Record<string, React.ElementType> = {
  heartbeat_complete: Zap,
  task_complete: ClipboardList,
  mission_complete: Calendar,
  report_submitted: FileText,
  default: Bell,
}

const STATUS_COLORS: Record<string, string> = {
  ok: 'bg-emerald-500/10 text-emerald-500',
  warning: 'bg-amber-500/10 text-amber-500',
  critical: 'bg-red-500/10 text-red-500',
  info: 'bg-blue-500/10 text-blue-500',
}

export function NotificationBell() {
  const [open, setOpen] = useState(false)
  const queryClient = useQueryClient()

  // Fetch unread count for badge
  const { data: countData } = useQuery({
    queryKey: ['notifications', 'unread-count'],
    queryFn: () => apiClient.get('/api/notifications/unread-count'),
    refetchInterval: 30000, // Poll every 30s
  })

  // Fetch notifications when dropdown opens
  const { data: notificationsData, isLoading } = useQuery({
    queryKey: ['notifications', 'list'],
    queryFn: () => apiClient.get('/api/notifications?limit=20'),
    enabled: open,
  })

  const markAsRead = useMutation({
    mutationFn: (id: string) => apiClient.post(`/api/notifications/${id}/read`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['notifications'] })
    },
  })

  const markAllAsRead = useMutation({
    mutationFn: () => apiClient.post('/api/notifications/read-all'),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['notifications'] })
    },
  })

  const dismiss = useMutation({
    mutationFn: (id: string) => apiClient.post(`/api/notifications/${id}/dismiss`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['notifications'] })
    },
  })

  const unreadCount = countData?.count ?? 0
  const notifications: Notification[] = notificationsData?.notifications ?? []

  const handleNotificationClick = (notif: Notification) => {
    if (!notif.read_at) {
      markAsRead.mutate(notif.id)
    }
    
    // Navigate to linked resource
    if (notif.link_type && notif.link_id) {
      const routes: Record<string, string> = {
        report: `/activity/reports/${notif.link_id}`,
        task: `/activity/tasks/${notif.link_id}`,
        mission: `/activity/missions/${notif.link_id}`,
        file: `/workspace?path=${encodeURIComponent(notif.link_id)}`,
      }
      const route = routes[notif.link_type]
      if (route) {
        window.location.href = route
      }
    }
    
    setOpen(false)
  }

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button
          variant="ghost"
          size="icon"
          className="relative h-9 w-9"
          aria-label={`Notifications${unreadCount > 0 ? ` (${unreadCount} unread)` : ''}`}
        >
          <Bell className="h-5 w-5" />
          {unreadCount > 0 && (
            <span className="absolute -top-0.5 -right-0.5 flex h-4 w-4 items-center justify-center rounded-full bg-primary text-[10px] font-medium text-primary-foreground">
              {unreadCount > 9 ? '9+' : unreadCount}
            </span>
          )}
        </Button>
      </PopoverTrigger>
      
      <PopoverContent
        className="w-80 p-0"
        align="end"
        sideOffset={8}
      >
        {/* Header */}
        <div className="flex items-center justify-between border-b px-4 py-3">
          <h3 className="text-sm font-semibold">Notifications</h3>
          {unreadCount > 0 && (
            <Button
              variant="ghost"
              size="sm"
              className="h-auto px-2 py-1 text-xs text-muted-foreground hover:text-foreground"
              onClick={() => markAllAsRead.mutate()}
            >
              <Check className="mr-1 h-3 w-3" />
              Mark all read
            </Button>
          )}
        </div>

        {/* Notification list */}
        <ScrollArea className="h-[400px]">
          {isLoading ? (
            <div className="flex items-center justify-center py-8 text-sm text-muted-foreground">
              Loading...
            </div>
          ) : notifications.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-12 text-center">
              <Bell className="mb-2 h-8 w-8 text-muted-foreground/50" />
              <p className="text-sm text-muted-foreground">No notifications</p>
            </div>
          ) : (
            <div className="divide-y">
              {notifications.map((notif) => {
                const Icon = EVENT_ICONS[notif.event_type] || EVENT_ICONS.default
                const isUnread = !notif.read_at

                return (
                  <div
                    key={notif.id}
                    className={cn(
                      'group relative flex gap-3 px-4 py-3 cursor-pointer transition-colors hover:bg-muted/50',
                      isUnread && 'bg-primary/5'
                    )}
                    onClick={() => handleNotificationClick(notif)}
                  >
                    {/* Icon */}
                    <div className={cn('flex h-8 w-8 shrink-0 items-center justify-center rounded-full', STATUS_COLORS[notif.status])}>
                      <Icon className="h-4 w-4" />
                    </div>

                    {/* Content */}
                    <div className="min-w-0 flex-1">
                      <div className="flex items-start justify-between gap-2">
                        <p className={cn('text-sm leading-tight', isUnread && 'font-medium')}>
                          {notif.title}
                        </p>
                        {isUnread && (
                          <span className="h-2 w-2 shrink-0 rounded-full bg-primary" />
                        )}
                      </div>
                      
                      {notif.agent_name && (
                        <p className="mt-0.5 text-xs text-muted-foreground">
                          {notif.agent_name}
                        </p>
                      )}
                      
                      <p className="mt-1 text-xs text-muted-foreground">
                        {formatDistanceToNow(new Date(notif.created_at), { addSuffix: true })}
                      </p>
                    </div>

                    {/* Dismiss button */}
                    <button
                      className="absolute right-2 top-2 hidden h-6 w-6 items-center justify-center rounded-md hover:bg-muted group-hover:flex"
                      onClick={(e) => {
                        e.stopPropagation()
                        dismiss.mutate(notif.id)
                      }}
                      aria-label="Dismiss"
                    >
                      <X className="h-3 w-3 text-muted-foreground" />
                    </button>
                  </div>
                )
              })}
            </div>
          )}
        </ScrollArea>

        {/* Footer */}
        <div className="border-t px-4 py-2">
          <Button
            variant="ghost"
            size="sm"
            className="w-full text-xs"
            onClick={() => {
              window.location.href = '/settings/notifications'
              setOpen(false)
            }}
          >
            Notification Settings
          </Button>
        </div>
      </PopoverContent>
    </Popover>
  )
}
```

### 4.2 Notification Settings Page

File: `frontend/app/settings/notifications/page.tsx`

```tsx
'use client'

import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Bell, MessageSquare, Webhook, Volume2, VolumeX } from 'lucide-react'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Switch } from '@/components/ui/switch'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { apiClient } from '@/lib/api-client'

const EVENT_TYPES = [
  { id: 'heartbeat_complete', label: 'Heartbeat Complete', description: 'When an agent completes a scheduled heartbeat' },
  { id: 'task_complete', label: 'Task Complete', description: 'When a task on the board is marked complete' },
  { id: 'mission_step_complete', label: 'Mission Step', description: 'When a mission step completes (can be noisy)' },
  { id: 'mission_complete', label: 'Mission Complete', description: 'When an entire mission finishes' },
  { id: 'playbook_step_complete', label: 'Playbook Step', description: 'When a playbook step runs' },
  { id: 'playbook_complete', label: 'Playbook Complete', description: 'When a playbook finishes all steps' },
  { id: 'trigger_fired', label: 'Trigger Fired', description: 'When a webhook trigger executes' },
  { id: 'report_submitted', label: 'Report Submitted', description: 'When an agent submits a report' },
  { id: 'agent_error', label: 'Agent Error', description: 'When an agent encounters an error' },
]

const DESTINATIONS = [
  { id: 'in_app', label: 'In-App', icon: Bell },
  { id: 'telegram', label: 'Telegram', icon: MessageSquare },
  { id: 'slack', label: 'Slack', icon: MessageSquare },
  { id: 'webhook', label: 'Webhook', icon: Webhook },
  { id: 'silent', label: 'Silent', icon: VolumeX },
]

export default function NotificationSettingsPage() {
  const queryClient = useQueryClient()

  const { data: prefsData, isLoading } = useQuery({
    queryKey: ['notification-preferences'],
    queryFn: () => apiClient.get('/api/notification-preferences'),
  })

  const updatePrefs = useMutation({
    mutationFn: (prefs: any[]) => apiClient.put('/api/notification-preferences', { preferences: prefs }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['notification-preferences'] })
    },
  })

  const preferences = prefsData?.preferences ?? []

  const getPref = (eventType: string) => {
    return preferences.find((p: any) => p.event_type === eventType) ?? {
      event_type: eventType,
      destination: 'in_app',
      enabled: true,
    }
  }

  const updatePref = (eventType: string, updates: Partial<{ destination: string; enabled: boolean }>) => {
    const existing = [...preferences]
    const idx = existing.findIndex((p: any) => p.event_type === eventType)
    
    if (idx >= 0) {
      existing[idx] = { ...existing[idx], ...updates }
    } else {
      existing.push({ event_type: eventType, destination: 'in_app', enabled: true, ...updates })
    }
    
    updatePrefs.mutate(existing)
  }

  if (isLoading) {
    return <div className="p-8 text-center text-muted-foreground">Loading...</div>
  }

  return (
    <div className="container max-w-3xl py-8">
      <div className="mb-8">
        <h1 className="text-2xl font-bold">Notification Settings</h1>
        <p className="text-muted-foreground">
          Configure how you want to be notified when events occur.
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Event Notifications</CardTitle>
          <CardDescription>
            Choose where to receive notifications for each event type.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {EVENT_TYPES.map((event) => {
            const pref = getPref(event.id)
            
            return (
              <div key={event.id} className="flex items-center justify-between gap-4 py-2">
                <div className="min-w-0 flex-1">
                  <p className="font-medium">{event.label}</p>
                  <p className="text-sm text-muted-foreground">{event.description}</p>
                </div>
                
                <div className="flex items-center gap-4">
                  <Select
                    value={pref.destination}
                    onValueChange={(value) => updatePref(event.id, { destination: value })}
                    disabled={!pref.enabled}
                  >
                    <SelectTrigger className="w-32">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {DESTINATIONS.map((dest) => (
                        <SelectItem key={dest.id} value={dest.id}>
                          <div className="flex items-center gap-2">
                            <dest.icon className="h-4 w-4" />
                            {dest.label}
                          </div>
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  
                  <Switch
                    checked={pref.enabled}
                    onCheckedChange={(checked) => updatePref(event.id, { enabled: checked })}
                  />
                </div>
              </div>
            )
          })}
        </CardContent>
      </Card>
    </div>
  )
}
```

---

## 5. Migration

File: `orchestrator/alembic/versions/prd128_notifications.py`

```python
"""PRD-128: Unified Notification System

Revision ID: prd128_notifications
Revises: (previous)
Create Date: 2026-04-10
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB

revision = 'prd128_notifications'
down_revision = None  # Set to actual previous revision
branch_labels = None
depends_on = None


def upgrade():
    # notification_preferences table
    op.create_table(
        'notification_preferences',
        sa.Column('id', UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('workspace_id', UUID(as_uuid=True), sa.ForeignKey('workspaces.id', ondelete='CASCADE'), nullable=False),
        sa.Column('user_id', sa.Integer, sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=True),
        sa.Column('event_type', sa.String(50), nullable=False),
        sa.Column('destination', sa.String(30), nullable=False, server_default='in_app'),
        sa.Column('channel_connection_id', UUID(as_uuid=True), sa.ForeignKey('channel_connections.id', ondelete='SET NULL'), nullable=True),
        sa.Column('enabled', sa.Boolean, nullable=False, server_default='true'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint('workspace_id', 'user_id', 'event_type', name='uq_notif_prefs_ws_user_event'),
    )
    op.create_index('ix_notif_prefs_workspace', 'notification_preferences', ['workspace_id'])
    op.create_index('ix_notif_prefs_user', 'notification_preferences', ['user_id'], postgresql_where=sa.text('user_id IS NOT NULL'))

    # notifications table
    op.create_table(
        'notifications',
        sa.Column('id', UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('workspace_id', UUID(as_uuid=True), sa.ForeignKey('workspaces.id', ondelete='CASCADE'), nullable=False),
        sa.Column('user_id', sa.Integer, sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=True),
        sa.Column('event_type', sa.String(50), nullable=False),
        sa.Column('title', sa.String(255), nullable=False),
        sa.Column('message', sa.Text, nullable=True),
        sa.Column('link_type', sa.String(30), nullable=True),
        sa.Column('link_id', sa.String(255), nullable=True),
        sa.Column('agent_id', sa.Integer, sa.ForeignKey('agents.id', ondelete='SET NULL'), nullable=True),
        sa.Column('agent_name', sa.String(100), nullable=True),
        sa.Column('status', sa.String(20), nullable=False, server_default='ok'),
        sa.Column('read_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('dismissed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index('ix_notifications_workspace', 'notifications', ['workspace_id'])
    op.create_index('ix_notifications_user', 'notifications', ['user_id'], postgresql_where=sa.text('user_id IS NOT NULL'))
    op.create_index('ix_notifications_unread', 'notifications', ['workspace_id', 'user_id', 'created_at'], postgresql_where=sa.text('read_at IS NULL AND dismissed_at IS NULL'))
    op.create_index('ix_notifications_recent', 'notifications', ['workspace_id', sa.desc('created_at')])


def downgrade():
    op.drop_table('notifications')
    op.drop_table('notification_preferences')
```

---

## 6. Implementation Phases

### Phase 1: Database & Core Service (2 days)
- [ ] Create migration for `notification_preferences` and `notifications` tables
- [ ] Implement `NotificationDispatcher` service
- [ ] Write unit tests for dispatcher

### Phase 2: API Endpoints (1 day)
- [ ] Implement `/api/notifications` endpoints
- [ ] Implement `/api/notification-preferences` endpoints
- [ ] Add to router in `main.py`

### Phase 3: Event Hooks (2 days)
- [ ] Migrate `HeartbeatService._deliver_notification()` to use dispatcher
- [ ] Wire task completion to dispatcher
- [ ] Wire mission completion to dispatcher
- [ ] Wire playbook completion to dispatcher
- [ ] Wire report submission to dispatcher

### Phase 4: Frontend (2 days)
- [ ] Implement `NotificationBell` component
- [ ] Replace static bell icon in navbar
- [ ] Implement notification settings page
- [ ] Add to settings navigation

### Phase 5: Testing & Polish (1 day)
- [ ] E2E test: heartbeat → notification → bell → click → navigate
- [ ] Test Telegram/Slack routing
- [ ] Performance test: 100 notifications in dropdown

---

## 7. Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Notification delivery latency | < 500ms | P95 from event to notification table insert |
| Bell badge accuracy | 100% | Unread count matches actual unread |
| Click-through rate | > 30% | Notifications clicked / notifications delivered |
| External delivery success | > 95% | Telegram/Slack messages sent successfully |

---

## 8. Future Enhancements (Not in Scope)

- **Email notifications** — Add `email` destination, integrate with SendGrid/SES
- **Push notifications** — Mobile app scope
- **Digest mode** — Batch notifications into hourly/daily summaries
- **Rich actions** — Mark task done, snooze agent from notification
- **Notification templates** — Customizable message formats per event type
