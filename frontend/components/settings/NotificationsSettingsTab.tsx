'use client'

/**
 * Notification preferences settings (PRD-128 US-009).
 *
 * Shows a row per event type with a destination Select and enabled Switch.
 * Reads the merged workspace+user preferences from GET /api/notification-preferences
 * and PUTs the full list back on any change. Optimistic UI via react-query
 * invalidation.
 */

import { useEffect, useMemo, useState } from 'react'
import { Bell, Loader2 } from 'lucide-react'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Switch } from '@/components/ui/switch'
import { Label } from '@/components/ui/label'
import { Badge } from '@/components/ui/badge'
import {
  useNotificationPreferences,
  useUpdateNotificationPreferences,
  type PreferenceRow,
} from '@/hooks/use-notifications-api'

// Display metadata for each of the 9 supported event types. Order here is the
// render order on the settings page.
const EVENT_TYPES: Array<{
  id: string
  label: string
  description: string
}> = [
  {
    id: 'heartbeat_complete',
    label: 'Heartbeat complete',
    description: 'An agent finished its heartbeat cycle.',
  },
  {
    id: 'task_complete',
    label: 'Task complete',
    description: 'A board task was marked complete.',
  },
  {
    id: 'mission_step_complete',
    label: 'Mission step complete',
    description: 'A single step inside a mission finished (noisy).',
  },
  {
    id: 'mission_complete',
    label: 'Mission complete',
    description: 'A mission reached its terminal state.',
  },
  {
    id: 'playbook_step_complete',
    label: 'Playbook step complete',
    description: 'A single step inside a playbook finished (noisy).',
  },
  {
    id: 'playbook_complete',
    label: 'Playbook complete',
    description: 'A playbook run finished.',
  },
  {
    id: 'trigger_fired',
    label: 'Trigger fired',
    description: 'A Composio trigger fired.',
  },
  {
    id: 'report_submitted',
    label: 'Report submitted',
    description: 'An agent submitted a report.',
  },
  {
    id: 'agent_error',
    label: 'Agent error',
    description: 'An agent raised an error.',
  },
]

const DESTINATIONS: Array<{ value: string; label: string }> = [
  { value: 'in_app', label: 'In-app' },
  { value: 'telegram', label: 'Telegram' },
  { value: 'slack', label: 'Slack' },
  { value: 'webhook', label: 'Webhook' },
  { value: 'silent', label: 'Silent' },
]

/**
 * Build the initial form state: one row per event type, preferring the first
 * matching preference returned by the API, or falling back to an in_app row.
 */
function buildInitialRows(prefs: PreferenceRow[]): Record<string, PreferenceRow> {
  const byEvent: Record<string, PreferenceRow> = {}
  for (const p of prefs) {
    if (!byEvent[p.event_type]) {
      byEvent[p.event_type] = { ...p }
    }
  }
  for (const ev of EVENT_TYPES) {
    if (!byEvent[ev.id]) {
      byEvent[ev.id] = {
        event_type: ev.id,
        destination: 'in_app',
        enabled: true,
        channel_connection_id: null,
      }
    }
  }
  return byEvent
}

export function NotificationsSettingsTab() {
  const { data, isLoading, isError, error } = useNotificationPreferences()
  const updateMutation = useUpdateNotificationPreferences()

  const [rows, setRows] = useState<Record<string, PreferenceRow>>({})

  useEffect(() => {
    if (data?.preferences) {
      setRows(buildInitialRows(data.preferences))
    }
  }, [data])

  const orderedRows = useMemo(
    () => EVENT_TYPES.map((ev) => rows[ev.id]).filter(Boolean),
    [rows],
  )

  const persist = (next: Record<string, PreferenceRow>) => {
    const payload = EVENT_TYPES.map((ev) => next[ev.id]).filter(
      (r): r is PreferenceRow => Boolean(r),
    )
    updateMutation.mutate(payload)
  }

  const handleDestinationChange = (eventId: string, destination: string) => {
    const current = rows[eventId]
    if (!current) return
    const updated: PreferenceRow = { ...current, destination }
    const next = { ...rows, [eventId]: updated }
    setRows(next)
    persist(next)
  }

  const handleEnabledChange = (eventId: string, enabled: boolean) => {
    const current = rows[eventId]
    if (!current) return
    const updated: PreferenceRow = { ...current, enabled }
    const next = { ...rows, [eventId]: updated }
    setRows(next)
    persist(next)
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-12">
        <Loader2 className="w-6 h-6 animate-spin text-muted-foreground" />
      </div>
    )
  }

  if (isError) {
    return (
      <Card>
        <CardContent className="pt-6">
          <p className="text-sm text-destructive">
            Failed to load notification preferences:{' '}
            {error instanceof Error ? error.message : 'unknown error'}
          </p>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <Bell className="w-5 h-5" />
          <CardTitle>Notification preferences</CardTitle>
          {updateMutation.isLoading && (
            <Badge variant="secondary" className="ml-2">
              Saving…
            </Badge>
          )}
        </div>
        <CardDescription>
          Pick where each event type notifies you. Changes save automatically.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {orderedRows.map((row) => {
          const meta = EVENT_TYPES.find((e) => e.id === row.event_type)
          if (!meta) return null
          return (
            <div
              key={row.event_type}
              data-testid={`notification-pref-${row.event_type}`}
              className="flex flex-col gap-3 border rounded-lg p-4 md:flex-row md:items-center md:justify-between"
            >
              <div className="flex-1 min-w-0">
                <Label className="text-sm font-medium">{meta.label}</Label>
                <p className="text-xs text-muted-foreground mt-1">
                  {meta.description}
                </p>
              </div>
              <div className="flex items-center gap-4">
                <Select
                  value={row.destination}
                  onValueChange={(value) =>
                    handleDestinationChange(row.event_type, value)
                  }
                >
                  <SelectTrigger className="w-36" aria-label={`${meta.label} destination`}>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {DESTINATIONS.map((d) => (
                      <SelectItem key={d.value} value={d.value}>
                        {d.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <div className="flex items-center gap-2">
                  <Switch
                    checked={row.enabled}
                    onCheckedChange={(checked) =>
                      handleEnabledChange(row.event_type, checked)
                    }
                    aria-label={`${meta.label} enabled`}
                  />
                  <span className="text-xs text-muted-foreground w-14">
                    {row.enabled ? 'Enabled' : 'Disabled'}
                  </span>
                </div>
              </div>
            </div>
          )
        })}
        {updateMutation.isError && (
          <p className="text-sm text-destructive">
            Failed to save:{' '}
            {updateMutation.error instanceof Error
              ? updateMutation.error.message
              : 'unknown error'}
          </p>
        )}
      </CardContent>
    </Card>
  )
}

export default NotificationsSettingsTab
