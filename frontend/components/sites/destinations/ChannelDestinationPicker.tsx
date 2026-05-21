'use client'

import { useEffect, useState } from 'react'
import Link from 'next/link'
import { Loader2, Plus, Trash2, AlertCircle, Radio } from 'lucide-react'
import { Card, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { listCallbackChannels } from '@/lib/sites/api'
import {
  PLATFORM_LABEL,
  PLATFORM_TARGET_HINT,
  type ChannelConnection,
} from '@/lib/channels/types'
import type { CallbackDestination } from '@/lib/sites/types'

interface Props {
  value: CallbackDestination[]
  onChange: (next: CallbackDestination[]) => void
}

export function ChannelDestinationPicker({ value, onChange }: Props) {
  const [channels, setChannels] = useState<ChannelConnection[] | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    listCallbackChannels()
      .then((c) => { if (!cancelled) setChannels(c) })
      .catch((e) => {
        if (!cancelled) setError(e?.message ?? 'Failed to load channels')
      })
    return () => { cancelled = true }
  }, [])

  const addedIds = new Set(value.map((d) => d.connection_id))
  const available = (channels ?? []).filter((c) => !addedIds.has(c.id))

  function addDestination(c: ChannelConnection) {
    onChange([
      ...value,
      {
        type: 'channel_connection',
        connection_id: c.id,
        target: '',
        platform: c.platform,
        label: PLATFORM_LABEL[c.platform] ?? c.platform,
      },
    ])
  }

  function patchDestination(idx: number, patch: Partial<CallbackDestination>) {
    onChange(value.map((d, i) => (i === idx ? { ...d, ...patch } : d)))
  }

  function removeDestination(idx: number) {
    onChange(value.filter((_, i) => i !== idx))
  }

  if (error) {
    return (
      <Card>
        <CardContent className="py-4 flex items-start gap-2 text-sm text-destructive">
          <AlertCircle className="w-4 h-4 mt-0.5 shrink-0" />
          <div>
            <div className="font-medium">Couldn't load channel connections</div>
            <div className="text-xs text-muted-foreground mt-1">{error}</div>
          </div>
        </CardContent>
      </Card>
    )
  }

  if (channels === null) {
    return (
      <div className="flex items-center text-muted-foreground text-sm py-3">
        <Loader2 className="w-4 h-4 mr-2 animate-spin" /> Loading channels…
      </div>
    )
  }

  if (channels.length === 0) {
    return (
      <Card>
        <CardContent className="py-6 text-sm text-muted-foreground space-y-2">
          <div className="flex items-start gap-2">
            <Radio className="w-4 h-4 mt-0.5 shrink-0 text-muted-foreground" />
            <div>
              <div className="font-medium text-foreground">No channels connected yet</div>
              <p className="text-xs mt-1">
                Connect Slack, Telegram, WhatsApp, or another channel under{' '}
                <Link href="/settings" className="text-primary hover:underline">
                  Settings → Channels
                </Link>
                . They'll show up here as callback destinations.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <div className="space-y-3">
      {value.length === 0 && (
        <p className="text-xs text-muted-foreground italic">
          No destinations yet — callbacks are accepted but not delivered. Add one below.
        </p>
      )}

      <div className="space-y-2">
        {value.map((dest, idx) => {
          const channel = channels.find((c) => c.id === dest.connection_id)
          const platform = dest.platform ?? channel?.platform ?? 'unknown'
          const hint = PLATFORM_TARGET_HINT[platform]
          const stale = !channel
          return (
            <div
              key={`${dest.connection_id}-${idx}`}
              className="flex items-center gap-2 p-2 rounded-md border border-border/40 bg-card"
            >
              <Badge
                variant="outline"
                className="text-xs shrink-0 border-primary/40 bg-primary/10 text-foreground"
              >
                {PLATFORM_LABEL[platform] ?? platform}
              </Badge>
              <div className="flex-1">
                <Input
                  placeholder={hint?.placeholder ?? 'Target ID'}
                  value={dest.target}
                  onChange={(e) => patchDestination(idx, { target: e.target.value })}
                  aria-label={hint?.label ?? 'Target'}
                />
                {stale && (
                  <p className="text-[10px] text-destructive mt-1">
                    Channel connection no longer exists — remove and re-add.
                  </p>
                )}
              </div>
              <Button
                size="sm"
                variant="ghost"
                onClick={() => removeDestination(idx)}
                aria-label="Remove destination"
              >
                <Trash2 className="w-3.5 h-3.5" />
              </Button>
            </div>
          )
        })}
      </div>

      <div>
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button size="sm" variant="outline" disabled={available.length === 0}>
              <Plus className="w-3.5 h-3.5 mr-1" />
              {available.length === 0 ? 'All channels added' : 'Add destination'}
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="start">
            {available.map((c) => (
              <DropdownMenuItem key={c.id} onClick={() => addDestination(c)}>
                <span className="font-medium mr-2">{PLATFORM_LABEL[c.platform] ?? c.platform}</span>
                <span className="text-xs text-muted-foreground">
                  {c.status === 'active' ? 'active' : c.status}
                </span>
              </DropdownMenuItem>
            ))}
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
    </div>
  )
}
