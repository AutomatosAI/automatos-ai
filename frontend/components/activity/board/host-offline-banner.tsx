'use client'

import { TerminalSquare } from 'lucide-react'
import { useCliHostHealth } from '@/hooks/use-cli-host-health'

function since(iso: string | null): string {
  if (!iso) return 'never seen'
  const ms = Date.now() - new Date(iso).getTime()
  const mins = Math.floor(ms / 60000)
  if (mins < 1) return 'moments ago'
  if (mins < 60) return `${mins} min ago`
  const hours = Math.floor(mins / 60)
  if (hours < 48) return `${hours} h ago`
  return `${Math.floor(hours / 24)} days ago`
}

/**
 * PRD-235 W3: the board's executor warning. Claude Code agents run on the
 * operator's paired host; when none is online their tickets queue silently in
 * `assigned`. This says so once, at the top, with the way out. Renders nothing
 * when the workspace has no CLI agents, or a host is online.
 */
export function HostOfflineBanner() {
  const { data } = useCliHostHealth()
  if (!data || data.online || data.cli_agents === 0) return null
  const waiting = data.waiting_tickets
  return (
    <div
      role="status"
      data-testid="host-offline-banner"
      className="flex items-start gap-3 px-4 py-3 rounded-lg bg-[hsl(var(--warning))]/10 border border-[hsl(var(--warning))]/30"
    >
      <TerminalSquare className="w-4 h-4 mt-0.5 text-[hsl(var(--warning))] shrink-0" />
      <div className="text-sm">
        <p className="font-medium text-[hsl(var(--warning))]">
          Your CLI host is offline
          {data.paired_hosts > 0 ? ` (last seen ${since(data.last_seen_at)})` : ' (none paired yet)'}
          {waiting > 0 ? ` — ${waiting} ticket${waiting === 1 ? '' : 's'} waiting` : ''}
        </p>
        <p className="text-xs text-muted-foreground mt-0.5">
          {data.paired_hosts > 0
            ? 'Claude Code agents cannot start until it is back. Run `make cli-host`, or install it as a login service once with `make cli-host-install` so it is always on.'
            : 'Pair a host from Settings → Session mode (`make cli-host PAIR=<code>`), then `make cli-host-install` keeps it running.'}
        </p>
      </div>
    </div>
  )
}
