'use client'

/**
 * PRD-256 F430 — what a copy of a marketplace agent could not call as a CLI session.
 *
 * An install is an API agent, which has every platform tool. A copy switched to a CLI
 * session (local edition) gets the default session tools, and the item detail's
 * `dependencies.session_tool_gaps` lists what its skills name that those do not have:
 * one warning line per skill, before anyone installs it and switches it. Nothing in the
 * hosted edition, where agents stay on the API runtime.
 */
import { AlertTriangle } from 'lucide-react'
import { isLocal } from '@/lib/auth-edition'
import { GAP_WARNING_CLASS, sessionGapLines } from '@/components/agents/runtime-section'
import type { SessionToolGapEntry } from '@/types/session-tools'

interface WithSessionGaps {
  dependencies?: { session_tool_gaps?: SessionToolGapEntry[] | null } | null
}

/** The skill gaps a CLI-session copy would fail at; [] when there are none or the backend sent none. */
export function cliCopyGapLines(item: unknown): string[] {
  const gaps = (item as WithSessionGaps | null | undefined)?.dependencies?.session_tool_gaps
  return sessionGapLines(gaps).filter((entry) => entry.cannotUse).map((entry) => entry.line)
}

export function ItemSessionGapsSection({ item }: { item: unknown }) {
  const lines = cliCopyGapLines(item)
  if (!isLocal || lines.length === 0) return null
  return (
    <div className="space-y-2" data-testid="marketplace-session-gaps">
      <h3 className="flex items-center gap-2 text-sm font-semibold">
        <AlertTriangle className="h-4 w-4 text-[hsl(var(--warning))]" />
        As a CLI session
      </h3>
      {lines.map((line, index) => (
        <p key={`${index}-${line}`} className={GAP_WARNING_CLASS}>{line}</p>
      ))}
    </div>
  )
}
