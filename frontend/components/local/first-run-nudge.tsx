'use client'

/**
 * PRD-233 S3 — local-edition first-run nudge.
 *
 * The local edition boots with a seeded workspace (starter roster, demo
 * Playbook, welcome Deliverable) but no language model: nothing answers until
 * the operator stores an LLM key. This banner says so on the landing surface
 * (chat), and only when the key lists are POSITIVELY empty — an unknown state
 * (request failed, still loading) renders nothing rather than nagging on a
 * guess. Dismissal is remembered per browser in localStorage (try/catch —
 * storage may be unavailable).
 *
 * Reads the two queries Settings → API Keys already runs, under the SAME query
 * keys, so saving a key there hides the nudge without a reload. No new
 * api-client methods. SaaS: never renders, never fetches.
 */

import { useEffect, useState } from 'react'
import Link from 'next/link'
import { useQuery } from '@tanstack/react-query'
import { KeyRound, X } from 'lucide-react'
import { apiClient } from '@/lib/api-client'
import { isLocal } from '@/lib/auth-edition'

export const FIRST_RUN_NUDGE_STORAGE_KEY = 'automatos:local-first-run-nudge:dismissed'
export const FIRST_RUN_NUDGE_SETTINGS_HREF = '/settings'
export const FIRST_RUN_NUDGE_MESSAGE = 'Add an LLM key to bring Auto to life'
export const FIRST_RUN_NUDGE_LINK_LABEL = 'Settings → API Keys'

interface ApiKeyOut {
  id: number
  provider: string
  is_active: boolean
}

interface PlatformKeyStatus {
  platform_keys: Record<string, { configured: boolean } | undefined>
}

function readDismissed(): boolean {
  try {
    return window.localStorage.getItem(FIRST_RUN_NUDGE_STORAGE_KEY) === '1'
  } catch {
    return false
  }
}

function persistDismissed(): void {
  try {
    window.localStorage.setItem(FIRST_RUN_NUDGE_STORAGE_KEY, '1')
  } catch {
    // Storage unavailable (private mode, blocked) — the dismissal lasts this render only.
  }
}

/** True when the workspace has any usable LLM key: a BYOK row or a platform-level key. */
export function hasAnyLlmKey(
  keys: ApiKeyOut[] | undefined,
  platform: PlatformKeyStatus | undefined,
): boolean {
  const byok = (keys ?? []).length > 0
  const platformConfigured = Object.values(platform?.platform_keys ?? {}).some(
    (entry) => entry?.configured === true,
  )
  return byok || platformConfigured
}

export function FirstRunNudge() {
  // Start hidden and read storage after mount, so hydration never flashes a
  // banner the operator already dismissed.
  const [dismissed, setDismissed] = useState(true)
  useEffect(() => {
    setDismissed(readDismissed())
  }, [])

  const armed = isLocal && !dismissed

  const keys = useQuery<ApiKeyOut[]>({
    queryKey: ['api-keys'],
    queryFn: () => apiClient.get<ApiKeyOut[]>('/api/keys'),
    enabled: armed,
  })
  const platform = useQuery<PlatformKeyStatus>({
    queryKey: ['platform-key-status'],
    queryFn: () => apiClient.get<PlatformKeyStatus>('/api/keys/platform-status'),
    enabled: armed,
  })

  if (!armed || !keys.isSuccess || !platform.isSuccess) return null
  if (hasAnyLlmKey(keys.data, platform.data)) return null

  const dismiss = () => {
    persistDismissed()
    setDismissed(true)
  }

  return (
    // Zero-height anchor: overlays the top of whichever chat layout hosts it
    // (classic `relative` container or the studio flex column) without
    // taking space from the Chat. Inline flex/height beat the studio's
    // `.sh-chat-main > * { flex: 1 }` rule.
    <div className="relative z-30" style={{ flex: '0 0 auto', height: 0 }}>
      <div
        role="status"
        aria-live="polite"
        data-testid="first-run-nudge"
        className="absolute left-1/2 top-3 flex max-w-[calc(100vw-2rem)] -translate-x-1/2 items-center gap-3 whitespace-nowrap rounded-full border border-border bg-background/95 px-4 py-2 text-sm shadow-lg backdrop-blur"
      >
        <KeyRound className="h-4 w-4 shrink-0 text-warning" aria-hidden="true" />
        <span>{FIRST_RUN_NUDGE_MESSAGE}</span>
        <Link
          href={FIRST_RUN_NUDGE_SETTINGS_HREF}
          data-testid="first-run-nudge-link"
          className="font-medium underline underline-offset-4 hover:text-foreground"
        >
          {FIRST_RUN_NUDGE_LINK_LABEL}
        </Link>
        <button
          type="button"
          aria-label="Dismiss"
          data-testid="first-run-nudge-dismiss"
          onClick={dismiss}
          className="rounded-full p-1 text-muted-foreground transition-colors hover:text-foreground"
        >
          <X className="h-4 w-4" aria-hidden="true" />
        </button>
      </div>
    </div>
  )
}
