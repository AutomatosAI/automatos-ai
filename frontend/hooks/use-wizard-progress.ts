'use client'

/**
 * PRD-130: Wizard Progress Hook
 * ==============================
 *
 * Subscribes to `GET /api/wizard/progress/{profileId}` (Server-Sent Events)
 * and returns a streaming list of events the UI can render as a live feed.
 *
 * Why fetch + ReadableStream instead of EventSource?
 * ---------------------------------------------------
 * The browser's native `EventSource` cannot send custom headers, which
 * means it cannot attach our Clerk Bearer token or X-Workspace-ID header.
 * We use `fetch` with streaming body parsing instead — same wire format,
 * auth works, and we control backoff.
 *
 * Terminal handling
 * -----------------
 * The stream ends when the server emits a `stage === "complete"` or
 * `stage === "failed"` event (backend closes the generator). The hook
 * surfaces this through `state` so the wizard shell can advance to step 6
 * or render an error panel without needing to poll anything.
 */

import { useCallback, useEffect, useRef, useState } from 'react'
import { apiClient } from '@/lib/api-client'

export type WizardStage =
  | 'scan'
  | 'scrape'
  | 'ingest'
  | 'graphify'
  | 'profile'
  | 'plan'
  | 'complete'
  | 'failed'

export type WizardEventLevel = 'info' | 'warn' | 'error'

export interface WizardProgressEvent {
  ts: number
  stage: WizardStage
  level: WizardEventLevel
  message: string
  meta: Record<string, unknown>
}

export type WizardProgressState = 'idle' | 'streaming' | 'complete' | 'failed' | 'error'

interface UseWizardProgressOptions {
  profileId: string | null
  /** Set to `true` to open the stream. Flip back to `false` to stop. */
  active: boolean
}

interface UseWizardProgressResult {
  events: WizardProgressEvent[]
  state: WizardProgressState
  /** The latest event seen — handy for showing a headline status line. */
  latest: WizardProgressEvent | null
  /** Clear buffered events locally (does not reset the server feed). */
  reset: () => void
}

export function useWizardProgress({
  profileId,
  active,
}: UseWizardProgressOptions): UseWizardProgressResult {
  const [events, setEvents] = useState<WizardProgressEvent[]>([])
  const [state, setState] = useState<WizardProgressState>('idle')
  const abortRef = useRef<AbortController | null>(null)

  const reset = useCallback(() => {
    setEvents([])
    setState('idle')
  }, [])

  useEffect(() => {
    if (!active || !profileId) {
      return
    }

    let cancelled = false
    const controller = new AbortController()
    abortRef.current = controller
    setState('streaming')

    const run = async () => {
      try {
        const baseUrl = apiClient.getBaseUrl()
        const headers = await apiClient.getAuthHeaders()
        const url = `${baseUrl}/api/wizard/progress/${profileId}`

        const response = await fetch(url, {
          method: 'GET',
          headers: {
            ...headers,
            Accept: 'text/event-stream',
          },
          signal: controller.signal,
          cache: 'no-store',
        })

        if (!response.ok || !response.body) {
          if (!cancelled) setState('error')
          return
        }

        const reader = response.body.getReader()
        const decoder = new TextDecoder('utf-8')
        let buffer = ''

        while (true) {
          const { done, value } = await reader.read()
          if (done || cancelled) break

          buffer += decoder.decode(value, { stream: true })

          // SSE frames are separated by blank lines (`\n\n`)
          let sepIdx = buffer.indexOf('\n\n')
          while (sepIdx !== -1) {
            const frame = buffer.slice(0, sepIdx)
            buffer = buffer.slice(sepIdx + 2)
            sepIdx = buffer.indexOf('\n\n')

            // Each frame may have multiple `data:` lines + comment lines (`:`)
            const dataLines = frame
              .split('\n')
              .filter((line) => line.startsWith('data:'))
              .map((line) => line.slice(5).trimStart())

            if (dataLines.length === 0) continue
            const raw = dataLines.join('\n')

            try {
              const parsed = JSON.parse(raw) as WizardProgressEvent
              if (cancelled) return

              setEvents((prev) => [...prev, parsed])

              if (parsed.stage === 'complete') {
                setState('complete')
              } else if (parsed.stage === 'failed') {
                setState('failed')
              }
            } catch {
              // Ignore malformed frames — keep the stream alive
              continue
            }
          }
        }
      } catch (err: any) {
        if (cancelled || err?.name === 'AbortError') return
        console.error('[wizard-progress] stream error:', err)
        setState('error')
      }
    }

    run()

    return () => {
      cancelled = true
      controller.abort()
      abortRef.current = null
    }
  }, [profileId, active])

  const latest = events.length > 0 ? events[events.length - 1] : null

  return { events, state, latest, reset }
}
