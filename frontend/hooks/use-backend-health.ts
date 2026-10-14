'use client'

import { useCallback, useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import { apiClient } from '@/lib/api-client'

/**
 * PRD-260 T-004 — is the backend answering? Play's connection pill asks
 * GET /health through apiClient (the request SessionModeTab already makes),
 * again every BACKEND_HEALTH_POLL_MS. Ported from Automatos Play's
 * useBackendStatus, whose proxy check becomes the core's /health.
 *
 * One failed request is never the answer (P260-RVW-1): Play's frame swaps
 * its page for "Automatos is asleep" on 'down' (AsleepGate), so a blip — a
 * wake from sleep, a Wi-Fi hand-over, one 502 at the edge — must not unmount
 * it. The query asks again BACKEND_HEALTH_RETRIES times, BACKEND_HEALTH_RETRY_DELAY_MS
 * apart, before it reports an error; while those retries are pending after an
 * earlier answer react-query keeps the query a success, so it stays 'connected'.
 */

/** How often the pill asks again. */
export const BACKEND_HEALTH_POLL_MS = 30_000
/** How many times a failed ask is repeated before the backend is down. */
export const BACKEND_HEALTH_RETRIES = 3
/**
 * The wait before each repeat: the retries span 7.5 s, enough for a blip, and
 * finish inside the asleep screen's 10 s recheck (ASLEEP_RECHECK_MS).
 */
export const BACKEND_HEALTH_RETRY_DELAY_MS = 2_500
export const BACKEND_HEALTH_QUERY_KEY = ['health', 'connection'] as const

export type BackendConnection = 'checking' | 'connected' | 'down'

export interface BackendHealth {
  readonly connection: BackendConnection
  /** The backend's host name, for "Connected · <host>"; absent when it can't be read. */
  readonly host?: string
  /** Ask again now. */
  readonly check: () => void
}

/** An ask that failed on every retry is down (the pill says so in words); no answer yet is checking. */
export function connectionOf(query: { readonly isError: boolean; readonly isSuccess: boolean }): BackendConnection {
  if (query.isError) return 'down'
  if (query.isSuccess) return 'connected'
  return 'checking'
}

/** The API's host: the configured base URL, or this page's host when it is relative. */
export function apiHost(baseUrl: string, pageUrl: string): string | undefined {
  try {
    return new URL(baseUrl || '/', pageUrl).hostname || undefined
  } catch (error) {
    console.warn('[health] the API base URL is not a URL, so the pill shows no host:', error)
    return undefined
  }
}

export function useBackendHealth(): BackendHealth {
  const query = useQuery({
    queryKey: BACKEND_HEALTH_QUERY_KEY,
    queryFn: () => apiClient.request<unknown>('/health'),
    refetchInterval: BACKEND_HEALTH_POLL_MS,
    // Down only once every retry has failed: the pill says "Not running" until the next answer.
    retry: BACKEND_HEALTH_RETRIES,
    retryDelay: BACKEND_HEALTH_RETRY_DELAY_MS,
  })
  const { refetch } = query
  const check = useCallback(() => {
    void refetch()
  }, [refetch])
  const host = useMemo(
    () => (typeof window === 'undefined' ? undefined : apiHost(apiClient.getBaseUrl(), window.location.href)),
    [],
  )
  return { connection: connectionOf(query), host, check }
}
