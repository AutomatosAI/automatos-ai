'use client'

import { useCallback, useMemo } from 'react'
import { useQuery, type UseQueryResult } from '@tanstack/react-query'
import { apiClient } from '@/lib/api-client'

/**
 * PRD-260 T-004 — is the backend answering? Play's connection pill asks
 * GET /health through apiClient, again every BACKEND_HEALTH_POLL_MS. Ported
 * from Automatos Play's useBackendStatus, whose proxy check becomes the core's
 * /health.
 *
 * One query, one poll (P260-FX-27, M88): every caller reads the same cache
 * (BACKEND_HEALTH_QUERY_KEY, fetchBackendHealth), Settings → Session mode
 * included (useBackendHealthQuery), and only a caller that asks for `poll`
 * starts a timer: Play's AsleepGate, mounted with the frame on every Play
 * page. react-query runs one timer per polling observer, each in its own
 * phase, so a second poller would be a second request every period. An answer
 * stays fresh until the poll asks again, so a reader that mounts in between
 * reads it instead of asking.
 *
 * One failed request is never the answer (P260-RVW-1): Play's frame swaps
 * its page for "Automatos is asleep" on 'down' (AsleepGate), so a blip — a
 * wake from sleep, a Wi-Fi hand-over, one 502 at the edge — must not unmount
 * it. The query asks again BACKEND_HEALTH_RETRIES times, BACKEND_HEALTH_RETRY_DELAY_MS
 * apart, before it reports an error; while those retries are pending after an
 * earlier answer react-query keeps the query a success, so it stays 'connected'.
 *
 * A frozen backend (it takes the connection and never answers) is down too
 * (P260-FX-21, M43): each ask gives up after BACKEND_HEALTH_TIMEOUT_MS and
 * counts as a failure through the same retries. And once every retry has
 * failed the connection stays 'down' until the next answer (M89), also while
 * it is asked again and on the first render of a hook that mounts then.
 */

/** How often the poll owner (Play's AsleepGate) asks again. */
export const BACKEND_HEALTH_POLL_MS = 30_000
/** How many times a failed ask is repeated before the backend is down. */
export const BACKEND_HEALTH_RETRIES = 3
/** The wait before each repeat: the waits span 7.5 s, enough for a blip. */
export const BACKEND_HEALTH_RETRY_DELAY_MS = 2_500
/**
 * How long one ask may take before it counts as failed. It covers apiClient's
 * own wait of up to 2 s for a Clerk token, which runs inside the same window,
 * and 3 s for the answer. A frozen backend reads as down once four asks have
 * timed out with the waits between them: 4 × 5 s + 3 × 2.5 s = 27.5 s, inside
 * one BACKEND_HEALTH_POLL_MS. The asleep screen's countdown holds while a
 * check is in flight (useRecheckCountdown), so its recheck never cuts in.
 */
export const BACKEND_HEALTH_TIMEOUT_MS = 5_000
export const BACKEND_HEALTH_QUERY_KEY = ['health', 'connection'] as const

export type BackendConnection = 'checking' | 'connected' | 'down'

export interface BackendHealthOptions {
  /** Ask again every BACKEND_HEALTH_POLL_MS: the page's one poll owner (AsleepGate). Others read. */
  readonly poll?: boolean
}

export interface BackendHealth {
  readonly connection: BackendConnection
  /** The backend's host name, for "Connected · <host>"; absent when it can't be read. */
  readonly host?: string
  /** An ask (or one of its retries) is in flight: the asleep screen says Checking…. */
  readonly checking: boolean
  /** Ask again now. */
  readonly check: () => void
}

/** The parts of react-query's result the connection is read from. */
export interface HealthQueryState {
  readonly isError: boolean
  readonly isSuccess: boolean
  readonly errorUpdatedAt: number
  readonly dataUpdatedAt: number
}

/**
 * An ask that failed on every retry is down (the pill says so in words) until
 * the next answer; no answer yet is checking. react-query reads a query that has
 * never answered as loading again while it asks again, and so does the first
 * render of a hook that mounts then: its last answer, a failure, still decides.
 */
export function connectionOf(query: HealthQueryState): BackendConnection {
  if (query.isError) return 'down'
  if (query.isSuccess) return 'connected'
  return query.errorUpdatedAt > query.dataUpdatedAt ? 'down' : 'checking'
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

/** GET /health. A frozen backend never answers: the ask gives up, and fails like any other. */
export function fetchBackendHealth<T = unknown>(): Promise<T> {
  return apiClient.request<T>('/health', { signal: AbortSignal.timeout(BACKEND_HEALTH_TIMEOUT_MS) })
}

/** The one GET /health query, as react-query reports it; `T` is the answer's shape the caller reads. */
export function useBackendHealthQuery<T = unknown>({ poll = false }: BackendHealthOptions = {}): UseQueryResult<T> {
  return useQuery({
    queryKey: BACKEND_HEALTH_QUERY_KEY,
    queryFn: () => fetchBackendHealth<T>(),
    refetchInterval: poll ? BACKEND_HEALTH_POLL_MS : false,
    // Fresh until the poll asks again: a caller mounting in between reads the answer.
    staleTime: BACKEND_HEALTH_POLL_MS,
    // Down only once every retry has failed: the pill says "Not running" until the next answer.
    retry: BACKEND_HEALTH_RETRIES,
    retryDelay: BACKEND_HEALTH_RETRY_DELAY_MS,
  })
}

export function useBackendHealth({ poll = false }: BackendHealthOptions = {}): BackendHealth {
  const query = useBackendHealthQuery({ poll })
  const { refetch } = query
  const check = useCallback(() => {
    void refetch()
  }, [refetch])
  const host = useMemo(
    () => (typeof window === 'undefined' ? undefined : apiHost(apiClient.getBaseUrl(), window.location.href)),
    [],
  )
  return { connection: connectionOf(query), checking: query.isFetching, host, check }
}
