'use client'

/**
 * PRD-261 C-05j — the admin's theme revocations and the local edition's revocation sync: reads and
 * writes.
 * =================================================================================================
 *
 * React Query over the W-C4 opening commit's `apiClient` methods. The server decides every rule
 * (`caller_is_admin` first in each handler; a create with a bad digest, id, range, level or reason
 * is a 400 with its sentence; a feed row lifts only while sync is off or stale, else 409; the last
 * local disable of a deactivated Marketplace version never lifts, 409); these hooks only call it
 * and refresh what changed.
 *
 * Create and lift carry step-up on the server (`api/admin_theme_revocations.py` ADMIN_STEP_UP), so
 * they go through `useStepUp` (lib/theme-host/api.ts): the hosted edition shows Clerk's "confirm
 * it's you" and runs the call once more. The sync routes have none: they exist only in the local
 * edition (404 hosted), where step-up passes, so the hosted edition never asks for them.
 *
 * The keys sit under the review queue's (`themeReviewKeys.all`), so a take-down, which writes a
 * `disable` revocation, refreshes this list too.
 */
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { toast } from 'sonner'

import { apiClient } from '@/lib/api-client'
import type {
  ThemeRevocationCreate,
  ThemeRevocationCreateResult,
  ThemeRevocationLiftResult,
  ThemeRevocationSyncStatus,
} from '@/lib/api-client'
import { themeErrorMessage, useStepUp } from '@/lib/theme-host/api'

import { REVOCATION_CREATED, REVOCATION_LIFTED, REVOCATION_REPEATED, SYNC_CHECKED } from './review-words'

export const revocationKeys = {
  all: ['admin-themes', 'revocations'] as const,
  list: ['admin-themes', 'revocations', 'list'] as const,
  sync: ['admin-themes', 'revocations', 'sync'] as const,
}

/** Refusals that mean the list on screen is out of date (404 gone, 409 moved on): refetch it. */
const STALE_STATUSES: ReadonlySet<number> = new Set([404, 409])

function isStale(error: unknown): boolean {
  const status = (error as { status?: unknown } | null | undefined)?.status
  return typeof status === 'number' && STALE_STATUSES.has(status)
}

/** Every revocation, live ones first, then the newest. */
export function useThemeRevocations() {
  return useQuery({
    queryKey: revocationKeys.list,
    queryFn: () => apiClient.getThemeRevocations(),
    retry: false,
  })
}

/** The local edition's sync status; never asked for while `enabled` is false (the hosted edition). */
export function useRevocationSync(enabled: boolean) {
  return useQuery<ThemeRevocationSyncStatus>({
    queryKey: revocationKeys.sync,
    queryFn: () => apiClient.getThemeRevocationSync(),
    enabled,
    retry: false,
  })
}

function useRevocationWrite<Vars, Result>(run: (vars: Vars) => Promise<Result>, done: (result: Result) => string) {
  const client = useQueryClient()
  const refresh = () => client.invalidateQueries({ queryKey: revocationKeys.all })
  return useMutation<Result, unknown, Vars>({
    mutationFn: run,
    onSuccess: async (result) => {
      toast.success(done(result))
      await refresh()
    },
    onError: async (error) => {
      toast.error(themeErrorMessage(error))
      if (isStale(error)) await refresh()
    },
  })
}

/** Revoke a digest, or a theme id with an optional range (admin + step-up); a repeat answers the live row. */
export function useCreateRevocation() {
  const create = useStepUp(apiClient.createThemeRevocation.bind(apiClient))
  return useRevocationWrite<ThemeRevocationCreate, ThemeRevocationCreateResult>(
    (revocation) => create(revocation),
    (result) => (result.created ? REVOCATION_CREATED : REVOCATION_REPEATED),
  )
}

/** Lift a live revocation (admin + step-up). */
export function useLiftRevocation() {
  const lift = useStepUp(apiClient.liftThemeRevocation.bind(apiClient))
  return useRevocationWrite<string, ThemeRevocationLiftResult>((revocationId) => lift(revocationId), () => REVOCATION_LIFTED)
}

/**
 * Check now: one pull of the hosted feed. The answer is the new status; a failed pull is a 502 with
 * the error it recorded, so the status and the rows are read again either way.
 */
export function usePullRevocations() {
  const client = useQueryClient()
  const refresh = () => client.invalidateQueries({ queryKey: revocationKeys.all })
  return useMutation<ThemeRevocationSyncStatus, unknown, void>({
    mutationFn: () => apiClient.pullThemeRevocations(),
    onSuccess: async (status) => {
      client.setQueryData(revocationKeys.sync, status)
      toast.success(SYNC_CHECKED)
      await refresh()
    },
    onError: async (error) => {
      toast.error(themeErrorMessage(error))
      await refresh()
    },
  })
}
