'use client'

/**
 * PRD-261 C-03b — the admin theme review queue's reads and writes.
 * =================================================================
 *
 * React Query over the C-03a `apiClient` methods (W-C2 opening commit). The server decides every
 * rule: `caller_is_admin` in each handler (a non-admin gets 403 with its own sentence), approve
 * refuses a blocked or not-run scan and, on the hosted edition, a version without a passing
 * isolated-job note naming its digest. These hooks only call it and refresh what changed.
 *
 * The writes the server guards with step-up (`report-note`, `approve`, `deactivate`, `signature`;
 * `api/admin_themes.py` ADMIN_STEP_UP) go through `useStepUp` (lib/theme-host/api.ts), so the
 * hosted edition shows Clerk's "confirm it's you" and runs the call once more; the local edition
 * calls straight through. `reject` has no step-up on the server, so none here.
 *
 * The queue is the platform's, not a workspace's: its keys carry no workspace id.
 */

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { toast } from 'sonner'

import { apiClient } from '@/lib/api-client'
import type {
  ThemeApprovalResult,
  ThemeDeactivationResult,
  ThemeRejectionResult,
  ThemeReportNoteOutcome,
  ThemeReportNoteResult,
  ThemeSignatureResult,
} from '@/lib/api-client'
import { themeErrorMessage, useStepUp } from '@/lib/theme-host/api'

export const themeReviewKeys = {
  all: ['admin-themes'] as const,
  pending: ['admin-themes', 'pending'] as const,
  report: (packageId: string) => ['admin-themes', 'report', packageId] as const,
  // Outside `all`: a decision refreshes the queue, and must not make a new review link by the way.
  reviewPackage: (packageId: string) => ['admin-theme-review-package', packageId] as const,
}

export const APPROVED = 'Version approved. Its signature request is open below.'
export const REJECTED = 'Version rejected.'
export const DEACTIVATED = 'Version deactivated: every install falls back to its base on the next load.'
export const NOTE_SAVED = 'Review note saved.'
export const SIGNATURE_SAVED = 'Signature stored.'

/** Refusals that mean what the page shows is out of date (404 gone, 409 moved on): refetch it. */
const STALE_STATUSES: ReadonlySet<number> = new Set([404, 409])

function isStale(error: unknown): boolean {
  const status = (error as { status?: unknown } | null | undefined)?.status
  return typeof status === 'number' && STALE_STATUSES.has(status)
}

/** The versions waiting for review, oldest first. */
export function usePendingThemes() {
  return useQuery({
    queryKey: themeReviewKeys.pending,
    queryFn: () => apiClient.getPendingThemes(),
    retry: false,
  })
}

/** One version's report; idle until a version is chosen. */
export function useThemeReport(packageId: string | null) {
  return useQuery({
    queryKey: themeReviewKeys.report(packageId ?? ''),
    queryFn: () => apiClient.getThemeReport(packageId ?? ''),
    enabled: packageId !== null,
    retry: false,
  })
}

/**
 * C-05j: the kit review job's inputs for a version (C-03c `GET …/review-package`): a 15-minute link
 * to its package, rebuilt from the stored files, and its digest. Asked for only once `requested`
 * (each answer stores a fresh copy), never again on its own: a refetch would only make a new link.
 */
export function useReviewPackage(packageId: string, requested: boolean) {
  return useQuery({
    queryKey: themeReviewKeys.reviewPackage(packageId),
    queryFn: () => apiClient.getThemeReviewPackage(packageId),
    enabled: requested,
    retry: false,
    staleTime: Infinity,
    refetchOnWindowFocus: false,
  })
}

function useReviewWrite<Vars, Result>(run: (vars: Vars) => Promise<Result>, done: string) {
  const client = useQueryClient()
  const refresh = () => client.invalidateQueries({ queryKey: themeReviewKeys.all })
  return useMutation<Result, unknown, Vars>({
    mutationFn: run,
    onSuccess: async () => {
      toast.success(done)
      await refresh()
    },
    onError: async (error) => {
      toast.error(themeErrorMessage(error))
      if (isStale(error)) await refresh()
    },
  })
}

/** Approve a pending version (admin + step-up); the server refuses a blocked or not-run scan. */
export function useApproveTheme() {
  const approve = useStepUp(apiClient.approveTheme.bind(apiClient))
  return useReviewWrite<string, ThemeApprovalResult>((packageId) => approve(packageId), APPROVED)
}

/** Reject a pending version with the reason the submitter sees (admin). */
export function useRejectTheme() {
  return useReviewWrite<{ packageId: string; reason: string }, ThemeRejectionResult>(
    ({ packageId, reason }) => apiClient.rejectTheme(packageId, reason),
    REJECTED,
  )
}

/** Take an approved version down; its digest gets a `disable` revocation (admin + step-up). */
export function useDeactivateTheme() {
  const deactivate = useStepUp(apiClient.deactivateTheme.bind(apiClient))
  return useReviewWrite<{ packageId: string; reason: string }, ThemeDeactivationResult>(
    ({ packageId, reason }) => deactivate(packageId, reason),
    DEACTIVATED,
  )
}

/** Store the isolated review job's summary the reviewer pasted (admin + step-up). */
export function useAddReportNote() {
  const addNote = useStepUp(apiClient.addThemeReportNote.bind(apiClient))
  return useReviewWrite<{ packageId: string; summary: string; outcome: ThemeReportNoteOutcome }, ThemeReportNoteResult>(
    ({ packageId, summary, outcome }) => addNote(packageId, { summary, outcome }),
    NOTE_SAVED,
  )
}

/** Store the kit sign workflow's signature for the version's open request (admin + step-up). */
export function useStoreThemeSignature() {
  const store = useStepUp(apiClient.storeThemeSignature.bind(apiClient))
  return useReviewWrite<{ packageId: string; requestId: string; signature: string }, ThemeSignatureResult>(
    ({ packageId, requestId, signature }) => store(packageId, requestId, signature),
    SIGNATURE_SAVED,
  )
}
