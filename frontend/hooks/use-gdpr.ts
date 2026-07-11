/**
 * GDPR self-service hooks — PRD-196 S7 (P2-15, governance I.4).
 *
 * The ws-admin buttons the GDPR API already guards: export the bundle,
 * erase a data subject (with the honest gaps + untagged-history report), and
 * erase the whole workspace (the API enforces the typed confirmation echo).
 */

import { useMutation } from '@tanstack/react-query'
import { apiClient } from '@/lib/api-client'
import type { GdprSubjectErasureResult, GdprWorkspaceErasureResult } from '@/lib/api-client'

export function useGdprEraseSubject() {
  return useMutation<GdprSubjectErasureResult, Error, string>({
    mutationFn: (subjectId) => apiClient.eraseGdprSubject(subjectId),
  })
}

export function useGdprEraseWorkspace() {
  return useMutation<GdprWorkspaceErasureResult, Error, string>({
    mutationFn: (confirmWorkspaceId) => apiClient.eraseGdprWorkspace(confirmWorkspaceId),
  })
}
