'use client'

/**
 * PRD-251B US-B302 and US-B303 — the brand kit's style references and the style profile.
 *
 * Every change to the references answers with the new list, and the server reads the
 * profile again in the background: until that read lands (its `read_at` moves), the query
 * asks again every STYLE_POLL_MS, for at most STYLE_POLL_LIMIT_MS. "Read the references
 * again" waits for the read itself (502 or 504 say why it failed).
 */
import { useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { toast } from 'sonner'

import { useWorkspace } from '@/components/workspace-provider'
import { apiClient } from '@/lib/api-client'
import type { BrandReferenceChange, BrandReferenceStance, BrandStyleResponse } from '@/lib/brand-style-types'

export const STYLE_POLL_MS = 5_000
export const STYLE_POLL_LIMIT_MS = 120_000

export const brandStyleKey = (workspaceId: string | null) => ['brand-kit', workspaceId, 'style'] as const

/** A change the profile has not been read again for yet: the read it replaced, and when. */
export interface PendingRead {
  readBefore: string | null
  at: number
}

/** Whether a background read is still expected for `pending`. */
export function readPending(style: BrandStyleResponse | undefined, pending: PendingRead | null, now = Date.now()): boolean {
  if (!style || !pending || style.references.length === 0) return false
  if (now - pending.at > STYLE_POLL_LIMIT_MS) return false
  return (style.profile?.read_at ?? null) === pending.readBefore
}

function useStyleWrite<Input>(run: (input: Input) => Promise<BrandStyleResponse>, success: string, failure: string, onChanged: () => void) {
  const { workspace } = useWorkspace()
  const client = useQueryClient()
  return useMutation<BrandStyleResponse, Error, Input>({
    mutationFn: run,
    onSuccess: (style) => {
      client.setQueryData(brandStyleKey(workspace?.id ?? null), style)
      onChanged()
      if (success) toast.success(success)
    },
    onError: (error) => {
      toast.error(error.message || failure)
    },
  })
}

export function useBrandStyle() {
  const { workspace } = useWorkspace()
  const workspaceId = workspace?.id ?? null
  const [pending, setPending] = useState<PendingRead | null>(null)
  const query = useQuery<BrandStyleResponse>({
    queryKey: brandStyleKey(workspaceId),
    enabled: !!workspaceId,
    queryFn: () => apiClient.getBrandStyle(),
    refetchInterval: (data) => (readPending(data, pending) ? STYLE_POLL_MS : false),
  })
  const changed = () => setPending({ readBefore: query.data?.profile?.read_at ?? null, at: Date.now() })
  const settled = () => setPending(null)
  return {
    query,
    reading: readPending(query.data, pending),
    upload: useStyleWrite(
      ({ file, note, stance }: { file: File; note: string; stance: BrandReferenceStance }) => apiClient.uploadBrandReference(file, note, stance),
      'Reference added: Auto reads it in a moment', 'Could not add the reference', changed,
    ),
    update: useStyleWrite(
      ({ id, changes }: { id: string; changes: BrandReferenceChange }) => apiClient.updateBrandReference(id, changes),
      '', 'Could not change the reference', changed,
    ),
    remove: useStyleWrite((id: string) => apiClient.deleteBrandReference(id), 'Reference removed', 'Could not remove the reference', changed),
    readAgain: useStyleWrite(() => apiClient.readBrandStyle(), 'Auto read the references again', 'Auto could not read the references', settled),
    sendLiked: useStyleWrite((on: boolean) => apiClient.setBrandStyleSendLiked(on), 'Saved', 'Could not save the setting', () => undefined),
  }
}
