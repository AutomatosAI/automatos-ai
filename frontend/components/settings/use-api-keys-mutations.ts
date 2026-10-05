'use client'

/**
 * Mutations shared by the "Your API Keys" list (PRD-54, issue #830):
 * delete, test (`POST /api/keys/{id}/test`), and the per-provider BYOK
 * toggle. Split out of `ApiKeysSettingsTab` to keep that component under
 * the 150-line React component limit.
 */
import { useState } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { toast } from 'sonner'
import { apiClient } from '@/lib/api-client'
import type { ApiKeyTestResult } from './api-keys-types'

export function useApiKeysMutations() {
  const queryClient = useQueryClient()
  const [testingKeyId, setTestingKeyId] = useState<number | null>(null)

  const deleteKeyMutation = useMutation({
    mutationFn: (id: number) => apiClient.delete(`/api/keys/${id}`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['api-keys'] })
      toast.success('API key deleted')
    },
    onError: (err: Error) => {
      toast.error(`Failed to delete key: ${err.message}`)
    },
  })

  const testKeyMutation = useMutation({
    mutationFn: (id: number) => {
      setTestingKeyId(id)
      return apiClient.post<ApiKeyTestResult>(`/api/keys/${id}/test`)
    },
    onSuccess: (result) => {
      if (result.valid) {
        toast.success('Key is valid')
      } else {
        toast.error(result.message || 'Key test failed')
      }
      setTestingKeyId(null)
    },
    onError: (err: Error) => {
      toast.error(`Test failed: ${err.message}`)
      setTestingKeyId(null)
    },
  })

  const saveByokMutation = useMutation({
    mutationFn: (overrides: Record<string, boolean>) =>
      apiClient.put('/api/workspaces/current/byok-preferences', { byok_overrides: overrides }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['byok-preferences'] })
    },
    onError: (err: Error) => {
      toast.error(`Failed to save preference: ${err.message}`)
    },
  })

  return { deleteKeyMutation, testKeyMutation, saveByokMutation, testingKeyId }
}
