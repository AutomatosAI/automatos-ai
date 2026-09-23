/**
 * Socials API hooks (PRD-251 Wave 0)
 * ==================================
 *
 * React Query over the apiClient socials methods — the workspace switch and the
 * post lifecycle (create, edit, submit, approve, request changes, reject). The
 * server enforces every rule (the gate, the role, the status machine, D6's
 * approval hash); these hooks only call it and refresh the list.
 *
 * Query keys are scoped by workspace id so switching workspaces clears cache.
 */

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { toast } from 'sonner'

import { apiClient } from '@/lib/api-client'
import type {
  CreateSocialPostInput,
  SocialPost,
  SocialPostsResponse,
  UpdateSocialPostInput,
} from '@/lib/api-client'
import { useWorkspace } from '@/components/workspace-provider'

// ============= QUERY KEYS =============

export const socialsQueryKeys = {
  all: (workspaceId: string | null) => ['socials', workspaceId] as const,
  posts: (workspaceId: string | null) => ['socials', workspaceId, 'posts'] as const,
}

function useWorkspaceId(): string | null {
  return useWorkspace().workspace?.id ?? null
}

function useInvalidateSocials() {
  const workspaceId = useWorkspaceId()
  const queryClient = useQueryClient()
  return () => queryClient.invalidateQueries({ queryKey: socialsQueryKeys.all(workspaceId) })
}

// ============= QUERY HOOKS =============

/** The workspace's posts, newest first. Fetched only while both switches are
 * on — the gated routes answer 404 otherwise (D1). */
export function useSocialPosts() {
  const { workspace } = useWorkspace()
  const workspaceId = workspace?.id ?? null
  const socialsOn = !!workspace?.socials?.available && !!workspace?.socials?.enabled
  return useQuery<SocialPostsResponse>({
    queryKey: socialsQueryKeys.posts(workspaceId),
    enabled: socialsOn && !!workspaceId,
    queryFn: () => apiClient.listSocialPosts(),
    staleTime: 15_000,
  })
}

// ============= MUTATION HOOKS =============

/** Turn Socials on for this workspace, then refetch the workspace so the tab
 * switches to the list without a page reload. */
export function useEnableSocials() {
  const { refreshWorkspace } = useWorkspace()
  return useMutation<unknown, Error, void>({
    mutationFn: () => apiClient.setWorkspaceSocialsEnabled(true),
    onSuccess: async () => {
      await refreshWorkspace()
      toast.success('Socials is on for this workspace')
    },
    onError: (error) => {
      toast.error(error.message || 'Could not turn on Socials')
    },
  })
}

export function useCreateSocialPost() {
  const invalidate = useInvalidateSocials()
  return useMutation<SocialPost, Error, CreateSocialPostInput>({
    mutationFn: (input) => apiClient.createSocialPost(input),
    onSuccess: async () => {
      await invalidate()
      toast.success('Draft created')
    },
    onError: (error) => {
      toast.error(error.message || 'Could not create the draft')
    },
  })
}

export function useUpdateSocialPost() {
  const invalidate = useInvalidateSocials()
  return useMutation<SocialPost, Error, { postId: string; changes: UpdateSocialPostInput }>({
    mutationFn: ({ postId, changes }) => apiClient.updateSocialPost(postId, changes),
    onSuccess: async () => {
      await invalidate()
      toast.success('Post saved')
    },
    onError: (error) => {
      toast.error(error.message || 'Could not save the post')
    },
  })
}

/** The review and submit actions, each one server call. */
export type SocialPostAction =
  | { kind: 'submit' }
  | { kind: 'approve' }
  | { kind: 'request_changes'; comment: string }
  | { kind: 'reject'; reason?: string }

const ACTION_DONE: Record<SocialPostAction['kind'], string> = {
  submit: 'Sent for approval',
  approve: 'Approved',
  request_changes: 'Changes requested',
  reject: 'Rejected',
}

function runAction(postId: string, action: SocialPostAction): Promise<SocialPost> {
  switch (action.kind) {
    case 'submit':
      return apiClient.submitSocialPost(postId)
    case 'approve':
      return apiClient.approveSocialPost(postId)
    case 'request_changes':
      return apiClient.requestSocialPostChanges(postId, action.comment)
    case 'reject':
      return apiClient.rejectSocialPost(postId, action.reason)
  }
}

export function useSocialPostAction() {
  const invalidate = useInvalidateSocials()
  return useMutation<SocialPost, Error, { postId: string; action: SocialPostAction }>({
    mutationFn: ({ postId, action }) => runAction(postId, action),
    onSuccess: async (_post, { action }) => {
      await invalidate()
      toast.success(ACTION_DONE[action.kind])
    },
    onError: (error) => {
      toast.error(error.message || 'The action failed')
    },
  })
}
