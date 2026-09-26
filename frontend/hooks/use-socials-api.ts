/**
 * Socials API hooks (PRD-251 Wave 0; rendering and render minutes, Wave 1 S1.1c)
 * ==============================================================================
 *
 * React Query over the apiClient socials methods — the workspace switch and the
 * post lifecycle (create, edit, submit, approve, request changes, reject,
 * render), this month's render minutes, and (S1.5) the voices a post can be
 * spoken with. While any post renders, the list
 * polls until the render ends (needs approval, or failed). The
 * server enforces every rule (the gate, the role, the status machine, D6's
 * approval hash); these hooks only call it and refresh the list. Approve sends
 * the content_hash of the post on screen, and every write commits only if the
 * post is still the version the server loaded. So a 409 from any write means
 * the post changed since it was fetched: the list is refetched and the user
 * looks again.
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
  SocialsUsageResponse,
  SocialToolkitVoicesResponse,
  SocialVoiceSourcesResponse,
  UpdateSocialPostInput,
} from '@/lib/api-client'
import { useWorkspace } from '@/components/workspace-provider'
import { anyRendering } from '@/components/deliverables/socials/socials-status'

// ============= QUERY KEYS =============

export const socialsQueryKeys = {
  all: (workspaceId: string | null) => ['socials', workspaceId] as const,
  posts: (workspaceId: string | null) => ['socials', workspaceId, 'posts'] as const,
  usage: (workspaceId: string | null) => ['socials', workspaceId, 'usage'] as const,
  voices: (workspaceId: string | null) => ['socials', workspaceId, 'voices'] as const,
  toolkitVoices: (workspaceId: string | null, toolkit: string | null, query: string) =>
    ['socials', workspaceId, 'voices', toolkit, query] as const,
}

/** How often the list refetches while a post renders. */
export const SOCIALS_RENDER_POLL_MS = 5_000

/** The list's refetch interval: poll while any post renders, otherwise not at all. */
export function renderPollInterval(data: SocialPostsResponse | undefined): number | false {
  return data && anyRendering(data.posts) ? SOCIALS_RENDER_POLL_MS : false
}

function useWorkspaceId(): string | null {
  return useWorkspace().workspace?.id ?? null
}

const HTTP_CONFLICT = 409

/** Shown when an action answers 409: the post changed since this screen loaded it. */
export const SOCIAL_POST_CHANGED_MESSAGE =
  'This post changed since you opened it. Review the latest version, then try again.'

/** The HTTP status apiClient.request() puts on the Error it throws, if any. */
function httpStatusOf(error: unknown): number | undefined {
  const status = (error as { status?: unknown } | null | undefined)?.status
  return typeof status === 'number' ? status : undefined
}

function useInvalidateSocials() {
  const workspaceId = useWorkspaceId()
  const queryClient = useQueryClient()
  return () => queryClient.invalidateQueries({ queryKey: socialsQueryKeys.all(workspaceId) })
}

/** onError for a post write: a 409 says the post changed and refetches the
 * posts; anything else shows the server's message, or `fallback`. */
function usePostWriteErrorHandler(fallback: string) {
  const invalidate = useInvalidateSocials()
  return async (error: Error) => {
    if (httpStatusOf(error) === HTTP_CONFLICT) {
      toast.error(SOCIAL_POST_CHANGED_MESSAGE)
      await invalidate()
      return
    }
    toast.error(error.message || fallback)
  }
}

// ============= QUERY HOOKS =============

function useSocialsOn(): { workspaceId: string | null; socialsOn: boolean } {
  const { workspace } = useWorkspace()
  const workspaceId = workspace?.id ?? null
  const socialsOn = !!workspace?.socials?.available && !!workspace?.socials?.enabled
  return { workspaceId, socialsOn: socialsOn && !!workspaceId }
}

/** The workspace's posts, newest first. Fetched only while both switches are
 * on — the gated routes answer 404 otherwise (D1). Polls while a post renders. */
export function useSocialPosts() {
  const { workspaceId, socialsOn } = useSocialsOn()
  return useQuery<SocialPostsResponse>({
    queryKey: socialsQueryKeys.posts(workspaceId),
    enabled: socialsOn,
    queryFn: () => apiClient.listSocialPosts(),
    staleTime: 15_000,
    refetchInterval: renderPollInterval,
  })
}

/** This month's render minutes used and the plan's quota (S1.1c). */
export function useSocialsUsage() {
  const { workspaceId, socialsOn } = useSocialsOn()
  return useQuery<SocialsUsageResponse>({
    queryKey: socialsQueryKeys.usage(workspaceId),
    enabled: socialsOn,
    queryFn: () => apiClient.getSocialsUsage(),
    staleTime: 30_000,
  })
}

/** What a post can be spoken with (S1.5, D11): Kokoro, the voice toolkits the
 * workspace has connected in Composio, and the ones to connect. */
export function useSocialVoiceSources() {
  const { workspaceId, socialsOn } = useSocialsOn()
  return useQuery<SocialVoiceSourcesResponse>({
    queryKey: socialsQueryKeys.voices(workspaceId),
    enabled: socialsOn,
    queryFn: () => apiClient.getSocialVoiceSources(),
    staleTime: 60_000,
  })
}

/** A connected voice toolkit's voices, read only while `toolkit` is set. */
export function useSocialToolkitVoices(toolkit: string | null, query: string) {
  const { workspaceId, socialsOn } = useSocialsOn()
  return useQuery<SocialToolkitVoicesResponse>({
    queryKey: socialsQueryKeys.toolkitVoices(workspaceId, toolkit, query),
    enabled: socialsOn && !!toolkit,
    queryFn: () => apiClient.listSocialToolkitVoices(toolkit as string, query),
    staleTime: 60_000,
    retry: false,
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
  const onError = usePostWriteErrorHandler('Could not save the post')
  return useMutation<SocialPost, Error, { postId: string; changes: UpdateSocialPostInput }>({
    mutationFn: ({ postId, changes }) => apiClient.updateSocialPost(postId, changes),
    onSuccess: async () => {
      await invalidate()
      toast.success('Post saved')
    },
    onError,
  })
}

/** The review and submit actions, each one server call. Approve carries the
 * content_hash of the version on screen (D6). */
export type SocialPostAction =
  | { kind: 'submit' }
  | { kind: 'approve'; contentHash: string }
  | { kind: 'request_changes'; comment: string }
  | { kind: 'reject'; reason?: string }

/** Start a render (S1.1c). A 429 (no render minutes left) or 503 (no renderer)
 * shows the server's reason; a 409 means the post changed, as for any write. */
export function useRenderSocialPost() {
  const invalidate = useInvalidateSocials()
  const onError = usePostWriteErrorHandler('Could not start the render')
  return useMutation<SocialPost, Error, { postId: string }>({
    mutationFn: ({ postId }) => apiClient.renderSocialPost(postId),
    onSuccess: async () => {
      await invalidate()
      toast.success('Rendering started')
    },
    onError,
  })
}

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
      return apiClient.approveSocialPost(postId, action.contentHash)
    case 'request_changes':
      return apiClient.requestSocialPostChanges(postId, action.comment)
    case 'reject':
      return apiClient.rejectSocialPost(postId, action.reason)
  }
}

export function useSocialPostAction() {
  const invalidate = useInvalidateSocials()
  const onError = usePostWriteErrorHandler('The action failed')
  return useMutation<SocialPost, Error, { postId: string; action: SocialPostAction }>({
    mutationFn: ({ postId, action }) => runAction(postId, action),
    onSuccess: async (_post, { action }) => {
      await invalidate()
      toast.success(ACTION_DONE[action.kind])
    },
    onError,
  })
}
