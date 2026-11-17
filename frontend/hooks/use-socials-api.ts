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
  SocialPostMediaLink,
  SocialPostsResponse,
  SocialsUsageResponse,
  SocialToolkitVoicesResponse,
  SocialVoiceSourcesResponse,
  UpdateSocialPostInput,
} from '@/lib/api-client'
import type {
  SocialCampaignApprovalMode,
  SocialCampaignsResponse,
  SocialCampaignWithPosts,
  SocialSeriesApproval,
  SocialSeriesShownPost,
} from '@/lib/api-client'
import { useWorkspace } from '@/components/workspace-provider'
import { activityQueryKeys } from '@/hooks/use-activity-api'
import { anyInFlight } from '@/components/deliverables/socials/socials-status'
import { FILE_OLDER_THAN_WORDS, unsourcedClaimsOf } from '@/components/deliverables/socials/socials-review'

// ============= QUERY KEYS =============

export const socialsQueryKeys = {
  all: (workspaceId: string | null) => ['socials', workspaceId] as const,
  posts: (workspaceId: string | null) => ['socials', workspaceId, 'posts'] as const,
  usage: (workspaceId: string | null) => ['socials', workspaceId, 'usage'] as const,
  voices: (workspaceId: string | null) => ['socials', workspaceId, 'voices'] as const,
  toolkitVoices: (workspaceId: string | null, toolkit: string | null, query: string) =>
    ['socials', workspaceId, 'voices', toolkit, query] as const,
  media: (workspaceId: string | null, postId: string, contentHash: string) =>
    ['socials', workspaceId, 'media', postId, contentHash] as const,
}

/** How often the list refetches while a post renders. */
export const SOCIALS_RENDER_POLL_MS = 5_000

/** The list's refetch interval: poll while any post renders or publishes (US-308), otherwise not at all. */
export function renderPollInterval(data: SocialPostsResponse | undefined): number | false {
  return data && anyInFlight(data.posts) ? SOCIALS_RENDER_POLL_MS : false
}

function useWorkspaceId(): string | null {
  return useWorkspace().workspace?.id ?? null
}

export const HTTP_CONFLICT = 409
const HTTP_UNPROCESSABLE = 422

/** Shown when an action answers 409: the post changed since this screen loaded it. */
export const SOCIAL_POST_CHANGED_MESSAGE =
  'This post changed since you opened it. Review the latest version, then try again.'

/** Shown when Approve answers 409 (US-206): the approver was shown an older version. */
export const SOCIAL_POST_REVIEW_STALE_MESSAGE =
  'This post changed while you were reviewing it — review the new version'

/** The HTTP status apiClient.request() puts on the Error it throws, if any. */
export function httpStatusOf(error: unknown): number | undefined {
  const status = (error as { status?: unknown } | null | undefined)?.status
  return typeof status === 'number' ? status : undefined
}

/** After a post changed: the Socials queries and the Command Center's (its calendar shows the
 * workspace's coming posts) are both read again, so a post moved, approved or deleted here
 * never stays at its old slot there. It was done for a schedule write only. */
export function useInvalidateSocials() {
  const workspaceId = useWorkspaceId()
  const queryClient = useQueryClient()
  return () => Promise.all([
    queryClient.invalidateQueries({ queryKey: socialsQueryKeys.all(workspaceId) }),
    queryClient.invalidateQueries({ queryKey: activityQueryKeys.all }),
  ])
}

/** onError for a post write: a 409 says the post changed and refetches the
 * posts; anything else shows the server's message, or `fallback`. */
export function usePostWriteErrorHandler(fallback: string) {
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

export function useSocialsOn(): { workspaceId: string | null; socialsOn: boolean } {
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

/** The post's media as presigned inline links (D9, S3.4): the exact files the
 * approval view shows, fetched again whenever the post's content changes. */
export function useSocialPostMedia(postId: string, contentHash: string, enabled: boolean) {
  const { workspaceId, socialsOn } = useSocialsOn()
  return useQuery<SocialPostMediaLink[]>({
    queryKey: socialsQueryKeys.media(workspaceId, postId, contentHash),
    enabled: socialsOn && enabled,
    queryFn: () => apiClient.getSocialPostMedia(postId),
    staleTime: 60_000,
    retry: false,
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

/** Turn Socials on or off for this workspace (PRD-251 D1; PRD-251B US-B106), then
 * refetch the workspace so every gate reads the new state without a page reload:
 * the Socials tab switches between the list and the turn-on card, and off means
 * invisible (the Socials calendar, Auto's Socials tools, the marketplace package). */
export function useSetWorkspaceSocials() {
  const { refreshWorkspace } = useWorkspace()
  return useMutation<unknown, Error, boolean>({
    mutationFn: (enabled) => apiClient.setWorkspaceSocialsEnabled(enabled),
    onSuccess: async (_result, enabled) => {
      await refreshWorkspace()
      toast.success(enabled ? 'Socials is on for this workspace' : 'Socials is off for this workspace')
    },
    onError: (error, enabled) => {
      toast.error(error.message || (enabled ? 'Could not turn on Socials' : 'Could not turn off Socials'))
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

/** The review and submit actions, each one server call. Approve has its own
 * hook (useApproveSocialPost): it carries the version on screen (D6). */
export type SocialPostAction =
  | { kind: 'submit' }
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
  request_changes: 'Changes requested',
  reject: 'Rejected',
}

function runAction(postId: string, action: SocialPostAction): Promise<SocialPost> {
  switch (action.kind) {
    case 'submit':
      return apiClient.submitSocialPost(postId)
    case 'request_changes':
      return apiClient.requestSocialPostChanges(postId, action.comment)
    case 'reject':
      return apiClient.rejectSocialPost(postId, action.reason)
  }
}

export const POST_DELETED = 'Post deleted.'

/** Delete a post (owner or admin): the lists refresh; a refusal shows the server's reason. */
export function useDeleteSocialPost() {
  const invalidate = useInvalidateSocials()
  const onError = usePostWriteErrorHandler('The post could not be deleted')
  return useMutation<void, Error, { postId: string }>({
    mutationFn: ({ postId }) => apiClient.deleteSocialPost(postId),
    onSuccess: async () => {
      await invalidate()
      toast.success(POST_DELETED)
    },
    onError,
  })
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

export interface ApproveSocialPostInput {
  postId: string
  /** The content_hash of the version on screen (D6). */
  contentHash: string
  /** The second confirmation: approve with the named unsourced claims (D7). */
  overrideUnsourced?: boolean
  /** Its button said "Approve & post now" (`approvePostsNow`): the server publishes it at once only if it agrees. */
  postNow?: boolean
}

/** The approve's answer: the post, and whether a publish it asked for did not start. */
type ApprovedPost = SocialPost & { posting_not_started?: boolean }

interface ApproveHandlers {
  /** 409: the post changed while it was reviewed; the posts are refetched. */
  onStale: () => void
  /** 422: claims the server counts as unsourced, to confirm by name. */
  onUnsourced: (claims: string[]) => void
}

function approve({ postId, contentHash, overrideUnsourced, postNow }: ApproveSocialPostInput): Promise<ApprovedPost> {
  const options = { ...(overrideUnsourced ? { overrideUnsourced: true } : {}), ...(postNow ? { postNow: true } : {}) }
  return Object.keys(options).length > 0
    ? apiClient.approveSocialPost(postId, contentHash, options)
    : apiClient.approveSocialPost(postId, contentHash)
}

/** What an approval did, said once it is done (one Approve, 17 Nov 2026). */
export const APPROVED = 'Approved'
export const APPROVED_AND_SCHEDULED = 'Approved. It posts at its slot.'
export const APPROVED_AND_PUBLISHING = 'Approved and publishing now. Each channel shows its receipt when it is done.'
/** The approval stands, but the publish its click asked for did not start (the server logged why). */
export const POSTING_NOT_STARTED = "Approved, but posting didn't start. Use Publish now."
/** The same, for a post the server ended failed, each channel never tried: Retry publishes it. */
export const POSTING_NOT_STARTED_RETRY = "Approved, but posting didn't start. Use Retry."

/** The toast for the post as the approve answered it: the server schedules a post into a slot still
 * ahead, publishes one with no slot at once, and otherwise leaves it approved. */
export function approvedMessage(post: Pick<SocialPost, 'status'> | undefined): string {
  if (post?.status === 'publishing') return APPROVED_AND_PUBLISHING
  return post?.status === 'scheduled' ? APPROVED_AND_SCHEDULED : APPROVED
}

/** What to say when the publish an approval asked for did not start (`posting_not_started`), else null. */
export function postingNotStarted(post: Pick<ApprovedPost, 'status' | 'posting_not_started'> | undefined): string | null {
  if (!post?.posting_not_started) return null
  return post.status === 'failed' ? POSTING_NOT_STARTED_RETRY : POSTING_NOT_STARTED
}

/** Approve the version on screen (US-206). A 409 says the post changed while it
 * was reviewed and reloads it; a 422 naming claims asks for the second
 * confirmation; anything else shows the server's message. */
export function useApproveSocialPost({ onStale, onUnsourced }: ApproveHandlers) {
  const invalidate = useInvalidateSocials()
  return useMutation<ApprovedPost, Error, ApproveSocialPostInput>({
    mutationFn: approve,
    onSuccess: async (post) => {
      await invalidate()
      const notStarted = postingNotStarted(post)
      if (notStarted) toast.error(notStarted)
      else toast.success(approvedMessage(post))
    },
    onError: async (error) => {
      const claims = httpStatusOf(error) === HTTP_UNPROCESSABLE ? unsourcedClaimsOf(error) : null
      if (claims && claims.length > 0) {
        onUnsourced(claims)
        return
      }
      if (httpStatusOf(error) === HTTP_CONFLICT && error.message === FILE_OLDER_THAN_WORDS) {
        // PRD-251D US-D108: the post is the one on screen, but its file is older than its words. The
        // server's own sentence, and the post read again so the controls offer Render again.
        toast.error(error.message)
        await invalidate()
        return
      }
      if (httpStatusOf(error) === HTTP_CONFLICT) {
        toast.error(SOCIAL_POST_REVIEW_STALE_MESSAGE)
        onStale()
        await invalidate()
        return
      }
      toast.error(error.message || 'Could not approve the post')
    },
  })
}

// ============= CAMPAIGNS AND SERIES APPROVAL (S2.4, D6) =============

export const socialsCampaignKeys = {
  list: (workspaceId: string | null) => ['socials', workspaceId, 'campaigns'] as const,
  one: (workspaceId: string | null, campaignId: string | null) =>
    ['socials', workspaceId, 'campaigns', campaignId] as const,
}

/** The workspace's campaigns, newest first, each with its post count. */
export function useSocialCampaigns() {
  const { workspaceId, socialsOn } = useSocialsOn()
  return useQuery<SocialCampaignsResponse>({
    queryKey: socialsCampaignKeys.list(workspaceId),
    enabled: socialsOn,
    queryFn: () => apiClient.listSocialCampaigns(),
    staleTime: 15_000,
  })
}

/** One campaign with its posts, read only while `campaignId` is set. */
export function useSocialCampaign(campaignId: string | null) {
  const { workspaceId, socialsOn } = useSocialsOn()
  return useQuery<SocialCampaignWithPosts>({
    queryKey: socialsCampaignKeys.one(workspaceId, campaignId),
    enabled: socialsOn && !!campaignId,
    queryFn: () => apiClient.getSocialCampaign(campaignId as string),
    staleTime: 15_000,
  })
}

export function useCreateSocialCampaign() {
  const invalidate = useInvalidateSocials()
  return useMutation<SocialCampaignWithPosts, Error, { name: string; approval_mode: SocialCampaignApprovalMode }>({
    mutationFn: (input) => apiClient.createSocialCampaign(input),
    onSuccess: async () => {
      await invalidate()
      toast.success('Campaign created')
    },
    onError: (error) => {
      toast.error(error.message || 'Could not create the campaign')
    },
  })
}

export function useUpdateSocialCampaign() {
  const invalidate = useInvalidateSocials()
  return useMutation<
    SocialCampaignWithPosts,
    Error,
    { campaignId: string; changes: { name?: string; approval_mode?: SocialCampaignApprovalMode } }
  >({
    mutationFn: ({ campaignId, changes }) => apiClient.updateSocialCampaign(campaignId, changes),
    onSuccess: async () => {
      await invalidate()
      toast.success('Campaign saved')
    },
    onError: (error) => {
      toast.error(error.message || 'Could not save the campaign')
    },
  })
}

/** Put a post in a campaign, or take it out. */
export function useSocialCampaignPost() {
  const invalidate = useInvalidateSocials()
  return useMutation<SocialPost, Error, { campaignId: string; postId: string; action: 'add' | 'remove' }>({
    mutationFn: ({ campaignId, postId, action }) =>
      action === 'add'
        ? apiClient.addSocialCampaignPost(campaignId, postId)
        : apiClient.removeSocialCampaignPost(campaignId, postId),
    onSuccess: async (_post, { action }) => {
      await invalidate()
      toast.success(action === 'add' ? 'Post added to the campaign' : 'Post taken out of the campaign')
    },
    onError: (error) => {
      toast.error(error.message || 'Could not change the campaign')
    },
  })
}

/** Turn the workspace's series approval on or off, then refetch the workspace. */
export function useSetSeriesApproval() {
  const { refreshWorkspace } = useWorkspace()
  return useMutation<unknown, Error, boolean>({
    mutationFn: (on) => apiClient.setWorkspaceSeriesApproval(on),
    onSuccess: async (_result, on) => {
      await refreshWorkspace()
      toast.success(on ? 'Series approval is on' : 'Series approval is off')
    },
    onError: (error) => {
      toast.error(error.message || 'Could not change series approval')
    },
  })
}

/** Approve the posts on screen as one series (D6). The answer names each post left
 * unapproved and why; the posts and campaigns are refetched either way. */
export function useApproveSocialSeries() {
  const invalidate = useInvalidateSocials()
  return useMutation<SocialSeriesApproval, Error, { campaignId: string; posts: SocialSeriesShownPost[] }>({
    mutationFn: ({ campaignId, posts }) => apiClient.approveSocialCampaignSeries(campaignId, posts),
    onSuccess: async (result) => {
      await invalidate()
      if (result.approved.length > 0) toast.success(`Approved ${result.approved.length} post(s)`)
    },
    onError: async (error) => {
      await invalidate()
      toast.error(error.message || 'Could not approve the series')
    },
  })
}
