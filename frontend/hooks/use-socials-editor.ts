/**
 * Socials editor hooks (PRD-251B US-B109)
 * =======================================
 *
 * What the post editor reads (the template gallery for a format, the workspace's image
 * and video Deliverables, what footage a template's slots can be filled with) and what it
 * writes, each through apiClient: Save draft creates the post (POST /posts) or edits it
 * (PATCH), then sets its channels (PUT /targets) and its slot (PUT /slot); Render preview
 * and Submit for approval save first; Redraft with Auto asks the composer with the
 * editor's choices; an upload or a Library pick saves a new post first, then sets its
 * media. The server checks everything again.
 */
import { useMutation, useQuery } from '@tanstack/react-query'
import { toast } from 'sonner'

import { apiClient } from '@/lib/api-client'
import type {
  CreateSocialPostInput,
  DeliverableSummary,
  SocialComposeInput,
  SocialComposeProposal,
  SocialFootageSources,
  SocialPost,
  SocialPostTargetInput,
  SocialTemplateSummary,
} from '@/lib/api-client'
import { useInvalidateSocials, useSocialsOn } from '@/hooks/use-socials-api'
import { useWorkspace } from '@/components/workspace-provider'

export const socialsEditorKeys = {
  brandName: (workspaceId: string | null) => ['socials', workspaceId, 'brand-name'] as const,
  templates: (workspaceId: string | null, format: string | null) => ['socials', workspaceId, 'templates', format] as const,
  library: (workspaceId: string | null) => ['socials', workspaceId, 'library'] as const,
  footage: (workspaceId: string | null) => ['socials', workspaceId, 'footage'] as const,
}

export const LIBRARY_KINDS = ['image', 'video'] as const
export const LIBRARY_LIMIT = 24

/** The name the preview's post frame shows (PRD-251B US-B110): the brand kit's, else the
 * workspace's. */
export function useBrandName(): string {
  const { workspace } = useWorkspace()
  const { workspaceId, socialsOn } = useSocialsOn()
  const { data } = useQuery<string>({
    queryKey: socialsEditorKeys.brandName(workspaceId),
    enabled: socialsOn,
    queryFn: async () => (await apiClient.get<{ name?: string }>('/api/documents/brand-kit'))?.name ?? '',
    staleTime: 5 * 60_000,
  })
  return data || workspace?.name || 'Your brand'
}

/** The workspace's templates for a format (GET /api/socials/templates?format=). */
export function useSocialTemplates(format: string | null) {
  const { workspaceId, socialsOn } = useSocialsOn()
  return useQuery<SocialTemplateSummary[]>({
    queryKey: socialsEditorKeys.templates(workspaceId, format),
    enabled: socialsOn,
    queryFn: () => apiClient.listSocialTemplates(format ?? undefined),
    staleTime: 60_000,
  })
}

/** The workspace's image and video Deliverables, newest first (the Library). */
export function useSocialLibrary(enabled: boolean) {
  const { workspaceId, socialsOn } = useSocialsOn()
  return useQuery<DeliverableSummary[]>({
    queryKey: socialsEditorKeys.library(workspaceId),
    enabled: socialsOn && enabled,
    queryFn: async () => {
      const pages = await Promise.all(LIBRARY_KINDS.map((kind) => apiClient.listDeliverables({ artifact_type: kind, limit: LIBRARY_LIMIT })))
      return pages.flatMap((page) => page.deliverables).sort((a, b) => b.created_at.localeCompare(a.created_at))
    },
    staleTime: 30_000,
  })
}

/** What footage a template's slots can be filled with here (GET /api/socials/footage). */
export function useSocialFootageSources(enabled: boolean) {
  const { workspaceId, socialsOn } = useSocialsOn()
  return useQuery<SocialFootageSources>({
    queryKey: socialsEditorKeys.footage(workspaceId),
    enabled: socialsOn && enabled,
    queryFn: () => apiClient.getSocialFootageSources(),
    staleTime: 60_000,
  })
}

export interface EditorSave {
  /** The post being edited, or null for a new one (created on the first save). */
  postId: string | null
  fields: CreateSocialPostInput
  targets: SocialPostTargetInput[]
  /** The slot to set (ISO, UTC) and its zone; undefined leaves the slot as it is. */
  slot?: { plannedFor: string | null; timezone: string }
}

/** Save the editor: create or edit the post, then its channels, then its slot. */
export async function saveEditor({ postId, fields, targets, slot }: EditorSave): Promise<SocialPost> {
  let saved = postId ? await apiClient.updateSocialPost(postId, fields) : await apiClient.createSocialPost(fields)
  if (targets.length > 0 || postId) saved = await apiClient.setSocialPostTargets(saved.id, targets)
  if (slot) saved = await apiClient.setSocialPostSlot(saved.id, slot.plannedFor, slot.timezone)
  return saved
}

function useEditorMutation<Input>(run: (input: Input) => Promise<SocialPost>, success: string | null, failure: string) {
  const invalidate = useInvalidateSocials()
  return useMutation<SocialPost, Error, Input>({
    mutationFn: run,
    onSuccess: async () => {
      await invalidate()
      if (success) toast.success(success)
    },
    onError: async (error) => {
      await invalidate()
      toast.error(error.message || failure)
    },
  })
}

export function useSaveEditor() {
  return useEditorMutation<EditorSave>(saveEditor, 'Draft saved', 'Could not save the draft')
}

/** Save, then render: a video at half resolution (the preview), an image for real. */
export function useRenderEditorPreview() {
  return useEditorMutation<EditorSave & { video: boolean }>(
    async ({ video, ...save }) => apiClient.renderSocialPost((await saveEditor(save)).id, video ? { preview: true } : {}),
    null,
    'Could not render the preview',
  )
}

export function useSubmitEditor() {
  return useEditorMutation<EditorSave>(
    async (save) => apiClient.submitSocialPost((await saveEditor(save)).id),
    'Sent for approval',
    'Could not send the post for approval',
  )
}

/** The post's visual from a file: a new post is saved first. */
export function useUploadEditorMedia() {
  return useEditorMutation<EditorSave & { file: File }>(
    async ({ file, ...save }) => apiClient.uploadSocialPostMedia((await saveEditor(save)).id, file),
    'File uploaded',
    'Could not upload the file',
  )
}

/** PRD-251B US-B305: four AI options for an image slot, made in the background; the draft is saved first. */
export function useMakeAiOptions() {
  return useEditorMutation<EditorSave & { imageSlot: string; prompt: string }>(
    async ({ imageSlot, prompt, ...save }) => apiClient.makeSocialAiOptions((await saveEditor(save)).id, imageSlot, prompt),
    'Making four options: they appear here in a minute',
    'Could not make AI options',
  )
}

/** The option picked becomes the slot's file. A render setting: an approval stands. */
export function usePickAiOption() {
  return useEditorMutation<{ postId: string; slot: string; name: string }>(
    ({ postId, slot, name }) => apiClient.pickSocialAiOption(postId, slot, name),
    'Option chosen',
    'Could not use that option',
  )
}

/** The post's visual from the Library: one of the workspace's Deliverables. */
export function usePickLibraryMedia() {
  return useEditorMutation<EditorSave & { item: DeliverableSummary }>(
    async ({ item, ...save }) => {
      const saved = await saveEditor(save)
      const format = item.artifact_type === 'video' ? 'video' : 'image'
      return apiClient.updateSocialPost(saved.id, { media: { original: [item.id] }, template_id: null, length_seconds: null, format })
    },
    'Visual chosen from the Library',
    'Could not use that file',
  )
}

/** Redraft with Auto: the composer's proposal for the editor's brief and choices (not saved). */
export function useRedraft() {
  return useMutation<SocialComposeProposal, Error, SocialComposeInput>({
    mutationFn: (input) => apiClient.composeSocialPost(input),
    onError: (error) => {
      toast.error(error.message || 'Auto could not redraft the post')
    },
  })
}
