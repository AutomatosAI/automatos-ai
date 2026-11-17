/**
 * Socials instant preview hooks (PRD-251D US-D201, E8)
 * ====================================================
 *
 * What the editor's instant preview of an image post reads, each through apiClient:
 * - the template as one self-contained page for the words on screen
 *   (POST /api/socials/templates/{id}/preview-page); the editor keeps the last page on screen
 *   while the next one loads (studio/use-instant-preview.ts);
 * - the brand kit's font faces (GET /api/socials/brand-kit/preview-fonts), fetched once per
 *   kit: keyed on the page's `fonts_version`, and outside the 'socials' keys, which every
 *   post save refreshes, so their megabytes are not fetched again with each save;
 * - each photo the post holds in a photo spot, read through the post's media route and handed
 *   back as a data: URI. The page runs in a frame with no same origin, which can load neither a
 *   route behind sign-in nor a blob: URL of the app's.
 *
 * Query keys are scoped by workspace id so switching workspaces clears cache.
 */
import { useQuery } from '@tanstack/react-query'

import { apiClient } from '@/lib/api-client'
import type { SocialPreviewFonts, SocialPreviewPage, SocialPreviewPageInput } from '@/lib/api-client'
import { useSocialsOn } from '@/hooks/use-socials-api'

export const socialsPreviewKeys = {
  page: (workspaceId: string | null, templateId: string, input: SocialPreviewPageInput | null) =>
    ['socials', workspaceId, 'preview-page', templateId, input] as const,
  fonts: (workspaceId: string | null, version: string) => ['socials-preview-fonts', workspaceId, version] as const,
  photos: (workspaceId: string | null, postId: string, files: string) => ['socials-preview-photos', workspaceId, postId, files] as const,
}

// A page per pause in typing: each is let go soon after the words move on (it can carry the logos).
const PAGE_KEPT_MS = 30_000

/** A photo spot and the name of the post's file in it. */
export interface PreviewPhotoFile {
  slot: string
  name: string
}

/** The post's media route for one of its files (GET /api/socials/posts/{id}/media/{name}). */
export function postMediaPath(postId: string, name: string): string {
  return `/api/socials/posts/${postId}/media/${encodeURIComponent(name)}`
}

/** A file's bytes as a data: URI. */
export function blobToDataUri(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(String(reader.result))
    reader.onerror = () => reject(reader.error ?? new Error('The photo could not be read.'))
    reader.readAsDataURL(blob)
  })
}

/** The page for `input`, asked for only while both are set. A refusal is not retried: it says why. */
export function useSocialPreviewPage(templateId: string | null, input: SocialPreviewPageInput | null) {
  const { workspaceId, socialsOn } = useSocialsOn()
  return useQuery<SocialPreviewPage>({
    queryKey: socialsPreviewKeys.page(workspaceId, templateId ?? '', input),
    enabled: socialsOn && !!templateId && !!input,
    queryFn: async () => apiClient.getSocialTemplatePreviewPage(templateId as string, input as SocialPreviewPageInput),
    retry: false,
    staleTime: PAGE_KEPT_MS,
    cacheTime: PAGE_KEPT_MS,
  })
}

/** The kit's font faces, once per `version` (the page's `fonts_version`). */
export function useSocialPreviewFonts(version: string | null) {
  const { workspaceId, socialsOn } = useSocialsOn()
  return useQuery<SocialPreviewFonts>({
    queryKey: socialsPreviewKeys.fonts(workspaceId, version ?? ''),
    enabled: socialsOn && !!version,
    queryFn: async () => apiClient.getSocialPreviewFonts(),
    staleTime: Infinity,
    refetchOnWindowFocus: false,
    retry: false,
  })
}

/** Each file in `files` as a data: URI by its spot, read once per post and set of files. */
export function useSocialPreviewPhotos(postId: string | null, files: ReadonlyArray<PreviewPhotoFile>) {
  const { workspaceId, socialsOn } = useSocialsOn()
  const named = files.map((file) => `${file.slot}=${file.name}`).join('&')
  return useQuery<Record<string, string>>({
    queryKey: socialsPreviewKeys.photos(workspaceId, postId ?? '', named),
    enabled: socialsOn && !!postId && files.length > 0,
    queryFn: async () => {
      const read = await Promise.all(files.map(async (file) => {
        const blob = await apiClient.requestBlob(postMediaPath(postId as string, file.name))
        return [file.slot, await blobToDataUri(blob)] as const
      }))
      return Object.fromEntries(read)
    },
    staleTime: Infinity,
    refetchOnWindowFocus: false,
    retry: false,
  })
}
