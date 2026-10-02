'use client'

/**
 * PRD-251B US-B110 — the files the preview shows: a video's half-resolution preview (the
 * post's `preview`), else the post's rendered or uploaded media as presigned inline links
 * (GET /api/socials/posts/{id}/media), each with its aspect.
 */
import type { SocialPost } from '@/lib/api-client'
import { useSocialPostMedia } from '@/hooks/use-socials-api'
import { aspectOfSize, type PreviewFile } from './preview-model'

export function usePreviewMedia(post: SocialPost | null, video: boolean): PreviewFile[] {
  const previewFiles = video && post?.preview?.status === 'done' ? post.preview.files : []
  const hasMedia = !!post && previewFiles.length === 0 && Object.keys(post.media ?? {}).length > 0
  const media = useSocialPostMedia(post?.id ?? '', post?.content_hash ?? '', hasMedia)
  if (!post) return []
  if (previewFiles.length > 0) {
    return previewFiles.map((f) => ({
      key: f.name, url: f.url, name: f.name, contentType: f.content_type ?? '', aspect: f.aspect ?? aspectOfSize(f.width, f.height),
    }))
  }
  return (media.data ?? [])
    .filter((f) => f.url)
    .map((f) => ({ key: `${f.aspect}-${f.deliverable_id}`, url: f.url as string, name: f.name ?? '', contentType: f.content_type ?? '', aspect: f.aspect }))
}
