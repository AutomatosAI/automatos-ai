/**
 * PRD-251B US-B110 — the preview's reading of a post (pure): the file a channel shows (the
 * render at its aspect, else the first), whether the preview is out of date, and the note
 * under the frame (what was rendered, or what the full render will cost in minutes).
 */
import { closestShape } from './channel-shape'
import type { SocialPost } from '@/lib/api-client'
import { formatRenderMinutes } from '../socials-status'
import { lengthLabel } from './socials-calendar-model'
import { draftFromPost, type EditorDraft } from './editor-model'

export const PREVIEW_STALE = 'Preview out of date'
export const TEXT_NOTE = 'Text only: nothing to render.'
const RENDER_DONE = 'render_done'

export interface PreviewFile {
  key: string
  url: string
  name: string
  contentType: string
  /** "9:16", "1:1"; "original" for an uploaded file. */
  aspect: string
}

/** 1080 × 1920 → "9:16". */
export function aspectOfSize(width: number | undefined, height: number | undefined): string {
  if (!width || !height) return 'original'
  const gcd = (a: number, b: number): number => (b === 0 ? a : gcd(b, a % b))
  const d = gcd(width, height)
  return `${width / d}:${height / d}`
}

/** The file a channel shows: the render at the channel's aspect, else the one closest in
 * shape (the one the server publishes to it, channel-shape.ts), else the first one. */
export function fileForAspect(files: ReadonlyArray<PreviewFile>, aspect: string | null): PreviewFile | null {
  return files.find((file) => file.aspect === aspect) ?? closestShape(files, (file) => file.aspect, aspect) ?? files[0] ?? null
}

/** What a render reads: an edit to any of these since the render makes it out of date. PRD-251D
 * US-D301: the post's look too (light or dark, the accent, the logo). */
function renderInputs(draft: EditorDraft): string {
  return JSON.stringify([draft.brief, draft.variables, draft.templateId, draft.lengthSeconds, draft.format, draft.look])
}

/** The preview no longer shows the post on screen: an edit to its brief, variables,
 * template, length or format since it was saved, or a saved change since the preview. */
export function previewStale(post: SocialPost | null, draft: EditorDraft): boolean {
  if (!post) return false
  if (renderInputs(draftFromPost(post)) !== renderInputs(draft)) return true
  return post.preview?.status === 'done' && post.preview.content_hash !== post.content_hash
}

/** When the post was last rendered ("HH:MM"), or null. */
export function renderedAt(post: SocialPost | null): string | null {
  const entry = [...(post?.review_log ?? [])].reverse().find((e) => e.action === RENDER_DONE)
  if (!entry?.at) return null
  return new Date(entry.at).toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' })
}

interface NoteInput {
  format: string
  lengthSeconds: number | null
  /** The ratios the ticked channels render in. */
  ratios: ReadonlyArray<string>
  /** How many sizes the full render makes (the template's). */
  sizes: number
  renderedAt: string | null
}

const SECONDS_PER_MINUTE = 60

/** "about 1.5 render minutes (3 sizes x 0:30)": what a video's full render costs. The note under
 * the preview and the line under Submit (editor-submit.ts) both say it in these words. */
export function fullRenderCost(lengthSeconds: number | null, sizes: number): string {
  const length = lengthSeconds ? lengthLabel(lengthSeconds) : 'its length'
  const count = Math.max(1, sizes)
  const minutes = lengthSeconds ? formatRenderMinutes((lengthSeconds / SECONDS_PER_MINUTE) * count, null) : 'a few'
  return `about ${minutes} render minutes (${count} ${count === 1 ? 'size' : 'sizes'} x ${length})`
}

/** The note under the frame: what an image render made, what a video's full render costs. */
export function renderNote({ format, lengthSeconds, ratios, sizes, renderedAt: at }: NoteInput): string {
  if (format === 'text') return TEXT_NOTE
  if (format === 'video') {
    const length = lengthSeconds ? lengthLabel(lengthSeconds) : 'its length'
    return `Half-resolution preview, ${length}. The full render is made when you submit: ${fullRenderCost(lengthSeconds, sizes)}.`
  }
  const when = at ? `Rendered ${at}.` : 'Not rendered yet.'
  return `${when} One PNG per size (${ratios.length ? ratios.join(', ') : 'no channel ticked'}). Images use no render minutes.`
}
