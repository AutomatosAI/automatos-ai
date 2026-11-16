/**
 * PRD-251D (E1, US-D102) — what "Submit for approval" does (pure): it makes the file.
 *
 * A post with a visual to render (a template, "Let Auto pick", or a still of the person's own,
 * cropped per channel) is saved, then given the full render (POST /posts/{id}/render with no
 * `preview`), which ends it in needs_approval by itself. A text post, and a post whose visual
 * is the person's own video or slides, have nothing to render: they are saved and submitted.
 *
 * Every visual post renders again on Submit. The post does not record which content its
 * rendered file was made from: the content hash a render binds is not kept with the render,
 * an edit keeps the old `media`, and a person's saved edit leaves no history entry. So "the
 * file still shows these words" cannot be told from "the words changed since", and E1 says
 * Submit never sends a file that does not show the post's current words.
 */
import type { SocialPost } from '@/lib/api-client'
import { formatRenderMinutes } from '../socials-status'
import type { EditorDraft } from './editor-model'
import { previewStale } from './preview-model'
import { lengthLabel } from './socials-calendar-model'

export const SUBMIT_RENDERS = 'Submit for approval renders the post first, then it waits for approval. Images use no render minutes.'
export const SUBMIT_CROPS = "Submit for approval crops your picture to each channel's shape first, then it waits for approval. No render minutes."
export const RENDERING_FOR_APPROVAL = 'Rendering. The post goes for approval once it is made.'
export const SENT_FOR_APPROVAL = 'Sent for approval'
/** Under the approve controls after a saved change to what the file shows: why Approve waits. */
export const FILE_OLDER_THAN_WORDS = 'You saved a change after this file was made, so it no longer shows the post as it is. Submit for approval makes it again; approve it after that.'
const SECONDS_PER_MINUTE = 60

/**
 * Whether saving `draft` leaves the post's file older than its words: the post already has a
 * rendered file, and the save changes something a render reads (preview-model's `previewStale`).
 * The server keeps the old file after an edit and does not record which content it was made
 * from, so the editor holds this for the sitting: Approve waits and Submit makes the file again.
 */
export function saveOutdatesFile(post: SocialPost | null, draft: EditorDraft): boolean {
  const hasFile = !!post && Object.keys(post.media ?? {}).length > 0
  return hasFile && previewStale(post, draft)
}

export interface SubmitInput {
  format: string
  /** The person's own file is the post (no template): an upload or a Library pick. */
  ownFile: boolean
  lengthSeconds: number | null
  /** How many sizes the full render makes (the chosen template's). */
  sizes: number
}

/** Whether Submit renders the post: everything but a text post and a file of the person's own
 * that is not a still (their video or slides go as they are). */
export function submitRenders({ format, ownFile }: Pick<SubmitInput, 'format' | 'ownFile'>): boolean {
  if (format === 'text') return false
  return !ownFile || format === 'image'
}

/** "about 1.5 render minutes (3 sizes x 0:30)": what a video's full render costs. */
export function fullRenderCost(lengthSeconds: number | null, sizes: number): string {
  const length = lengthSeconds ? lengthLabel(lengthSeconds) : 'its length'
  const count = Math.max(1, sizes)
  const minutes = lengthSeconds ? formatRenderMinutes((lengthSeconds / SECONDS_PER_MINUTE) * count, null) : 'a few'
  return `about ${minutes} render minutes (${count} ${count === 1 ? 'size' : 'sizes'} x ${length})`
}

/** The line under Submit: what it will render first, and a video's render minutes; null when it
 * renders nothing. */
export function submitNote(input: SubmitInput): string | null {
  if (!submitRenders(input)) return null
  if (input.ownFile) return SUBMIT_CROPS
  if (input.format === 'video') {
    return `Submit for approval makes the full render first, then it waits for approval: ${fullRenderCost(input.lengthSeconds, input.sizes)}.`
  }
  return SUBMIT_RENDERS
}
