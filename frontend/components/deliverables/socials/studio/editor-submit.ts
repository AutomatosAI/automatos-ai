/**
 * PRD-251D (E1, US-D102, US-D108) — what "Submit for approval" does (pure): it makes the file when
 * the post needs one.
 *
 * A post with a visual to render (a template, "Let Auto pick", or a still of the person's own,
 * cropped per channel) is saved, then given the full render (POST /posts/{id}/render with no
 * `preview`), which ends it in needs_approval by itself, when its file does not show it yet: it has
 * no rendered file, its file is not current (the post's `file_current`: the server records what each
 * render was made from, US-D108), or the editor holds unsaved changes to what the file shows
 * (`unsavedFileChanges`: a field, the template, the length, the format or the look; not the brief,
 * which is what Auto writes from and is not on the file). Otherwise its file already shows it, and it is saved and
 * submitted as it was before Wave 1. A still of the person's own is cropped every time: its crops
 * follow its channels, which the record does not cover. A text post, and a post whose visual is the
 * person's own video or slides, have nothing to render: they are saved and submitted.
 */
import type { SocialPost } from '@/lib/api-client'
import { OWN_FILE_ASPECT, draftFromPost, type EditorDraft } from './editor-model'
import { fullRenderCost } from './preview-model'

export const SUBMIT_RENDERS = 'Submit for approval renders the post first, then it waits for approval. Images use no render minutes.'
export const SUBMIT_CROPS = "Submit for approval crops your picture to each channel's shape first, then it waits for approval. No render minutes."
export const RENDERING_FOR_APPROVAL = 'Rendering. The post goes for approval once it is made.'
export const SENT_FOR_APPROVAL = 'Sent for approval'

export interface SubmitInput {
  format: string
  /** The person's own file is the post (no template): an upload or a Library pick. */
  ownFile: boolean
  lengthSeconds: number | null
  /** How many sizes the full render makes (the chosen template's). */
  sizes: number
  /** The post as the server has it; null for a new one. */
  post: SocialPost | null
  /** The editor holds unsaved changes to what the file shows (`unsavedFileChanges`). */
  unsaved: boolean
}

/** What the file shows of a draft: its fields, template, length and format, and its own look (US-D301). */
function fileInputs(draft: EditorDraft): string {
  return JSON.stringify([draft.variables, draft.templateId, draft.lengthSeconds, draft.format, draft.look])
}

/** Whether the editor holds unsaved changes to what the post's file shows. An edit to the brief alone
 * is not one: the file would be made from the same fields. A new post has nothing saved to differ from. */
export function unsavedFileChanges(post: SocialPost | null, draft: EditorDraft): boolean {
  return !!post && fileInputs(draftFromPost(post)) !== fileInputs(draft)
}

type RenderInput = Pick<SubmitInput, 'format' | 'ownFile' | 'post' | 'unsaved'>

/** Whether the post has a visual to make: everything but a text post and a file of the person's own
 * that is not a still (their video or slides go as they are). */
export function makesFile({ format, ownFile }: Pick<SubmitInput, 'format' | 'ownFile'>): boolean {
  if (format === 'text') return false
  return !ownFile || format === 'image'
}

/** A render has made a file for the post: an aspect beside the person's own original. */
function hasRenderedFile(post: SocialPost): boolean {
  return Object.keys(post.media ?? {}).some((aspect) => aspect !== OWN_FILE_ASPECT)
}

/** Whether Submit renders the post first (the module note). */
export function submitRenders({ format, ownFile, post, unsaved }: RenderInput): boolean {
  if (!makesFile({ format, ownFile })) return false
  if (ownFile || !post || !hasRenderedFile(post)) return true
  return post.file_current === false || unsaved
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
