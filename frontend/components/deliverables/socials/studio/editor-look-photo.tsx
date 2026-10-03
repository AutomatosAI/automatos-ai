'use client'

/**
 * PRD-251B (3 Oct 2026 pass) — where an upload or a Library pick goes when the chosen template
 * shows a photo: into one of its photo spots, so the template's words sit over the picture
 * (POST /posts/{id}/media with `slot`, PUT /posts/{id}/photos/{slot}), or the whole post, with
 * no template, cropped for each channel (as before). The spot's picture now shows beside it.
 * Nothing shows for a template without a photo spot: the file is the whole post.
 */
import type { SocialPost, SocialTemplateSummary } from '@/lib/api-client'
import { isOwnPicture, OPTION_CHOSEN, OWN_PICTURE, SlotPicture } from './editor-look-ai'
import { Hint, Segmented } from './editor-ui'

export interface PhotoSpot {
  name: string
  label: string
}

export const WHOLE_POST = 'whole-post'
export const WHOLE_POST_LABEL = 'The whole post'
export const PHOTO_TARGET = 'Where your picture goes'
export const PHOTO_SPOT_HINT = "Your picture goes behind the template's words: a PNG, JPEG or WebP image."
export const WHOLE_POST_HINT = 'Your file is the whole post, with no template: it is cropped for each channel.'

/** The chosen template's photo spots, each with its label ("Photo", "Before photo"). */
export function photoSpotsOf(template: SocialTemplateSummary | null | undefined): PhotoSpot[] {
  return (template?.image_slots ?? []).map((name) => ({ name, label: template?.image_slot_labels?.[name] ?? name }))
}

/** Where a pick goes: the spot asked for while the template has it, else its first spot, else the whole post. */
export function photoTarget(spots: ReadonlyArray<PhotoSpot>, asked: string | null): string {
  if (asked === WHOLE_POST || (asked && spots.some((spot) => spot.name === asked))) return asked
  return spots[0]?.name ?? WHOLE_POST
}

interface PhotoTargetProps {
  spots: ReadonlyArray<PhotoSpot>
  target: string
  post: SocialPost | null
  onTarget: (target: string) => void
}

export function PhotoTargetPicker({ spots, target, post, onTarget }: PhotoTargetProps) {
  if (spots.length === 0) return null
  const choices = [...spots.map((spot) => ({ value: spot.name, label: spot.label })), { value: WHOLE_POST, label: WHOLE_POST_LABEL }]
  const held = target === WHOLE_POST ? undefined : post?.footage?.[target]
  const shown = held?.status === 'done' && held.name ? held.name : null
  return (
    <div className="flex flex-col gap-2">
      <Segmented label={PHOTO_TARGET} choices={choices} value={target} onChange={onTarget} />
      <Hint>{target === WHOLE_POST ? WHOLE_POST_HINT : PHOTO_SPOT_HINT}</Hint>
      {post && shown && <SlotPicture postId={post.id} name={shown} caption={isOwnPicture(held) ? OWN_PICTURE : OPTION_CHOSEN} />}
    </div>
  )
}
