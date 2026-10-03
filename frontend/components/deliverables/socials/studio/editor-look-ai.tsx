'use client'

/**
 * PRD-251B US-B305 — AI-made in the editor's Look. AI image: for each of the template's
 * image slots, a prompt and "Make 4 options" (stills through the workspace's AI images
 * tool, the brand kit's style after the prompt, priced and capped like a render's footage);
 * the four arrive in a minute, and the one picked becomes the slot's file. AI footage
 * fills the hook and b-roll: the switch in Format. Every word on screen stays template
 * text (D12), so a prompt describes a picture, never words.
 */
import { useState } from 'react'
import Link from 'next/link'
import { Check, Loader2, Sparkles } from 'lucide-react'

import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import type { SocialFootageKind, SocialPost, SocialPostFootage } from '@/lib/api-client'
import type { SocialAiOption } from '@/lib/brand-style-types'
import { BRAND_KIT_HREF } from '@/lib/deliverables/tabs'
import { useAuthedImage } from '@/hooks/use-authed-image'
import { Hint } from './editor-ui'

export const NO_IMAGE_SLOT = 'This template shows no picture. Pick a template marked Photo, and AI makes its pictures here. For a video, AI footage fills the hook and b-roll: switch it on in Format.'
export const WORDS_NOTE = 'Describe the picture only: every word on screen is template text.'
const PROMPT_MAX = 1000

function mediaPath(postId: string, name: string): string {
  return `/api/socials/posts/${postId}/media/${encodeURIComponent(name)}`
}

function OptionThumb({ postId, option, index, disabled, onPick }: {
  postId: string; option: SocialAiOption; index: number; disabled: boolean; onPick: () => void
}) {
  const image = useAuthedImage(mediaPath(postId, option.name))
  return (
    <li>
      <button
        type="button" aria-label={`Use option ${index + 1}`} disabled={disabled} onClick={onPick}
        className="group relative block aspect-square w-full overflow-hidden rounded-lg border-2 border-transparent bg-muted hover:border-accent"
      >
        {image && (
          // eslint-disable-next-line @next/next/no-img-element
          <img src={image} alt={`Option ${index + 1}`} className="h-full w-full object-cover" />
        )}
        <Check className="absolute right-1.5 top-1.5 hidden h-4 w-4 rounded-full bg-accent text-accent-foreground group-hover:block" aria-hidden />
      </button>
    </li>
  )
}

export const OPTION_CHOSEN = 'Option chosen: the next render uses it.'
export const OWN_PICTURE = 'Your picture: the next render uses it.'

/** Whether a slot's file is the person's own (an upload or a Library picture), not an AI option. */
export function isOwnPicture(asked: SocialPostFootage | undefined): boolean {
  return asked?.toolkit === 'upload' || asked?.toolkit === 'library'
}

/** The picture a slot holds now, with what it is. */
export function SlotPicture({ postId, name, caption }: { postId: string; name: string; caption: string }) {
  const image = useAuthedImage(mediaPath(postId, name))
  return (
    <div className="flex items-center gap-2">
      <div className="h-16 w-16 overflow-hidden rounded-md border-2 border-accent bg-muted">
        {image && (
          // eslint-disable-next-line @next/next/no-img-element
          <img src={image} alt={caption} className="h-full w-full object-cover" />
        )}
      </div>
      <span className="text-xs text-muted-foreground">{caption}</span>
    </div>
  )
}

interface AiSlotProps {
  slot: string
  post: SocialPost | null
  asked: SocialPostFootage | undefined
  busy: boolean
  onMake: (slot: string, prompt: string) => void
  onPick: (slot: string, name: string) => void
}

function AiSlot({ slot, post, asked, busy, onMake, onPick }: AiSlotProps) {
  const own = isOwnPicture(asked)
  // The person's own picture has no prompt to make options from: theirs starts empty.
  const [prompt, setPrompt] = useState(own ? '' : asked?.prompt ?? '')
  const making = asked?.options_state === 'making'
  const options = asked?.options ?? []
  const chosenName = asked?.status === 'done' ? asked.name : null
  return (
    <div className="flex flex-col gap-2 rounded-lg border border-border p-3" aria-label={`AI image for ${slot}`} role="group">
      <span className="text-sm font-medium text-foreground">{slot.replace(/_/g, ' ')}</span>
      <Textarea aria-label={`Prompt for ${slot}`} value={prompt} maxLength={PROMPT_MAX} rows={2} onChange={(e) => setPrompt(e.target.value)} placeholder="A harbour at first light, boats moored, soft mist" />
      <div className="flex flex-wrap items-center gap-2">
        <Button type="button" size="sm" variant="secondary" disabled={busy || making || !prompt.trim()} onClick={() => onMake(slot, prompt.trim())}>
          {making ? <Loader2 className="mr-1.5 h-4 w-4 animate-spin" aria-hidden /> : <Sparkles className="mr-1.5 h-4 w-4" aria-hidden />}
          {making ? 'Making four options…' : 'Make 4 options'}
        </Button>
      </div>
      {post && chosenName && <SlotPicture postId={post.id} name={chosenName} caption={own ? OWN_PICTURE : OPTION_CHOSEN} />}
      {asked?.options_state === 'failed' && <p role="alert" className="text-xs text-destructive">{asked.options_error || 'No options were made.'}</p>}
      {asked?.options_state === 'ready' && asked.options_error && <p className="text-xs text-muted-foreground">Some options were not made: {asked.options_error}</p>}
      {post && options.length > 0 && (
        <ul aria-label={`Options for ${slot}`} className="grid grid-cols-4 gap-2">
          {options.map((option, index) => (
            <OptionThumb key={option.name} postId={post.id} option={option} index={index} disabled={busy} onPick={() => onPick(slot, option.name)} />
          ))}
        </ul>
      )}
    </div>
  )
}

interface EditorLookAiProps {
  post: SocialPost | null
  imageSlots: ReadonlyArray<string>
  images: SocialFootageKind | undefined
  busy: boolean
  onMake: (slot: string, prompt: string) => void
  onPick: (slot: string, name: string) => void
}

export function EditorLookAi({ post, imageSlots, images, busy, onMake, onPick }: EditorLookAiProps) {
  if (imageSlots.length === 0) return <Hint>{NO_IMAGE_SLOT}</Hint>
  if (!images?.available) {
    return (
      <Hint>
        {images?.reason ?? 'No AI images tool is connected.'} Choose one in the{' '}
        <Link href={BRAND_KIT_HREF as any} className="underline underline-offset-4">Brand kit&apos;s AI tools</Link>.
      </Hint>
    )
  }
  return (
    <div className="flex flex-col gap-3">
      <Hint>{`Made with ${images.label ?? images.toolkit}, in your brand kit's style. ${WORDS_NOTE}`}</Hint>
      {imageSlots.map((slot) => (
        <AiSlot key={`${post?.id ?? 'new'}-${slot}`} slot={slot} post={post} asked={post?.footage?.[slot]} busy={busy} onMake={onMake} onPick={onPick} />
      ))}
    </div>
  )
}
