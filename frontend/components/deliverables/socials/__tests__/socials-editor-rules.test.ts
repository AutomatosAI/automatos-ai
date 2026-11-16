/**
 * PRD-251D Wave 1 — the editor's new rules, pure.
 *
 * * US-D104 (E4) `startingKinds`: a new post starts with the channels of the workspace's most
 *   recent post that are still connected and can take the format, each at the format's own kind;
 *   with no earlier post, the one connected channel when there is exactly one; otherwise none.
 * * US-D102 (E1) `submitRenders` / `submitNote`: Submit renders every post with a visual (a
 *   template, Let Auto pick, a still of the person's own) and says so, with a video's render
 *   minutes in the preview note's own words; a text post and a video or slides of the person's
 *   own submit as they are.
 * * US-D105 (E5) `unconnectedChannels`: the supported channels the workspace has not connected.
 */
import { describe, it, expect } from 'vitest'

import type { SocialChannel, SocialPost, SocialPostTarget } from '@/lib/api-client'
import { startingKinds, withStartingChannels, newDraft } from '@/components/deliverables/socials/studio/editor-model'
import {
  SUBMIT_CROPS, SUBMIT_RENDERS, fullRenderCost, submitNote, submitRenders,
} from '@/components/deliverables/socials/studio/editor-submit'
import { SUPPORTED_CHANNELS, unconnectedChannels } from '@/components/deliverables/socials/studio/editor-connect-channels'
import { renderNote } from '@/components/deliverables/socials/studio/preview-model'

const kind = (k: string) => ({ kind: k, available: true, reason: null, needs_public_storage: false })
const channel = (toolkit: string, kinds: string[]) =>
  ({ toolkit, label: toolkit, post_kinds: kinds.map(kind), verified: true, setup_note: null, copy_limits: {} }) as unknown as SocialChannel
const X = channel('twitter', ['text', 'image', 'video'])
const LINKEDIN = channel('linkedin', ['text', 'image', 'video'])
const INSTAGRAM = channel('instagram', ['image', 'carousel', 'reel', 'story'])
const TIKTOK = channel('tiktok', ['video'])
const target = (toolkit: string, postKind: string) => ({ toolkit, post_kind: postKind }) as SocialPostTarget
const posted = (...targets: SocialPostTarget[]) => ({ targets }) as Pick<SocialPost, 'targets'>

describe('the channels a new post starts with (US-D104)', () => {
  it('with no earlier post, ticks the one connected channel when it can take the format', () => {
    expect(startingKinds([X], [], 'image')).toEqual({ twitter: 'image' })
    expect(startingKinds([TIKTOK], [], 'video')).toEqual({ tiktok: 'video' })
    expect(startingKinds([TIKTOK], [], 'image')).toEqual({})  // TikTok takes video only
  })

  it('with no earlier post and two channels, ticks none', () => {
    expect(startingKinds([X, LINKEDIN], [], 'image')).toEqual({})
  })

  it("ticks the most recent post's channels that can take the format, each at the format's own kind", () => {
    const latest = posted(target('instagram', 'story'), target('linkedin', 'video'), target('tiktok', 'video'))
    const older = posted(target('twitter', 'image'))
    // Instagram was a story and LinkedIn a video: each starts at the image kind. TikTok takes no
    // image, and X was only on an older post.
    expect(startingKinds([X, LINKEDIN, INSTAGRAM, TIKTOK], [latest, older], 'image')).toEqual({ linkedin: 'image', instagram: 'image' })
    expect(startingKinds([X, LINKEDIN, INSTAGRAM, TIKTOK], [latest, older], 'video')).toEqual({ linkedin: 'video', instagram: 'reel', tiktok: 'video' })
  })

  it('leaves out a channel the most recent post used that is no longer connected', () => {
    expect(startingKinds([X], [posted(target('twitter', 'image'), target('linkedin', 'image'))], 'image')).toEqual({ twitter: 'image' })
  })

  it('takes the most recent post at its word: one with no channel ticks none, even with one channel connected', () => {
    expect(startingKinds([X], [posted(), posted(target('twitter', 'image'))], 'image')).toEqual({})
  })

  it('a new draft takes them as its kinds, at its own format', () => {
    const draft = withStartingChannels({ ...newDraft(), format: 'text' }, [X, INSTAGRAM], [posted(target('twitter', 'image'), target('instagram', 'image'))])
    expect(draft.kinds).toEqual({ twitter: 'text' })
    expect(draft.format).toBe('text')
  })
})

describe('what Submit for approval makes first (US-D102)', () => {
  it('renders every post with a visual, and says so', () => {
    expect(submitRenders({ format: 'image', ownFile: false })).toBe(true)
    expect(submitRenders({ format: 'carousel', ownFile: false })).toBe(true)
    expect(submitNote({ format: 'image', ownFile: false, lengthSeconds: null, sizes: 2 })).toBe(SUBMIT_RENDERS)
  })

  it("crops a still of the person's own, at no render minutes", () => {
    expect(submitRenders({ format: 'image', ownFile: true })).toBe(true)
    expect(submitNote({ format: 'image', ownFile: true, lengthSeconds: null, sizes: 0 })).toBe(SUBMIT_CROPS)
  })

  it("submits a text post and a video or slides of the person's own as they are, with no line", () => {
    for (const input of [{ format: 'text', ownFile: false }, { format: 'video', ownFile: true }, { format: 'carousel', ownFile: true }]) {
      expect(submitRenders(input)).toBe(false)
      expect(submitNote({ ...input, lengthSeconds: 30, sizes: 2 })).toBeNull()
    }
  })

  it("names a video's full render and its render minutes, in the preview note's own words", () => {
    expect(submitNote({ format: 'video', ownFile: false, lengthSeconds: 30, sizes: 2 })).toBe(
      'Submit for approval makes the full render first, then it waits for approval: about 1 render minutes (2 sizes x 0:30).',
    )
    expect(fullRenderCost(null, 0)).toBe('about a few render minutes (1 size x its length)')
    // The note under the preview says the same cost: the two never drift apart.
    const note = renderNote({ format: 'video', lengthSeconds: 30, ratios: [], sizes: 3, renderedAt: null })
    expect(note).toContain(`The full render is made when you submit: ${fullRenderCost(30, 3)}.`)
  })
})

describe('the channels to connect (US-D105)', () => {
  it('are the supported ones the workspace has not connected, in the brand kit order', () => {
    expect(SUPPORTED_CHANNELS).toEqual(['linkedin', 'twitter', 'instagram', 'tiktok', 'youtube'])
    expect(unconnectedChannels([])).toEqual(SUPPORTED_CHANNELS)
    expect(unconnectedChannels([{ toolkit: 'twitter' }, { toolkit: 'reddit' }, { toolkit: 'youtube' }])).toEqual(['linkedin', 'instagram', 'tiktok'])
  })
})
