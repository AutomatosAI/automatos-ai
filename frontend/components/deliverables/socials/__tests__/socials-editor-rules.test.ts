/**
 * PRD-251D Wave 1 — the editor's new rules, pure.
 *
 * * US-D104 (E4) `startingKinds`: a new post starts with the channels of the workspace's most
 *   recent post that are still connected and can take the format, each at the format's own kind;
 *   with no earlier post, the one connected channel when there is exactly one; otherwise none.
 * * US-D102 (E1) and US-D108 `submitRenders` / `submitNote`: Submit renders a post with a visual
 *   (a template, Let Auto pick) when its file does not show it yet (no rendered file, its own
 *   `file_current` false, or unsaved changes to what a render reads) and says so, with a video's
 *   render minutes in the preview note's own words; a post whose file still shows it submits as
 *   it is; a still of the person's own is cropped every time; a text post and a video or slides of
 *   the person's own submit as they are.
 * * US-D105 (E5) `unconnectedChannels`: the supported channels the workspace has not connected.
 */
import { describe, it, expect } from 'vitest'

import type { SocialChannel, SocialPost, SocialPostTarget } from '@/lib/api-client'
import { draftFromPost, startingKinds, withStartingChannels, newDraft } from '@/components/deliverables/socials/studio/editor-model'
import {
  SUBMIT_CROPS, SUBMIT_RENDERS, submitNote, submitRenders, unsavedFileChanges,
} from '@/components/deliverables/socials/studio/editor-submit'
import { SUPPORTED_CHANNELS, unconnectedChannels } from '@/components/deliverables/socials/studio/editor-connect-channels'
import { fullRenderCost, renderNote } from '@/components/deliverables/socials/studio/preview-model'

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

describe('what Submit for approval makes first (US-D102, US-D108)', () => {
  /** A post as the server has it, with a rendered file unless `media` says otherwise. */
  const rendered = (over: Partial<SocialPost> = {}) => ({ media: { '1:1': ['d-square'] }, ...over }) as unknown as SocialPost
  const NEW = { post: null, unsaved: false }

  it('an unsaved change to a field or to the look changes what the file shows; the brief alone does not (US-D301)', () => {
    const saved = {
      title: 'Open late', brief: 'Say we open late', format: 'image', template_id: 'tpl-img', length_seconds: null,
      variables: { headline: { value: 'Open late', claim: false } }, sources: {}, targets: [], look: null,
    } as unknown as SocialPost
    const onScreen = draftFromPost(saved)
    expect(unsavedFileChanges(saved, onScreen)).toBe(false)
    expect(unsavedFileChanges(saved, { ...onScreen, brief: 'Another angle' })).toBe(false)
    expect(unsavedFileChanges(saved, { ...onScreen, variables: { headline: { value: 'Open early', claim: false } } })).toBe(true)
    expect(unsavedFileChanges(saved, { ...onScreen, look: { logo: false } })).toBe(true)
    expect(unsavedFileChanges(null, onScreen)).toBe(false)  // a new post has nothing saved to differ from
  })

  it('renders a post with a visual that has no rendered file yet, and says so', () => {
    expect(submitRenders({ format: 'image', ownFile: false, ...NEW })).toBe(true)
    expect(submitRenders({ format: 'carousel', ownFile: false, post: rendered({ media: {} }), unsaved: false })).toBe(true)
    // A template chosen over a picture of the person's own: the picture is no render of the template.
    expect(submitRenders({ format: 'image', ownFile: false, post: rendered({ media: { original: ['d-own'] } }), unsaved: false })).toBe(true)
    expect(submitNote({ format: 'image', ownFile: false, lengthSeconds: null, sizes: 2, ...NEW })).toBe(SUBMIT_RENDERS)
  })

  it('renders a post whose file is older than its words, or whose edits on screen change what a render reads', () => {
    expect(submitRenders({ format: 'image', ownFile: false, post: rendered({ file_current: false }), unsaved: false })).toBe(true)
    expect(submitRenders({ format: 'image', ownFile: false, post: rendered({ file_current: true }), unsaved: true })).toBe(true)
    expect(submitNote({ format: 'image', ownFile: false, lengthSeconds: null, sizes: 2, post: rendered({ file_current: false }), unsaved: false }))
      .toBe(SUBMIT_RENDERS)
  })

  it('submits a post whose file still shows it as it is, with no line about a render', () => {
    // A post read without the flag counts as current, as the server counts one with no render record.
    for (const post of [rendered({ file_current: true }), rendered()]) {
      expect(submitRenders({ format: 'image', ownFile: false, post, unsaved: false })).toBe(false)
      expect(submitNote({ format: 'image', ownFile: false, lengthSeconds: null, sizes: 2, post, unsaved: false })).toBeNull()
    }
    const video = rendered({ media: { '9:16': ['d-video'] }, file_current: true })
    expect(submitRenders({ format: 'video', ownFile: false, post: video, unsaved: false })).toBe(false)
  })

  it("crops a still of the person's own every time, at no render minutes: its crops follow its channels", () => {
    const own = rendered({ media: { original: ['d-own'], '1:1': ['d-crop'] }, file_current: true })
    expect(submitRenders({ format: 'image', ownFile: true, post: own, unsaved: false })).toBe(true)
    expect(submitNote({ format: 'image', ownFile: true, lengthSeconds: null, sizes: 0, post: own, unsaved: false })).toBe(SUBMIT_CROPS)
  })

  it("submits a text post and a video or slides of the person's own as they are, with no line", () => {
    for (const input of [{ format: 'text', ownFile: false }, { format: 'video', ownFile: true }, { format: 'carousel', ownFile: true }]) {
      expect(submitRenders({ ...input, ...NEW, unsaved: true })).toBe(false)
      expect(submitNote({ ...input, lengthSeconds: 30, sizes: 2, ...NEW })).toBeNull()
    }
  })

  it("names a video's full render and its render minutes, in the preview note's own words", () => {
    expect(submitNote({ format: 'video', ownFile: false, lengthSeconds: 30, sizes: 2, ...NEW })).toBe(
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
