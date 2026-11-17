/**
 * PRD-251D US-D201 (E8) — the instant preview's rules (pure): which posts show it and which keep
 * today's preview; what the editor asks for (the values on screen, the active channel's size, a
 * story, the photos); a page is shown only when it opens with its own policy; the kit's faces and a
 * photo go only where the server marked them, and only as what is safe there.
 */
import { describe, it, expect } from 'vitest'

import type { SocialChannel, SocialPost } from '@/lib/api-client'
import { newDraft, type EditorDraft } from '@/components/deliverables/socials/studio/editor-model'
import {
  FONTS_MARKER, PAGE_POLICY_META, hasPagePolicy, pageWithFiles, photoFilesOf, photoMarker, previewInput, renderIsCurrent, sizeOf,
  valuesOf, wantsInstantPreview,
} from '@/components/deliverables/socials/studio/instant-preview-model'
import { CHANNELS, PHOTO_TEMPLATE, post } from './socials-editor-harness'

const channel = (toolkit: string) => CHANNELS.find((c) => c.toolkit === toolkit) as SocialChannel
const draft = (over: Partial<EditorDraft> = {}): EditorDraft => ({ ...newDraft(), templateId: 'tpl-img', ...over })
const PHOTO = 'data:image/png;base64,iVBORw0KGgo='

describe('which posts show the instant preview', () => {
  it('an image post with a template whose file is not current', () => {
    expect(wantsInstantPreview(draft(), false)).toBe(true)
  })

  it('not a video, a carousel, a text post, a post with no template ("Let Auto pick", the person\'s own file) or a current render', () => {
    expect(wantsInstantPreview(draft({ format: 'video' }), false)).toBe(false)
    expect(wantsInstantPreview(draft({ format: 'carousel' }), false)).toBe(false)
    expect(wantsInstantPreview(draft({ format: 'text', templateId: null }), false)).toBe(false)
    expect(wantsInstantPreview(draft({ templateId: null }), false)).toBe(false)
    expect(wantsInstantPreview(draft(), true)).toBe(false)
  })

  it('a render is current while the post has a file and nothing a render reads changed since the save', () => {
    const saved = post({ template_id: 'tpl-img', variables: { headline: { value: 'Open late' } }, media: { '1:1': ['d-1'] } })
    const onScreen = draft({ brief: saved.brief ?? '', variables: { headline: { value: 'Open late' } } })
    expect(renderIsCurrent(saved, onScreen)).toBe(true)
    expect(renderIsCurrent(saved, { ...onScreen, variables: { headline: { value: 'Open early' } } })).toBe(false)
    expect(renderIsCurrent({ ...saved, media: {} }, onScreen)).toBe(false)
    // A saved edit after the render: the server says the file is older than the words (US-D108).
    expect(renderIsCurrent({ ...saved, file_current: false }, onScreen)).toBe(false)
    expect(renderIsCurrent(null, onScreen)).toBe(false)
  })
})

describe('what the editor asks for', () => {
  it('the values on screen: text, a finite number or a switch', () => {
    const variables = {
      headline: { value: 'Open late' }, members: { value: 412 }, open: { value: true }, half: { value: Number.NaN },
    } as EditorDraft['variables']
    expect(valuesOf({ variables })).toEqual({ headline: 'Open late', members: 412, open: true })
  })

  it('the size the active channel gets, a story when any ticked channel posts one, and the photos in order', () => {
    const sizes = ['1080x1350', '1080x1920', '1920x1080']
    const onScreen = draft({ kinds: { twitter: 'image', instagram: 'story' } })
    expect(previewInput(onScreen, sizes, channel('twitter'), ['photo', 'after'])).toEqual({
      values: {}, size: '1920x1080', story: true, photos: ['after', 'photo'],
    })
    expect(previewInput(onScreen, sizes, channel('instagram'), []).size).toBe('1080x1920')
    expect(previewInput(draft({ kinds: { linkedin: 'image' } }), sizes, null, [])).toEqual({ values: {}, size: '1080x1350', story: false, photos: [] })
    expect(previewInput(draft(), [], null, []).size).toBeNull()
  })

  it('the photos the post holds in the template\'s photo spots, chosen and stored', () => {
    const held = post({
      footage: {
        photo: { prompt: 'p', status: 'done', toolkit: 'upload', name: 'upload-1.png' },
        hook: { prompt: 'h', status: 'done', name: 'hook.mp4' },
      },
    } as Partial<SocialPost>)
    expect(photoFilesOf(held, PHOTO_TEMPLATE)).toEqual([{ slot: 'photo', name: 'upload-1.png' }])
    expect(photoFilesOf(post({ footage: { photo: { prompt: 'p', options_state: 'making' } } } as Partial<SocialPost>), PHOTO_TEMPLATE)).toEqual([])
    expect(photoFilesOf(held, null)).toEqual([])
  })
})

describe('the page', () => {
  it('is shown only when it opens with its own policy, before any script', () => {
    expect(hasPagePolicy(`<!doctype html>${PAGE_POLICY_META}<script>x()</script>`)).toBe(true)
    expect(hasPagePolicy(`<!doctype html><script>x()</script>${PAGE_POLICY_META}`)).toBe(false)
    expect(hasPagePolicy('<!doctype html><h1>No policy</h1>')).toBe(false)
    expect(hasPagePolicy(PAGE_POLICY_META.replace("default-src 'none'", 'default-src *'))).toBe(false)
  })

  it('takes the kit\'s faces at the font marker and each photo at its own marker, only as what is safe there', () => {
    const page = `<style>:root{}${FONTS_MARKER}</style><img src="${photoMarker('photo')}"><img src="${photoMarker('after')}">`
    const css = '@font-face{font-family:"Brand Sans";src:url("data:font/woff2;base64,d09G") format("woff2")}'
    expect(pageWithFiles(page, css, { photo: PHOTO })).toBe(
      `<style>:root{}${css}</style><img src="${PHOTO}"><img src="${photoMarker('after')}">`,
    )
    // A face that would close the style, and a "photo" that would end the quoted src, never go in.
    expect(pageWithFiles(page, '</style><script>x()</script>', { photo: 'data:image/png;base64,AA" onload="x()' })).toBe(
      `<style>:root{}</style><img src="${photoMarker('photo')}"><img src="${photoMarker('after')}">`,
    )
    expect(pageWithFiles(page, css, { photo: 'data:text/html;base64,PGgxPg==' })).toContain(photoMarker('photo'))
  })

  it('is drawn at the template\'s size', () => {
    expect(sizeOf('1080x1350')).toEqual({ width: 1080, height: 1350 })
    expect(sizeOf('1080 x 1350')).toBeNull()
    expect(sizeOf(null)).toBeNull()
  })
})
