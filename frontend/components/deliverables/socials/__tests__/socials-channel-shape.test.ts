/**
 * 3 Oct 2026 — each channel its own shape (the server's modules/socials/channel_sizes.py):
 * of a template's sizes, the one closest in shape to the channel's; the editor labels each
 * channel with it, "Renders" names the sizes the render makes, and the preview shows each
 * channel the file the server publishes to it.
 */
import { describe, it, expect } from 'vitest'

import { closestShape, shapeRatio, sizeAspect } from '@/components/deliverables/socials/studio/channel-shape'
import { renderRatios, sizeFor } from '@/components/deliverables/socials/studio/editor-model'
import { fileForAspect } from '@/components/deliverables/socials/studio/preview-model'

const TEXT_CARD = ['1080x1350', '1080x1920', '1200x628', '1600x900']
const PHOTO_CARD = ['1080x1350', '1080x1920', '1080x1080', '1200x628']
const same = (s: string) => s

describe('the closest shape', () => {
  it('reads sizes and aspects alike, and picks the nearest', () => {
    expect(shapeRatio('4:5')).toBe(0.8)
    expect(shapeRatio('1080x1350')).toBe(0.8)
    expect(shapeRatio('text')).toBeNull()
    expect(closestShape(TEXT_CARD, same, '1:1')).toBe('1080x1350')
    expect(closestShape(TEXT_CARD, same, '16:9')).toBe('1600x900')
    expect(closestShape(TEXT_CARD, same, '9:16')).toBe('1080x1920')
    expect(closestShape(PHOTO_CARD, same, '1:1')).toBe('1080x1080')
    expect(closestShape(PHOTO_CARD, same, '16:9')).toBe('1200x628')
    expect(sizeAspect('1200x628')).toBe('300:157')
  })
})

describe('the editor', () => {
  it('labels each channel with the size it gets, and Renders names the sizes made', () => {
    expect(sizeFor('instagram', 'image', TEXT_CARD)).toBe('1080 × 1350')
    expect(sizeFor('twitter', 'image', PHOTO_CARD)).toBe('1200 × 628')
    const draft = { kinds: { instagram: 'image', twitter: 'image', linkedin: 'image' } } as const
    expect(renderRatios(draft, TEXT_CARD)).toEqual(['4:5', '16:9'])
    expect(renderRatios(draft, PHOTO_CARD)).toEqual(['1:1', '300:157'])
    expect(renderRatios(draft)).toEqual(['1:1', '16:9'])  // no template yet: the channels' own
  })

  it('previews each channel the file published to it', () => {
    const files = [{ aspect: '4:5', name: 'image-4x5.png' }, { aspect: '16:9', name: 'image-16x9.png' }] as any[]
    expect(fileForAspect(files, '16:9')?.name).toBe('image-16x9.png')
    expect(fileForAspect(files, '1:1')?.name).toBe('image-4x5.png')
    expect(fileForAspect(files, '9:16')?.name).toBe('image-4x5.png')
    expect(fileForAspect(files, null)?.name).toBe('image-4x5.png')
  })
})
