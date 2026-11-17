/**
 * PRD-251D US-D301 (E10) — a post's own look, the pure model (look-options-model.ts) and where the
 * draft carries it (editor-model.ts, preview-model.ts):
 *
 * * Light or dark is offered only where the chosen template's gallery entry names 'mode' (a
 *   template built on paper); the accent and the logo for every template and for "Let Auto pick".
 * * A choice that is the kit's own (light, the brand kit's accent, the logo on) is left out of the
 *   look, kept in the server's key order, and a look with nothing left saves as null.
 * * The draft starts from the post's look, saves it with the post's fields, and a change to it
 *   marks the preview out of date.
 */
import { describe, it, expect } from 'vitest'

import {
  lookKeys, lookSegments, lookToSave, orderedLook, withLookChoice,
} from '@/components/deliverables/socials/studio/look-options-model'
import { draftFromPost, newDraft, postFields } from '@/components/deliverables/socials/studio/editor-model'
import { previewStale } from '@/components/deliverables/socials/studio/preview-model'
import { post } from './socials-editor-harness'

describe('which look choices a template takes', () => {
  it('light or dark only where the gallery entry names it; the accent and the logo always', () => {
    expect(lookKeys({ looks: ['mode', 'accent', 'logo'] })).toEqual(['mode', 'accent', 'logo'])
    expect(lookKeys({ looks: ['accent', 'logo'] })).toEqual(['accent', 'logo'])
    expect(lookKeys(null)).toEqual(['accent', 'logo']) // Let Auto pick
    expect(lookKeys({})).toEqual(['accent', 'logo']) // an entry that names none
  })
})

describe('a choice', () => {
  it("the kit's own is left out; any other is kept, in the server's order", () => {
    expect(withLookChoice({}, 'mode', 'dark')).toEqual({ mode: 'dark' })
    expect(withLookChoice({ mode: 'dark' }, 'mode', 'light')).toEqual({})
    expect(withLookChoice({}, 'accent', 'bold')).toEqual({ accent: 'bold' })
    expect(withLookChoice({ accent: 'bold' }, 'accent', 'sparing')).toEqual({ accent: 'sparing' })
    expect(withLookChoice({ accent: 'bold' }, 'accent', 'kit')).toEqual({})
    expect(withLookChoice({}, 'logo', 'off')).toEqual({ logo: false })
    expect(withLookChoice({ logo: false }, 'logo', 'on')).toEqual({})
    expect(Object.keys(withLookChoice({ logo: false, accent: 'bold' }, 'mode', 'dark'))).toEqual(['mode', 'accent', 'logo'])
  })

  it('never changes the look it was given', () => {
    const look = { accent: 'bold' as const }
    withLookChoice(look, 'mode', 'dark')
    expect(look).toEqual({ accent: 'bold' })
  })

  it("each control shows the kit's own when the look leaves it out", () => {
    expect(lookSegments({})).toEqual({ mode: 'light', accent: 'kit', logo: 'on' })
    expect(lookSegments({ mode: 'dark', accent: 'sparing', logo: false })).toEqual({ mode: 'dark', accent: 'sparing', logo: 'off' })
  })

  it("a look of the kit's own saves as null; any other as a copy", () => {
    expect(lookToSave({})).toBeNull()
    const look = { mode: 'dark' as const }
    expect(lookToSave(look)).toEqual(look)
    expect(lookToSave(look)).not.toBe(look)
    expect(orderedLook(null)).toEqual({})
  })
})

describe('the draft carries the look', () => {
  it('starts from the post, in the server order, and saves with the post fields', () => {
    expect(newDraft().look).toEqual({})
    expect(draftFromPost(post()).look).toEqual({})
    const draft = draftFromPost(post({ look: { logo: false, mode: 'dark' } }))
    expect(Object.keys(draft.look)).toEqual(['mode', 'logo'])
    expect(postFields(draft, []).look).toEqual({ mode: 'dark', logo: false })
    expect(postFields(newDraft(), []).look).toBeNull()
  })

  it('a change to it marks the preview out of date; the saved look does not', () => {
    const saved = post({ look: { mode: 'dark' } })
    const draft = draftFromPost(saved)
    expect(previewStale(saved, draft)).toBe(false)
    expect(previewStale(saved, { ...draft, look: withLookChoice(draft.look, 'mode', 'light') })).toBe(true)
    expect(previewStale(saved, { ...draft, look: withLookChoice(draft.look, 'logo', 'off') })).toBe(true)
  })
})
