/**
 * F377 (night 11, 7 Oct) — the videos told on a software product's own screens (UI story,
 * Cinematic product promo, App promo) carry `made_for: 'software'`; the Look gallery says so
 * under their name, and says nothing under a template any business can use.
 */
import { describe, it, expect } from 'vitest'

import { SOFTWARE_NOTE, templateNote } from '@/components/deliverables/socials/studio/editor-look-card'

describe('whom a template is for', () => {
  it('names a software template, and leaves any other without a note', () => {
    expect(templateNote({ made_for: 'software' })).toBe(SOFTWARE_NOTE)
    expect(templateNote({ made_for: null })).toBeUndefined()
    expect(templateNote({})).toBeUndefined()
  })
})
