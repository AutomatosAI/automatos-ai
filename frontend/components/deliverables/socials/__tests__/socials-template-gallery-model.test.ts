/**
 * PRD-251D US-D107 (E7) — the template gallery's model: a search over each template's name and
 * what it is for (its description), the chips All, With a photo and No photo, the chosen
 * template kept in view (in its own place) when they would hide it, and the line that says
 * nothing matches. The gallery itself is tested in socials-editor-look.test.tsx.
 */
import { describe, it, expect } from 'vitest'

import {
  galleryView, hasPhoto, matchesFilter, matchesSearch, noMatchLine, type GalleryTemplate,
} from '@/components/deliverables/socials/studio/template-gallery-model'

const OFFER: GalleryTemplate = {
  id: 'tpl-offer', name: 'Offer', image_slots: ['photo'],
  description: 'A deal over your photo: the offer in big type, what it is for, the small print and a button. For a sale, a discount, a bundle or an opening offer.',
}
const QUOTE: GalleryTemplate = {
  id: 'tpl-quote', name: 'Quote card', image_slots: [],
  description: "A pull-quote set large on the brand's paper, one phrase in the brand colour, with who said it.",
}
const REVIEW: GalleryTemplate = {
  id: 'tpl-review', name: 'Review', image_slots: ['photo'],
  description: "A customer's words on a card over your photo, with their stars and name. For a review, a testimonial or a thank-you.",
}
/** A workspace's own template: no description, and no image_slots in the answer at all. */
const OWN: GalleryTemplate = { id: 'tpl-own', name: 'Team news', description: null }
/** In the server's order, by name. */
const TEMPLATES = [OFFER, QUOTE, REVIEW, OWN]
const ids = (list: ReadonlyArray<GalleryTemplate>) => list.map((template) => template.id)

describe('the search', () => {
  it('finds a template by its name or by what it is for, in any case', () => {
    expect(matchesSearch(QUOTE, 'quote')).toBe(true)
    expect(matchesSearch(REVIEW, 'TESTIMONIAL')).toBe(true)
    expect(matchesSearch(OFFER, 'testimonial')).toBe(false)
  })

  it('needs every word of the search, each in the name or the description', () => {
    expect(matchesSearch(REVIEW, '  review   stars ')).toBe(true)
    expect(matchesSearch(REVIEW, 'review discount')).toBe(false)
  })

  it('finds a template without a description by its name, and an empty search finds every one', () => {
    expect(matchesSearch(OWN, 'team')).toBe(true)
    expect(matchesSearch(OWN, 'photo')).toBe(false)
    expect(matchesSearch(OWN, '   ')).toBe(true)
  })
})

describe('the chips', () => {
  it('With a photo keeps the templates with an image slot, No photo the rest, All every one', () => {
    expect([OFFER, QUOTE, REVIEW, OWN].map(hasPhoto)).toEqual([true, false, true, false])
    expect(ids(TEMPLATES.filter((template) => matchesFilter(template, 'photo')))).toEqual(['tpl-offer', 'tpl-review'])
    expect(ids(TEMPLATES.filter((template) => matchesFilter(template, 'no_photo')))).toEqual(['tpl-quote', 'tpl-own'])
    expect(ids(TEMPLATES.filter((template) => matchesFilter(template, 'all')))).toEqual(ids(TEMPLATES))
  })
})

describe('what the gallery shows', () => {
  it('shows every template in the server order until the search or a chip narrows it', () => {
    const view = galleryView(TEMPLATES, { search: '', filter: 'all', chosen: null })
    expect(ids(view.shown)).toEqual(ids(TEMPLATES))
    expect(view.noMatch).toBe(false)
    expect(ids(galleryView(TEMPLATES, { search: 'photo', filter: 'no_photo', chosen: null }).shown)).toEqual([])
    expect(ids(galleryView(TEMPLATES, { search: 'for a', filter: 'photo', chosen: null }).shown)).toEqual(['tpl-offer', 'tpl-review'])
  })

  it('keeps the chosen template, in its own place, when the search or a chip would hide it', () => {
    const words = galleryView(TEMPLATES, { search: '', filter: 'no_photo', chosen: 'tpl-review' })
    expect(ids(words.shown)).toEqual(['tpl-quote', 'tpl-review', 'tpl-own'])
    expect(words.noMatch).toBe(false)
    const none = galleryView(TEMPLATES, { search: 'zebra', filter: 'all', chosen: 'tpl-review' })
    expect(ids(none.shown)).toEqual(['tpl-review'])
    expect(none.noMatch).toBe(true) // the chosen one is shown, not counted as a match
  })

  it('says nothing matches only when the search or a chip narrowed a gallery that has templates', () => {
    expect(galleryView(TEMPLATES, { search: 'zebra', filter: 'all', chosen: null }).noMatch).toBe(true)
    expect(galleryView([QUOTE, OWN], { search: '', filter: 'photo', chosen: null }).noMatch).toBe(true)
    expect(galleryView(TEMPLATES, { search: '  ', filter: 'all', chosen: null }).noMatch).toBe(false)
    expect(galleryView([], { search: 'zebra', filter: 'photo', chosen: null }).noMatch).toBe(false)
  })
})

describe('the no-match line', () => {
  it("names the search's words, and the chip when it is not All", () => {
    expect(noMatchLine(' zebra ', 'all')).toBe('No template matches "zebra".')
    expect(noMatchLine('zebra', 'photo')).toBe('No template with a photo matches "zebra".')
    expect(noMatchLine('zebra', 'no_photo')).toBe('No template without a photo matches "zebra".')
    expect(noMatchLine('', 'photo')).toBe('There is no template with a photo for this format.')
    expect(noMatchLine('', 'no_photo')).toBe('There is no template without a photo for this format.')
  })
})
