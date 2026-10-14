/**
 * PRD-260 T-007b — /chat?draft=<text> (lib/play/draft.ts), ported from
 * Automatos Play's lib/chat/params.ts: the link a Deliverable's "Ask Auto about
 * this" opens, and the words Play's chat reads from it.
 */
import { describe, expect, it } from 'vitest'

import { cutDraft, DRAFT_PARAM, draftFrom, draftHref, MAX_DRAFT_CHARS } from '@/lib/play/draft'

function paramsOf(href: string): URLSearchParams {
  return new URL(href, 'http://play.invalid').searchParams
}

describe('draftHref and draftFrom', () => {
  it('round-trips the words through the chat URL', () => {
    const words = 'About my Deliverable "Laptop & bag: 50% off?": '
    const href = draftHref(words)
    expect(href.startsWith('/chat?draft=')).toBe(true)
    expect(draftFrom(paramsOf(href))).toBe(words)
  })

  it('reads nothing from a link without words, or with only spaces', () => {
    expect(draftFrom(new URLSearchParams(''))).toBeNull()
    expect(draftFrom(new URLSearchParams(`${DRAFT_PARAM}=%20%20`))).toBeNull()
    expect(draftFrom(null)).toBeNull()
  })

  it('cuts a long draft, in the link and when read', () => {
    const long = 'y'.repeat(MAX_DRAFT_CHARS + 40)
    expect(draftFrom(paramsOf(draftHref(long)))).toBe('y'.repeat(MAX_DRAFT_CHARS))
    expect(draftFrom(new URLSearchParams({ [DRAFT_PARAM]: long }))).toHaveLength(MAX_DRAFT_CHARS)
  })
})

describe('cutDraft', () => {
  it('never cuts an emoji in half', () => {
    const emoji = '😀'.repeat(MAX_DRAFT_CHARS + 1)
    expect(Array.from(cutDraft(emoji))).toHaveLength(MAX_DRAFT_CHARS)
  })

  it('replaces a lone surrogate, so the link can always be built', () => {
    const broken = `a\uD83Db\uDE00c`
    expect(cutDraft(broken)).toBe('a�b�c')
    expect(() => draftHref(broken)).not.toThrow()
  })
})
