/**
 * Who made a deliverable (4 Oct 2026): the agent when it has one, else where it came from.
 * Socials renders and uploads carry no agent and read "Unknown agent" no more.
 */
import { describe, expect, it } from 'vitest'
import { MADE_BY_FALLBACK, madeBy } from '../made-by'

describe('madeBy', () => {
  it('names the agent when there is one', () => {
    expect(madeBy('ATLAS', 'social_post')).toBe('ATLAS')
  })

  it('names the source when there is no agent', () => {
    expect(madeBy(null, 'social_post')).toBe('Socials')
    expect(madeBy(undefined, 'upload')).toBe('Uploaded')
    expect(madeBy('', 'mission')).toBe('Mission')
  })

  it('names the generated, template and trigger sources too (issue #947)', () => {
    expect(madeBy(null, 'agent_output')).toBe('Generated')
    expect(madeBy(null, 'document')).toBe('Templates')
    expect(madeBy(null, 'trigger')).toBe('Trigger')
    expect(madeBy('ATLAS', 'agent_output')).toBe('ATLAS')
  })

  it('falls back only when neither is known', () => {
    expect(madeBy(null, 'something_new')).toBe(MADE_BY_FALLBACK)
    expect(madeBy(null, null)).toBe(MADE_BY_FALLBACK)
  })
})
