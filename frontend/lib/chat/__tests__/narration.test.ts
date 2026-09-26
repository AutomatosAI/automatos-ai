/**
 * F186 (night 6) — pre-tool narration leaves the answer and goes to the activity trail.
 */
import { describe, it, expect } from 'vitest'
import type { MessagePart } from '@/types'
import { narrationLines, withoutNarration } from '../narration'

const NARRATION =
  'I see ticket #1110 on your board, waiting for the Shopify Support Agent. I will check the current activity for you.'
const ANSWER = 'You got it, Gerard! Ticket #1110 was completed at 03:04.'

describe('withoutNarration', () => {
  it('takes the marked round out of what has streamed so far', () => {
    expect(withoutNarration(NARRATION, NARRATION)).toBe('')
    expect(withoutNarration(`${NARRATION}${ANSWER}`, NARRATION)).toBe(ANSWER)
  })

  it('takes the last occurrence: an earlier identical line stays', () => {
    expect(withoutNarration('Checking.Done.Checking.', 'Checking.')).toBe('Checking.Done.')
  })

  it('leaves the text alone when the marked round is not in it', () => {
    expect(withoutNarration(ANSWER, NARRATION)).toBe(ANSWER)
    expect(withoutNarration(ANSWER, '')).toBe(ANSWER)
  })
})

describe('narrationLines', () => {
  it('reads a saved reply’s narration part as trail lines, and nothing else', () => {
    const parts: MessagePart[] = [
      { type: 'reasoning', reasoning: 'hmm' },
      { type: 'narration', narration: `${NARRATION}\n\nLet me look at the board too.` },
      { type: 'text', text: ANSWER },
    ]
    expect(narrationLines(parts)).toEqual([NARRATION, 'Let me look at the board too.'])
  })

  it('is empty for a reply without narration', () => {
    expect(narrationLines([{ type: 'text', text: ANSWER }])).toEqual([])
    expect(narrationLines(undefined)).toEqual([])
  })
})
