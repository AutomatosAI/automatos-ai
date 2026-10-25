// @vitest-environment node
/**
 * Theme voice, web quick-ask — the sentence chunker (ported with automatos-go's sentences.test.ts, PR #15): each
 * sentence handed over once, as soon as the next word begins, never at the end of the text so far.
 */
import { describe, expect, it } from 'vitest'

import { sentenceChunker } from '@/lib/voice/quick-ask/sentences'

/** Streams the text in the given tokens: each push carries the whole text so far. */
function stream(tokens: readonly string[]): { pushed: string[][]; flushed: readonly string[] } {
  const chunker = sentenceChunker()
  const pushed = tokens.map((_token, index) => [...chunker.push(tokens.slice(0, index + 1).join(''))])
  return { pushed, flushed: chunker.flush() }
}

/** Word by word, the way a model streams: each token after the first carries its leading space. */
function words(text: string): readonly string[] {
  return text.split(/(?= )/)
}

describe('sentenceChunker', () => {
  it('hands over each sentence once, as soon as the next word begins', () => {
    const { pushed, flushed } = stream(words('Hello there. How are you today? I am fine! Thanks for asking.'))

    expect(pushed.flat()).toEqual(['Hello there.', 'How are you today?', 'I am fine!'])
    expect(flushed).toEqual(['Thanks for asking.'])
    expect(pushed[1]).toEqual([])
    expect(pushed[2]).toEqual(['Hello there.'])
  })

  it('never ends a sentence at the end of the text so far', () => {
    const chunker = sentenceChunker()

    expect(chunker.push('It is 3.')).toEqual([])
    expect(chunker.push('It is 3.5 degrees. ')).toEqual(['It is 3.5 degrees.'])
  })

  it('keeps decimals, versions and ticket numbers whole, token by token', () => {
    const { pushed, flushed } = stream([
      'Version 2', '.', '0', '.', '8 fixes ticket #0419', '.', ' It takes 3', '.', '5 seconds', '.',
    ])

    expect(pushed.flat()).toEqual(['Version 2.0.8 fixes ticket #0419.'])
    expect(flushed).toEqual(['It takes 3.5 seconds.'])
  })

  it('does not split after e.g., i.e., Mr., Dr., a.m. or an initial', () => {
    const text =
      'Dr. Lee and Mr. Brown meet at 9 a.m. tomorrow, e.g. in room four. Bring notes, i.e. the draft. J. R. R. Tolkien wrote it. Ms. Kay agrees. '

    expect(sentenceChunker().push(text)).toEqual([
      'Dr. Lee and Mr. Brown meet at 9 a.m. tomorrow, e.g. in room four.',
      'Bring notes, i.e. the draft.',
      'J. R. R. Tolkien wrote it.',
      'Ms. Kay agrees.',
    ])
  })

  it('ends at ?, !, an ellipsis and a run of them, with closing quotes and brackets', () => {
    expect(sentenceChunker().push('Wait… really?! He said "go." (Then left.) Yes... ok. ')).toEqual([
      'Wait…', 'really?!', 'He said "go."', '(Then left.)', 'Yes...', 'ok.',
    ])
  })

  it('emits nothing twice when the same text comes again', () => {
    const chunker = sentenceChunker()

    expect(chunker.push('One. Two. ')).toEqual(['One.', 'Two.'])
    expect(chunker.push('One. Two. ')).toEqual([])
    expect(chunker.push('One. Two. Three')).toEqual([])
    expect(chunker.flush()).toEqual(['Three'])
    expect(chunker.push('One. Two. Three')).toEqual([])
    expect(chunker.flush()).toEqual([])
  })

  it('restarts from the common prefix when narration leaves the answer', () => {
    const chunker = sentenceChunker()

    expect(chunker.push('Let me check your calendar. ')).toEqual(['Let me check your calendar.'])
    expect(chunker.push('')).toEqual([])
    expect(chunker.push('You have two meetings. ')).toEqual(['You have two meetings.'])
  })

  it('keeps what still stands in the common prefix', () => {
    const chunker = sentenceChunker()

    expect(chunker.push('Sure. Let me look. ')).toEqual(['Sure.', 'Let me look.'])
    expect(chunker.push('Sure. Your meeting is at 3. ')).toEqual(['Your meeting is at 3.'])
  })

  it('skips what has nothing to say, and flushes the rest once the turn finishes', () => {
    const chunker = sentenceChunker()

    expect(chunker.push('... Hello. ')).toEqual(['Hello.'])
    expect(chunker.push('... Hello. And one more thing')).toEqual([])
    expect(chunker.flush()).toEqual(['And one more thing'])
    expect(sentenceChunker().flush()).toEqual([])
  })
})
