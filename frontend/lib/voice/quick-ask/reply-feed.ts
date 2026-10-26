/**
 * Theme voice, web quick-ask: one reply, fed as it streams (automatos-go's reply-feed.ts, PR #15). Each view of
 * Auto's answer so far becomes the sentences new since the last one: toSpeakable over the whole text (markdown,
 * code and URLs out even when spokenMode already speechified it), cut by the chunker as sentences complete, so Auto
 * starts speaking before the turn ends; finish() hands over the rest. One feed per turn.
 */
import { sentenceChunker } from './sentences'
import { toSpeakable } from './speakable'

export interface ReplyFeed {
  /** The sentences new in this view of the answer so far. */
  readonly next: (answerSoFar: string) => readonly string[]
  /** The rest, once the turn has finished. */
  readonly finish: () => readonly string[]
}

export function createReplyFeed(): ReplyFeed {
  const chunker = sentenceChunker()
  let said = ''
  return Object.freeze({
    next: (answerSoFar: string) => {
      const text = toSpeakable(answerSoFar)
      if (text === said) return []
      said = text
      return chunker.push(text)
    },
    finish: () => chunker.flush(),
  })
}
