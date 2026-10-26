/**
 * Theme voice, web quick-ask: "Auto, stop". Said as the WHOLE utterance while holding, a stop phrase acts as Stop and
 * nothing is sent. Ported from automatos-go src/voice/quick-ask/stop-phrase.ts (PR #15). The words are lower-cased,
 * punctuation and symbols become spaces, and a leading wake word ('hey auto', 'auto') is dropped; what is left must be
 * one of STOP_PHRASES exactly. 'Auto, stop.' and 'Hey Auto, cancel!' stop; 'stop the deploy' and 'can you stop' are
 * requests, sent to Auto as said.
 */
import { STOP_PHRASES } from './config'

/** Punctuation and symbols, in any script: 'Auto, stop.' and 'Auto — stop!' read alike. */
const NOT_WORDS = /[\p{P}\p{S}]+/gu

/** What may come before the phrase, longest first ('hey auto stop' is not 'hey' + 'auto stop'). */
const WAKE_WORDS: readonly string[] = Object.freeze(['hey auto', 'auto'])

/** The utterance as bare words: lower case, no punctuation, single spaces. */
export function utteranceWords(text: string): string {
  return text.toLowerCase().replace(NOT_WORDS, ' ').replace(/\s+/g, ' ').trim()
}

/** The words after a leading wake word; the words as they are when none leads. */
function withoutWakeWord(words: string): string {
  const wake = WAKE_WORDS.find((word) => words === word || words.startsWith(`${word} `))
  return wake === undefined ? words : words.slice(wake.length).trim()
}

const PHRASES: ReadonlySet<string> = new Set(STOP_PHRASES.map(utteranceWords))

/** The whole utterance is a stop phrase, with or without the wake word before it. */
export function isStopPhrase(text: string): boolean {
  const words = utteranceWords(text)
  if (!words) return false
  return PHRASES.has(words) || PHRASES.has(withoutWakeWord(words))
}
