/**
 * PRD-268 §6.12.6, US-K02 (SDK 1.4.0) — which of the brief's rows Listen is reading, for a theme's `▸ SPEAKING` line
 * and the row it marks (HUD's BRIEFING builds its numbered rows in time with speech).
 *
 * What Listen reads is the server's (services/brief/build.py spoken_words): the greeting, then "1. Title." for each
 * shown row, numbered in the sections' order (needs you, suggestions, today, since your last brief, couldn't check),
 * or the empty brief's sentence. Listen speaks it sentence by sentence (use-brief-voice.ts spokenSentences), and the
 * speaker cuts a long sentence into pieces (speechPieces). So each piece's row is known before anything plays:
 * - a sentence that starts with the next row's number is that row's, its index the number less one;
 * - a sentence that doesn't continues the row before it (a title the chunker cut), and a number that isn't the next
 *   one (a year in a title) starts nothing;
 * - a row whose title has nothing to say leaves its number before the next one's ("3. 4. Payroll."): the numbers
 *   still count on, so the sentence is the last one's row, and the silent row is never reported;
 * - what comes before the first row (the greeting, or the empty brief's sentence) is no row's.
 *
 * The row is reported as Auto reads: a row's index when its first piece starts, and null when its last piece ends,
 * so it is null between rows, during the greeting, and once the reading ends or stops. It is read off the port Listen
 * speaks through, wrapped here with its events passed on unchanged: the voice engine (the speaker, the server voice
 * and the browser's speech behind it) is not changed. A reading's late events, once it was stopped or another began,
 * report nothing.
 */
import { speechPieces, type SynthesisPort, type UtteranceEvents } from '@/lib/voice/quick-ask/speaker'

/** A brief row's index while Auto reads it; null while it reads none. */
export type SpokenRow = number | null

/** The port Listen speaks through, which also says which row each piece it queues belongs to. */
export interface RowTrackingPort extends SynthesisPort {
  /** A reading of `sentences` starts: the pieces queued next are its pieces, in the speaker's order. */
  begin(sentences: readonly string[]): void
  /** The reading is over (stopped, or another begins): no row, and its late events report nothing. */
  end(): void
}

interface Reading {
  readonly id: number
  /** Each piece's row, in the order the speaker queues them. */
  readonly rows: readonly SpokenRow[]
  /** The pieces of this reading queued so far. */
  readonly queued: number
  /** The piece whose audio started last; null before the first. */
  readonly current: number | null
}

/** The row numbers a sentence starts with: "12. Payroll." starts with 12, "3. 4. Payroll." with 3 and 4. */
const LEADING_NUMBERS = /^(?:\d+\.(?:\s+|$))+/
const DIGITS = /\d+/g

function numbersBegun(sentence: string): readonly number[] {
  const leading = LEADING_NUMBERS.exec(sentence.trim())?.[0] ?? ''
  return (leading.match(DIGITS) ?? []).map(Number)
}

/**
 * The row a sentence starts: the last of its leading numbers that count on from the next row's (`next` is its index),
 * so "1. 2026. Plan." is the first row's and "3. 4. Payroll." the fourth's; null when the first doesn't count on.
 */
function rowBegun(sentence: string, next: number): SpokenRow {
  const numbers = numbersBegun(sentence)
  const stop = numbers.findIndex((number, index) => number !== next + 1 + index)
  const counted = stop < 0 ? numbers.length : stop
  return counted > 0 ? next + counted - 1 : null
}

/** Each sentence's row: the next row's number starts it, other sentences continue the row before; none before the first. */
export function sentenceRows(sentences: readonly string[]): readonly SpokenRow[] {
  return sentences.reduce<readonly SpokenRow[]>((rows, sentence) => {
    const before = rows.length > 0 ? rows[rows.length - 1] : null
    return [...rows, rowBegun(sentence, before === null ? 0 : before + 1) ?? before]
  }, [])
}

/** Each piece's row, in the order the speaker queues the pieces (each sentence cut by speechPieces). */
export function pieceRows(sentences: readonly string[]): readonly SpokenRow[] {
  const rows = sentenceRows(sentences)
  return sentences.flatMap((sentence, index) => speechPieces(sentence).map(() => rows[index]))
}

/** `port`, telling `report` the row Auto reads as each piece starts and ends; every event is passed on unchanged. */
export function rowTrackingPort(port: SynthesisPort, report: (row: SpokenRow) => void): RowTrackingPort {
  let reading: Reading = { id: 0, rows: [], queued: 0, current: null }

  function tracked(events: UtteranceEvents): UtteranceEvents {
    const { id, rows, queued: piece } = reading
    reading = { ...reading, queued: piece + 1 }
    const row = rows[piece] ?? null
    const live = (): boolean => reading.id === id
    return {
      onStart: () => {
        events.onStart()
        if (!live()) return
        reading = { ...reading, current: piece }
        report(row)
      },
      onBoundary: () => events.onBoundary(),
      onEnd: () => {
        events.onEnd()
        const rowEnds = (rows[piece + 1] ?? null) !== row
        if (live() && reading.current === piece && rowEnds) report(null)
      },
    }
  }

  function restart(rows: readonly SpokenRow[]): void {
    reading = { id: reading.id + 1, rows, queued: 0, current: null }
    report(null)
  }

  return Object.freeze({
    speak: (text: string, events: UtteranceEvents) => port.speak(text, tracked(events)),
    cancel: () => port.cancel(),
    prime: () => port.prime(),
    begin: (sentences: readonly string[]) => restart(pieceRows(sentences)),
    end: () => restart([]),
  })
}
