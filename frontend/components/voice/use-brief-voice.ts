'use client'

/**
 * PRD-268 §6.12.6, K11, K17 (US-C21d) — Listen's speech: the brief's spoken text in Auto's voice.
 *
 * The voice is quick-ask's (lib/voice/quick-ask/deps.ts speechPort): the server voice (server-speech.ts, POST
 * /api/voice/speech in the page session's voice for the page's language), and the browser's speech for the rest of
 * the brief once the server refuses (a refusal code: tts_off also keeps the server off for the page session) or
 * fails. The port is built on the first press, so a mount asks nothing of the server and plays nothing: only listen()
 * speaks, and it primes the audio inside the press's gesture. The text goes sentence by sentence, so the first
 * sentence plays while the next is made; a row's number ("1.") goes with its title, never alone.
 *
 * speaking is true from the press until the last sentence has ended or stop() cancels every request and the audio;
 * unmounting stops it too, and so does the tab's voice session starting (`sessionRunning`: a hold or a call), so
 * Auto never talks over itself and the microphone never hears Listen. Only a reading still going is cancelled: the
 * browser's speech is the page's one queue, so a finished Listen never cuts off another voice (a quick ask's reply).
 *
 * row (SDK 1.4.0) is the brief row Auto is reading, its index in the brief's rows in the sections' order, or null
 * while it reads none (brief-voice-rows.ts: read off the port's own events, the engine unchanged).
 */
import { type MutableRefObject, useCallback, useEffect, useRef, useState } from 'react'
import { SYSTEM_CLOCK } from '@/lib/voice/quick-ask/clock'
import { speechPort } from '@/lib/voice/quick-ask/deps'
import { sentenceChunker } from '@/lib/voice/quick-ask/sentences'
import { createSpeaker, type Speaker } from '@/lib/voice/quick-ask/speaker'

import { rowTrackingPort, type RowTrackingPort, type SpokenRow } from './brief-voice-rows'

interface BriefSpeech {
  readonly port: RowTrackingPort
  readonly speaker: Speaker
}

export interface BriefVoiceControl {
  /** Auto is reading the brief. */
  readonly speaking: boolean
  /** The brief row Auto is reading (its index, in the sections' order); null while it reads none. */
  readonly row: SpokenRow
  /** Reads the spoken text from its start; call it only from the person's press. */
  readonly listen: () => void
  readonly stop: () => void
}

const LOG_PREFIX = '[voice]'
/** A sentence that is only a row's number: "1.", "12." (spoken_words numbers the rows). */
const ROW_NUMBER = /^\d+\.$/

/** The spoken text as sentences (quick-ask's chunker), each row's number joined to its title. */
export function spokenSentences(spoken: string): readonly string[] {
  const chunker = sentenceChunker()
  return [...chunker.push(spoken), ...chunker.flush()].reduce<readonly string[]>((joined, sentence) => {
    const last = joined[joined.length - 1]
    return last !== undefined && ROW_NUMBER.test(last) ? [...joined.slice(0, -1), `${last} ${sentence}`] : [...joined, sentence]
  }, [])
}

/** Cancels the reading if it is still going. */
function stopSpeech(speech: MutableRefObject<BriefSpeech | null>): void {
  const current = speech.current
  if (current?.speaker.speaking()) current.speaker.stop()
}

/** Stops the reading and ends it, so no row is being read; its late events report nothing. */
function endSpeech(speech: MutableRefObject<BriefSpeech | null>): void {
  stopSpeech(speech)
  speech.current?.port.end()
}

/** The voice for Listen, made on the first press; null (logged) when this browser can't make one. */
function newSpeech(onDrained: () => void, onRow: (row: SpokenRow) => void): BriefSpeech | null {
  try {
    const port = rowTrackingPort(speechPort(), onRow)
    const speaker = createSpeaker(port, SYSTEM_CLOCK, (event) => {
      if (event.type === 'drained') onDrained()
    })
    return Object.freeze({ port, speaker })
  } catch (error) {
    console.warn(LOG_PREFIX, "Listen couldn't start a voice", error)
    return null
  }
}

/** Tells `handler` the row, logging (never rethrowing) a failure: a theme's handler never breaks Auto's speech. */
function tell(handler: ((row: SpokenRow) => void) | undefined, row: SpokenRow): void {
  try {
    handler?.(row)
  } catch (error) {
    console.warn(LOG_PREFIX, 'onSpeakingRow failed', error)
  }
}

/**
 * The row Auto reads: its state, for the root's attribute, and `report`, which also tells `onRow` straight from the
 * speech's event, each change once, so a clip that starts and ends in one render batch is still heard.
 */
function useRowReport(onRow: ((row: SpokenRow) => void) | undefined) {
  const [row, setRow] = useState<SpokenRow>(null)
  const handler = useRef(onRow)
  const reported = useRef<SpokenRow>(null)
  handler.current = onRow
  const report = useCallback((next: SpokenRow) => {
    setRow(next)
    if (next === reported.current) return
    reported.current = next
    tell(handler.current, next)
  }, [])
  return { row, report }
}

/**
 * Listen and Stop over the brief's spoken text; `sessionRunning` stops Listen while the tab's voice session runs, and
 * `onRow` hears the row Auto reads each time it changes (null once it reads none, unmounting included).
 */
export function useBriefVoice(spoken: string, sessionRunning = false, onRow?: (row: SpokenRow) => void): BriefVoiceControl {
  const [speaking, setSpeaking] = useState(false)
  const { row, report } = useRowReport(onRow)
  const speech = useRef<BriefSpeech | null>(null)
  const drained = useCallback(() => {
    setSpeaking(false)
    report(null)
  }, [report])
  const listen = useCallback(() => {
    const sentences = spokenSentences(spoken)
    if (sentences.length === 0) return
    speech.current = speech.current ?? newSpeech(drained, report)
    if (speech.current === null) return
    stopSpeech(speech)
    speech.current.port.begin(sentences)
    speech.current.port.prime()
    setSpeaking(true)
    speech.current.speaker.speak(sentences)
  }, [spoken, drained, report])
  const stop = useCallback(() => {
    endSpeech(speech)
    setSpeaking(false)
  }, [])
  useEffect(() => {
    if (sessionRunning) stop()
  }, [sessionRunning, stop])
  useEffect(() => () => endSpeech(speech), [])
  return { speaking, row, listen, stop }
}
