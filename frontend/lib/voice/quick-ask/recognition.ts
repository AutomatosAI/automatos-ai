/**
 * Theme voice, web quick-ask: the browser's speech recognition (SpeechRecognition, or webkitSpeechRecognition in
 * Safari and older Chrome) behind a small port, so the engine and its tests never touch the browser object.
 *
 * One port per hold: continuous with interim results, so the member's words show as they are recognised and the
 * capture lasts until they let go (stop()), not until the first pause. The browser holds the microphone itself and
 * does not share the audio; the member's level is therefore an envelope of recognition activity (levels.ts). The
 * browser's recognition may send the audio to its vendor (Chrome: Google) — the open privacy decision D1 in the theme
 * voice research note.
 */
import type { QuickAskErrorKey } from './copy'

/** Why recognition could not go on, as quick-ask's error copy names it. */
export type RecognitionFailure = Extract<
  QuickAskErrorKey,
  'permission' | 'noSpeech' | 'noMicrophone' | 'unsupported' | 'serviceOff' | 'recognitionNetwork' | 'recognizer'
>

export interface RecognitionEvents {
  /** The browser began capturing audio: the microphone is open. Said once or twice (start, audiostart). */
  readonly onCaptureStart: () => void
  /** Capture ended: the microphone is closed. */
  readonly onCaptureEnd: () => void
  /** Everything recognised so far in this hold: the final words, then the current guess. */
  readonly onWords: (text: string) => void
  readonly onError: (failure: RecognitionFailure) => void
  /** Recognition is over: no more words will come. */
  readonly onEnd: () => void
}

export interface RecognitionPort {
  /** Starts listening; throws when the browser refuses to start. */
  start(events: RecognitionEvents): void
  /** Stops capture: the last words, then onEnd, follow. */
  stop(): void
  /** Ends now: no more words. */
  abort(): void
}

/** A new port for one hold, or null when this browser has no speech recognition. */
export type RecognitionFactory = () => RecognitionPort | null

interface BrowserAlternative {
  readonly transcript: string
}
interface BrowserResult {
  readonly length: number
  readonly [index: number]: BrowserAlternative | undefined
}
interface BrowserResultList {
  readonly length: number
  readonly [index: number]: BrowserResult | undefined
}

/** The slice of the browser's SpeechRecognition quick-ask uses (TypeScript's DOM library has no type for it). */
export interface BrowserRecognition {
  lang: string
  continuous: boolean
  interimResults: boolean
  maxAlternatives: number
  onstart: (() => void) | null
  onaudiostart: (() => void) | null
  onaudioend: (() => void) | null
  onresult: ((event: { readonly results: BrowserResultList }) => void) | null
  onerror: ((event: { readonly error: string }) => void) | null
  onend: (() => void) | null
  start(): void
  stop(): void
  abort(): void
}

export type BrowserRecognitionConstructor = new () => BrowserRecognition

/** The browser's error codes; null: our own abort, not a failure. Any other code is the recognizer stopping. */
const FAILURES: Readonly<Record<string, RecognitionFailure | null>> = Object.freeze({
  'not-allowed': 'permission',
  'service-not-allowed': 'serviceOff',
  'language-not-supported': 'unsupported',
  'no-speech': 'noSpeech',
  'audio-capture': 'noMicrophone',
  network: 'recognitionNetwork',
  aborted: null,
})

export function recognitionFailure(code: string): RecognitionFailure | null {
  return Object.prototype.hasOwnProperty.call(FAILURES, code) ? FAILURES[code] ?? null : 'recognizer'
}

/** Every result's best guess, in order, on one line. */
export function transcriptOf(results: BrowserResultList): string {
  const words: string[] = []
  for (let index = 0; index < results.length; index += 1) words.push(results[index]?.[0]?.transcript ?? '')
  return words.join(' ').replace(/\s+/g, ' ').trim()
}

function bind(recognition: BrowserRecognition, events: RecognitionEvents): void {
  recognition.onstart = () => events.onCaptureStart()
  recognition.onaudiostart = () => events.onCaptureStart()
  recognition.onaudioend = () => events.onCaptureEnd()
  recognition.onresult = (event) => events.onWords(transcriptOf(event.results))
  recognition.onerror = (event) => {
    const failure = recognitionFailure(event.error)
    if (failure !== null) events.onError(failure)
  }
  recognition.onend = () => {
    events.onCaptureEnd()
    events.onEnd()
  }
}

/** A port over one browser recognition object, listening in `lang`. */
export function browserRecognitionPort(Recognition: BrowserRecognitionConstructor, lang: string): RecognitionPort {
  const recognition = new Recognition()
  recognition.lang = lang
  recognition.continuous = true
  recognition.interimResults = true
  recognition.maxAlternatives = 1
  return Object.freeze({
    start: (events: RecognitionEvents) => {
      bind(recognition, events)
      recognition.start()
    },
    stop: () => recognition.stop(),
    abort: () => recognition.abort(),
  })
}

/** The browser's recognition constructor, either name, or null (Firefox, the server). */
export function browserRecognitionConstructor(): BrowserRecognitionConstructor | null {
  if (typeof window === 'undefined') return null
  const speech = window as unknown as Record<string, BrowserRecognitionConstructor | undefined>
  return speech.SpeechRecognition ?? speech.webkitSpeechRecognition ?? null
}
