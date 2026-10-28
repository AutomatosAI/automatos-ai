/**
 * Theme voice, web quick-ask: which server voice speaks Auto's replies, by the page's language (the one recognition
 * listens in; automatic reply-language detection is PRD-269). GET /api/voice/voices is asked once a page session:
 *
 *   1. the server's default voice, when it speaks the page's language in any accent (en-GB hears af_heart, en-US);
 *   2. else the first voice in exactly the page's accent (pt-BR hears a pt-BR voice);
 *   3. else, for a page language that names no region ('pt'), the first voice in that language.
 *
 * A page language with a region the server has no voice in exactly (pt-PT, where Kokoro has only pt-BR) gets none,
 * and so does a list that can't be had (tts_off, down, malformed): the browser's voice then speaks every piece, and no
 * piece is asked of the server. A tts_off answer to a piece turns the server voice off for the page session too.
 */
export interface ServerVoiceListing {
  readonly id: string
  /** A BCP 47 tag: en-US, en-GB, pt-BR, es-ES, ... */
  readonly accent: string
}

export interface ServerVoiceList {
  readonly defaultId: string | null
  readonly voices: readonly ServerVoiceListing[]
}

/** The page session's server voice, shared by every engine in the tab. */
export interface ServerVoiceSession {
  /** The voice for the page's language: the list is fetched on the first call only; none (null) on any failure. */
  voice(): Promise<string | null>
  /** The session knows the server can't speak here: it answered tts_off, or has no voice for the language. */
  serverless(): boolean
  /** The server answered tts_off: no piece is asked of it again this page session. */
  turnOff(): void
}

const LOG_PREFIX = '[voice]'
const REGION_SEPARATOR = '-'

function languageOf(tag: string): string {
  return tag.toLowerCase().split(REGION_SEPARATOR)[0]
}

function isListing(entry: unknown): entry is ServerVoiceListing {
  if (typeof entry !== 'object' || entry === null) return false
  const { id, accent } = entry as { readonly id?: unknown; readonly accent?: unknown }
  return typeof id === 'string' && id.length > 0 && typeof accent === 'string' && accent.length > 0
}

/** GET /api/voice/voices's body, checked: {default, voices: [{id, accent, ...}]}; a malformed voice is dropped. */
export function parseVoiceList(body: unknown): ServerVoiceList | null {
  if (typeof body !== 'object' || body === null) return null
  const { default: defaultId, voices } = body as { readonly default?: unknown; readonly voices?: unknown }
  if (!Array.isArray(voices)) return null
  const listed = voices.filter(isListing).map(({ id, accent }) => Object.freeze({ id, accent }))
  return Object.freeze({ defaultId: typeof defaultId === 'string' ? defaultId : null, voices: Object.freeze(listed) })
}

/** The voice for the page's language (the rules above), or null for the browser's voice. */
export function chooseVoice(list: ServerVoiceList | null, pageLanguage: string): string | null {
  if (list === null) return null
  const wanted = pageLanguage.toLowerCase()
  const language = languageOf(wanted)
  const serverDefault = list.voices.find((voice) => voice.id === list.defaultId)
  if (serverDefault !== undefined && languageOf(serverDefault.accent) === language) return serverDefault.id
  const exact = list.voices.find((voice) => voice.accent.toLowerCase() === wanted)
  if (exact !== undefined) return exact.id
  if (wanted !== language) return null
  return list.voices.find((voice) => languageOf(voice.accent) === language)?.id ?? null
}

/** A page session over `fetchVoices` (GET /api/voice/voices) for the page's language. */
export function createServerVoiceSession(
  fetchVoices: () => Promise<unknown>,
  pageLanguage: string,
): ServerVoiceSession {
  let chosen: Promise<string | null> | null = null
  // undefined until the list has been read: then the voice, or null for none.
  let known: string | null | undefined
  let off = false

  function voice(): Promise<string | null> {
    if (chosen !== null) return chosen
    chosen = Promise.resolve()
      .then(fetchVoices)
      .then(
        (body) => chooseVoice(parseVoiceList(body), pageLanguage),
        (error: unknown) => {
          console.warn(LOG_PREFIX, "the server voice's list could not be had; the browser's voice speaks", error)
          return null
        },
      )
      .then((id) => {
        known = id
        return id
      })
    return chosen
  }

  return Object.freeze({
    voice,
    serverless: () => off || known === null,
    turnOff: () => {
      off = true
    },
  })
}
