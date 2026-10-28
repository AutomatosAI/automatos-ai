/**
 * Theme voice, web quick-ask: its numbers in one place (the values ported from automatos-go's
 * src/voice/config.ts, PR #15, where they apply in a browser too).
 */

/** Listening stops itself after this long, as if the member had let go. */
export const HOLD_MAX_MS = 60_000

/**
 * After release the browser's recognition hands over its last words and ends; when no end comes within this long
 * (a browser that never fires it), the hold ends with the words heard so far.
 */
export const RELEASE_GRACE_MS = 2_000

/** durationSec counts the hold in whole seconds, re-published this often while the microphone is open. */
export const DURATION_TICK_MS = 1_000

/**
 * The longest piece of a sentence handed to speechSynthesis as one utterance; a longer sentence goes in pieces cut
 * at a space. Chrome's network voices stop part way through a long utterance and never say it ended (automatos-go's
 * speaker-core.ts fitToEngine does the same for Android's input limit).
 */
export const SPEECH_MAX_CHARS = 200

/**
 * Auto's server voice (POST /api/voice/speech, server-speech.ts): how many pieces after the one playing have their
 * MP3 asked for. One: the next sentence is synthesised while this one plays (about 0.5 s for a short sentence, up to
 * 2 s for a long one), so there is no gap, and a reply never asks for all its sentences at once.
 */
export const SERVER_SPEECH_PREFETCH_AHEAD = 1

/**
 * How long the piece due to play next (nothing playing before it) waits for its MP3; then the browser's voice speaks
 * it, and the next piece asks the server again. automatos-go's SERVER_VOICE_WAIT_MS.
 */
export const SERVER_VOICE_WAIT_MS = 4_000

/**
 * Said as the WHOLE utterance while holding, these stop Auto and send nothing (stop-phrase.ts drops a leading
 * 'auto' or 'hey auto' first). 'stop the deploy' is a request and goes to Auto as said.
 */
export const STOP_PHRASES: readonly string[] = Object.freeze(['stop', 'auto stop', 'stop auto', 'stop it', 'cancel'])

/** This tab's sessionStorage key: set once the session has heard the AI notice (ai-notice.ts). */
export const AI_NOTICE_STORAGE_KEY = 'automatos:voice:ai-notice'
/** The value stored under AI_NOTICE_STORAGE_KEY. */
export const AI_NOTICE_HEARD = '1'

/**
 * levels(): the browser's recognition does not share its audio, so the member's level is an envelope of recognition
 * activity (each recognised word lifts it to the peak, then it falls away over LEVEL_DECAY_MS), and Auto's is an
 * envelope of speech synthesis (word boundaries lift it; it never drops below the floor while Auto speaks).
 */
export const USER_LEVEL_PEAK = 0.8
export const AGENT_LEVEL_PEAK = 0.9
export const AGENT_LEVEL_FLOOR = 0.3
export const LEVEL_DECAY_MS = 350
