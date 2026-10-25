/**
 * Theme voice, web quick-ask: every word it says or shows. Errors are plain sentences for the member (the voice
 * contract's snapshot.error), never a code or an object.
 */
import { FALLBACK_TURN_ERROR } from '@/lib/chat/errors'

/**
 * The AI notice (PRD-264 EUAI-04), spoken once per browser session before the first reply. The backend's own words:
 * orchestrator/modules/policy/ai_disclosure.py VOICE_GREETING, the default of config.py VOICE_AI_GREETING that Auto
 * Live speaks as its opening line. No route serves the greeting to the browser, so this is the one frontend copy;
 * copy.test.ts fails when the backend's words change and this one does not. ai-notice.ts puts the answering agent's
 * name in place of Auto.
 */
export const AI_VOICE_GREETING = 'Hi, this is Auto, your AI assistant.'

/** What the voice says in place of what speech cannot carry; the chat shows the real thing. */
export const SPOKEN_COPY = Object.freeze({
  code: "There's code in the chat.",
  table: "There's a table in the chat.",
  link: 'a link',
})

/** Why quick-ask could not go on, as the member reads it. */
export const QUICK_ASK_ERRORS = Object.freeze({
  permission:
    "Automatos can't hear you: the microphone is blocked for this site. Allow it in your browser's site settings, then try again.",
  noSpeech: 'Nothing was heard. Hold to talk and try again.',
  noMicrophone: 'No microphone was found. Connect one, then hold to talk again.',
  unsupported: "This browser can't recognise speech, so voice isn't available here. You can type to Auto instead.",
  serviceOff:
    "Speech recognition is turned off for this browser. Turn on dictation in your device's settings, or type to Auto instead.",
  recognitionNetwork: "Speech recognition couldn't connect. Check your connection and try again.",
  recognizer: 'Speech recognition stopped. Hold to talk again.',
  chatNetwork: "Auto can't be reached. Check your connection and try again.",
  turn: FALLBACK_TURN_ERROR,
})

export type QuickAskErrorKey = keyof typeof QUICK_ASK_ERRORS
