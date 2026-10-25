/**
 * What this browser can do for voice, read where the host decides which engine to offer (the host voice provider's
 * availability) and where the quick-ask engine checks itself before it starts.
 */

interface SpeechCapableWindow {
  readonly SpeechRecognition?: unknown
  readonly webkitSpeechRecognition?: unknown
  readonly speechSynthesis?: unknown
}

/**
 * Web quick-ask runs on the browser's own speech: recognition to hear the member (`SpeechRecognition`, prefixed
 * `webkitSpeechRecognition` in Chrome and Safari) and synthesis to speak Auto's reply. False on the server.
 */
export function quickAskSupported(): boolean {
  if (typeof window === 'undefined') return false
  const speech = window as unknown as SpeechCapableWindow
  return Boolean((speech.SpeechRecognition || speech.webkitSpeechRecognition) && speech.speechSynthesis)
}
