/** Whether this browser can run quick-ask voice: speech recognition for the member, speech synthesis for Auto. */
export function quickAskSupported(): boolean {
  if (typeof window === 'undefined') return false
  const speech = window as unknown as Record<string, unknown>
  return ('SpeechRecognition' in speech || 'webkitSpeechRecognition' in speech) && 'speechSynthesis' in window
}
