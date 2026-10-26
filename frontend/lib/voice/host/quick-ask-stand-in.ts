/**
 * Theme voice — the stand-in for the quick-ask lane's engine while this lane builds alone (see ./quick-engine.ts):
 * the same factory name and contract, and a start that fails at once with a plain sentence, so nothing listens and
 * nothing is sent. The lanes' merge deletes this file.
 */
import { IDLE_VOICE, type VoiceEngine, type VoiceLevels, type VoiceSnapshot } from '@/lib/voice/engine'

export const QUICK_ASK_NOT_BUILT = "Quick voice isn't in this build yet. Try Auto Live."

const ZERO_LEVELS: VoiceLevels = Object.freeze({ user: 0, agent: 0 })

/** A quick-ask engine that refuses every start in words. */
export function createQuickAskEngine(): VoiceEngine {
  let shown: VoiceSnapshot = IDLE_VOICE
  const listeners = new Set<(snapshot: VoiceSnapshot) => void>()
  const show = (next: VoiceSnapshot) => {
    shown = next
    listeners.forEach((listener) => listener(shown))
  }
  return Object.freeze({
    start: async ({ chatId }: { readonly chatId: string | null }) => {
      show(Object.freeze({ ...IDLE_VOICE, mode: 'quick' as const, phase: 'error' as const, error: QUICK_ASK_NOT_BUILT, chatId }))
    },
    release: () => undefined,
    stop: () => show(IDLE_VOICE),
    snapshot: () => shown,
    subscribe: (listener: (snapshot: VoiceSnapshot) => void) => {
      listeners.add(listener)
      return () => {
        listeners.delete(listener)
      }
    },
    levels: () => ZERO_LEVELS,
    dispose: () => listeners.clear(),
  })
}
