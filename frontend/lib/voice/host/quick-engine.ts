'use client'

/**
 * Theme voice — the web quick-ask engine (hold to talk, the browser's own recognition, `POST /api/chat` in spoken
 * mode, the reply spoken back), created once per tab on first use. This is the host's one import of it.
 *
 * The engine is the quick-ask lane's (lib/voice/quick-ask, `createQuickAskEngine(deps?)`). Until the two lanes merge
 * this lane builds without it, so the import below reads a stand-in; the merge swaps that one line for
 * `import { createQuickAskEngine } from '@/lib/voice/quick-ask'` and deletes ./quick-ask-stand-in.ts with its test.
 */
import type { VoiceEngine } from '@/lib/voice/engine'

import { createQuickAskEngine } from './quick-ask-stand-in'

let shared: VoiceEngine | null = null

/** The tab's one quick-ask engine. */
export function quickAskEngine(): VoiceEngine {
  if (!shared) shared = createQuickAskEngine()
  return shared
}
