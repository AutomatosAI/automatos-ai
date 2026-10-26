'use client'

/**
 * Theme voice — the web quick-ask engine (hold to talk, the browser's own recognition, `POST /api/chat` in spoken
 * mode, the reply spoken back), created once per tab on first use. This is the host's one import of it.
 *
 * The engine is lib/voice/quick-ask's (`createQuickAskEngine(deps?)`), with the browser's own deps.
 */
import type { VoiceEngine } from '@/lib/voice/engine'

import { createQuickAskEngine } from '@/lib/voice/quick-ask'

let shared: VoiceEngine | null = null

/** The tab's one quick-ask engine. */
export function quickAskEngine(): VoiceEngine {
  if (!shared) shared = createQuickAskEngine()
  return shared
}
