'use client'

/**
 * Theme voice — the tab's one voice host, wired to the app: the tab's engines (one Retell session, one quick-ask),
 * the chat session store (the conversation on screen, and a new chat taking the one the session names), and the
 * live-turns store, whose exchange in progress is settled when its conversation's saved rows land (chat_changed).
 *
 * Every voice surface (Studio's chat under app/chat/layout.tsx, a theme's routes under app/x/[...path]/page.tsx)
 * mounts a VoiceProvider over this one host, so moving between them keeps the session. A provider holds the host
 * while it is mounted; when none holds it for VOICE_HANDOVER_MS (the member left every surface with voice, and no
 * route change is mounting the next one) the session ends: a call is never left running with no controls on screen.
 */
import type { VoiceLevels } from '@/lib/voice/engine'
import { liveEngine } from '@/lib/voice/live/live-engine'
import { useLiveTurnStore, watchSavedVoiceTurns } from '@/lib/voice/live-turns'
import { useChatSessionStore } from '@/stores/chat-session-store'

import { quickAskEngine } from './quick-engine'
import { VoiceHost, type VoiceHostDeps } from './voice-host'

/** How long a session outlives its last provider: a route change unmounts one and mounts the next in this time. */
export const VOICE_HANDOVER_MS = 1000

/** The host's ties to the app. */
export function appVoiceHostDeps(): VoiceHostDeps {
  return {
    liveEngine,
    quickEngine: quickAskEngine,
    onScreenChatId: () => useChatSessionStore.getState().session.activeChatId,
    adoptChat: (chatId) => useChatSessionStore.getState().chatIdAssigned(chatId),
    publishTurn: (turn) => useLiveTurnStore.getState().publish(turn),
    clearTurn: () => useLiveTurnStore.getState().clear(),
  }
}

let shared: VoiceHost | null = null

/** The tab's one voice host; the first call also starts settling saved exchanges for the tab. */
export function sharedVoiceHost(): VoiceHost {
  if (shared) return shared
  shared = new VoiceHost(appVoiceHostDeps())
  if (typeof window !== 'undefined') watchSavedVoiceTurns(window)
  return shared
}

/** The tab's voice levels, read inside an animation frame (a theme's waveform, a mascot): never triggers a render. */
export function readHostVoiceLevels(): VoiceLevels {
  return sharedVoiceHost().readLevels()
}

const holds = new WeakMap<VoiceHost, number>()

/** A provider holds `host` while mounted; returns the release. The last release ends the session after the handover. */
export function holdVoiceHost(host: VoiceHost): () => void {
  holds.set(host, (holds.get(host) ?? 0) + 1)
  return () => {
    holds.set(host, Math.max(0, (holds.get(host) ?? 0) - 1))
    setTimeout(() => {
      if ((holds.get(host) ?? 0) === 0) host.stop()
    }, VOICE_HANDOVER_MS)
  }
}
