'use client'

/**
 * Theme voice — the tab's one voice host, wired to the app: the tab's engines (one Retell session, one quick-ask),
 * the chat session store (the conversation on screen, and a new chat taking the one the session names), the
 * live-turns store, and the window's chat_changed signal that a quick-ask turn's saved rows are ready.
 *
 * Every voice surface (Studio's chat under app/chat/layout.tsx, a theme's routes under app/x/[...path]/page.tsx)
 * mounts a VoiceProvider over this one host, so moving between them keeps the session. A provider holds the host
 * while it is mounted; when none holds it for VOICE_HANDOVER_MS (the member left every surface with voice, and no
 * route change is mounting the next one) the session ends: a call is never left running with no controls on screen.
 */
import { liveEngine } from '@/lib/voice/live/live-engine'
import { useLiveTurnStore } from '@/lib/voice/live-turns'
import { useChatSessionStore } from '@/stores/chat-session-store'

import { quickAskEngine } from './quick-engine'
import { VoiceHost, type VoiceHostDeps } from './voice-host'

/** The window event the SSE lane fans chat_changed out as (hooks/use-board-event-stream.ts). */
export const CHAT_CHANGED_EVENT = 'automatos:chat-changed'
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
    turnLanded: (chatId) => {
      window.dispatchEvent(new CustomEvent(CHAT_CHANGED_EVENT, { detail: { chat_id: chatId } }))
    },
  }
}

let shared: VoiceHost | null = null

/** The tab's one voice host. */
export function sharedVoiceHost(): VoiceHost {
  if (!shared) shared = new VoiceHost(appVoiceHostDeps())
  return shared
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
