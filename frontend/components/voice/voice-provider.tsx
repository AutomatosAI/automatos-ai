'use client'

/**
 * Theme voice — the host voice provider: the tab's one voice session (lib/voice/host/voice-host.ts) for every chat
 * inside it. app/chat/layout.tsx mounts it around the chat page's slot (Studio's chat, or a theme's /chat) and
 * app/x/[...path]/page.tsx around a theme's routes, so Studio's Live strip, Studio's presence wave and a theme's voice
 * controls all read the same session through useHostVoice().
 *
 * It binds the session to the conversation on screen (the chat session store), offers Live or quick-ask by what is
 * available (live-status and the browser), and holds the host while mounted. Nothing here waits on the AI label:
 * voice is never locked on the label's visibility (the product owner's transparency policy, 23 Oct 2026).
 */
import { type ReactNode, createContext, useContext, useEffect, useMemo, useSyncExternalStore } from 'react'

import type { VoiceSnapshot, VoiceLevels } from '@/lib/voice/engine'
import { type VoiceAvailability, type VoiceStartMode, resolveVoiceMode } from '@/lib/voice/host/availability'
import { holdVoiceHost, sharedVoiceHost } from '@/lib/voice/host/shared-host'
import { useVoiceAvailability } from '@/lib/voice/host/use-voice-availability'
import type { VoiceHost, VoiceHostState } from '@/lib/voice/host/voice-host'
import type { LiveCallDetails } from '@/lib/voice/live/live-engine'
import { useChatSessionStore } from '@/stores/chat-session-store'

export type { VoiceAvailability, VoiceStartMode } from '@/lib/voice/host/availability'
export { readHostVoiceLevels } from '@/lib/voice/host/shared-host'

export interface HostVoiceStartOptions {
  /** `auto` (the default): Live when it can start, else quick-ask. */
  readonly mode?: VoiceStartMode
  /** The conversation the session talks into (null: a new chat); the one on screen when left out. */
  readonly chatId?: string | null
}

/** The live call's microphone: what the strip shows and the two things it changes. */
export interface HostLiveControls extends LiveCallDetails {
  readonly toggleMute: () => void
  readonly pickDevice: (deviceId: string) => void
}

/** The host's voice as a chat surface reads it. */
export interface HostVoice {
  readonly snapshot: VoiceSnapshot
  readonly available: VoiceAvailability
  /** Read inside an animation frame; never triggers a render. */
  readonly readLevels: () => VoiceLevels
  /** Starts a session on the conversation on screen, or the one named; call it from the member's gesture. */
  readonly start: (options?: HostVoiceStartOptions) => Promise<void>
  /** Hold-to-talk: the member let go. */
  readonly release: () => void
  /** Ends the session or the reply now. */
  readonly stop: () => void
  readonly live: HostLiveControls
}

interface VoiceContextValue {
  readonly host: VoiceHost
  readonly available: VoiceAvailability
}

const VoiceContext = createContext<VoiceContextValue | null>(null)

const OUTSIDE_PROVIDER =
  'useHostVoice is called outside VoiceProvider: the provider is mounted around the chat route (app/chat/layout.tsx) and theme routes (app/x/[...path]/page.tsx).'

/** The session follows the conversation on screen: moving to another one ends it. */
function useScreenBinding(host: VoiceHost): void {
  const onScreen = useChatSessionStore((s) => s.session.activeChatId)
  useEffect(() => {
    host.screenChanged(onScreen)
  }, [host, onScreen])
}

export interface VoiceProviderProps {
  readonly children: ReactNode
  /** The host to run; the tab's one host when omitted (a test passes its own). */
  readonly host?: VoiceHost
}

export function VoiceProvider({ children, host: given }: VoiceProviderProps) {
  const host = useMemo(() => given ?? sharedVoiceHost(), [given])
  useEffect(() => holdVoiceHost(host), [host])
  useScreenBinding(host)
  const available = useVoiceAvailability()
  const value = useMemo(() => ({ host, available }), [host, available])
  return <VoiceContext.Provider value={value}>{children}</VoiceContext.Provider>
}

function hostVoiceOf(host: VoiceHost, state: VoiceHostState, available: VoiceAvailability): HostVoice {
  return Object.freeze({
    snapshot: state.snapshot,
    available,
    readLevels: host.readLevels,
    start: (options?: HostVoiceStartOptions) =>
      host.start(resolveVoiceMode(options?.mode ?? 'auto', available), options?.chatId),
    release: host.release,
    stop: host.stop,
    live: Object.freeze({ ...state.live, toggleMute: host.toggleMute, pickDevice: host.pickDevice }),
  })
}

const NO_HOST_SUBSCRIBE = () => () => undefined
const NO_HOST_STATE = (): VoiceHostState | null => null

/** The host's voice, or null outside a VoiceProvider (a page with no voice surface). */
export function useOptionalHostVoice(): HostVoice | null {
  const context = useContext(VoiceContext)
  const host = context?.host ?? null
  const state = useSyncExternalStore<VoiceHostState | null>(
    host ? host.subscribe : NO_HOST_SUBSCRIBE,
    host ? host.getState : NO_HOST_STATE,
    host ? host.getState : NO_HOST_STATE,
  )
  const available = context?.available ?? null
  return useMemo(() => (host && state && available ? hostVoiceOf(host, state, available) : null), [host, state, available])
}

/** The host's voice: a frozen view, remade only when the session or the availability changes. */
export function useHostVoice(): HostVoice {
  const voice = useOptionalHostVoice()
  if (!voice) throw new Error(OUTSIDE_PROVIDER)
  return voice
}
