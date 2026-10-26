/**
 * Theme voice — the host's one voice session per tab, whichever engine runs it (Auto Live, web quick-ask), bound to
 * the conversation on screen. The host voice provider (components/voice/voice-provider.tsx) is its React face; this
 * class holds the rules, so they are tested without React:
 *
 * - A start binds the session to the conversation on screen (null for a new chat) and runs the engine asked for; a
 *   start of the other engine stops the one running.
 * - A new chat takes the conversation the session names (the mint's chat_id, quick-ask's new chat): the host puts it
 *   on screen (the session store's chatIdAssigned), as a typed first message does.
 * - Voice belongs to the conversation on screen: moving to another one ends the session.
 * - The session's words in progress go to the conversation they belong to (lib/voice/live-turns.ts), so every chat
 *   on the host's stream shows them as they are spoken; a new session starts with none. The saved rows come with
 *   chat_changed: the backend's for a live call, the quick-ask engine's own announcement for a quick-ask turn.
 */
import {
  IDLE_VOICE,
  type VoiceEngine,
  type VoiceLevels,
  type VoiceMode,
  type VoicePhase,
  type VoiceSnapshot,
} from '@/lib/voice/engine'
import type { LiveCallDetails } from '@/lib/voice/live/live-engine'
import type { LiveVoiceTurn } from '@/lib/voice/live-turns'

/** The Live engine's extras beyond the shared contract: the call's microphone controls. */
export interface LiveVoiceEngine extends VoiceEngine {
  details(): LiveCallDetails
  toggleMute(): void
  pickDevice(deviceId: string): void
}

export interface VoiceHostDeps {
  readonly liveEngine: () => LiveVoiceEngine
  readonly quickEngine: () => VoiceEngine
  /** The conversation on screen; null for a new chat. */
  readonly onScreenChatId: () => string | null
  /** Puts on screen the conversation a session started from a new chat writes into. */
  readonly adoptChat: (chatId: string) => void
  readonly publishTurn: (turn: LiveVoiceTurn) => void
  readonly clearTurn: () => void
}

export interface VoiceHostState {
  readonly snapshot: VoiceSnapshot
  /** The live call's microphone details; empty while no live session is attached. */
  readonly live: LiveCallDetails
}

export const NO_LIVE_DETAILS: LiveCallDetails = Object.freeze({
  micSilent: false,
  inputDevices: Object.freeze([]),
  captureDeviceId: null,
})
const ZERO_LEVELS: VoiceLevels = Object.freeze({ user: 0, agent: 0 })
const RUNNING: ReadonlySet<VoicePhase> = new Set<VoicePhase>(['connecting', 'listening', 'thinking', 'speaking'])

export class VoiceHost {
  private engine: VoiceEngine | null = null
  private mode: VoiceMode | null = null
  private detach: (() => void) | null = null
  private boundChatId: string | null = null
  private state: VoiceHostState = Object.freeze({ snapshot: IDLE_VOICE, live: NO_LIVE_DETAILS })
  private readonly listeners = new Set<() => void>()

  constructor(private readonly deps: VoiceHostDeps) {}

  /** Starts a session of `mode` on the conversation on screen. Call it from the member's gesture. */
  readonly start = (mode: VoiceMode): Promise<void> => {
    const engine = mode === 'live' ? this.deps.liveEngine() : this.deps.quickEngine()
    if (this.engine && this.engine !== engine) this.engine.stop()
    this.attach(engine, mode)
    this.boundChatId = this.deps.onScreenChatId()
    this.deps.clearTurn()
    return engine.start({ chatId: this.boundChatId })
  }

  readonly release = (): void => {
    this.engine?.release()
  }

  readonly stop = (): void => {
    this.engine?.stop()
  }

  readonly readLevels = (): VoiceLevels => this.engine?.levels() ?? ZERO_LEVELS

  readonly toggleMute = (): void => {
    this.attachedLive()?.toggleMute()
  }

  readonly pickDevice = (deviceId: string): void => {
    this.attachedLive()?.pickDevice(deviceId)
  }

  readonly getState = (): VoiceHostState => this.state

  readonly subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener)
    return () => {
      this.listeners.delete(listener)
    }
  }

  /** The conversation on screen is now `chatId`: a session bound to another one ends. */
  readonly screenChanged = (chatId: string | null): void => {
    if (this.state.snapshot.phase === 'idle' || chatId === this.boundChatId) return
    this.stop()
  }

  private attachedLive(): LiveVoiceEngine | null {
    return this.mode === 'live' && this.engine ? (this.engine as LiveVoiceEngine) : null
  }

  private attach(engine: VoiceEngine, mode: VoiceMode): void {
    this.mode = mode
    if (this.engine === engine) return
    this.detach?.()
    this.engine = engine
    this.detach = engine.subscribe((snapshot) => this.onEngine(snapshot))
    this.setState(engine.snapshot())
  }

  private onEngine(next: VoiceSnapshot): void {
    this.adopt(next)
    if (next.chatId) this.deps.publishTurn({ chatId: next.chatId, userText: next.heard, agentText: next.reply })
    this.setState(next)
  }

  /** A running session that names another conversation than the one it was bound to takes it on screen. */
  private adopt(next: VoiceSnapshot): void {
    if (!next.chatId || next.chatId === this.boundChatId || !RUNNING.has(next.phase)) return
    // Bound first: putting it on screen tells screenChanged, which must find the session where it now is.
    this.boundChatId = next.chatId
    this.deps.adoptChat(next.chatId)
  }

  private setState(snapshot: VoiceSnapshot): void {
    const live = this.attachedLive()?.details() ?? NO_LIVE_DETAILS
    if (snapshot === this.state.snapshot && live === this.state.live) return
    this.state = Object.freeze({ snapshot, live })
    this.listeners.forEach((listener) => listener())
  }
}
