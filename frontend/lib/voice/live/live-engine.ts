'use client'

/**
 * Auto Live as a VoiceEngine (PRD-207 S5; theme voice): the live call the host voice provider runs for Studio's chat
 * and a theme's, replacing the useRetellCall hook that lived inside Studio's call strip.
 *
 * A start mints a short-lived access token server-side (`POST /api/voice/web-call`: the Retell key never reaches the
 * browser; the answer names the conversation the call writes into), then Retell's WebRTC SDK carries the audio both
 * ways (retell-call.ts) and drives Auto's own agent loop through the custom-LLM socket. Retell's events walk the
 * presence state machine (lib/voice/orb-state.ts: "thinking" is dead air after the member spoke); the running
 * transcript gives the exchange in progress (transcript.ts); a local analyser tap measures the member's microphone
 * for the levels and the silent-capture banner (mic-meter.ts, lib/voice/mic-health.ts). The member picks the capture
 * device (devices.ts) and can mute.
 *
 * One engine per tab (liveEngine()), so one Retell session per tab. Every call has a generation: stop() and a new
 * start() retire the old one, and a late answer or event of a retired call changes nothing (a stop during the mint
 * never connects). A stop the member asks for ends in `idle`; a call Retell ends is `ended`; a failure is `error`
 * with a plain sentence.
 */
import { apiClient } from '@/lib/api-client'
import { useAiLabelStore } from '@/lib/ai-label/store'
import { IDLE_VOICE, type VoiceEngine, type VoiceLevels, type VoiceSnapshot } from '@/lib/voice/engine'
import { feedMicLevel, initialMicHealth, type MicHealth } from '@/lib/voice/mic-health'
import { initialOrb, orbReducer, rmsLevel, USER_SPEECH_THRESHOLD, type OrbEvent, type OrbSnapshot } from '@/lib/voice/orb-state'

import {
  type MicInputDevice,
  listInputDevices,
  liveCallSupported,
  readStoredCaptureDevice,
  storeCaptureDevice,
  watchInputDevices,
} from './devices'
import { type MicMeter, isMicPermissionRefusal, openMicMeter } from './mic-meter'
import { LIVE_BROWSER_UNSUPPORTED, LIVE_CONNECT_FAILED, LIVE_MIC_DENIED, liveRefusalSentence } from './refusal'
import { type RetellCall, type RetellCallEvents, openRetellCall } from './retell-call'
import { currentExchange } from './transcript'

export const LIVE_MINT_PATH = '/api/voice/web-call'
const DURATION_TICK_MS = 1000
const ORB_TICK_MS = 250
const MS_PER_SECOND = 1000
/** Rebinding the SDK's capture needs a fresh call: the gap between ending one and starting the next. */
export const DEVICE_RESTART_MS = 200

export interface LiveCallDetails {
  /** The capture delivered a full window of digital silence while unmuted: the mic hears nothing. */
  readonly micSilent: boolean
  /** The audio inputs the browser offers (labelled once permission exists). */
  readonly inputDevices: readonly MicInputDevice[]
  /** The picked capture device; null for the browser default. */
  readonly captureDeviceId: string | null
}

export interface LiveEngineOptions {
  /** The agent the owner picked for the chat on screen; null while Auto answers. Read at each start. */
  readonly agentId: () => number | null
}

interface MintAnswer {
  readonly call_id: string
  readonly access_token: string
  readonly chat_id?: string | null
}

interface LiveView extends LiveCallDetails {
  readonly orb: OrbSnapshot
  readonly muted: boolean
  readonly heard: string
  readonly reply: string
  readonly durationSec: number
  readonly error: string | null
  readonly chatId: string | null
}

const ZERO_LEVELS: VoiceLevels = Object.freeze({ user: 0, agent: 0 })
const NO_DEVICES: readonly MicInputDevice[] = Object.freeze([])
const LIVE_PHASES: ReadonlySet<string> = new Set(['listening', 'thinking', 'speaking'])
const RUNNING_PHASES: ReadonlySet<string> = new Set(['connecting', 'listening', 'thinking', 'speaking'])
const INITIAL_VIEW: LiveView = Object.freeze({
  orb: initialOrb, muted: false, heard: '', reply: '', durationSec: 0, error: null, chatId: null,
  micSilent: false, inputDevices: NO_DEVICES, captureDeviceId: null,
})

function snapshotOf(view: LiveView): VoiceSnapshot {
  const phase = view.orb.state
  return Object.freeze({
    mode: phase === 'idle' ? null : 'live',
    phase,
    micOpen: LIVE_PHASES.has(phase) && !view.muted,
    muted: view.muted,
    heard: view.heard,
    reply: view.reply,
    durationSec: view.durationSec,
    error: phase === 'error' ? view.error : null,
    chatId: view.chatId,
  })
}

function detailsOf(view: LiveView): LiveCallDetails {
  return Object.freeze({ micSilent: view.micSilent, inputDevices: view.inputDevices, captureDeviceId: view.captureDeviceId })
}

function sameSnapshot(a: VoiceSnapshot, b: VoiceSnapshot): boolean {
  return (Object.keys(a) as (keyof VoiceSnapshot)[]).every((key) => a[key] === b[key])
}

function sameDetails(a: LiveCallDetails, b: LiveCallDetails): boolean {
  return a.micSilent === b.micSilent && a.inputDevices === b.inputDevices && a.captureDeviceId === b.captureDeviceId
}

function endCall(call: RetellCall): void {
  try {
    call.stop()
  } catch (error) {
    console.warn('[voice] the live call was already down', error)
  }
}

export class LiveEngine implements VoiceEngine {
  private view: LiveView = INITIAL_VIEW
  private shown: VoiceSnapshot = IDLE_VOICE
  private shownDetails: LiveCallDetails = detailsOf(INITIAL_VIEW)
  private readonly listeners = new Set<(snapshot: VoiceSnapshot) => void>()
  private levelsNow: VoiceLevels = ZERO_LEVELS
  private generation = 0
  private call: RetellCall | null = null
  private meter: MicMeter | null = null
  private timers: ReturnType<typeof setInterval>[] = []
  private micHealth: MicHealth = initialMicHealth(0)
  private stopWatchingDevices: (() => void) | null = null
  private storedDeviceRead = false

  constructor(private readonly options: LiveEngineOptions) {}

  async start({ chatId }: { readonly chatId: string | null }): Promise<void> {
    const gen = this.begin(chatId)
    if (!liveCallSupported()) return this.fail(LIVE_BROWSER_UNSUPPORTED)
    const minted = await this.mint(chatId, gen)
    if (!minted || gen !== this.generation) return
    if (minted.chat_id) this.update({ chatId: minted.chat_id })
    await this.connect(minted.access_token, gen)
    if (gen !== this.generation || this.view.orb.state === 'error') return
    await this.openMeter(gen)
    // Permission now exists, so the device labels are real.
    void this.refreshDevices()
  }

  /** Live has no hold-to-talk: the call listens until it ends. */
  release(): void {}

  stop(): void {
    this.releaseResources()
    this.update({ orb: initialOrb, muted: false, error: null, micSilent: false })
  }

  snapshot(): VoiceSnapshot {
    return this.shown
  }

  details(): LiveCallDetails {
    return this.shownDetails
  }

  subscribe(listener: (snapshot: VoiceSnapshot) => void): () => void {
    this.listeners.add(listener)
    return () => {
      this.listeners.delete(listener)
    }
  }

  levels(): VoiceLevels {
    return this.levelsNow
  }

  dispose(): void {
    this.releaseResources()
    this.stopWatchingDevices?.()
    this.stopWatchingDevices = null
    this.listeners.clear()
    this.view = INITIAL_VIEW
    this.shown = IDLE_VOICE
  }

  toggleMute(): void {
    const call = this.call
    if (!call) return
    const muted = !this.view.muted
    try {
      if (muted) call.mute()
      else call.unmute()
    } catch (error) {
      console.warn(`[voice] the microphone could not be ${muted ? 'muted' : 'unmuted'}`, error)
      return
    }
    this.update({ muted })
  }

  /** Captures from `deviceId` from now on (kept for the next call); a running call restarts on it. */
  pickDevice(deviceId: string): void {
    storeCaptureDevice(deviceId)
    this.update({ captureDeviceId: deviceId })
    if (!RUNNING_PHASES.has(this.view.orb.state)) return
    const chatId = this.view.chatId
    this.releaseResources()
    this.dispatch({ type: 'call_connecting' })
    const gen = this.generation
    setTimeout(() => {
      if (gen === this.generation) void this.start({ chatId })
    }, DEVICE_RESTART_MS)
  }

  /** Retires any call, opens a fresh one in `connecting`; returns its generation. */
  private begin(chatId: string | null): number {
    this.releaseResources()
    if (!this.stopWatchingDevices) this.stopWatchingDevices = watchInputDevices(() => void this.refreshDevices())
    const captureDeviceId = this.storedDeviceRead ? this.view.captureDeviceId : readStoredCaptureDevice()
    this.storedDeviceRead = true
    this.generation += 1
    this.update({
      ...INITIAL_VIEW,
      orb: orbReducer(initialOrb, { type: 'call_connecting' }),
      chatId,
      inputDevices: this.view.inputDevices,
      captureDeviceId,
    })
    return this.generation
  }

  private async mint(chatId: string | null, gen: number): Promise<MintAnswer | null> {
    const agentId = this.options.agentId()
    const body = { ...(chatId ? { chat_id: chatId } : {}), ...(agentId ? { agent_id: agentId } : {}) }
    try {
      return await apiClient.request<MintAnswer>(LIVE_MINT_PATH, { method: 'POST', body: body as unknown as BodyInit })
    } catch (error) {
      // Every refusal is the member's truth (plan, switched off, not armed, workspace off, budget used).
      if (gen === this.generation) this.fail(liveRefusalSentence(error))
      return null
    }
  }

  private async connect(accessToken: string, gen: number): Promise<void> {
    try {
      const call = await openRetellCall(this.callEvents(gen))
      if (gen !== this.generation) return
      this.call = call
      await call.start(accessToken, this.view.captureDeviceId)
    } catch (error) {
      if (gen !== this.generation) return
      console.error('[voice] the live call could not connect', error)
      this.fail(isMicPermissionRefusal(error) ? LIVE_MIC_DENIED : LIVE_CONNECT_FAILED)
    }
  }

  /** Retell's events for the call of generation `gen`; once it is retired they change nothing. */
  private callEvents(gen: number): RetellCallEvents {
    const current = <A extends unknown[]>(handler: (...args: A) => void) => (...args: A): void => {
      if (gen === this.generation) handler(...args)
    }
    return {
      onStarted: current(() => this.callStarted()),
      onEnded: current(() => {
        this.releaseResources()
        this.dispatch({ type: 'call_ended' })
      }),
      onError: current((message: unknown) => {
        console.error('[voice] the live call failed:', message)
        this.fail(LIVE_CONNECT_FAILED)
      }),
      onAgentTalking: current((talking: boolean) =>
        this.dispatch(talking ? { type: 'agent_start_talking' } : { type: 'agent_stop_talking', now: Date.now() }),
      ),
      onAgentAudio: current((samples: Float32Array) => {
        this.levelsNow = { ...this.levelsNow, agent: rmsLevel(samples) }
      }),
      onTranscript: current((transcript: unknown) => {
        const { userText, agentText } = currentExchange(transcript)
        this.update({ heard: userText, reply: agentText })
      }),
    }
  }

  private callStarted(): void {
    const startedAt = Date.now()
    this.dispatch({ type: 'call_started' })
    this.timers = [
      setInterval(() => this.update({ durationSec: Math.floor((Date.now() - startedAt) / MS_PER_SECOND) }), DURATION_TICK_MS),
      setInterval(() => this.dispatch({ type: 'tick', now: Date.now() }), ORB_TICK_MS),
    ]
  }

  private async openMeter(gen: number): Promise<void> {
    try {
      this.micHealth = initialMicHealth(Date.now())
      const meter = await openMicMeter(this.view.captureDeviceId, (level) => this.micLevel(level))
      if (gen !== this.generation) return meter.stop()
      this.meter = meter
    } catch (error) {
      if (gen !== this.generation) return
      if (isMicPermissionRefusal(error)) return this.fail(LIVE_MIC_DENIED)
      // No meter is cosmetic: the call works, the orb just won't react to the member's voice.
      console.warn('[voice] the microphone meter could not open; the call goes on without it', error)
    }
  }

  private micLevel(level: number): void {
    const now = Date.now()
    this.levelsNow = { ...this.levelsNow, user: level }
    if (level >= USER_SPEECH_THRESHOLD) this.dispatch({ type: 'user_voice', now })
    // Muted silence is intentional (the window stays fresh); unmuted digital silence for a full window raises it.
    this.micHealth = this.view.muted ? initialMicHealth(now) : feedMicLevel(this.micHealth, level, now)
    this.update({ micSilent: this.micHealth.silent })
  }

  private async refreshDevices(): Promise<void> {
    const inputDevices = await listInputDevices()
    const picked = this.view.captureDeviceId
    // A stored device that no longer exists must not pin the call to nothing.
    const gone = picked !== null && inputDevices.length > 0 && !inputDevices.some((device) => device.deviceId === picked)
    if (gone) storeCaptureDevice(null)
    this.update(gone ? { inputDevices, captureDeviceId: null } : { inputDevices })
  }

  private fail(message: string): void {
    this.releaseResources()
    this.update({ orb: orbReducer(this.view.orb, { type: 'error' }), error: message, micSilent: false })
  }

  /** Ends the call's audio, timers and meter, and retires its generation. */
  private releaseResources(): void {
    this.generation += 1
    this.timers.forEach((timer) => clearInterval(timer))
    this.timers = []
    this.meter?.stop()
    this.meter = null
    const call = this.call
    this.call = null
    if (call) endCall(call)
    this.levelsNow = ZERO_LEVELS
  }

  private dispatch(event: OrbEvent): void {
    this.update({ orb: orbReducer(this.view.orb, event) })
  }

  private update(patch: Partial<LiveView>): void {
    this.view = Object.freeze({ ...this.view, ...patch })
    const snapshot = snapshotOf(this.view)
    const details = detailsOf(this.view)
    const snapshotChanged = !sameSnapshot(snapshot, this.shown)
    const detailsChanged = !sameDetails(details, this.shownDetails)
    if (!snapshotChanged && !detailsChanged) return
    // Unchanged halves keep their identity, so a reader comparing by reference re-renders only on a change.
    if (snapshotChanged) this.shown = snapshot
    if (detailsChanged) this.shownDetails = details
    this.listeners.forEach((listener) => listener(this.shown))
  }
}

let shared: LiveEngine | null = null

/** The tab's one Live engine, so one Retell session per tab. */
export function liveEngine(): LiveEngine {
  if (!shared) shared = new LiveEngine({ agentId: () => useAiLabelStore.getState().agentId })
  return shared
}
