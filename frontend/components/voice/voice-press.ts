'use client'

/**
 * Theme voice — what a press on a voice control does, for every control that starts the host's voice: Studio's Live
 * chip (chat-mode-bar.tsx) and the voice button core draws for themes (voice-button.tsx). One copy, so a toggle, a
 * hold and the hotkey behave the same wherever they are pressed.
 *
 * - A control owns the tab's session when it runs (or shows its end) in the engine the control starts and, when the
 *   control names a conversation, on that conversation.
 * - Toggle: a press starts a session; the next one sends what was heard (quick-ask, while the microphone is open) or
 *   ends it (a live call; quick-ask while Auto replies starts a new turn instead, cutting Auto off).
 * - Hold: pressing the pointer starts a session and letting go sends what was heard (Live ignores the release, so a
 *   held Live press starts a call and the next press ends it). A click that no pointer made (a focused button's Space
 *   or Enter, a screen reader's activation) counts as a toggle, so hold-to-talk never shuts anyone out; the hotkey
 *   (voice-hotkey.ts) holds from the keyboard.
 *
 * The press reads the host through the HostVoice it is handed (useHostVoice or useOptionalHostVoice); without one,
 * every handler does nothing.
 */
import type { MouseEvent as ReactMouseEvent, PointerEvent as ReactPointerEvent } from 'react'
import { useMemo, useRef } from 'react'

import type { HostVoice } from '@/components/voice/voice-provider'
import type { VoicePhase, VoiceSnapshot } from '@/lib/voice/engine'
import type { VoiceStartMode } from '@/lib/voice/host/availability'

export type VoicePressKind = 'hold' | 'toggle'
export type VoicePressAction = 'start' | 'release' | 'stop' | 'none'

export interface VoicePressOptions {
  /** The engine a press starts: `auto` is Live when it can start, else quick-ask. */
  readonly mode: VoiceStartMode
  readonly press: VoicePressKind
  /** The conversation the control talks into (null: a new chat); left out: the one on screen. */
  readonly chatId?: string | null
}

export interface VoicePressHandlers {
  readonly onClick: (event: ReactMouseEvent<HTMLElement>) => void
  readonly onPointerDown: (event: ReactPointerEvent<HTMLElement>) => void
  readonly onPointerUp: () => void
  readonly onPointerCancel: () => void
  readonly onPointerLeave: () => void
}

export interface VoicePress {
  /** The tab's session is this control's. */
  readonly active: boolean
  /** A press begins (the pointer, the hotkey). */
  readonly down: () => void
  /** The press ends. */
  readonly up: () => void
  readonly handlers: VoicePressHandlers
}

const HEARING: ReadonlySet<VoicePhase> = new Set<VoicePhase>(['connecting', 'listening'])
const PRIMARY_BUTTON = 0
/** A click's `detail` is the pointer's click count; a click no pointer made (a key, a screen reader) has none. */
const NO_POINTER_CLICKS = 0
const NO_OP = (): void => undefined

/** Whether the tab's session is the control's: running or ended in its engine, on its conversation when it names one. */
export function ownsSession(snapshot: VoiceSnapshot, options: Pick<VoicePressOptions, 'mode' | 'chatId'>): boolean {
  if (snapshot.phase === 'idle' || snapshot.mode === null) return false
  if (options.mode !== 'auto' && snapshot.mode !== options.mode) return false
  const named = options.chatId
  return named === undefined || named === null || snapshot.chatId === null || snapshot.chatId === named
}

/** What a press does now, for a control of these options. */
export function pressAction(snapshot: VoiceSnapshot, options: VoicePressOptions): VoicePressAction {
  if (!ownsSession(snapshot, options)) return 'start'
  if (snapshot.mode === 'live') return 'stop'
  if (HEARING.has(snapshot.phase)) return options.press === 'toggle' ? 'release' : 'none'
  return 'start'
}

/** Carries out `action` on the host: a start on the control's conversation, when it names one. */
export function runPressAction(voice: HostVoice, action: VoicePressAction, options: VoicePressOptions): void {
  const { mode, chatId } = options
  if (action === 'start') void voice.start(chatId === undefined ? { mode } : { mode, chatId })
  else if (action === 'release') voice.release()
  else if (action === 'stop') voice.stop()
}

function holdPress(voice: HostVoice, options: VoicePressOptions, holding: { current: boolean }): VoicePress {
  const down = () => {
    const action = pressAction(voice.snapshot, options)
    holding.current = action === 'start'
    runPressAction(voice, action, options)
  }
  const up = () => {
    if (!holding.current) return
    holding.current = false
    voice.release()
  }
  const handlers: VoicePressHandlers = {
    // A pointer's click follows its own press and release, which already acted.
    onClick: (event) => {
      if (event.detail !== NO_POINTER_CLICKS) return
      runPressAction(voice, pressAction(voice.snapshot, { ...options, press: 'toggle' }), options)
    },
    // Only the main button (or a touch, or a pen) talks; a right or middle press does nothing.
    onPointerDown: (event) => {
      if (event.button > PRIMARY_BUTTON) return
      down()
    },
    onPointerUp: up,
    onPointerCancel: up,
    onPointerLeave: up,
  }
  return { active: ownsSession(voice.snapshot, options), down, up, handlers }
}

const INERT_HANDLERS: VoicePressHandlers = Object.freeze({
  onClick: NO_OP, onPointerDown: NO_OP, onPointerUp: NO_OP, onPointerCancel: NO_OP, onPointerLeave: NO_OP,
})

function togglePress(voice: HostVoice, options: VoicePressOptions): VoicePress {
  const down = () => runPressAction(voice, pressAction(voice.snapshot, options), options)
  // A focused button's own Space and Enter click it, so the click is the whole press.
  return { active: ownsSession(voice.snapshot, options), down, up: NO_OP, handlers: { ...INERT_HANDLERS, onClick: down } }
}

const INERT: VoicePress = Object.freeze({ active: false, down: NO_OP, up: NO_OP, handlers: INERT_HANDLERS })

/** A voice control's press over the host's voice (null: no voice here, and the press does nothing). */
export function useVoicePress(voice: HostVoice | null, options: VoicePressOptions): VoicePress {
  const holding = useRef(false)
  const { mode, press, chatId } = options
  return useMemo(() => {
    if (voice === null) return INERT
    const current = { mode, press, chatId }
    return press === 'hold' ? holdPress(voice, current, holding) : togglePress(voice, current)
  }, [voice, mode, press, chatId])
}
