/**
 * Theme voice (TV-C5) — the SDK's `voice.VoiceButton`: the voice button core draws (components/voice/voice-button.tsx)
 * with the SDK's read-only props.
 *
 * Core draws the press, the microphone prompt, connecting, the "AI voice assistant" caption and the stop control (a
 * protected surface, `voice-control`), mute and the device for a live call, a failure with Try again, the
 * silent-microphone notice and the state for screen readers; the theme places, styles and animates it (its
 * `data-voice-*` attributes and `--voice-button-*` custom properties). The props come from a theme's code, so they are
 * checked here: an unknown mode, press or size, or a conversation, hotkey or label of the wrong kind throws, naming
 * the prop, and the theme's error boundary reports it.
 */
import type { ReactElement } from 'react'

import { VoiceButton as HostVoiceButton } from '@/components/voice/voice-button'

/** `auto`: Live when it can start for this workspace, else quick-ask. */
export type VoiceButtonMode = 'auto' | 'quick' | 'live'
/** `hold`: hold to talk; `toggle`: press to start, press again to send or end. */
export type VoiceButtonPress = 'hold' | 'toggle'
export type VoiceButtonSize = 'sm' | 'md' | 'lg'

export interface VoiceButtonProps {
  /** The conversation the button talks into (null: a new chat); left out: the conversation on screen. */
  readonly conversation?: string | null
  /** Default `auto`. */
  readonly mode?: VoiceButtonMode
  /** Default `hold`. */
  readonly press?: VoiceButtonPress
  /** A KeyboardEvent.code that presses the button anywhere but a field the member types in; default `Space`, null for none. */
  readonly hotkey?: string | null
  /** Default `md`. */
  readonly size?: VoiceButtonSize
  /** The button's name for everyone; default: what a press does ("Hold to talk", "Talk to Auto"). */
  readonly label?: string
}

const MODES: readonly VoiceButtonMode[] = ['auto', 'quick', 'live']
const PRESSES: readonly VoiceButtonPress[] = ['hold', 'toggle']
const SIZES: readonly VoiceButtonSize[] = ['sm', 'md', 'lg']

function rejected(what: string): TypeError {
  return new TypeError(`Automatos theme SDK: VoiceButton needs ${what}.`)
}

function choice<T extends string>(value: unknown, allowed: readonly T[], prop: string): T | undefined {
  if (value === undefined) return undefined
  const found = allowed.find((option) => option === value)
  if (found === undefined) throw rejected(`${prop} to be one of ${allowed.join(', ')}`)
  return found
}

/** Some text, or null where null is allowed; left out stays left out. */
function text(value: unknown, prop: string, nullable: boolean): string | null | undefined {
  if (value === undefined) return undefined
  if (nullable && value === null) return null
  if (typeof value === 'string' && value.trim()) return value
  throw rejected(`${prop} to be some text${nullable ? ' or null' : ''}`)
}

/** The props a theme gave, checked; throws a TypeError naming the first prop of the wrong kind. */
export function checkedVoiceButtonProps(props: VoiceButtonProps): VoiceButtonProps {
  return {
    conversation: text(props.conversation, 'conversation', true),
    mode: choice(props.mode, MODES, 'mode'),
    press: choice(props.press, PRESSES, 'press'),
    hotkey: text(props.hotkey, 'hotkey', true),
    size: choice(props.size, SIZES, 'size'),
    label: text(props.label, 'label', false) ?? undefined,
  }
}

/** The host's voice button, with the theme's props checked. */
export function VoiceButton(props: VoiceButtonProps): ReactElement {
  const { conversation, mode, press, hotkey, size, label } = checkedVoiceButtonProps(props)
  const placed = conversation === undefined ? {} : { conversation }
  return <HostVoiceButton {...placed} mode={mode} press={press} hotkey={hotkey} size={size} label={label} />
}
