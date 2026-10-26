'use client'

/**
 * Theme voice — the voice button core draws for a theme (the SDK's `voice.VoiceButton`, lib/theme-sdk/host-cards/
 * voice-button.tsx): one button that talks to Auto through the host's voice (the tab's one session, voice-provider.tsx).
 *
 * Core draws all of it, so every theme's voice keeps the same promises:
 * - the press (voice-press.ts, Studio's Live chip's own), as a hold or a toggle, and its hotkey (voice-hotkey.ts);
 * - the browser's microphone prompt named while the session waits for it, then connecting;
 * - while the session is the button's: the "AI voice assistant" caption (AiVoiceLabel: a tiny caption, never a pill,
 *   and no lock on the button) and the stop control, with mute and the device for a live call, inside a protected
 *   surface (`voice-control`), so no theme stylesheet hides them while the microphone is open;
 * - the state for screen readers, a failure's sentence with Try again, and the provider's silent-microphone notice;
 * - every control a plain button (`type="button"`), so none submits a form the button sits in (Studio's composer).
 *
 * A theme places, styles and animates it: `data-voice-state` (the session's phase while it is the button's, else
 * idle), `data-voice-mode` (quick or live), `data-voice-press`, `data-voice-size` and `data-voice-mic` (open or
 * closed) on the root (`data-voice-button="root"`; the press is `data-voice-button="press"`), and the
 * `--voice-button-*` custom properties (VOICE_BUTTON_CSS_VARS) for its size, shape and colours. Outside a voice
 * provider the button is drawn disabled, saying voice isn't available on the page.
 */
import { useMemo } from 'react'
import { Loader2, Mic, Square } from 'lucide-react'

import { ProtectedSurface } from '@/components/theme-host/protected-surface'
import { Button } from '@/components/ui/button'
import { AiVoiceLabel } from '@/components/voice/AiVoiceLabel'
import { MicDevicePicker, MicSilentBanner } from '@/components/voice/MicHealthControl'
import { VoiceErrorBoundary } from '@/components/voice/VoiceErrorBoundary'
import { type HostVoice, useOptionalHostVoice } from '@/components/voice/voice-provider'
import type { VoicePhase } from '@/lib/voice/engine'
import type { VoiceStartMode } from '@/lib/voice/host/availability'
import { LIVE_UNAVAILABLE } from '@/lib/voice/live/refusal'
import { cn } from '@/lib/utils'

import { DEFAULT_LABELS, VOICE_BUTTON_WORDS, type VoiceButtonSize, type VoiceButtonView, voiceButtonView } from './voice-button-view'
import { EndOrClose, LiveCallFailure, MicState, MuteToggle, formatDuration } from './voice-controls'
import { DEFAULT_VOICE_HOTKEY, useVoiceHotkey } from './voice-hotkey'
import { type VoicePress, type VoicePressKind, type VoicePressOptions, runPressAction, useVoicePress } from './voice-press'

export interface VoiceButtonProps {
  /** The conversation the button talks into (null: a new chat); left out: the one on screen. */
  readonly conversation?: string | null
  /** `auto` (the default): Live when it can start, else quick-ask. */
  readonly mode?: VoiceStartMode
  /** `hold` (the default): hold to talk; `toggle`: press to start, press again to send or end. */
  readonly press?: VoicePressKind
  /** A KeyboardEvent.code that presses the button from anywhere but a field (default Space); null for none. */
  readonly hotkey?: string | null
  readonly size?: VoiceButtonSize
  /** The button's name for everyone (default: what a press does). */
  readonly label?: string
}

const RUNNING: ReadonlySet<VoicePhase> = new Set<VoicePhase>(['connecting', 'listening', 'thinking', 'speaking'])
const ON_THE_CALL: ReadonlySet<VoicePhase> = new Set<VoicePhase>(['listening', 'thinking', 'speaking'])

const SIZE_CLASSES: Readonly<Record<VoiceButtonSize, string>> = {
  sm: 'h-[var(--voice-button-size,2rem)] w-[var(--voice-button-size,2rem)]',
  md: 'h-[var(--voice-button-size,2.5rem)] w-[var(--voice-button-size,2.5rem)]',
  lg: 'h-[var(--voice-button-size,3.5rem)] w-[var(--voice-button-size,3.5rem)]',
}
const ICON_CLASSES: Readonly<Record<VoiceButtonSize, string>> = { sm: 'h-4 w-4', md: 'h-5 w-5', lg: 'h-6 w-6' }
const PRESS_CLASS =
  'inline-flex shrink-0 select-none touch-none items-center justify-center rounded-[var(--voice-button-radius,9999px)] ' +
  'transition-colors focus-visible:outline-none focus-visible:ring-2 ' +
  'focus-visible:ring-[color:var(--voice-button-ring,hsl(var(--ring)))] disabled:cursor-not-allowed disabled:opacity-50'
const IDLE_CLASS =
  'bg-[color:var(--voice-button-bg,hsl(var(--primary)))] text-[color:var(--voice-button-fg,hsl(var(--primary-foreground)))]'
const ACTIVE_CLASS =
  'bg-[color:var(--voice-button-active-bg,hsl(var(--warning)))] ' +
  'text-[color:var(--voice-button-active-fg,hsl(var(--warning-foreground)))]'
const SMALL_ICON = 'h-3.5 w-3.5'

interface PressButtonProps {
  readonly voicePress: VoicePress
  readonly view: VoiceButtonView
  readonly size: VoiceButtonSize
  readonly label: string
  readonly disabled: boolean
}

/** The press itself: a spinner while the session connects, else the microphone. */
function PressButton({ voicePress, view, size, label, disabled }: PressButtonProps) {
  const Icon = view.state === 'connecting' ? Loader2 : Mic
  return (
    <button
      type="button"
      data-voice-button="press"
      aria-label={label}
      aria-pressed={view.owned}
      title={view.title}
      disabled={disabled}
      className={cn(PRESS_CLASS, SIZE_CLASSES[size], view.owned ? ACTIVE_CLASS : IDLE_CLASS)}
      {...voicePress.handlers}
    >
      <Icon aria-hidden className={cn(ICON_CLASSES[size], view.state === 'connecting' && 'animate-spin')} />
    </button>
  )
}

/** Stop while quick-ask hears or Auto speaks, End or Close for a call; nothing while a failure offers its own Close. */
function StopControl({ voice }: { voice: HostVoice }) {
  const { snapshot } = voice
  const running = RUNNING.has(snapshot.phase)
  if (snapshot.mode === 'live') return <EndOrClose running={running} onStop={voice.stop} />
  if (!running) return null
  return (
    <Button
      type="button"
      variant="ghost"
      size="icon"
      className="h-7 w-7 rounded-full"
      onClick={voice.stop}
      aria-label={VOICE_BUTTON_WORDS.stop}
    >
      <Square className={SMALL_ICON} />
    </Button>
  )
}

/** The live call's microphone: mute and the device, while on the call. */
function LiveMicControls({ voice }: { voice: HostVoice }) {
  const { snapshot, live } = voice
  if (snapshot.mode !== 'live' || !ON_THE_CALL.has(snapshot.phase)) return null
  return (
    <>
      <MuteToggle muted={snapshot.muted} onToggle={live.toggleMute} />
      <MicDevicePicker devices={live.inputDevices} activeDeviceId={live.captureDeviceId} onPickDevice={live.pickDevice} />
    </>
  )
}

/**
 * The caption and the stop control, inside the protected surface `voice-control` (the inline !important guard, and
 * the server's CSS scope check refusing any theme selector that names a protected surface). Its parent is core's.
 */
function VoiceControl({ voice }: { voice: HostVoice }) {
  const { snapshot } = voice
  const failed = snapshot.phase === 'error'
  return (
    <div className="w-fit" data-voice-control-slot="">
      <ProtectedSurface surface="voice-control">
        <div className="flex items-center gap-2 px-1">
          <AiVoiceLabel />
          {snapshot.micOpen && <MicState muted={snapshot.muted} />}
          {snapshot.micOpen && snapshot.durationSec > 0 && (
            <span aria-hidden className="font-mono text-[10px] text-muted-foreground/60">
              {formatDuration(snapshot.durationSec)}
            </span>
          )}
          <LiveMicControls voice={voice} />
          {!failed && <StopControl voice={voice} />}
        </div>
      </ProtectedSurface>
    </div>
  )
}

/** The microphone prompt named, the silent-microphone notice (the provider's), and a failure with Try again. */
function VoiceNotices({ voice, options }: { voice: HostVoice; options: VoicePressOptions }) {
  const { snapshot, live } = voice
  const fallback = snapshot.mode === 'live' ? LIVE_UNAVAILABLE : VOICE_BUTTON_WORDS.failed
  const failure = snapshot.phase === 'error' ? snapshot.error ?? fallback : null
  const waitingForMic = snapshot.phase === 'connecting' && !snapshot.micOpen
  const silent = !failure && snapshot.mode === 'live' && ON_THE_CALL.has(snapshot.phase) && live.micSilent && !snapshot.muted
  const retry = () => runPressAction(voice, 'start', { ...options, mode: snapshot.mode ?? options.mode })
  return (
    <>
      {waitingForMic && (
        <p className="text-[11px] text-muted-foreground" data-voice-notice="allow-microphone">
          {VOICE_BUTTON_WORDS.allowMic}
        </p>
      )}
      {silent && (
        <MicSilentBanner devices={live.inputDevices} activeDeviceId={live.captureDeviceId} onPickDevice={live.pickDevice} />
      )}
      {failure && <LiveCallFailure message={failure} onRetry={retry} onClose={voice.stop} />}
    </>
  )
}

function pressOptions(mode: VoiceStartMode, press: VoicePressKind, conversation: string | null | undefined): VoicePressOptions {
  return conversation === undefined ? { mode, press } : { mode, press, chatId: conversation }
}

/** The voice button: the press, the screen reader's line, and, while the session is the button's, its controls. */
export function VoiceButton({
  conversation,
  mode = 'auto',
  press = 'hold',
  hotkey = DEFAULT_VOICE_HOTKEY,
  size = 'md',
  label,
}: VoiceButtonProps) {
  const voice = useOptionalHostVoice()
  const options = useMemo(() => pressOptions(mode, press, conversation), [mode, press, conversation])
  const voicePress = useVoicePress(voice, options)
  useVoiceHotkey(hotkey, voicePress, voice !== null)
  const name = label ?? DEFAULT_LABELS[press]
  const view = voiceButtonView(voice, options, name)
  return (
    <VoiceErrorBoundary>
      <div
        className="inline-flex flex-col items-center gap-1"
        data-voice-button="root"
        data-voice-state={view.state}
        data-voice-mode={view.mode}
        data-voice-press={press}
        data-voice-size={size}
        data-voice-mic={view.micOpen ? 'open' : 'closed'}
      >
        <PressButton voicePress={voicePress} view={view} size={size} label={name} disabled={voice === null} />
        <span role="status" aria-live="polite" className="sr-only" data-voice-status="">
          {view.status}
        </span>
        {voice && view.owned && <VoiceControl voice={voice} />}
        {voice && view.owned && <VoiceNotices voice={voice} options={options} />}
      </div>
    </VoiceErrorBoundary>
  )
}
