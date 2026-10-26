'use client'

/**
 * Theme voice — the small controls every voice surface the host draws shares: Studio's Live strip (LiveVoiceMode.tsx)
 * and the voice button core draws for themes (voice-button.tsx). The microphone's state, mute, end or close, and a
 * failure in plain words with Try again. Theme tokens only. Every control is a plain button (`type="button"`), so none
 * submits a form it sits in (the voice button in Studio's composer, or in a theme's own form). Stop, End call and Close
 * say their word beside their icon: a bare icon beside the caption read as a glitch, not a control.
 */

import { Mic, MicOff, PhoneOff, Square, X } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'

const SECONDS_PER_MINUTE = 60
const SECONDS_PAD = 2

export const MIC_LIVE = 'Mic live'
export const MIC_MUTED = 'Mic muted'
export const TRY_AGAIN = 'Try again'
export const CLOSE = 'Close'
export const STOP = 'Stop'
export const END_CALL = 'End call'
export const CLOSE_LIVE_VOICE = 'Close live voice'

/** A small labelled control (its icon and its word), the height of the icon buttons beside it. */
const LABELLED_CONTROL_CLASS = 'h-7 gap-1 rounded-full px-2 text-xs'
const CONTROL_ICON_CLASS = 'h-3.5 w-3.5'

/** A session's length as m:ss. */
export function formatDuration(seconds: number): string {
  const m = Math.floor(seconds / SECONDS_PER_MINUTE)
  const s = seconds % SECONDS_PER_MINUTE
  return `${m}:${s.toString().padStart(SECONDS_PAD, '0')}`
}

/** The microphone is open: a dot and "Mic live", or "Mic muted". */
export function MicState({ muted }: { muted: boolean }) {
  return (
    <span className="flex items-center gap-1.5 text-[11px] text-muted-foreground">
      <span className={`h-1.5 w-1.5 rounded-full ${muted ? 'bg-muted-foreground/40' : 'bg-warning animate-pulse'}`} />
      {muted ? MIC_MUTED : MIC_LIVE}
    </span>
  )
}

export function MuteToggle({ muted, onToggle }: { muted: boolean; onToggle: () => void }) {
  return (
    <Button
      type="button"
      variant="ghost"
      size="icon"
      className="h-7 w-7 rounded-full"
      onClick={onToggle}
      aria-label={muted ? 'Unmute microphone' : 'Mute microphone'}
      aria-pressed={muted}
    >
      {muted ? <MicOff className="h-3.5 w-3.5 text-destructive" /> : <Mic className="h-3.5 w-3.5" />}
    </Button>
  )
}

/** Failure states — honest copy, always a way out. */
export function LiveCallFailure({ message, onRetry, onClose }: { message: string; onRetry: () => void; onClose: () => void }) {
  return (
    <div className="mx-auto max-w-md rounded-lg border border-destructive/30 bg-destructive/10 px-4 py-2 text-center">
      <p className="text-xs text-destructive">{message}</p>
      <div className="mt-2 flex justify-center gap-2">
        <Button type="button" size="sm" variant="outline" onClick={onRetry}>
          {TRY_AGAIN}
        </Button>
        <Button type="button" size="sm" variant="ghost" onClick={onClose}>
          {CLOSE}
        </Button>
      </div>
    </div>
  )
}

/** Stop: the quick ask's way out while it hears or Auto speaks. */
export function StopButton({ onStop }: { onStop: () => void }) {
  return (
    <Button type="button" variant="outline" size="sm" className={LABELLED_CONTROL_CLASS} onClick={onStop} aria-label={STOP}>
      <Square aria-hidden className={CONTROL_ICON_CLASS} />
      {STOP}
    </Button>
  )
}

/** End call while the call runs (or connects); Close once it has ended. */
export function EndOrClose({ running, onStop }: { running: boolean; onStop: () => void }) {
  return running ? (
    <Button
      type="button"
      variant="outline"
      size="sm"
      className={cn(LABELLED_CONTROL_CLASS, 'text-destructive hover:text-destructive')}
      onClick={onStop}
      aria-label={END_CALL}
    >
      <PhoneOff aria-hidden className={CONTROL_ICON_CLASS} />
      {END_CALL}
    </Button>
  ) : (
    <Button
      type="button"
      variant="outline"
      size="sm"
      className={LABELLED_CONTROL_CLASS}
      onClick={onStop}
      aria-label={CLOSE_LIVE_VOICE}
    >
      <X aria-hidden className={CONTROL_ICON_CLASS} />
      {CLOSE}
    </Button>
  )
}
