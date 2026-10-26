'use client'

/**
 * Theme voice — the small controls every voice surface the host draws shares: Studio's Live strip (LiveVoiceMode.tsx)
 * and the voice button core draws for themes (voice-button.tsx). The microphone's state, mute, end or close, and a
 * failure in plain words with Try again. Theme tokens only.
 */

import { Mic, MicOff, PhoneOff, X } from 'lucide-react'
import { Button } from '@/components/ui/button'

const SECONDS_PER_MINUTE = 60
const SECONDS_PAD = 2

export const MIC_LIVE = 'Mic live'
export const MIC_MUTED = 'Mic muted'
export const TRY_AGAIN = 'Try again'
export const CLOSE = 'Close'

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
        <Button size="sm" variant="outline" onClick={onRetry}>
          {TRY_AGAIN}
        </Button>
        <Button size="sm" variant="ghost" onClick={onClose}>
          {CLOSE}
        </Button>
      </div>
    </div>
  )
}

/** End while the call runs (or connects); Close once it has ended. */
export function EndOrClose({ running, onStop }: { running: boolean; onStop: () => void }) {
  return running ? (
    <Button
      variant="ghost"
      size="icon"
      className="h-7 w-7 rounded-full text-destructive hover:text-destructive"
      onClick={onStop}
      aria-label="End call"
    >
      <PhoneOff className="h-3.5 w-3.5" />
    </Button>
  ) : (
    <Button variant="ghost" size="icon" className="h-7 w-7 rounded-full" onClick={onStop} aria-label="Close live voice">
      <X className="h-3.5 w-3.5" />
    </Button>
  )
}
