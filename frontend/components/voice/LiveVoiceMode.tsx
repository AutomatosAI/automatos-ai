'use client'

/**
 * LiveVoiceMode — Studio's Auto Live strip (PRD-207 S5), drawn from the host voice provider (theme voice).
 *
 * The call is the host's (components/voice/voice-provider.tsx: one session per tab, run outside React by the Live
 * engine, lib/voice/live/live-engine.ts), so the strip only shows it: it is on screen while the tab's session is a
 * live call (connecting, on the call, ended by Retell, or failed) and gone once the session stops. Studio's Live chip
 * starts the call (chat-mode-bar.tsx); the wave is the chat's background (host-presence-orb.tsx); the words are the
 * thread's own (lib/chat/live-voice-turns.ts). What remains here is a slim strip above the message box (state,
 * timer, mic, mute, device, end) and a failure in plain words with Try again.
 *
 * PRD-264 EUAI-04: the strip opens with the small "AI voice assistant" caption for the whole call, a failure
 * included.
 */

import { AnimatePresence, motion } from 'framer-motion'
import { Mic, MicOff, PhoneOff, X } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { AiVoiceLabel } from '@/components/voice/AiVoiceLabel'
import { MicDevicePicker, MicSilentBanner } from '@/components/voice/MicHealthControl'
import { VoiceErrorBoundary } from '@/components/voice/VoiceErrorBoundary'
import { type HostVoice, useHostVoice } from '@/components/voice/voice-provider'
import type { VoicePhase } from '@/lib/voice/engine'
import { LIVE_UNAVAILABLE } from '@/lib/voice/live/refusal'
import { ORB_STATE_LABELS } from '@/lib/voice/orb-state'

const SECONDS_PER_MINUTE = 60
const ON_THE_CALL: ReadonlySet<VoicePhase> = new Set<VoicePhase>(['listening', 'thinking', 'speaking'])

function formatDuration(seconds: number): string {
  const m = Math.floor(seconds / SECONDS_PER_MINUTE)
  const s = seconds % SECONDS_PER_MINUTE
  return `${m}:${s.toString().padStart(2, '0')}`
}

function MuteToggle({ muted, onToggle }: { muted: boolean; onToggle: () => void }) {
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
function LiveCallFailure({ message, onRetry, onClose }: { message: string; onRetry: () => void; onClose: () => void }) {
  return (
    <div className="mx-auto max-w-md rounded-lg border border-destructive/30 bg-destructive/10 px-4 py-2 text-center">
      <p className="text-xs text-destructive">{message}</p>
      <div className="mt-2 flex justify-center gap-2">
        <Button size="sm" variant="outline" onClick={onRetry}>
          Try again
        </Button>
        <Button size="sm" variant="ghost" onClick={onClose}>
          Close
        </Button>
      </div>
    </div>
  )
}

/** End while the call runs (or connects); Close once it has ended. */
function EndOrClose({ running, onStop }: { running: boolean; onStop: () => void }) {
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

/** The call's quiet facts and controls: state and timer, the mic, mute, the device, end. */
function LiveCallControls({ voice }: { voice: HostVoice }) {
  const { snapshot, live } = voice
  const onTheCall = ON_THE_CALL.has(snapshot.phase)
  return (
    <>
      <p aria-live="polite" className="text-[11px] tracking-wide text-muted-foreground/60">
        {ORB_STATE_LABELS[snapshot.phase]}
        {onTheCall && (
          <span className="ml-2 font-mono text-[10px] text-muted-foreground/50">{formatDuration(snapshot.durationSec)}</span>
        )}
      </p>
      {onTheCall && (
        <span className="flex items-center gap-1.5 text-[11px] text-muted-foreground">
          <span className={`h-1.5 w-1.5 rounded-full ${snapshot.muted ? 'bg-muted-foreground/40' : 'bg-warning animate-pulse'}`} />
          {snapshot.muted ? 'Mic muted' : 'Mic live'}
        </span>
      )}
      {onTheCall && <MuteToggle muted={snapshot.muted} onToggle={live.toggleMute} />}
      {onTheCall && (
        <MicDevicePicker devices={live.inputDevices} activeDeviceId={live.captureDeviceId} onPickDevice={live.pickDevice} />
      )}
      <EndOrClose running={onTheCall || snapshot.phase === 'connecting'} onStop={voice.stop} />
    </>
  )
}

function LiveStrip({ voice }: { voice: HostVoice }) {
  const { snapshot, live } = voice
  const failure = snapshot.phase === 'error' ? snapshot.error ?? LIVE_UNAVAILABLE : null
  const silent = !failure && ON_THE_CALL.has(snapshot.phase) && live.micSilent && !snapshot.muted
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: 8 }}
      transition={{ duration: 0.25 }}
      className="relative w-full"
      data-testid="live-voice-mode"
      data-voice-phase={snapshot.phase}
    >
      {/* Whisper strip — no card, no border: the room glows behind the whole chat, the words stream in the thread
          like any typed reply; up here only the quiet facts and the buttons that matter. */}
      <div className="mx-auto flex w-fit items-center gap-3 px-2 py-0.5">
        <AiVoiceLabel />
        {!failure && <LiveCallControls voice={voice} />}
      </div>
      {/* The mic is bound to a device delivering silence: name it, fix it here. */}
      {silent && (
        <MicSilentBanner devices={live.inputDevices} activeDeviceId={live.captureDeviceId} onPickDevice={live.pickDevice} />
      )}
      {failure && (
        <LiveCallFailure message={failure} onRetry={() => void voice.start({ mode: 'live' })} onClose={voice.stop} />
      )}
    </motion.div>
  )
}

/** Studio's Live strip: shown while the tab's voice session is a live call. */
export function LiveVoiceMode() {
  const voice = useHostVoice()
  return (
    <AnimatePresence>
      {voice.snapshot.mode === 'live' && (
        <VoiceErrorBoundary key="live-voice-strip">
          <LiveStrip voice={voice} />
        </VoiceErrorBoundary>
      )}
    </AnimatePresence>
  )
}
