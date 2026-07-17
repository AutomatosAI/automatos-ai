'use client'

/**
 * LiveVoiceMode — the chat screen's live block (PRD-207 S5).
 *
 * Mounts top-center when Live is on: the existing welcome/messages content
 * animates lower beneath it ("Auto is in the room"). Owns the whole call
 * lifecycle UX: connect → orb + captions + controls; every failure state has
 * designed copy (mint refused with the honest reason, mic denied,
 * unsupported browser, mid-call disconnect) — never a dead button.
 */

import { useEffect } from 'react'
import { motion } from 'framer-motion'
import { Mic, MicOff, PhoneOff, X } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { PresenceOrb } from '@/components/voice/PresenceOrb'
import { useRetellCall } from '@/hooks/use-retell-call'

interface LiveVoiceModeProps {
  /** The on-screen thread to bind — omit when the chat has no server row yet. */
  chatId?: string | null
  agentId?: number | null
  onExit: () => void
}

function formatDuration(seconds: number): string {
  const m = Math.floor(seconds / 60)
  const s = seconds % 60
  return `${m}:${s.toString().padStart(2, '0')}`
}

export function LiveVoiceMode({ chatId, agentId, onExit }: LiveVoiceModeProps) {
  const {
    orbState,
    stateLabel,
    levelsRef,
    captions,
    durationSec,
    muted,
    refusal,
    error,
    isLive,
    start,
    stop,
    toggleMute,
  } = useRetellCall({ chatId, agentId })

  // Live mode IS the call: mounting connects, one gesture from the Live pill.
  useEffect(() => {
    void start()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const handleExit = () => {
    stop()
    onExit()
  }

  const failure = refusal || error

  return (
    <motion.div
      initial={{ opacity: 0, y: -16 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -16 }}
      transition={{ duration: 0.35 }}
      className="flex w-full flex-col items-center gap-2 pt-6 pb-2"
      data-testid="live-voice-mode"
    >
      <PresenceOrb state={orbState} levelsRef={levelsRef} size={120} />

      {/* State for ears and eyes — the a11y surface for the canvas. */}
      <p aria-live="polite" className="text-sm text-muted-foreground">
        {stateLabel}
        {isLive && (
          <span className="ml-2 font-mono text-xs text-muted-foreground/70">
            {formatDuration(durationSec)}
          </span>
        )}
      </p>

      {/* Live captions (accessibility + trust) */}
      {isLive && captions.length > 0 && (
        <div className="max-w-md space-y-0.5 text-center">
          {captions.map((line, i) => (
            <p
              key={`${line.role}-${i}`}
              className={`truncate text-xs ${
                line.role === 'agent' ? 'text-warning' : 'text-muted-foreground'
              }`}
            >
              <span className="font-medium">{line.role === 'agent' ? 'Auto' : 'You'}:</span>{' '}
              {line.text}
            </p>
          ))}
        </div>
      )}

      {/* Failure states — honest copy, always a way out. */}
      {failure && (
        <div className="max-w-md rounded-lg border border-destructive/30 bg-destructive/10 px-4 py-2 text-center">
          <p className="text-xs text-destructive">{failure}</p>
          <div className="mt-2 flex justify-center gap-2">
            <Button size="sm" variant="outline" onClick={() => void start()}>
              Try again
            </Button>
            <Button size="sm" variant="ghost" onClick={handleExit}>
              Close
            </Button>
          </div>
        </div>
      )}

      {/* Controls: mic-live indicator, mute, end — all keyboard-reachable. */}
      {!failure && (
        <div className="flex items-center gap-3">
          {isLive && (
            <span className="flex items-center gap-1.5 text-xs text-muted-foreground">
              <span
                className={`h-2 w-2 rounded-full ${
                  muted ? 'bg-muted-foreground/40' : 'bg-warning animate-pulse'
                }`}
              />
              {muted ? 'Mic muted' : 'Mic live'}
            </span>
          )}
          {isLive && (
            <Button
              variant="ghost"
              size="icon"
              className="h-10 w-10 rounded-full"
              onClick={toggleMute}
              aria-label={muted ? 'Unmute microphone' : 'Mute microphone'}
              aria-pressed={muted}
            >
              {muted ? <MicOff className="h-4 w-4 text-destructive" /> : <Mic className="h-4 w-4" />}
            </Button>
          )}
          {isLive || orbState === 'connecting' ? (
            <Button
              variant="ghost"
              size="icon"
              className="h-10 w-10 rounded-full text-destructive hover:text-destructive"
              onClick={handleExit}
              aria-label="End call"
            >
              <PhoneOff className="h-4 w-4" />
            </Button>
          ) : (
            <Button
              variant="ghost"
              size="icon"
              className="h-10 w-10 rounded-full"
              onClick={handleExit}
              aria-label="Close live voice"
            >
              <X className="h-4 w-4" />
            </Button>
          )}
        </div>
      )}
    </motion.div>
  )
}
