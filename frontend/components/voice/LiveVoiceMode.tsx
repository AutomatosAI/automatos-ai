'use client'

/**
 * LiveVoiceMode — the composer's voice half (PRD-207 S5).
 *
 * Docked directly ON TOP of the message box, same width, same column: the
 * wave and the text input are ONE conversation surface (Gerard: "the chat
 * text box and the voice wave need to look the same conversation"). The
 * thread flows above; words type out in it as they're spoken; typing
 * mid-call goes to the very same thread. Owns the call lifecycle UX —
 * every failure state has designed copy, never a dead button.
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
  /** The call's thread id (created server-side when none was bound) — the
   * chat screen points at it so the spoken conversation is the visible one. */
  onChatId?: (chatId: string) => void
  /** The current exchange as it streams — the chat types it out live. */
  onLiveTurn?: (turn: { userText: string; agentText: string }) => void
  onExit: () => void
}

function formatDuration(seconds: number): string {
  const m = Math.floor(seconds / 60)
  const s = seconds % 60
  return `${m}:${s.toString().padStart(2, '0')}`
}

export function LiveVoiceMode({ chatId, agentId, onChatId, onLiveTurn, onExit }: LiveVoiceModeProps) {
  const {
    orbState,
    stateLabel,
    levelsRef,
    durationSec,
    muted,
    refusal,
    error,
    isLive,
    start,
    stop,
    toggleMute,
  } = useRetellCall({ chatId, agentId, onChatId, onLiveTurn })

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
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: 12 }}
      transition={{ duration: 0.3 }}
      className="relative w-full"
      data-testid="live-voice-mode"
    >
      {/* Soft gold breath behind the composer — the wave and the input share
          one stage. Pure CSS, pointer-transparent. */}
      <div
        aria-hidden="true"
        className="pointer-events-none absolute -inset-x-6 -top-10 bottom-0 opacity-70"
        style={{
          background:
            'radial-gradient(ellipse at center, hsl(var(--warning) / 0.12) 0%, hsl(var(--warning) / 0.04) 50%, transparent 75%)',
          filter: 'blur(2px)',
        }}
      />

      {/* The wave sits flush on the message box — one surface, two ways in. */}
      <div className="relative rounded-t-2xl border border-b-0 border-warning/25 bg-background/60 px-3 pt-2 backdrop-blur-sm">
        <PresenceOrb state={orbState} levelsRef={levelsRef} size={110} />

        {/* One compact strip: state · timer · mic · controls */}
        <div className="flex items-center justify-center gap-3 pb-2 pt-1">
          <p aria-live="polite" className="text-xs font-medium tracking-wide text-warning/90">
            {stateLabel}
            {isLive && (
              <span className="ml-2 font-mono text-[11px] text-muted-foreground/70">
                {formatDuration(durationSec)}
              </span>
            )}
          </p>
          {!failure && isLive && (
            <span className="flex items-center gap-1.5 text-[11px] text-muted-foreground">
              <span
                className={`h-1.5 w-1.5 rounded-full ${
                  muted ? 'bg-muted-foreground/40' : 'bg-warning animate-pulse'
                }`}
              />
              {muted ? 'Mic muted' : 'Mic live'}
            </span>
          )}
          {!failure && isLive && (
            <Button
              variant="ghost"
              size="icon"
              className="h-7 w-7 rounded-full"
              onClick={toggleMute}
              aria-label={muted ? 'Unmute microphone' : 'Mute microphone'}
              aria-pressed={muted}
            >
              {muted ? (
                <MicOff className="h-3.5 w-3.5 text-destructive" />
              ) : (
                <Mic className="h-3.5 w-3.5" />
              )}
            </Button>
          )}
          {!failure &&
            (isLive || orbState === 'connecting' ? (
              <Button
                variant="ghost"
                size="icon"
                className="h-7 w-7 rounded-full text-destructive hover:text-destructive"
                onClick={handleExit}
                aria-label="End call"
              >
                <PhoneOff className="h-3.5 w-3.5" />
              </Button>
            ) : (
              <Button
                variant="ghost"
                size="icon"
                className="h-7 w-7 rounded-full"
                onClick={handleExit}
                aria-label="Close live voice"
              >
                <X className="h-3.5 w-3.5" />
              </Button>
            ))}
        </div>

        {/* Failure states — honest copy, always a way out. */}
        {failure && (
          <div className="mx-auto mb-2 max-w-md rounded-lg border border-destructive/30 bg-destructive/10 px-4 py-2 text-center">
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
      </div>
    </motion.div>
  )
}
