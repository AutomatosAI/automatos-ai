'use client'

/**
 * LiveVoiceMode — the call's control strip + presence feed (PRD-207 S5).
 *
 * The wave is NOT a widget anymore — it renders as the chat window's
 * BACKGROUND (Gerard: "a live feeling background that only comes alive when
 * speaking"). This component owns the call (the one useRetellCall instance)
 * and feeds presence upward: `onPresence` (state) + `onLevelsRef` (the live
 * audio levels ref) let the chat render the ambient background itself.
 * What remains visible here is a slim strip above the message box —
 * state · timer · mic · mute · end — plus honest failure copy.
 */

import { useEffect } from 'react'
import { motion } from 'framer-motion'
import { Mic, MicOff, PhoneOff, X } from 'lucide-react'
import type { MutableRefObject } from 'react'
import { Button } from '@/components/ui/button'
import { useRetellCall, type VoiceLevels } from '@/hooks/use-retell-call'
import type { OrbState } from '@/lib/voice/orb-state'

interface LiveVoiceModeProps {
  /** The on-screen thread to bind — omit when the chat has no server row yet. */
  chatId?: string | null
  agentId?: number | null
  /** The call's thread id (created server-side when none was bound). */
  onChatId?: (chatId: string) => void
  /** The current exchange as it streams — the chat types it out live. */
  onLiveTurn?: (turn: { userText: string; agentText: string }) => void
  /** Presence feed for the ambient background: state transitions… */
  onPresence?: (state: OrbState) => void
  /** …and the (stable) live audio-levels ref, sent once on mount. */
  onLevelsRef?: (levels: MutableRefObject<VoiceLevels>) => void
  onExit: () => void
}

function formatDuration(seconds: number): string {
  const m = Math.floor(seconds / 60)
  const s = seconds % 60
  return `${m}:${s.toString().padStart(2, '0')}`
}

export function LiveVoiceMode({
  chatId,
  agentId,
  onChatId,
  onLiveTurn,
  onPresence,
  onLevelsRef,
  onExit,
}: LiveVoiceModeProps) {
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
    onLevelsRef?.(levelsRef)
    void start()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    onPresence?.(orbState)
  }, [orbState, onPresence])

  const handleExit = () => {
    stop()
    onExit()
  }

  const failure = refusal || error

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: 8 }}
      transition={{ duration: 0.25 }}
      className="relative w-full"
      data-testid="live-voice-mode"
    >
      {/* Whisper strip — no card, no border: the room glows behind the whole
          chat, the words stream in the thread like any typed reply; up here
          only the quiet facts and the two buttons that matter. */}
      {!failure && (
        <div className="mx-auto flex w-fit items-center gap-3 px-2 py-0.5">
          <p aria-live="polite" className="text-[11px] tracking-wide text-muted-foreground/60">
            {stateLabel}
            {isLive && (
              <span className="ml-2 font-mono text-[10px] text-muted-foreground/50">
                {formatDuration(durationSec)}
              </span>
            )}
          </p>
          {isLive && (
            <span className="flex items-center gap-1.5 text-[11px] text-muted-foreground">
              <span
                className={`h-1.5 w-1.5 rounded-full ${
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
          {isLive || orbState === 'connecting' ? (
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
          )}
        </div>
      )}

      {/* Failure states — honest copy, always a way out. */}
      {failure && (
        <div className="mx-auto max-w-md rounded-lg border border-destructive/30 bg-destructive/10 px-4 py-2 text-center">
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
    </motion.div>
  )
}
