'use client'

/**
 * VoiceCallPanel — the docked live-call variant (PRD-74 shell, PRD-207 transport).
 *
 * S5 rewired this panel off the dead Pipecat-pod WebSocket (`useVoiceStream`,
 * deleted) onto `useRetellCall` — the same mint + Retell WebRTC lane the chat
 * screen's orb rides. The props interface is unchanged so hosts
 * (multimodal-input) need nothing; `workspaceId` stays for callers but the
 * server now derives the workspace from the authenticated mint.
 */

import { useEffect, useRef } from 'react'
import { motion } from 'framer-motion'
import { Loader2, Mic, MicOff, Phone, PhoneOff } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { PresenceOrb } from '@/components/voice/PresenceOrb'
import { useRetellCall } from '@/hooks/use-retell-call'
import type { OrbState } from '@/lib/voice/orb-state'

interface VoiceCallPanelProps {
  workspaceId: string
  agentId?: number | null
  conversationId?: string
  agentName?: string
  onClose?: () => void
}

const STATE_COLORS: Record<OrbState, string> = {
  idle: 'text-muted-foreground',
  connecting: 'text-warning',
  listening: 'text-success',
  thinking: 'text-warning',
  speaking: 'text-warning',
  ended: 'text-muted-foreground',
  error: 'text-destructive',
}

function formatDuration(seconds: number): string {
  const m = Math.floor(seconds / 60)
  const s = seconds % 60
  return `${m}:${s.toString().padStart(2, '0')}`
}

export function VoiceCallPanel({
  workspaceId: _workspaceId,
  agentId,
  conversationId,
  agentName = 'Auto',
  onClose,
}: VoiceCallPanelProps) {
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
  } = useRetellCall({ chatId: conversationId, agentId })

  const captionsEndRef = useRef<HTMLDivElement>(null)
  useEffect(() => {
    captionsEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [captions])

  const isActive = isLive || orbState === 'connecting'
  const failure = refusal || error

  const handleToggleCall = async () => {
    if (isActive) stop()
    else await start()
  }

  const handleEndCall = () => {
    stop()
    onClose?.()
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: 20 }}
      className="fixed inset-x-0 bottom-0 z-50 mx-auto max-w-lg p-4 md:bottom-4"
    >
      <div className="rounded-3xl border-2 border-warning/30 bg-background/95 backdrop-blur-xl shadow-2xl shadow-warning/10 overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-5 pt-4 pb-2">
          <div className="flex items-center gap-3">
            <div className="relative">
              <div
                className={[
                  'h-3 w-3 rounded-full',
                  isLive ? 'bg-success' : orbState === 'error' ? 'bg-destructive' : 'bg-muted',
                ].join(' ')}
              />
              {orbState === 'speaking' && (
                <div className="absolute inset-0 h-3 w-3 animate-ping rounded-full bg-warning/50" />
              )}
            </div>
            <div>
              <h3 className="text-sm font-semibold">{agentName}</h3>
              <p className={`text-xs ${STATE_COLORS[orbState]}`} aria-live="polite">
                {stateLabel}
              </p>
            </div>
          </div>
          {isActive && (
            <span className="text-xs text-muted-foreground font-mono">
              {formatDuration(durationSec)}
            </span>
          )}
        </div>

        {/* Live captions */}
        {isLive && captions.length > 0 && (
          <div className="mx-4 mb-2 max-h-40 overflow-y-auto rounded-xl bg-secondary/30 p-3 space-y-2">
            {captions.map((line, i) => (
              <div
                key={`${line.role}-${i}`}
                className={`text-xs ${line.role === 'user' ? 'text-info/80' : 'text-warning'}`}
              >
                <span className="font-medium">{line.role === 'user' ? 'You' : agentName}:</span>{' '}
                {line.text}
              </div>
            ))}
            <div ref={captionsEndRef} />
          </div>
        )}

        {/* The presence — same orb, docked size */}
        <div className="flex items-center justify-center py-3">
          <PresenceOrb state={orbState} levelsRef={levelsRef} size={88} />
        </div>

        {/* Failure copy (mint refusal reason, mic denied, disconnects) */}
        {failure && (
          <p className="text-center text-xs text-destructive pb-2 px-4">{failure}</p>
        )}

        {/* Controls */}
        <div className="flex items-center justify-center gap-4 px-5 pb-5">
          {isLive && (
            <Button
              variant="ghost"
              size="icon"
              className="h-12 w-12 rounded-full"
              onClick={toggleMute}
              aria-label={muted ? 'Unmute microphone' : 'Mute microphone'}
              aria-pressed={muted}
            >
              {muted ? (
                <MicOff className="w-5 h-5 text-destructive" />
              ) : (
                <Mic className="w-5 h-5" />
              )}
            </Button>
          )}

          <Button
            size="icon"
            className={[
              'h-16 w-16 rounded-full shadow-lg transition-all',
              isActive
                ? 'bg-destructive hover:bg-destructive/80 shadow-destructive/25'
                : 'bg-gradient-to-br from-green-500 to-green-600 hover:from-green-600 hover:to-green-700 shadow-green-500/25',
            ].join(' ')}
            onClick={handleToggleCall}
            aria-label={isActive ? 'Hang up' : 'Start call'}
          >
            {isActive ? (
              <PhoneOff className="w-6 h-6" />
            ) : orbState === 'connecting' ? (
              <Loader2 className="w-6 h-6 animate-spin" />
            ) : (
              <Phone className="w-6 h-6" />
            )}
          </Button>

          {isActive && (
            <Button
              variant="ghost"
              size="icon"
              className="h-12 w-12 rounded-full"
              onClick={handleEndCall}
              aria-label="End call and close"
            >
              <PhoneOff className="w-5 h-5 text-muted-foreground" />
            </Button>
          )}
        </div>
      </div>
    </motion.div>
  )
}
