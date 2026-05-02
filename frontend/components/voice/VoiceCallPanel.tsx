'use client'

/**
 * VoiceCallPanel — Real-time voice conversation UI (PRD-74 Phase 3)
 *
 * Full-screen or docked panel for live voice mode.
 * Shows live transcript, connection state, and call controls.
 */

import { useState, useCallback, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Phone,
  PhoneOff,
  Mic,
  MicOff,
  Volume2,
  Loader2,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { useVoiceStream, type VoiceStreamState } from '@/hooks/use-voice-stream'

interface VoiceCallPanelProps {
  workspaceId: string
  agentId?: number | null
  conversationId?: string
  agentName?: string
  onClose?: () => void
}

interface TranscriptEntry {
  role: 'user' | 'assistant'
  text: string
  timestamp: number
}

const STATE_LABELS: Record<VoiceStreamState, string> = {
  disconnected: 'Ready to call',
  connecting: 'Connecting...',
  connected: 'Listening...',
  speaking: 'You are speaking',
  processing: 'Thinking...',
  responding: 'Agent is speaking',
  error: 'Connection error',
}

const STATE_COLORS: Record<VoiceStreamState, string> = {
  disconnected: 'text-muted-foreground',
  connecting: 'text-warning',
  connected: 'text-success',
  speaking: 'text-info',
  processing: 'text-orange-400',
  responding: 'text-agent',
  error: 'text-destructive',
}

function formatDuration(seconds: number): string {
  const m = Math.floor(seconds / 60)
  const s = seconds % 60
  return `${m}:${s.toString().padStart(2, '0')}`
}

export function VoiceCallPanel({
  workspaceId,
  agentId,
  conversationId,
  agentName = 'Auto',
  onClose,
}: VoiceCallPanelProps) {
  const [transcript, setTranscript] = useState<TranscriptEntry[]>([])
  const [muted, setMuted] = useState(false)
  const transcriptEndRef = useRef<HTMLDivElement>(null)

  const handleTranscript = useCallback((text: string, isFinal: boolean) => {
    if (isFinal) {
      setTranscript((prev) => [...prev, { role: 'user', text, timestamp: Date.now() }])
    }
  }, [])

  const handleResponseText = useCallback((text: string) => {
    setTranscript((prev) => [...prev, { role: 'assistant', text, timestamp: Date.now() }])
  }, [])

  const { state, connect, disconnect, error, sessionDuration } = useVoiceStream({
    workspaceId,
    agentId,
    conversationId,
    onTranscript: handleTranscript,
    onResponseText: handleResponseText,
  })

  // Auto-scroll transcript
  useEffect(() => {
    transcriptEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [transcript])

  const isActive = state !== 'disconnected' && state !== 'error'

  const handleToggleCall = async () => {
    if (isActive) {
      disconnect()
    } else {
      await connect()
    }
  }

  const handleEndCall = () => {
    disconnect()
    onClose?.()
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: 20 }}
      className="fixed inset-x-0 bottom-0 z-50 mx-auto max-w-lg p-4 md:bottom-4"
    >
      <div className="rounded-3xl border-2 border-orange-500/30 bg-background/95 backdrop-blur-xl shadow-2xl shadow-orange-500/10 overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-5 pt-4 pb-2">
          <div className="flex items-center gap-3">
            {/* Pulsing indicator */}
            <div className="relative">
              <div
                className={[
                  'h-3 w-3 rounded-full',
                  isActive ? 'bg-success' : state === 'error' ? 'bg-destructive' : 'bg-muted',
                ].join(' ')}
              />
              {(state === 'speaking' || state === 'responding') && (
                <div className="absolute inset-0 h-3 w-3 animate-ping rounded-full bg-success/50" />
              )}
            </div>
            <div>
              <h3 className="text-sm font-semibold">{agentName}</h3>
              <p className={`text-xs ${STATE_COLORS[state]}`}>
                {STATE_LABELS[state]}
              </p>
            </div>
          </div>
          {isActive && (
            <span className="text-xs text-muted-foreground font-mono">
              {formatDuration(sessionDuration)}
            </span>
          )}
        </div>

        {/* Live Transcript */}
        {isActive && transcript.length > 0 && (
          <div className="mx-4 mb-2 max-h-40 overflow-y-auto rounded-xl bg-secondary/30 p-3 space-y-2">
            {transcript.slice(-6).map((entry, i) => (
              <div
                key={`${entry.timestamp}-${i}`}
                className={`text-xs ${
                  entry.role === 'user'
                    ? 'text-info/80'
                    : 'text-orange-300'
                }`}
              >
                <span className="font-medium">
                  {entry.role === 'user' ? 'You' : agentName}:
                </span>{' '}
                {entry.text}
              </div>
            ))}
            <div ref={transcriptEndRef} />
          </div>
        )}

        {/* Visualiser / State indicator */}
        <div className="flex items-center justify-center py-4">
          <AnimatePresence mode="wait">
            {state === 'connecting' ? (
              <motion.div
                key="connecting"
                initial={{ scale: 0.8, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                exit={{ scale: 0.8, opacity: 0 }}
              >
                <Loader2 className="w-12 h-12 animate-spin text-orange-400" />
              </motion.div>
            ) : state === 'speaking' ? (
              <motion.div
                key="speaking"
                initial={{ scale: 0.8 }}
                animate={{ scale: [1, 1.15, 1] }}
                transition={{ repeat: Infinity, duration: 0.8 }}
                className="flex items-center gap-1"
              >
                {[...Array(5)].map((_, i) => (
                  <motion.div
                    key={i}
                    className="w-1.5 bg-info rounded-full"
                    animate={{ height: [8, 24, 8] }}
                    transition={{
                      repeat: Infinity,
                      duration: 0.6,
                      delay: i * 0.1,
                    }}
                  />
                ))}
              </motion.div>
            ) : state === 'responding' ? (
              <motion.div
                key="responding"
                className="flex items-center gap-1"
              >
                {[...Array(5)].map((_, i) => (
                  <motion.div
                    key={i}
                    className="w-1.5 bg-orange-400 rounded-full"
                    animate={{ height: [8, 20, 8] }}
                    transition={{
                      repeat: Infinity,
                      duration: 0.7,
                      delay: i * 0.12,
                    }}
                  />
                ))}
              </motion.div>
            ) : state === 'processing' ? (
              <motion.div
                key="processing"
                className="flex items-center gap-2"
              >
                {[0, 1, 2].map((i) => (
                  <motion.div
                    key={i}
                    className="w-2 h-2 rounded-full bg-orange-400"
                    animate={{ opacity: [0.3, 1, 0.3] }}
                    transition={{
                      repeat: Infinity,
                      duration: 1,
                      delay: i * 0.3,
                    }}
                  />
                ))}
              </motion.div>
            ) : (
              <motion.div
                key="idle"
                initial={{ scale: 0.8, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
              >
                <Volume2 className="w-10 h-10 text-muted-foreground/40" />
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Error message */}
        {error && (
          <p className="text-center text-xs text-destructive pb-2 px-4">
            {error}
          </p>
        )}

        {/* Controls */}
        <div className="flex items-center justify-center gap-4 px-5 pb-5">
          {isActive && (
            <Button
              variant="ghost"
              size="icon"
              className="h-12 w-12 rounded-full"
              onClick={() => setMuted(!muted)}
            >
              {muted ? (
                <MicOff className="w-5 h-5 text-destructive" />
              ) : (
                <Mic className="w-5 h-5" />
              )}
            </Button>
          )}

          {/* Main call button */}
          <Button
            size="icon"
            className={[
              'h-16 w-16 rounded-full shadow-lg transition-all',
              isActive
                ? 'bg-destructive hover:bg-destructive/80 shadow-destructive/25'
                : 'bg-gradient-to-br from-green-500 to-green-600 hover:from-green-600 hover:to-green-700 shadow-green-500/25',
            ].join(' ')}
            onClick={handleToggleCall}
          >
            {isActive ? (
              <PhoneOff className="w-6 h-6" />
            ) : state === 'connecting' ? (
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
            >
              <PhoneOff className="w-5 h-5 text-muted-foreground" />
            </Button>
          )}
        </div>
      </div>
    </motion.div>
  )
}
