'use client'

import { Mic, Volume2 } from 'lucide-react'
import { motion } from 'framer-motion'
import { VoicePlayer } from './VoicePlayer'

interface VoiceMessageProps {
  transcript: string
  audioUrl?: string | null
  isUser: boolean
  sttLatencyMs?: number
  ttsLatencyMs?: number
  durationMs?: number
}

export function VoiceMessage({
  transcript,
  audioUrl,
  isUser,
  sttLatencyMs,
  ttsLatencyMs,
  durationMs,
}: VoiceMessageProps) {
  const showLatency =
    (sttLatencyMs !== undefined && sttLatencyMs > 0) ||
    (ttsLatencyMs !== undefined && ttsLatencyMs > 0)

  return (
    <motion.div
      initial={{ opacity: 0, y: 4 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-2"
    >
      {isUser ? (
        /* User voice message: icon + label + transcript */
        <div className="space-y-2">
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <Mic className="w-3.5 h-3.5 text-orange-400" />
            <span className="font-medium">Voice message</span>
          </div>
          {transcript && (
            <p className="text-sm text-foreground/80 italic leading-relaxed">
              {transcript}
            </p>
          )}
        </div>
      ) : (
        /* Assistant voice message: player + text response */
        <div className="space-y-2">
          {audioUrl && (
            <div className="flex items-center gap-2">
              <Volume2 className="w-3.5 h-3.5 text-orange-400 flex-shrink-0" />
              <VoicePlayer
                audioUrl={audioUrl}
                duration={durationMs}
                compact
                className="flex-1"
              />
            </div>
          )}
          {transcript && (
            <p className="text-sm text-foreground leading-relaxed">
              {transcript}
            </p>
          )}
        </div>
      )}

      {/* Latency badges */}
      {showLatency && (
        <div className="flex items-center gap-2">
          {sttLatencyMs !== undefined && sttLatencyMs > 0 && (
            <span className="rounded-full border border-orange-500/20 bg-orange-500/10 px-2 py-0.5 text-[10px] text-orange-300/80 tabular-nums">
              STT: {sttLatencyMs}ms
            </span>
          )}
          {ttsLatencyMs !== undefined && ttsLatencyMs > 0 && (
            <span className="rounded-full border border-orange-500/20 bg-orange-500/10 px-2 py-0.5 text-[10px] text-orange-300/80 tabular-nums">
              TTS: {ttsLatencyMs}ms
            </span>
          )}
        </div>
      )}
    </motion.div>
  )
}
