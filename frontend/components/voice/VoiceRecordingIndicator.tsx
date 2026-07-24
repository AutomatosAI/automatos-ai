'use client'

import { Square, X } from 'lucide-react'
import { motion } from 'framer-motion'
import { Button } from '@/components/ui/button'

interface VoiceRecordingIndicatorProps {
  durationMs: number
  onStop: () => void
  onCancel: () => void
}

function formatDuration(ms: number): string {
  const totalSeconds = Math.floor(ms / 1000)
  const minutes = Math.floor(totalSeconds / 60)
  const seconds = totalSeconds % 60
  return `${minutes}:${seconds.toString().padStart(2, '0')}`
}

export function VoiceRecordingIndicator({
  durationMs,
  onStop,
  onCancel,
}: VoiceRecordingIndicatorProps) {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.95 }}
      transition={{ duration: 0.2 }}
      className="flex items-center justify-between px-4 py-3 w-full"
    >
      {/* Left: recording indicator + timer */}
      <div className="flex items-center gap-3">
        {/* Pulsing red dot */}
        <div className="relative flex items-center justify-center">
          <span className="absolute inline-flex h-3 w-3 rounded-full bg-destructive opacity-75 animate-ping" />
          <span className="relative inline-flex h-3 w-3 rounded-full bg-destructive" />
        </div>

        <span className="text-sm font-medium text-foreground">
          Recording...
        </span>
        <span className="text-sm tabular-nums text-muted-foreground font-mono">
          {formatDuration(durationMs)}
        </span>
      </div>

      {/* Center: waveform animation */}
      <div className="flex items-center gap-[3px] h-6 mx-4">
        {Array.from({ length: 16 }).map((_, i) => (
          <div
            key={i}
            className="w-[3px] rounded-full bg-gradient-to-t from-warning to-red-500"
            style={{
              animation: `voice-bar 0.8s ease-in-out ${i * 0.05}s infinite alternate`,
              height: '4px',
            }}
          />
        ))}
      </div>

      {/* Right: stop + cancel buttons */}
      <div className="flex items-center gap-2">
        <Button
          type="button"
          variant="ghost"
          size="sm"
          onClick={onCancel}
          className="h-8 w-8 p-0 text-muted-foreground hover:text-foreground"
          title="Cancel recording"
        >
          <X className="w-4 h-4" />
        </Button>

        <Button
          type="button"
          size="sm"
          onClick={onStop}
          className="h-8 px-3 bg-gradient-to-br from-warning to-red-500 hover:from-warning hover:to-red-600 text-white rounded-xl shadow-[0_0_12px_rgba(249,115,22,0.3)]"
          title="Stop and send"
        >
          <Square className="w-3.5 h-3.5 mr-1.5 fill-current" />
          <span className="text-xs font-medium">Send</span>
        </Button>
      </div>

      {/* CSS keyframes for waveform bars */}
      <style jsx>{`
        @keyframes voice-bar {
          0% {
            height: 4px;
          }
          100% {
            height: 20px;
          }
        }
      `}</style>
    </motion.div>
  )
}
