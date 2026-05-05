'use client'

import { useCallback, useEffect } from 'react'
import { Mic, Square, Loader2 } from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'
import { Button } from '@/components/ui/button'
import type { RecordingState } from '@/hooks/use-voice-recorder'
import { toast } from 'sonner'

interface VoiceMicButtonProps {
  /** Current recorder state — managed by parent via useVoiceRecorder */
  state: RecordingState
  durationMs: number
  onStartRecording: () => void
  onStopRecording: () => void
  error: string | null
  disabled?: boolean
  className?: string
}

export function VoiceMicButton({
  state,
  durationMs,
  onStartRecording,
  onStopRecording,
  error,
  disabled = false,
  className = '',
}: VoiceMicButtonProps) {
  // Show toast on error
  useEffect(() => {
    if (error) {
      toast.error(error)
    }
  }, [error])

  const handleClick = useCallback(() => {
    if (state === 'idle') {
      onStartRecording()
    } else if (state === 'recording') {
      onStopRecording()
    }
  }, [state, onStartRecording, onStopRecording])

  const formatDuration = (ms: number): string => {
    const totalSeconds = Math.floor(ms / 1000)
    const minutes = Math.floor(totalSeconds / 60)
    const seconds = totalSeconds % 60
    return `${minutes}:${seconds.toString().padStart(2, '0')}`
  }

  const isRecording = state === 'recording'
  const isProcessing = state === 'processing'

  return (
    <div className={`relative inline-flex items-center ${className}`}>
      <AnimatePresence mode="wait">
        {isRecording && (
          <motion.span
            key="timer"
            initial={{ opacity: 0, x: -4 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -4 }}
            className="mr-1.5 text-xs tabular-nums font-mono text-destructive"
          >
            {formatDuration(durationMs)}
          </motion.span>
        )}
      </AnimatePresence>

      <Button
        type="button"
        variant="ghost"
        size="sm"
        disabled={disabled || isProcessing}
        onClick={handleClick}
        className={[
          'h-8 w-8 p-0 relative',
          isRecording
            ? 'text-destructive hover:text-destructive/80'
            : 'text-muted-foreground hover:text-foreground',
        ].join(' ')}
        title={
          isProcessing
            ? 'Processing voice...'
            : isRecording
              ? 'Stop recording'
              : 'Record voice message'
        }
      >
        {/* Pulsing ring behind button when recording */}
        <AnimatePresence>
          {isRecording && (
            <motion.span
              key="pulse"
              initial={{ scale: 0.8, opacity: 0 }}
              animate={{ scale: 1.6, opacity: 0 }}
              transition={{
                duration: 1.2,
                repeat: Infinity,
                ease: 'easeOut',
              }}
              className="absolute inset-0 rounded-full bg-destructive/30"
            />
          )}
        </AnimatePresence>

        <AnimatePresence mode="wait">
          {isProcessing ? (
            <motion.span
              key="processing"
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.8 }}
            >
              <Loader2 className="w-4 h-4 animate-spin" />
            </motion.span>
          ) : isRecording ? (
            <motion.span
              key="recording"
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.8 }}
            >
              <Square className="w-3.5 h-3.5 fill-current" />
            </motion.span>
          ) : (
            <motion.span
              key="idle"
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.8 }}
            >
              <Mic className="w-4 h-4" />
            </motion.span>
          )}
        </AnimatePresence>
      </Button>
    </div>
  )
}
