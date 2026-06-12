'use client'

import { useState, useCallback, useMemo, useEffect, useRef } from 'react'
import { Play, Pause, Volume2 } from 'lucide-react'
import { motion } from 'framer-motion'
import { Button } from '@/components/ui/button'
import { useVoicePlayback } from '@/hooks/use-voice-playback'

interface VoicePlayerProps {
  audioUrl: string
  audioBase64?: string  // inline base64 audio (preferred, avoids fetch)
  autoPlay?: boolean
  duration?: number // in ms, for display before loading
  compact?: boolean
  className?: string
}

const PLAYBACK_RATES = [1, 1.5, 2, 0.5] as const

function formatTime(seconds: number): string {
  if (!isFinite(seconds) || seconds < 0) return '0:00'
  const mins = Math.floor(seconds / 60)
  const secs = Math.floor(seconds % 60)
  return `${mins}:${secs.toString().padStart(2, '0')}`
}

export function VoicePlayer({
  audioUrl,
  audioBase64,
  autoPlay = false,
  duration: initialDurationMs,
  compact = false,
  className = '',
}: VoicePlayerProps) {
  const {
    state,
    currentTime,
    duration: audioDuration,
    play,
    pause,
    resume,
    setPlaybackRate,
    seek,
  } = useVoicePlayback()

  const [rateIndex, setRateIndex] = useState(0)
  const currentRate = PLAYBACK_RATES[rateIndex]
  const didAutoPlay = useRef(false)

  // Auto-play on mount when autoPlay is set
  useEffect(() => {
    if (autoPlay && !didAutoPlay.current && (audioBase64 || audioUrl)) {
      didAutoPlay.current = true
      const src = audioBase64 ? `data:audio/mp3;base64,${audioBase64}` : audioUrl
      play(src)
    }
  }, [autoPlay, audioBase64, audioUrl, play])

  const displayDuration = useMemo(() => {
    if (audioDuration > 0) return audioDuration
    if (initialDurationMs && initialDurationMs > 0) return initialDurationMs / 1000
    return 0
  }, [audioDuration, initialDurationMs])

  const progress = displayDuration > 0 ? (currentTime / displayDuration) * 100 : 0

  const handlePlayPause = useCallback(() => {
    if (state === 'playing') {
      pause()
    } else if (state === 'paused') {
      resume()
    } else {
      // Prefer inline base64 (same approach as voice preview — no fetch needed)
      if (audioBase64) {
        play(`data:audio/mp3;base64,${audioBase64}`)
      } else {
        play(audioUrl)
      }
    }
  }, [state, audioUrl, audioBase64, play, pause, resume])

  const handleSeek = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      if (displayDuration <= 0) return
      const rect = e.currentTarget.getBoundingClientRect()
      const x = e.clientX - rect.left
      const ratio = Math.max(0, Math.min(1, x / rect.width))
      seek(ratio * displayDuration)
    },
    [displayDuration, seek]
  )

  const handleRateChange = useCallback(() => {
    const nextIndex = (rateIndex + 1) % PLAYBACK_RATES.length
    setRateIndex(nextIndex)
    setPlaybackRate(PLAYBACK_RATES[nextIndex])
  }, [rateIndex, setPlaybackRate])

  const isActive = state === 'playing' || state === 'paused'

  return (
    <motion.div
      initial={{ opacity: 0, y: 4 }}
      animate={{ opacity: 1, y: 0 }}
      className={[
        'flex items-center gap-2',
        compact ? 'py-1' : 'py-2 px-3 rounded-xl bg-warning/5 border border-warning/15',
        className,
      ].join(' ')}
    >
      {/* Play/Pause */}
      <Button
        type="button"
        variant="ghost"
        size="sm"
        onClick={handlePlayPause}
        disabled={state === 'loading'}
        className="h-7 w-7 p-0 flex-shrink-0 text-warning hover:text-warning hover:bg-warning/10"
      >
        {state === 'loading' ? (
          <div className="w-3.5 h-3.5 rounded-full border-2 border-warning border-t-transparent animate-spin" />
        ) : state === 'playing' ? (
          <Pause className="w-3.5 h-3.5 fill-current" />
        ) : (
          <Play className="w-3.5 h-3.5 fill-current" />
        )}
      </Button>

      {/* Progress bar */}
      <div
        className="flex-1 h-1.5 rounded-full bg-warning/10 cursor-pointer relative group"
        onClick={handleSeek}
      >
        <div
          className="absolute inset-y-0 left-0 rounded-full bg-gradient-to-r from-warning to-red-500 transition-[width] duration-100"
          style={{ width: `${Math.min(progress, 100)}%` }}
        />
        {/* Scrubber dot on hover */}
        {isActive && (
          <div
            className="absolute top-1/2 -translate-y-1/2 w-2.5 h-2.5 rounded-full bg-warning shadow-[0_0_6px_rgba(249,115,22,0.4)] opacity-0 group-hover:opacity-100 transition-opacity"
            style={{ left: `calc(${Math.min(progress, 100)}% - 5px)` }}
          />
        )}
      </div>

      {/* Time display */}
      <span className="text-[11px] tabular-nums font-mono text-muted-foreground min-w-[70px] text-right">
        {formatTime(currentTime)} / {formatTime(displayDuration)}
      </span>

      {/* Playback rate */}
      <button
        type="button"
        onClick={handleRateChange}
        className="text-[10px] font-medium text-muted-foreground hover:text-foreground tabular-nums px-1 py-0.5 rounded hover:bg-secondary/40 transition-colors"
        title="Change playback speed"
      >
        {currentRate}x
      </button>

      {/* Volume icon indicator */}
      {!compact && (
        <Volume2 className="w-3.5 h-3.5 text-warning/50 flex-shrink-0" />
      )}
    </motion.div>
  )
}
