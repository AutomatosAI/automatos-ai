'use client'

import { useState, useRef, useCallback, useEffect } from 'react'

export type RecordingState = 'idle' | 'recording' | 'processing'

interface UseVoiceRecorderOptions {
  maxDurationMs?: number
  onRecordingComplete?: (blob: Blob, durationMs: number) => void
}

interface UseVoiceRecorderReturn {
  state: RecordingState
  durationMs: number
  startRecording: () => Promise<void>
  stopRecording: () => void
  cancelRecording: () => void
  error: string | null
}

function getSupportedMimeType(): string {
  const candidates = [
    'audio/webm;codecs=opus',
    'audio/webm',
    'audio/ogg;codecs=opus',
    'audio/ogg',
  ]
  for (const mime of candidates) {
    if (MediaRecorder.isTypeSupported(mime)) return mime
  }
  return ''
}

export function useVoiceRecorder(
  options: UseVoiceRecorderOptions = {}
): UseVoiceRecorderReturn {
  const { maxDurationMs = 120_000, onRecordingComplete } = options

  const [state, setState] = useState<RecordingState>('idle')
  const [durationMs, setDurationMs] = useState(0)
  const [error, setError] = useState<string | null>(null)

  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const startTimeRef = useRef<number>(0)
  const cancelledRef = useRef(false)

  const cleanup = useCallback(() => {
    if (timerRef.current) {
      clearInterval(timerRef.current)
      timerRef.current = null
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop())
      streamRef.current = null
    }
    mediaRecorderRef.current = null
    chunksRef.current = []
  }, [])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      cleanup()
    }
  }, [cleanup])

  const startRecording = useCallback(async () => {
    setError(null)
    cancelledRef.current = false

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      streamRef.current = stream

      const mimeType = getSupportedMimeType()
      const recorder = mimeType
        ? new MediaRecorder(stream, { mimeType })
        : new MediaRecorder(stream)

      mediaRecorderRef.current = recorder
      chunksRef.current = []

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data)
        }
      }

      recorder.onstop = () => {
        const elapsed = Date.now() - startTimeRef.current

        if (timerRef.current) {
          clearInterval(timerRef.current)
          timerRef.current = null
        }

        if (cancelledRef.current) {
          cleanup()
          setState('idle')
          setDurationMs(0)
          return
        }

        const blob = new Blob(chunksRef.current, {
          type: recorder.mimeType || 'audio/webm',
        })

        // Stop stream tracks
        if (streamRef.current) {
          streamRef.current.getTracks().forEach((track) => track.stop())
          streamRef.current = null
        }

        setState('processing')
        onRecordingComplete?.(blob, elapsed)
      }

      recorder.onerror = () => {
        setError('Recording failed unexpectedly')
        cleanup()
        setState('idle')
        setDurationMs(0)
      }

      // Start recording
      recorder.start(250) // collect chunks every 250ms
      startTimeRef.current = Date.now()
      setState('recording')
      setDurationMs(0)

      // Duration counter
      timerRef.current = setInterval(() => {
        const elapsed = Date.now() - startTimeRef.current
        setDurationMs(elapsed)

        // Auto-stop at max duration
        if (elapsed >= maxDurationMs) {
          recorder.stop()
        }
      }, 100)
    } catch (err: any) {
      if (err?.name === 'NotAllowedError' || err?.name === 'PermissionDeniedError') {
        setError('Microphone permission denied. Please allow microphone access.')
      } else if (err?.name === 'NotFoundError') {
        setError('No microphone found. Please connect a microphone.')
      } else {
        setError(err?.message || 'Failed to start recording')
      }
      cleanup()
      setState('idle')
    }
  }, [maxDurationMs, onRecordingComplete, cleanup])

  const stopRecording = useCallback(() => {
    cancelledRef.current = false
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.stop()
    }
  }, [])

  const cancelRecording = useCallback(() => {
    cancelledRef.current = true
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.stop()
    } else {
      cleanup()
      setState('idle')
      setDurationMs(0)
    }
  }, [cleanup])

  return {
    state,
    durationMs,
    startRecording,
    stopRecording,
    cancelRecording,
    error,
  }
}
