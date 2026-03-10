'use client'

/**
 * useVoiceStream — WebSocket-based real-time voice streaming (PRD-74 Phase 3)
 *
 * Connects to the voice-pipeline service via Pipecat's client SDK.
 * Handles mic capture, audio playback, and connection lifecycle.
 */

import { useState, useRef, useCallback, useEffect } from 'react'
import { useAuth } from '@clerk/nextjs'

export type VoiceStreamState =
  | 'disconnected'
  | 'connecting'
  | 'connected'
  | 'speaking'    // user is speaking (VAD active)
  | 'processing'  // waiting for AI response
  | 'responding'  // AI is speaking
  | 'error'

interface UseVoiceStreamOptions {
  workspaceId: string
  agentId?: number | null
  conversationId?: string
  onTranscript?: (text: string, isFinal: boolean) => void
  onResponseText?: (text: string) => void
  onStateChange?: (state: VoiceStreamState) => void
}

interface UseVoiceStreamReturn {
  state: VoiceStreamState
  connect: () => Promise<void>
  disconnect: () => void
  error: string | null
  sessionDuration: number
}

const PIPELINE_URL = process.env.NEXT_PUBLIC_VOICE_PIPELINE_URL || ''

export function useVoiceStream({
  workspaceId,
  agentId,
  conversationId,
  onTranscript,
  onResponseText,
  onStateChange,
}: UseVoiceStreamOptions): UseVoiceStreamReturn {
  const [state, setState] = useState<VoiceStreamState>('disconnected')
  const [error, setError] = useState<string | null>(null)
  const [sessionDuration, setSessionDuration] = useState(0)
  const { getToken } = useAuth()

  const wsRef = useRef<WebSocket | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const contextRef = useRef<AudioContext | null>(null)
  const processorRef = useRef<ScriptProcessorNode | null>(null)
  const playbackContextRef = useRef<AudioContext | null>(null)
  const startTimeRef = useRef<number>(0)
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const updateState = useCallback(
    (newState: VoiceStreamState) => {
      setState(newState)
      onStateChange?.(newState)
    },
    [onStateChange]
  )

  const disconnect = useCallback(() => {
    // Stop mic
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop())
      streamRef.current = null
    }
    // Close audio contexts
    if (contextRef.current?.state !== 'closed') {
      contextRef.current?.close()
    }
    contextRef.current = null
    processorRef.current = null
    if (playbackContextRef.current?.state !== 'closed') {
      playbackContextRef.current?.close()
    }
    playbackContextRef.current = null

    // Close WebSocket
    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }
    // Stop timer
    if (timerRef.current) {
      clearInterval(timerRef.current)
      timerRef.current = null
    }

    updateState('disconnected')
    setSessionDuration(0)
  }, [updateState])

  const connect = useCallback(async () => {
    if (state !== 'disconnected') return

    setError(null)
    updateState('connecting')

    try {
      // Get auth token
      const token = await getToken()

      // Build WebSocket URL
      const params = new URLSearchParams({ workspace_id: workspaceId })
      if (agentId) params.set('agent_id', String(agentId))
      if (conversationId) params.set('conversation_id', conversationId)
      if (token) params.set('token', token)

      const wsUrl = `${PIPELINE_URL.replace(/^http/, 'ws')}/ws/voice?${params}`

      // Request mic access
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: 16000,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      })
      streamRef.current = stream

      // Set up audio capture context
      const audioContext = new AudioContext({ sampleRate: 16000 })
      contextRef.current = audioContext
      const source = audioContext.createMediaStreamSource(stream)
      // ScriptProcessor for raw PCM capture (deprecated but widely supported)
      const processor = audioContext.createScriptProcessor(4096, 1, 1)
      processorRef.current = processor

      // Set up playback context
      playbackContextRef.current = new AudioContext({ sampleRate: 24000 })

      // Connect WebSocket
      const ws = new WebSocket(wsUrl)
      ws.binaryType = 'arraybuffer'
      wsRef.current = ws

      ws.onopen = () => {
        updateState('connected')
        startTimeRef.current = Date.now()
        // Start duration timer
        timerRef.current = setInterval(() => {
          setSessionDuration(Math.floor((Date.now() - startTimeRef.current) / 1000))
        }, 1000)

        // Start sending audio
        processor.onaudioprocess = (e) => {
          if (ws.readyState !== WebSocket.OPEN) return
          const input = e.inputBuffer.getChannelData(0)
          // Convert float32 to int16
          const int16 = new Int16Array(input.length)
          for (let i = 0; i < input.length; i++) {
            const s = Math.max(-1, Math.min(1, input[i]))
            int16[i] = s < 0 ? s * 0x8000 : s * 0x7fff
          }
          ws.send(int16.buffer)
        }
        source.connect(processor)
        processor.connect(audioContext.destination)
      }

      ws.onmessage = (event) => {
        if (event.data instanceof ArrayBuffer) {
          // Audio data from server — play it
          const pcmData = new Int16Array(event.data)
          if (playbackContextRef.current && pcmData.length > 0) {
            const float32 = new Float32Array(pcmData.length)
            for (let i = 0; i < pcmData.length; i++) {
              float32[i] = pcmData[i] / 0x7fff
            }
            const buffer = playbackContextRef.current.createBuffer(1, float32.length, 24000)
            buffer.getChannelData(0).set(float32)
            const bufferSource = playbackContextRef.current.createBufferSource()
            bufferSource.buffer = buffer
            bufferSource.connect(playbackContextRef.current.destination)
            bufferSource.start()

            if (state !== 'responding') updateState('responding')
          }
        } else if (typeof event.data === 'string') {
          // Control messages (JSON)
          try {
            const msg = JSON.parse(event.data)
            if (msg.type === 'transcript') {
              onTranscript?.(msg.text, msg.is_final)
            } else if (msg.type === 'response_text') {
              onResponseText?.(msg.text)
            } else if (msg.type === 'vad_start') {
              updateState('speaking')
            } else if (msg.type === 'vad_stop') {
              updateState('processing')
            } else if (msg.type === 'response_end') {
              updateState('connected')
            }
          } catch {
            // Not JSON, ignore
          }
        }
      }

      ws.onerror = () => {
        setError('Voice connection error')
        updateState('error')
      }

      ws.onclose = (event) => {
        if (state !== 'disconnected') {
          disconnect()
        }
      }
    } catch (err: any) {
      setError(err?.message || 'Failed to connect')
      updateState('error')
      disconnect()
    }
  }, [
    state,
    workspaceId,
    agentId,
    conversationId,
    getToken,
    updateState,
    disconnect,
    onTranscript,
    onResponseText,
  ])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      disconnect()
    }
  }, [disconnect])

  return {
    state,
    connect,
    disconnect,
    error,
    sessionDuration,
  }
}
