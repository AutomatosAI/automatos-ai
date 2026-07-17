'use client'

/**
 * useRetellCall — the live-voice transport (PRD-207 S5).
 *
 * Replaces the dead Pipecat-pod WebSocket (`useVoiceStream`, PRD-74) with
 * Retell's WebRTC SDK: S1 mints a short-lived access token server-side (the
 * Retell key never reaches the browser), the SDK carries audio both ways and
 * Retell drives Auto's own agent loop through the custom-LLM webhook.
 *
 * Audio levels feed the orb through a mutable ref — the canvas reads it in
 * its own rAF loop, so 20ms audio frames never cause React re-renders.
 */

import { useCallback, useEffect, useRef, useState, type MutableRefObject } from 'react'
import { apiClient } from '@/lib/api-client'
import {
  ORB_STATE_LABELS,
  OrbSnapshot,
  USER_SPEECH_THRESHOLD,
  initialOrb,
  orbReducer,
  rmsLevel,
  type OrbState,
} from '@/lib/voice/orb-state'

export interface VoiceLevels {
  agent: number
  user: number
}

export interface CaptionLine {
  role: 'user' | 'agent'
  text: string
}

export interface UseRetellCallReturn {
  orbState: OrbState
  stateLabel: string
  /** Read by the orb canvas each frame — never triggers renders. */
  levelsRef: MutableRefObject<VoiceLevels>
  captions: CaptionLine[]
  durationSec: number
  muted: boolean
  /** Honest refusal copy when the mint was refused (cap, toggle, unarmed…). */
  refusal: string | null
  error: string | null
  isLive: boolean
  start: () => Promise<void>
  stop: () => void
  toggleMute: () => void
}

interface UseRetellCallOptions {
  /** Bind the call to this on-screen thread (omit when the chat has no server row yet). */
  chatId?: string | null
  agentId?: number | null
}

export function useRetellCall({ chatId, agentId }: UseRetellCallOptions): UseRetellCallReturn {
  const [snap, setSnap] = useState<OrbSnapshot>(initialOrb)
  const [captions, setCaptions] = useState<CaptionLine[]>([])
  const [durationSec, setDurationSec] = useState(0)
  const [muted, setMuted] = useState(false)
  const [refusal, setRefusal] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const levelsRef = useRef<VoiceLevels>({ agent: 0, user: 0 })
  const clientRef = useRef<any>(null)
  const micStreamRef = useRef<MediaStream | null>(null)
  const audioCtxRef = useRef<AudioContext | null>(null)
  const analyserTimerRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const durationTimerRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const tickTimerRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const startedAtRef = useRef(0)

  const dispatch = useCallback((event: Parameters<typeof orbReducer>[1]) => {
    setSnap((prev) => orbReducer(prev, event))
  }, [])

  const teardown = useCallback(() => {
    if (analyserTimerRef.current) clearInterval(analyserTimerRef.current)
    if (durationTimerRef.current) clearInterval(durationTimerRef.current)
    if (tickTimerRef.current) clearInterval(tickTimerRef.current)
    analyserTimerRef.current = durationTimerRef.current = tickTimerRef.current = null

    micStreamRef.current?.getTracks().forEach((t) => t.stop())
    micStreamRef.current = null
    if (audioCtxRef.current && audioCtxRef.current.state !== 'closed') {
      void audioCtxRef.current.close()
    }
    audioCtxRef.current = null

    try {
      clientRef.current?.stopCall()
    } catch {
      // already down
    }
    clientRef.current = null
    levelsRef.current = { agent: 0, user: 0 }
  }, [])

  const stop = useCallback(() => {
    teardown()
    dispatch({ type: 'call_ended' })
  }, [teardown, dispatch])

  const toggleMute = useCallback(() => {
    const client = clientRef.current
    if (!client) return
    setMuted((prev) => {
      try {
        if (prev) client.unmute()
        else client.mute()
      } catch {
        return prev
      }
      return !prev
    })
  }, [])

  const start = useCallback(async () => {
    setRefusal(null)
    setError(null)
    setCaptions([])
    setMuted(false)

    if (typeof navigator === 'undefined' || !navigator.mediaDevices?.getUserMedia) {
      setError('This browser cannot do live voice — try a current Chrome, Edge or Safari.')
      dispatch({ type: 'error' })
      return
    }

    dispatch({ type: 'call_connecting' })

    // 1 — mint (every refusal reason here is user-facing truth: platform off,
    // not armed, workspace off, over budget).
    let minted: { call_id: string; access_token: string }
    try {
      minted = await apiClient.request<{ call_id: string; access_token: string }>(
        '/api/voice/web-call',
        {
          method: 'POST',
          body: {
            ...(chatId ? { chat_id: chatId } : {}),
            ...(agentId ? { agent_id: agentId } : {}),
          } as any,
        }
      )
    } catch (err: any) {
      setRefusal(err?.message || 'Live voice is unavailable right now')
      dispatch({ type: 'error' })
      return
    }

    // 2 — connect (the token dies in ~30s unused; go now).
    try {
      const { RetellWebClient } = await import('retell-client-js-sdk')
      const client = new RetellWebClient()
      clientRef.current = client

      client.on('call_started', () => {
        startedAtRef.current = Date.now()
        dispatch({ type: 'call_started' })
        durationTimerRef.current = setInterval(() => {
          setDurationSec(Math.floor((Date.now() - startedAtRef.current) / 1000))
        }, 1000)
        tickTimerRef.current = setInterval(() => {
          dispatch({ type: 'tick', now: Date.now() })
        }, 250)
      })
      client.on('call_ended', () => {
        stop()
      })
      client.on('error', (message: any) => {
        setError(typeof message === 'string' ? message : 'Voice connection error')
        teardown()
        dispatch({ type: 'error' })
      })
      client.on('agent_start_talking', () => dispatch({ type: 'agent_start_talking' }))
      client.on('agent_stop_talking', () =>
        dispatch({ type: 'agent_stop_talking', now: Date.now() })
      )
      // Auto's voice, raw PCM → the orb's agent side.
      client.on('audio', (samples: Float32Array) => {
        levelsRef.current.agent = rmsLevel(samples)
      })
      // Live captions from Retell's running transcript.
      client.on('update', (update: any) => {
        const transcript = update?.transcript
        if (!Array.isArray(transcript) || transcript.length === 0) return
        const lines: CaptionLine[] = transcript.slice(-2).map((t: any) => ({
          role: t?.role === 'agent' ? 'agent' : 'user',
          text: String(t?.content ?? ''),
        }))
        setCaptions(lines)
      })

      await client.startCall({
        accessToken: minted.access_token,
        emitRawAudioSamples: true,
      })
    } catch (err: any) {
      const message = String(err?.message || err || '')
      if (/permission|denied|notallowed/i.test(message)) {
        setError('Microphone permission was denied — allow the mic and try again.')
      } else {
        setError('Could not connect the call — check your connection and try again.')
      }
      teardown()
      dispatch({ type: 'error' })
      return
    }

    // 3 — the user's side of the orb: a local analyser tap (viz only; the SDK
    // owns the mic track it sends).
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      micStreamRef.current = stream
      const ctx = new AudioContext()
      audioCtxRef.current = ctx
      const analyser = ctx.createAnalyser()
      analyser.fftSize = 512
      ctx.createMediaStreamSource(stream).connect(analyser)
      const buf = new Float32Array(analyser.fftSize)
      analyserTimerRef.current = setInterval(() => {
        analyser.getFloatTimeDomainData(buf)
        const level = rmsLevel(buf)
        levelsRef.current.user = level
        if (level >= USER_SPEECH_THRESHOLD) {
          dispatch({ type: 'user_voice', now: Date.now() })
        }
      }, 60)
    } catch {
      // No analyser is cosmetic-only: the call still works, the orb just
      // won't react to the user's voice.
    }
  }, [agentId, chatId, dispatch, stop, teardown])

  useEffect(() => () => teardown(), [teardown])

  return {
    orbState: snap.state,
    stateLabel: ORB_STATE_LABELS[snap.state],
    levelsRef,
    captions,
    durationSec,
    muted,
    refusal,
    error,
    isLive: snap.state === 'listening' || snap.state === 'thinking' || snap.state === 'speaking',
    start,
    stop,
    toggleMute,
  }
}
