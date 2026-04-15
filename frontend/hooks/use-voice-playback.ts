'use client'

import { useState, useRef, useCallback, useEffect } from 'react'

export type PlaybackState = 'idle' | 'loading' | 'playing' | 'paused'

interface UseVoicePlaybackReturn {
  state: PlaybackState
  currentTime: number
  duration: number
  play: (url: string) => void
  pause: () => void
  resume: () => void
  stop: () => void
  setPlaybackRate: (rate: number) => void
  seek: (time: number) => void
}

export function useVoicePlayback(): UseVoicePlaybackReturn {
  const [state, setState] = useState<PlaybackState>('idle')
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)

  const ctxRef = useRef<AudioContext | null>(null)
  const sourceRef = useRef<AudioBufferSourceNode | null>(null)
  const bufferRef = useRef<AudioBuffer | null>(null)
  const startTimeRef = useRef(0)
  const offsetRef = useRef(0)
  const rafRef = useRef<number | null>(null)
  const currentUrlRef = useRef<string | null>(null)
  const rateRef = useRef(1)

  const stopAnimationFrame = useCallback(() => {
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current)
      rafRef.current = null
    }
  }, [])

  const startTimeTracking = useCallback(() => {
    const tick = () => {
      if (ctxRef.current && state === 'playing') {
        const elapsed = ctxRef.current.currentTime - startTimeRef.current
        setCurrentTime(offsetRef.current + elapsed * rateRef.current)
      }
      rafRef.current = requestAnimationFrame(tick)
    }
    rafRef.current = requestAnimationFrame(tick)
  }, [state])

  const destroyAudio = useCallback(() => {
    stopAnimationFrame()
    if (sourceRef.current) {
      try { sourceRef.current.stop() } catch { /* already stopped */ }
      sourceRef.current.disconnect()
      sourceRef.current = null
    }
    if (ctxRef.current) {
      ctxRef.current.close()
      ctxRef.current = null
    }
    bufferRef.current = null
    currentUrlRef.current = null
    startTimeRef.current = 0
    offsetRef.current = 0
    setState('idle')
    setCurrentTime(0)
    setDuration(0)
  }, [stopAnimationFrame])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      destroyAudio()
    }
  }, [destroyAudio])

  const playBuffer = useCallback((buffer: AudioBuffer, offset = 0) => {
    // Create fresh context + source for each play
    if (ctxRef.current) {
      ctxRef.current.close()
    }
    const ctx = new AudioContext()
    ctxRef.current = ctx

    const source = ctx.createBufferSource()
    source.buffer = buffer
    source.playbackRate.value = rateRef.current
    source.connect(ctx.destination)
    sourceRef.current = source

    offsetRef.current = offset
    startTimeRef.current = ctx.currentTime

    source.onended = () => {
      stopAnimationFrame()
      // Only set to idle if it played to the end (not manually stopped)
      if (sourceRef.current === source) {
        setCurrentTime(buffer.duration)
        setState('idle')
      }
    }

    source.start(0, offset)
    setState('playing')

    // Time tracking via rAF
    const tick = () => {
      if (ctxRef.current && sourceRef.current === source) {
        const elapsed = ctx.currentTime - startTimeRef.current
        const t = offset + elapsed * rateRef.current
        setCurrentTime(Math.min(t, buffer.duration))
      }
      rafRef.current = requestAnimationFrame(tick)
    }
    rafRef.current = requestAnimationFrame(tick)
  }, [stopAnimationFrame])

  const play = useCallback(
    async (url: string) => {
      // If we have the same buffer cached, replay from start
      if (bufferRef.current && currentUrlRef.current === url) {
        playBuffer(bufferRef.current, 0)
        return
      }

      // New URL — destroy old audio
      destroyAudio()
      setState('loading')
      currentUrlRef.current = url

      try {
        // Fetch audio bytes (follows redirects, avoids CORS issues with new Audio())
        const response = await fetch(url)
        if (!response.ok) {
          throw new Error(`Audio fetch failed: ${response.status}`)
        }
        const arrayBuffer = await response.arrayBuffer()

        // Decode with AudioContext
        const tempCtx = new AudioContext()
        const buffer = await tempCtx.decodeAudioData(arrayBuffer)
        tempCtx.close()

        bufferRef.current = buffer
        setDuration(buffer.duration)

        // Check we haven't been cancelled while loading
        if (currentUrlRef.current !== url) return

        playBuffer(buffer, 0)
      } catch (err) {
        console.error('Voice playback failed:', err)
        setState('idle')
      }
    },
    [destroyAudio, playBuffer]
  )

  const pause = useCallback(() => {
    if (sourceRef.current && ctxRef.current && state === 'playing') {
      // Record current position
      const elapsed = ctxRef.current.currentTime - startTimeRef.current
      offsetRef.current = offsetRef.current + elapsed * rateRef.current
      stopAnimationFrame()
      try { sourceRef.current.stop() } catch { /* ok */ }
      sourceRef.current.disconnect()
      sourceRef.current = null
      ctxRef.current.close()
      ctxRef.current = null
      setState('paused')
    }
  }, [state, stopAnimationFrame])

  const resume = useCallback(() => {
    if (bufferRef.current && state === 'paused') {
      playBuffer(bufferRef.current, offsetRef.current)
    }
  }, [state, playBuffer])

  const stop = useCallback(() => {
    stopAnimationFrame()
    if (sourceRef.current) {
      try { sourceRef.current.stop() } catch { /* ok */ }
      sourceRef.current.disconnect()
      sourceRef.current = null
    }
    if (ctxRef.current) {
      ctxRef.current.close()
      ctxRef.current = null
    }
    offsetRef.current = 0
    setCurrentTime(0)
    setState('idle')
  }, [stopAnimationFrame])

  const setPlaybackRate = useCallback((rate: number) => {
    rateRef.current = rate
    if (sourceRef.current) {
      sourceRef.current.playbackRate.value = rate
    }
  }, [])

  const seek = useCallback((time: number) => {
    if (bufferRef.current) {
      setCurrentTime(time)
      if (state === 'playing') {
        playBuffer(bufferRef.current, time)
      } else {
        offsetRef.current = time
      }
    }
  }, [state, playBuffer])

  return {
    state,
    currentTime,
    duration,
    play,
    pause,
    resume,
    stop,
    setPlaybackRate,
    seek,
  }
}
