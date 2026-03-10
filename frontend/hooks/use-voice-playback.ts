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

  const audioRef = useRef<HTMLAudioElement | null>(null)
  const rafRef = useRef<number | null>(null)
  const currentUrlRef = useRef<string | null>(null)

  const stopAnimationFrame = useCallback(() => {
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current)
      rafRef.current = null
    }
  }, [])

  const startTimeTracking = useCallback(() => {
    const tick = () => {
      if (audioRef.current) {
        setCurrentTime(audioRef.current.currentTime)
      }
      rafRef.current = requestAnimationFrame(tick)
    }
    rafRef.current = requestAnimationFrame(tick)
  }, [])

  const destroyAudio = useCallback(() => {
    stopAnimationFrame()
    if (audioRef.current) {
      audioRef.current.pause()
      audioRef.current.removeAttribute('src')
      audioRef.current.load()
      audioRef.current = null
    }
    currentUrlRef.current = null
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

  const play = useCallback(
    (url: string) => {
      // If playing the same URL, restart
      if (audioRef.current && currentUrlRef.current === url) {
        audioRef.current.currentTime = 0
        audioRef.current.play()
        setState('playing')
        startTimeTracking()
        return
      }

      // New URL — destroy old audio
      destroyAudio()

      setState('loading')
      const audio = new Audio(url)
      audioRef.current = audio
      currentUrlRef.current = url

      audio.onloadedmetadata = () => {
        setDuration(audio.duration)
      }

      audio.oncanplaythrough = () => {
        if (state === 'loading' || audioRef.current === audio) {
          audio.play()
          setState('playing')
          startTimeTracking()
        }
      }

      audio.onended = () => {
        stopAnimationFrame()
        setCurrentTime(audio.duration)
        setState('idle')
      }

      audio.onerror = () => {
        destroyAudio()
        setState('idle')
      }

      audio.load()
    },
    [destroyAudio, startTimeTracking, stopAnimationFrame, state]
  )

  const pause = useCallback(() => {
    if (audioRef.current && state === 'playing') {
      audioRef.current.pause()
      stopAnimationFrame()
      setState('paused')
    }
  }, [state, stopAnimationFrame])

  const resume = useCallback(() => {
    if (audioRef.current && state === 'paused') {
      audioRef.current.play()
      setState('playing')
      startTimeTracking()
    }
  }, [state, startTimeTracking])

  const stop = useCallback(() => {
    if (audioRef.current) {
      audioRef.current.pause()
      audioRef.current.currentTime = 0
      stopAnimationFrame()
      setCurrentTime(0)
      setState('idle')
    }
  }, [stopAnimationFrame])

  const setPlaybackRate = useCallback((rate: number) => {
    if (audioRef.current) {
      audioRef.current.playbackRate = rate
    }
  }, [])

  const seek = useCallback((time: number) => {
    if (audioRef.current) {
      audioRef.current.currentTime = time
      setCurrentTime(time)
    }
  }, [])

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
