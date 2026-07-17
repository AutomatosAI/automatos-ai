'use client'

/**
 * PresenceOrb — Auto in the room (PRD-207 S5).
 *
 * Canvas-rendered brand-orange presence: idle breathing, a user-reactive
 * listening ring, a thinking shimmer arc, and voice-reactive speaking
 * pulses. Levels arrive through a mutable ref read inside the rAF loop —
 * zero React re-renders per audio frame. `prefers-reduced-motion` collapses
 * everything to a static glow (state is still announced textually by the
 * parent's aria-live label).
 */

import { useEffect, useRef } from 'react'
import type { VoiceLevels } from '@/hooks/use-retell-call'
import type { OrbState } from '@/lib/voice/orb-state'

interface PresenceOrbProps {
  state: OrbState
  levelsRef: React.MutableRefObject<VoiceLevels>
  size?: number
}

function warningColor(): { h: number; s: number; l: number } {
  // The brand `warning` token (the orange→warning codemod rule). Fallback
  // matches the design system's amber if the var is unreadable (tests/SSR).
  if (typeof window !== 'undefined') {
    const raw = getComputedStyle(document.documentElement).getPropertyValue('--warning').trim()
    const m = raw.match(/^([\d.]+)\s+([\d.]+)%\s+([\d.]+)%$/)
    if (m) return { h: Number(m[1]), s: Number(m[2]), l: Number(m[3]) }
  }
  return { h: 38, s: 92, l: 50 }
}

export function PresenceOrb({ state, levelsRef, size = 120 }: PresenceOrbProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const stateRef = useRef<OrbState>(state)
  stateRef.current = state

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const dpr = typeof window !== 'undefined' ? window.devicePixelRatio || 1 : 1
    canvas.width = size * dpr
    canvas.height = size * dpr
    ctx.scale(dpr, dpr)

    const { h, s, l } = warningColor()
    const hsl = (alpha: number, dl = 0) => `hsla(${h}, ${s}%, ${l + dl}%, ${alpha}`
    const color = (alpha: number, dl = 0) => `${hsl(alpha, dl)})`

    const reducedMotion =
      typeof window !== 'undefined' &&
      window.matchMedia?.('(prefers-reduced-motion: reduce)').matches

    const cx = size / 2
    const cy = size / 2
    const baseR = size * 0.3
    let raf = 0
    let smoothAgent = 0
    let smoothUser = 0

    const drawStatic = () => {
      ctx.clearRect(0, 0, size, size)
      const glow = ctx.createRadialGradient(cx, cy, baseR * 0.2, cx, cy, baseR * 1.6)
      glow.addColorStop(0, color(0.9, 8))
      glow.addColorStop(0.65, color(0.35))
      glow.addColorStop(1, color(0))
      ctx.fillStyle = glow
      ctx.beginPath()
      ctx.arc(cx, cy, baseR * 1.6, 0, Math.PI * 2)
      ctx.fill()
      ctx.fillStyle = color(0.95, 5)
      ctx.beginPath()
      ctx.arc(cx, cy, baseR * 0.9, 0, Math.PI * 2)
      ctx.fill()
    }

    if (reducedMotion) {
      drawStatic()
      return () => undefined
    }

    const frame = (t: number) => {
      const st = stateRef.current
      const levels = levelsRef.current
      // Smooth the raw RMS so the orb feels alive, not jittery.
      smoothAgent += (levels.agent - smoothAgent) * 0.25
      smoothUser += (levels.user - smoothUser) * 0.25

      ctx.clearRect(0, 0, size, size)

      const breath = 1 + 0.04 * Math.sin(t / 900)
      let r = baseR * breath
      let coreAlpha = 0.85
      let glowReach = 1.55

      if (st === 'speaking') {
        r = baseR * (1 + Math.min(0.45, smoothAgent * 2.2))
        coreAlpha = 0.95
        glowReach = 1.8
      } else if (st === 'listening') {
        r = baseR * breath
        glowReach = 1.6
      } else if (st === 'connecting' || st === 'ended') {
        coreAlpha = 0.5
      } else if (st === 'error') {
        coreAlpha = 0.35
      }

      const glow = ctx.createRadialGradient(cx, cy, r * 0.2, cx, cy, r * glowReach)
      glow.addColorStop(0, color(coreAlpha, 8))
      glow.addColorStop(0.65, color(coreAlpha * 0.4))
      glow.addColorStop(1, color(0))
      ctx.fillStyle = glow
      ctx.beginPath()
      ctx.arc(cx, cy, r * glowReach, 0, Math.PI * 2)
      ctx.fill()

      ctx.fillStyle = color(coreAlpha, 5)
      ctx.beginPath()
      ctx.arc(cx, cy, r * 0.82, 0, Math.PI * 2)
      ctx.fill()

      // Listening: a ring that swells with the USER's voice — both voices live
      // in the one presence.
      if (st === 'listening' || st === 'thinking') {
        const ringR = r * (1.02 + Math.min(0.5, smoothUser * 3))
        ctx.strokeStyle = color(0.5 + Math.min(0.4, smoothUser * 2.5), 12)
        ctx.lineWidth = 2
        ctx.beginPath()
        ctx.arc(cx, cy, ringR, 0, Math.PI * 2)
        ctx.stroke()
      }

      // Thinking: a slow shimmer arc orbiting the core — visible work, not limbo.
      if (st === 'thinking' || st === 'connecting') {
        const a0 = (t / 700) % (Math.PI * 2)
        ctx.strokeStyle = color(0.9, 15)
        ctx.lineWidth = 3
        ctx.lineCap = 'round'
        ctx.beginPath()
        ctx.arc(cx, cy, r * 1.12, a0, a0 + Math.PI * 0.6)
        ctx.stroke()
      }

      raf = requestAnimationFrame(frame)
    }

    raf = requestAnimationFrame(frame)
    return () => cancelAnimationFrame(raf)
  }, [levelsRef, size])

  return (
    <canvas
      ref={canvasRef}
      style={{ width: size, height: size }}
      role="img"
      aria-hidden="true"
      data-orb-state={state}
    />
  )
}
