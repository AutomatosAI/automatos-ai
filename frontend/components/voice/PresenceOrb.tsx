'use client'

/**
 * PresenceOrb v3 — the voice-wave band (PRD-207 S5, to Gerard's waveform brief).
 *
 * The reference set is HORIZONTAL: a bright beam across the screen, dense
 * mirrored vertical bars breathing out of it, smooth ribbon waves flowing
 * through, a hot core flare at centre, drifting particles. Voice history
 * enters at the centre and travels outward to both edges, so speech
 * literally radiates from the middle of the beam.
 *
 * Canvas-2D, one rAF, zero React re-renders per audio frame (levels arrive
 * through a mutable ref). `prefers-reduced-motion` renders the full
 * composition once, static. All per-element "randomness" is a
 * deterministic index hash — stable frame to frame.
 *
 * `size` is the band HEIGHT; the band spans the container width (canvas
 * has a fixed internal resolution and stretches via CSS).
 */

import { useEffect, useRef, type MutableRefObject } from 'react'
import type { VoiceLevels } from '@/hooks/use-retell-call'
import type { OrbState } from '@/lib/voice/orb-state'

interface PresenceOrbProps {
  state: OrbState
  levelsRef: MutableRefObject<VoiceLevels>
  /** Band height in px; the band fills its container's width. */
  size?: number
  /** Ambient-background mode: no width cap, wider internal resolution —
   * the beam spans the whole container edge-to-edge with no canvas seam. */
  fullBleed?: boolean
}

function warningHsl(): { h: number; s: number; l: number } {
  if (typeof window !== 'undefined') {
    const raw = getComputedStyle(document.documentElement).getPropertyValue('--warning').trim()
    const m = raw.match(/^([\d.]+)\s+([\d.]+)%\s+([\d.]+)%$/)
    if (m) return { h: Number(m[1]), s: Number(m[2]), l: Number(m[3]) }
  }
  return { h: 38, s: 92, l: 50 }
}

/** Deterministic 0..1 hash per index — stable "randomness", no Math.random. */
function hash(i: number): number {
  const x = Math.sin(i * 127.1 + 311.7) * 43758.5453
  return x - Math.floor(x)
}

const SLOTS = 60 // half-width history slots (centre → edge)
const BAR_SPACING = 5.5

export function PresenceOrb({ state, levelsRef, size = 170, fullBleed = false }: PresenceOrbProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const stateRef = useRef<OrbState>(state)
  stateRef.current = state

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Internal resolution; CSS stretches to the container. Full-bleed uses a
    // wider raster so the beam/ribbons stay crisp across the whole window
    // while the voice bars keep their centred concentration.
    const W = fullBleed ? 1280 : 680
    const H = size
    const dpr = typeof window !== 'undefined' ? window.devicePixelRatio || 1 : 1
    canvas.width = W * dpr
    canvas.height = H * dpr
    ctx.scale(dpr, dpr)

    const { h, s, l } = warningHsl()
    // Full-bleed = the reference look: luminous, clearly THERE. The gain
    // lifts every stroke; the layer's opacity map still does the breathing.
    const gain = fullBleed ? 1.5 : 1
    const gold = (alpha: number, dl = 0) =>
      `hsla(${h}, ${s}%, ${Math.max(0, Math.min(100, l + dl))}%, ${Math.min(1, alpha * gain)})`
    const hot = (alpha: number) => `hsla(${h + 6}, 100%, 90%, ${Math.min(1, alpha * gain)})`

    const reducedMotion =
      typeof window !== 'undefined' &&
      window.matchMedia?.('(prefers-reduced-motion: reduce)').matches

    const cx = W / 2
    const cy = H / 2
    const maxBar = H * (fullBleed ? 0.44 : 0.36)

    // Voice history: slot 0 = centre (now), travelling outward each frame.
    const hist = new Float32Array(SLOTS)
    let smoothAgent = 0
    let smoothUser = 0
    let raf = 0

    const drawFrame = (t: number, animate: boolean) => {
      const st = stateRef.current
      const levels = levelsRef.current
      smoothAgent += (levels.agent - smoothAgent) * 0.35
      smoothUser += (levels.user - smoothUser) * 0.3
      const energy = Math.min(1, smoothAgent * 1.7 + smoothUser * 1.0)

      if (animate) {
        hist.copyWithin(1, 0) // history radiates outward from the centre
        hist[0] = energy
      }

      const dim = st === 'ended' || st === 'error' ? 0.35 : st === 'connecting' ? 0.65 : 1

      ctx.clearRect(0, 0, W, H)

      // 1 — ambient wash above/below the beam
      const wash = ctx.createLinearGradient(0, 0, 0, H)
      wash.addColorStop(0, gold(0))
      wash.addColorStop(0.5, gold(0.10 * dim, 6))
      wash.addColorStop(1, gold(0))
      ctx.fillStyle = wash
      ctx.fillRect(0, 0, W, H)

      // 2 — mirrored voice bars: history flows centre → edges (both sides)
      for (let sIdx = 0; sIdx < SLOTS; sIdx++) {
        const v = sIdx === 0 ? energy : hist[sIdx]
        const fade = 1 - (sIdx / SLOTS) * 0.8
        const jitter = hash(sIdx * 7) * 2
        const bh = 3 + v * maxBar * (0.45 + fade * 0.55) + jitter
        const hotBar = v > 0.45
        for (const dir of [-1, 1]) {
          const x = cx + dir * (sIdx * BAR_SPACING + 2)
          // soft wide pass + bright thin core = cheap glow
          ctx.strokeStyle = gold(0.14 * fade * dim, 8)
          ctx.lineWidth = 3.6
          ctx.beginPath(); ctx.moveTo(x, cy - bh); ctx.lineTo(x, cy + bh); ctx.stroke()
          ctx.strokeStyle = hotBar ? hot(0.9 * fade * dim) : gold(0.55 * fade * dim, 14)
          ctx.lineWidth = 1.4
          ctx.beginPath(); ctx.moveTo(x, cy - bh); ctx.lineTo(x, cy + bh); ctx.stroke()
        }
      }

      // 3 — ribbon waves flowing through the bars (three phase-shifted layers)
      const ribbonEnv = (x: number) => {
        const d = Math.abs(x - cx) / cx
        return (1 - d * d) * (0.35 + energy * 0.65)
      }
      for (let rIdx = 0; rIdx < 3; rIdx++) {
        const amp = H * (0.10 + rIdx * 0.045)
        const k = 0.012 + rIdx * 0.004
        const speed = (animate ? t : 900) / (900 + rIdx * 380)
        ctx.beginPath()
        for (let x = 0; x <= W; x += 6) {
          const y =
            cy +
            Math.sin(x * k + speed * (rIdx % 2 === 0 ? 1 : -1) * 2 + rIdx * 2.1) *
              amp * ribbonEnv(x)
          if (x === 0) ctx.moveTo(x, y)
          else ctx.lineTo(x, y)
        }
        ctx.strokeStyle = rIdx === 0 ? hot(0.35 * dim) : gold(0.22 * dim, 12 - rIdx * 6)
        ctx.lineWidth = rIdx === 0 ? 1.6 : 1.1
        ctx.stroke()
      }

      // 4 — the beam: a bright horizon line with a soft vertical glow
      const beamGlow = ctx.createLinearGradient(0, cy - 14, 0, cy + 14)
      beamGlow.addColorStop(0, gold(0))
      beamGlow.addColorStop(0.5, gold(0.5 * dim, 18))
      beamGlow.addColorStop(1, gold(0))
      ctx.fillStyle = beamGlow
      ctx.fillRect(0, cy - 14, W, 28)
      const beam = ctx.createLinearGradient(0, 0, W, 0)
      beam.addColorStop(0, gold(0.05 * dim))
      beam.addColorStop(0.5, hot(0.95 * dim))
      beam.addColorStop(1, gold(0.05 * dim))
      ctx.fillStyle = beam
      ctx.fillRect(0, cy - 0.8, W, 1.6)

      // 5 — core flare at centre: blooms with Auto's voice
      const flareR = 26 + smoothAgent * 95 + 6 * Math.sin((animate ? t : 600) / 700)
      const flare = ctx.createRadialGradient(cx, cy, 0, cx, cy, flareR)
      flare.addColorStop(0, hot(0.95 * dim))
      flare.addColorStop(0.3, gold(0.55 * dim, 20))
      flare.addColorStop(1, gold(0))
      ctx.fillStyle = flare
      ctx.beginPath()
      ctx.arc(cx, cy, flareR, 0, Math.PI * 2)
      ctx.fill()

      // 6 — drifting particles (the dust in the references)
      for (let p = 0; p < 26; p++) {
        const drift = animate ? (t / 1000) * (4 + hash(p) * 10) : 40
        const px = (hash(p * 3) * W + drift * (p % 2 === 0 ? 1 : -1) + W * 4) % W
        const py = cy + (hash(p * 5) - 0.5) * H * 0.75
        const tw = 0.25 + 0.6 * Math.abs(Math.sin((animate ? t : 500) / 900 + p * 1.7))
        ctx.fillStyle = p % 5 === 0 ? hot(tw * 0.8 * dim) : gold(tw * 0.5 * dim, 15)
        ctx.beginPath()
        ctx.arc(px, py, p % 4 === 0 ? 1.6 : 1, 0, Math.PI * 2)
        ctx.fill()
      }

      // 7 — thinking/connecting: a bright pulse travelling the beam
      if (st === 'thinking' || st === 'connecting') {
        const cycle = animate ? (t / 1600) % 1 : 0.35
        const xp = cycle * W
        const pg = ctx.createRadialGradient(xp, cy, 0, xp, cy, 22)
        pg.addColorStop(0, hot(0.9 * dim))
        pg.addColorStop(1, gold(0))
        ctx.fillStyle = pg
        ctx.beginPath()
        ctx.arc(xp, cy, 22, 0, Math.PI * 2)
        ctx.fill()
      }
    }

    if (reducedMotion) {
      for (let i = 0; i < SLOTS; i++) hist[i] = 0.15 + hash(i) * 0.45
      drawFrame(900, false)
      return () => undefined
    }

    const loop = (t: number) => {
      drawFrame(t, true)
      raf = requestAnimationFrame(loop)
    }
    raf = requestAnimationFrame(loop)
    return () => cancelAnimationFrame(raf)
  }, [levelsRef, size, fullBleed])

  return (
    <canvas
      ref={canvasRef}
      style={{ width: '100%', maxWidth: fullBleed ? undefined : 720, height: size, display: 'block' }}
      role="img"
      aria-hidden="true"
      data-orb-state={state}
    />
  )
}
