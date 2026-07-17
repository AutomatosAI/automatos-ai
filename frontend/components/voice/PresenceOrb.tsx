'use client'

/**
 * PresenceOrb v2 — the JARVIS ring (PRD-207 S5, rebuilt to Gerard's brief).
 *
 * Reference direction: Iron-Man-style circular interface — fragmented gold
 * arc rings rotating at different speeds, a radial audio waveform bent
 * around the core, particle blips, a white-hot turbulent centre, glow
 * everywhere. Canvas-2D, one rAF, zero React re-renders per audio frame
 * (levels arrive through a mutable ref). `prefers-reduced-motion` renders
 * the full composition once, static.
 *
 * All per-segment "randomness" is a deterministic index hash so the ring is
 * stable frame-to-frame and identical across mounts.
 */

import { useEffect, useRef, type MutableRefObject } from 'react'
import type { VoiceLevels } from '@/hooks/use-retell-call'
import type { OrbState } from '@/lib/voice/orb-state'

interface PresenceOrbProps {
  state: OrbState
  levelsRef: MutableRefObject<VoiceLevels>
  size?: number
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

const WAVE_BARS = 72
const RING_SEGMENTS = [26, 34, 18] // fragments per arc ring, inner→outer

export function PresenceOrb({ state, levelsRef, size = 260 }: PresenceOrbProps) {
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

    const { h, s, l } = warningHsl()
    const gold = (alpha: number, dl = 0, ds = 0) =>
      `hsla(${h}, ${Math.max(0, Math.min(100, s + ds))}%, ${Math.max(0, Math.min(100, l + dl))}%, ${alpha})`
    const hot = (alpha: number) => `hsla(${h + 6}, 100%, 88%, ${alpha})`

    const reducedMotion =
      typeof window !== 'undefined' &&
      window.matchMedia?.('(prefers-reduced-motion: reduce)').matches

    const cx = size / 2
    const cy = size / 2
    const R = size * 0.46 // outermost reach
    const coreR = size * 0.16
    const waveInner = size * 0.24
    const waveMax = size * 0.115

    // Rotating radial-waveform history: each frame writes the current energy
    // at the write head, so the bars carry a spinning trace of the last
    // seconds of BOTH voices (the audio-waveform references, bent circular).
    const waveHist = new Float32Array(WAVE_BARS)
    let waveHead = 0
    let smoothAgent = 0
    let smoothUser = 0
    let raf = 0

    const drawFrame = (t: number, animate: boolean) => {
      const st = stateRef.current
      const levels = levelsRef.current
      smoothAgent += (levels.agent - smoothAgent) * 0.3
      smoothUser += (levels.user - smoothUser) * 0.25
      const energy = Math.min(1, smoothAgent * 1.6 + smoothUser * 0.9)

      if (animate) {
        waveHist[waveHead] = energy
        waveHead = (waveHead + 1) % WAVE_BARS
      }

      const dim = st === 'ended' || st === 'error' ? 0.35 : st === 'connecting' ? 0.6 : 1
      const spin = st === 'speaking' ? 1.6 : st === 'thinking' ? 0.5 : 1

      ctx.clearRect(0, 0, size, size)

      // 1 — ambient halo, breathing with the room's energy
      const breath = 1 + 0.05 * Math.sin(t / 1100) + energy * 0.25
      const halo = ctx.createRadialGradient(cx, cy, coreR * 0.4, cx, cy, R * breath)
      halo.addColorStop(0, gold(0.16 * dim, 10))
      halo.addColorStop(0.5, gold(0.07 * dim))
      halo.addColorStop(1, gold(0))
      ctx.fillStyle = halo
      ctx.fillRect(0, 0, size, size)

      // 2 — fragmented arc rings (the JARVIS signature), counter-rotating
      RING_SEGMENTS.forEach((count, ringIdx) => {
        const rr = R * (0.78 + ringIdx * 0.09)
        const dir = ringIdx % 2 === 0 ? 1 : -1
        const rot = animate ? dir * (t / (9000 - ringIdx * 2600)) * spin : dir * 0.6
        const width = ringIdx === 1 ? 2.5 : 1.25
        for (let i = 0; i < count; i++) {
          const g0 = hash(i * 7 + ringIdx * 101)
          const g1 = hash(i * 13 + ringIdx * 211)
          const a0 = (i / count) * Math.PI * 2 + rot + g0 * 0.18
          const span = (Math.PI * 2 / count) * (0.25 + g1 * 0.55)
          // occasional bright "data blip" segments that flicker over time
          const blip = animate
            ? Math.max(0, Math.sin(t / 700 + i * 2.3 + ringIdx * 5)) ** 6
            : g1 * 0.4
          const alpha = (0.22 + g0 * 0.3 + blip * 0.5 + energy * 0.25) * dim
          ctx.strokeStyle = blip > 0.6 ? hot(Math.min(1, alpha)) : gold(Math.min(1, alpha), 8)
          ctx.lineWidth = width + blip * 1.5
          ctx.beginPath()
          ctx.arc(cx, cy, rr, a0, a0 + span)
          ctx.stroke()
        }
      })

      // 3 — radial waveform: the last seconds of voice, spun around the core.
      //     Agent speech renders hot gold; user presence warm amber.
      const userTint = st === 'listening' || st === 'thinking'
      for (let i = 0; i < WAVE_BARS; i++) {
        const slot = (waveHead - 1 - i + WAVE_BARS * 2) % WAVE_BARS
        const v = waveHist[slot]
        const live = i === 0 ? energy : v
        const len = 2 + live * waveMax + hash(i * 3) * 1.5
        const ang = (i / WAVE_BARS) * Math.PI * 2 + (animate ? t / 14000 : 0)
        const x0 = cx + Math.cos(ang) * waveInner
        const y0 = cy + Math.sin(ang) * waveInner
        const x1 = cx + Math.cos(ang) * (waveInner + len)
        const y1 = cy + Math.sin(ang) * (waveInner + len)
        const fade = 1 - (i / WAVE_BARS) * 0.75
        // wide soft pass + thin bright core = cheap glow
        ctx.strokeStyle = gold(0.12 * fade * dim, userTint ? 2 : 10)
        ctx.lineWidth = 3.4
        ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke()
        ctx.strokeStyle =
          live > 0.5 ? hot(0.85 * fade * dim) : gold(0.6 * fade * dim, userTint ? 6 : 16)
        ctx.lineWidth = 1.3
        ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke()
      }

      // 4 — turbulent core: white-hot centre, gold body, wobbled rim
      const pulse = coreR * (1 + 0.06 * Math.sin(t / 800) + smoothAgent * 0.5)
      ctx.beginPath()
      const pts = 48
      for (let i = 0; i <= pts; i++) {
        const a = (i / pts) * Math.PI * 2
        const wob = animate
          ? 0.05 * Math.sin(3 * a + t / 500) +
            0.035 * Math.sin(5 * a - t / 380) +
            smoothAgent * 0.12 * Math.sin(8 * a + t / 210)
          : 0.04 * Math.sin(3 * a)
        const r = pulse * (1 + wob)
        const x = cx + Math.cos(a) * r
        const y = cy + Math.sin(a) * r
        if (i === 0) ctx.moveTo(x, y)
        else ctx.lineTo(x, y)
      }
      ctx.closePath()
      const core = ctx.createRadialGradient(cx, cy, 0, cx, cy, pulse * 1.15)
      core.addColorStop(0, hot(0.95 * dim))
      core.addColorStop(0.35, gold(0.9 * dim, 18))
      core.addColorStop(0.8, gold(0.5 * dim, 4))
      core.addColorStop(1, gold(0))
      ctx.fillStyle = core
      ctx.fill()

      // 5 — thinking / connecting: a comet orbiting the waveform ring
      if (st === 'thinking' || st === 'connecting') {
        const base = animate ? t / 650 : 0.8
        const cometR = waveInner + waveMax * 0.5
        for (let i = 0; i < 10; i++) {
          const a = base - i * 0.09
          ctx.fillStyle = i === 0 ? hot(0.95 * dim) : gold((0.5 - i * 0.05) * dim, 12)
          ctx.beginPath()
          ctx.arc(cx + Math.cos(a) * cometR, cy + Math.sin(a) * cometR, i === 0 ? 3 : 2, 0, Math.PI * 2)
          ctx.fill()
        }
      }
    }

    if (reducedMotion) {
      // full composition, one static frame — designed, not a dot
      waveHist.forEach((_, i) => { waveHist[i] = hash(i) * 0.5 })
      drawFrame(1200, false)
      return () => undefined
    }

    const loop = (t: number) => {
      drawFrame(t, true)
      raf = requestAnimationFrame(loop)
    }
    raf = requestAnimationFrame(loop)
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
