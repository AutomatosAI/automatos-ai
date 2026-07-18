'use client'

/**
 * PresenceOrb v5 — the repair of the two-session splice (and the union of
 * both designs).
 *
 * Two parallel sessions rewrote v4 simultaneously; the merge stitched half
 * of each (`warningHsl`/`SLOTS` called, `cssHsl`/`MAX_SLOTS` defined) and
 * the resulting ReferenceError white-screened the app the moment Live
 * mounted. v5 is ONE coherent implementation honouring BOTH contracts:
 *
 * * `horizon` (chat background, the other session's API): the canvas fills
 *   its absolute host, the beamline sits at `horizon`×height and GLIDES
 *   when it changes (welcome 0.56 → thread 0.74 — words own the upper
 *   field); state intensity lives inside.
 * * `size` band (VoiceCallPanel): the compact beam-and-bars strip.
 *
 * The background draws GERARD'S REFERENCE: three woven ribbon bundles
 * (gold / magenta / blue, seven offset strands, bright leading strand),
 * tall three-pass luminous bars across the full width with energy
 * radiating from centre, floor glow, particles, a flare on Auto's voice —
 * no dominant beam. A slow envelope keeps the room lit between words.
 *
 * Safety: the rAF loop is try/caught — a draw fault logs once and stops
 * the loop; it can never take the page down again.
 */

import { useEffect, useRef, type MutableRefObject } from 'react'
import type { VoiceLevels } from '@/hooks/use-retell-call'
import type { OrbState } from '@/lib/voice/orb-state'

interface PresenceOrbProps {
  state: OrbState
  levelsRef: MutableRefObject<VoiceLevels>
  /** Compact-band height (VoiceCallPanel). Ignored in background mode. */
  size?: number
  /** Background mode (fills the host). Implied when `horizon` is given. */
  fullBleed?: boolean
  /** 0..1 vertical beamline placement in background mode; glides on change. */
  horizon?: number
}

function brandHsl(): { h: number; s: number; l: number } {
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
const BAR_SPACING = 5.5 // compact band
const HORIZON_GLIDE = 0.06 // per-frame lerp toward the horizon target

// State → how present the room feels (background mode).
const STATE_INTENSITY: Record<OrbState, number> = {
  speaking: 1,
  listening: 0.8,
  thinking: 0.9,
  connecting: 0.55,
  idle: 0.5,
  error: 0.32,
  ended: 0,
}

export function PresenceOrb({
  state,
  levelsRef,
  size = 170,
  fullBleed = false,
  horizon,
}: PresenceOrbProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const stateRef = useRef<OrbState>(state)
  stateRef.current = state
  const horizonTargetRef = useRef(horizon ?? 0.5)
  horizonTargetRef.current = horizon ?? 0.5

  const isBackground = fullBleed || horizon != null

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    const host = canvas.parentElement

    const dpr = typeof window !== 'undefined' ? window.devicePixelRatio || 1 : 1
    let W = 680
    let H = size

    const applySize = () => {
      if (isBackground && host) {
        const rect = host.getBoundingClientRect()
        W = Math.max(320, Math.round(rect.width)) || 1280
        H = Math.max(240, Math.round(rect.height)) || 560
      } else {
        W = 680
        H = size
      }
      canvas.width = W * dpr
      canvas.height = H * dpr
      ctx.setTransform(1, 0, 0, 1, 0, 0)
      ctx.scale(dpr, dpr)
    }
    applySize()

    let observer: ResizeObserver | null = null
    if (isBackground && host && typeof ResizeObserver !== 'undefined') {
      observer = new ResizeObserver(applySize)
      observer.observe(host)
    }

    const { h: brandH, s: brandS, l: brandL } = brandHsl()
    const HUES = [brandH, 318, 215] // gold anchor, magenta + blue companions
    const col = (hue: number, alpha: number, dl = 0, ds = 0) =>
      `hsla(${hue}, ${Math.max(0, Math.min(100, brandS + ds))}%, ${Math.max(
        0,
        Math.min(100, brandL + dl)
      )}%, ${Math.min(1, Math.max(0, alpha))})`
    const gold = (alpha: number, dl = 0) => col(brandH, alpha, dl)
    const hot = (alpha: number) =>
      `hsla(${brandH + 6}, 100%, 92%, ${Math.min(1, Math.max(0, alpha))})`

    const reducedMotion =
      typeof window !== 'undefined' &&
      window.matchMedia?.('(prefers-reduced-motion: reduce)').matches

    const hist = new Float32Array(SLOTS)
    let smoothAgent = 0
    let smoothUser = 0
    let energyEnv = 0
    let horizonY = horizonTargetRef.current
    let raf = 0
    let dead = false

    const drawReference = (t: number, animate: boolean) => {
      const st = stateRef.current
      const levels = levelsRef.current
      smoothAgent += (levels.agent - smoothAgent) * 0.35
      smoothUser += (levels.user - smoothUser) * 0.3
      const energy = Math.min(1, smoothAgent * 1.7 + smoothUser * 1.0)
      energyEnv += ((st === 'speaking' ? Math.max(0.35, energy) : energy) - energyEnv) * 0.06
      horizonY += (horizonTargetRef.current - horizonY) * HORIZON_GLIDE

      if (animate) {
        hist.copyWithin(1, 0)
        hist[0] = energy
      }

      const dim = STATE_INTENSITY[st] ?? 0.5
      ctx.clearRect(0, 0, W, H)
      if (dim <= 0) return

      const cx = W / 2
      const cy = H * horizonY

      // floor glow — grounds the scene like the reference's reflections
      const floor = ctx.createRadialGradient(cx, cy + H * 0.16, 0, cx, cy + H * 0.16, W * 0.45)
      floor.addColorStop(0, gold(0.14 * dim, 6))
      floor.addColorStop(1, gold(0))
      ctx.fillStyle = floor
      ctx.fillRect(0, 0, W, H)

      // TALL BARS across the full width — energy radiates from the centre
      const step = 16
      const barField = Math.min(H * 0.3, 260)
      for (let x = step / 2; x < W; x += step) {
        const i = Math.round(x / step)
        const distSlot = Math.min(SLOTS - 1, Math.floor((Math.abs(x - cx) / (W / 2)) * SLOTS))
        const v = distSlot === 0 ? Math.max(energy, hist[0]) : hist[distSlot]
        const idle = 0.1 + 0.14 * hash(i * 13) + 0.06 * Math.sin((animate ? t : 700) / 1200 + i)
        const amp = Math.min(1, idle + v * 1.1 + energyEnv * 0.25)
        const bh = amp * barField
        const hue = HUES[i % 7 === 0 ? 1 : i % 11 === 0 ? 2 : 0]
        ctx.strokeStyle = col(hue, 0.1 * dim, 10)
        ctx.lineWidth = 7
        ctx.beginPath(); ctx.moveTo(x, cy - bh); ctx.lineTo(x, cy + bh * 0.9); ctx.stroke()
        ctx.strokeStyle = col(hue, 0.3 * dim, 16)
        ctx.lineWidth = 2.6
        ctx.beginPath(); ctx.moveTo(x, cy - bh); ctx.lineTo(x, cy + bh * 0.9); ctx.stroke()
        ctx.strokeStyle = v > 0.45 ? hot(0.85 * dim) : col(hue, 0.5 * dim, 24)
        ctx.lineWidth = 1.2
        ctx.beginPath(); ctx.moveTo(x, cy - bh); ctx.lineTo(x, cy + bh * 0.9); ctx.stroke()
      }

      // WOVEN RIBBON BUNDLES — the hero of the reference
      const sway = 0.55 + 0.75 * energyEnv
      for (let b = 0; b < 3; b++) {
        const hue = HUES[b]
        const baseAmp = Math.min(H * 0.11, 90) * (1 + b * 0.4) * sway
        const k = 0.0075 + b * 0.0022
        const dir = b % 2 === 0 ? 1 : -1
        const speed = (animate ? t : 900) / (1150 + b * 420)
        for (let sIdx = 0; sIdx < 7; sIdx++) {
          const ampJ = baseAmp * (0.8 + 0.4 * hash(b * 31 + sIdx))
          const phase = sIdx * 0.33 + hash(b * 7 + sIdx) * 2.2
          const lead = sIdx === 3
          ctx.beginPath()
          for (let x = 0; x <= W; x += 8) {
            const env = 0.55 + 0.45 * Math.sin((x / W) * Math.PI)
            const y =
              cy -
              H * 0.015 +
              Math.sin(x * k + dir * speed * 2 + phase) * ampJ * env +
              (sIdx - 3) * 2.4
            if (x === 0) ctx.moveTo(x, y)
            else ctx.lineTo(x, y)
          }
          ctx.strokeStyle = lead ? col(hue, 0.55 * dim, 30, 8) : col(hue, 0.14 * dim, 14)
          ctx.lineWidth = lead ? 2.1 : 1.1
          ctx.stroke()
        }
      }

      // centre flare on Auto's voice
      const flareR = 20 + smoothAgent * 120
      const flare = ctx.createRadialGradient(cx, cy, 0, cx, cy, flareR)
      flare.addColorStop(0, hot(0.9 * dim * (0.25 + smoothAgent)))
      flare.addColorStop(1, gold(0))
      ctx.fillStyle = flare
      ctx.beginPath()
      ctx.arc(cx, cy, flareR, 0, Math.PI * 2)
      ctx.fill()

      // particles
      for (let p = 0; p < 34; p++) {
        const drift = animate ? (t / 1000) * (4 + hash(p) * 10) : 40
        const px = (hash(p * 3) * W + drift * (p % 2 === 0 ? 1 : -1) + W * 8) % W
        const py = cy + (hash(p * 5) - 0.5) * Math.min(H * 0.7, 500)
        const tw = 0.3 + 0.6 * Math.abs(Math.sin((animate ? t : 500) / 900 + p * 1.7))
        ctx.fillStyle = p % 6 === 0 ? col(HUES[1], tw * 0.7 * dim, 20) : gold(tw * 0.55 * dim, 18)
        ctx.beginPath()
        ctx.arc(px, py, p % 4 === 0 ? 1.7 : 1.1, 0, Math.PI * 2)
        ctx.fill()
      }

      // thinking / connecting: a bright pulse crossing the scene
      if (st === 'thinking' || st === 'connecting') {
        const cycle = animate ? (t / 1600) % 1 : 0.35
        const xp = cycle * W
        const pg = ctx.createRadialGradient(xp, cy, 0, xp, cy, 26)
        pg.addColorStop(0, hot(0.9 * dim))
        pg.addColorStop(1, gold(0))
        ctx.fillStyle = pg
        ctx.beginPath()
        ctx.arc(xp, cy, 26, 0, Math.PI * 2)
        ctx.fill()
      }
    }

    const drawBand = (t: number, animate: boolean) => {
      const st = stateRef.current
      const levels = levelsRef.current
      smoothAgent += (levels.agent - smoothAgent) * 0.35
      smoothUser += (levels.user - smoothUser) * 0.3
      const energy = Math.min(1, smoothAgent * 1.7 + smoothUser * 1.0)

      if (animate) {
        hist.copyWithin(1, 0)
        hist[0] = energy
      }

      const dim = st === 'ended' || st === 'error' ? 0.35 : st === 'connecting' ? 0.65 : 1
      ctx.clearRect(0, 0, W, H)
      const cx = W / 2
      const cy = H / 2

      for (let sIdx = 0; sIdx < SLOTS; sIdx++) {
        const v = sIdx === 0 ? energy : hist[sIdx]
        const fade = 1 - (sIdx / SLOTS) * 0.8
        const bh = 3 + v * H * 0.36 * (0.45 + fade * 0.55) + hash(sIdx * 7) * 2
        for (const dir of [-1, 1]) {
          const x = cx + dir * (sIdx * BAR_SPACING + 2)
          ctx.strokeStyle = gold(0.14 * fade * dim, 8)
          ctx.lineWidth = 3.6
          ctx.beginPath(); ctx.moveTo(x, cy - bh); ctx.lineTo(x, cy + bh); ctx.stroke()
          ctx.strokeStyle = v > 0.45 ? hot(0.9 * fade * dim) : gold(0.55 * fade * dim, 14)
          ctx.lineWidth = 1.4
          ctx.beginPath(); ctx.moveTo(x, cy - bh); ctx.lineTo(x, cy + bh); ctx.stroke()
        }
      }

      const beam = ctx.createLinearGradient(0, 0, W, 0)
      beam.addColorStop(0, gold(0.05 * dim))
      beam.addColorStop(0.5, hot(0.95 * dim))
      beam.addColorStop(1, gold(0.05 * dim))
      ctx.fillStyle = beam
      ctx.fillRect(0, cy - 0.8, W, 1.6)

      const flareR = 26 + smoothAgent * 60
      const flare = ctx.createRadialGradient(cx, cy, 0, cx, cy, flareR)
      flare.addColorStop(0, hot(0.9 * dim))
      flare.addColorStop(1, gold(0))
      ctx.fillStyle = flare
      ctx.beginPath()
      ctx.arc(cx, cy, flareR, 0, Math.PI * 2)
      ctx.fill()
    }

    const draw = isBackground ? drawReference : drawBand

    if (reducedMotion) {
      for (let i = 0; i < SLOTS; i++) hist[i] = 0.15 + hash(i) * 0.45
      energyEnv = 0.35
      try {
        draw(900, false)
      } catch (err) {
        console.error('[PresenceOrb] static draw failed', err)
      }
      return () => {
        observer?.disconnect()
      }
    }

    const loop = (t: number) => {
      if (dead) return
      try {
        draw(t, true)
      } catch (err) {
        // A draw fault is cosmetic — log once, stop the loop, never crash React.
        dead = true
        console.error('[PresenceOrb] draw loop stopped', err)
        return
      }
      raf = requestAnimationFrame(loop)
    }
    raf = requestAnimationFrame(loop)
    return () => {
      dead = true
      cancelAnimationFrame(raf)
      observer?.disconnect()
    }
  }, [levelsRef, size, isBackground])

  return (
    <canvas
      ref={canvasRef}
      style={
        isBackground
          ? { width: '100%', height: '100%', display: 'block' }
          : { width: '100%', maxWidth: 720, height: size, display: 'block' }
      }
      role="img"
      aria-hidden="true"
      data-orb-state={state}
    />
  )
}
