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

/** Read a `H S% L%` design token off :root. Falls back when unreadable. */
function cssHsl(varName: string, fb: { h: number; s: number; l: number }): { h: number; s: number; l: number } {
  if (typeof window !== 'undefined') {
    const raw = getComputedStyle(document.documentElement).getPropertyValue(varName).trim()
    const m = raw.match(/^([\d.]+)\s+([\d.]+)%\s+([\d.]+)%$/)
    if (m) return { h: Number(m[1]), s: Number(m[2]), l: Number(m[3]) }
  }
  return fb
}

function brandHsl(): { h: number; s: number; l: number } {
  return cssHsl('--warning', { h: 38, s: 92, l: 50 })
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

    // Palette read live from the brand tokens. The Automatos ORANGE
    // (--primary) anchors the warm end; gold (--warning) the mid; magenta →
    // violet → indigo are the reference's jewel companions on the cool end.
    // Rendered as a horizontal canvas gradient so colour sweeps ALONG the
    // wave (RGB interpolation keeps orange→magenta pink, never green).
    const { h: brandH, s: brandS, l: brandL } = brandHsl()
    const orange = cssHsl('--primary', { h: 16, s: 100, l: 58 })
    const gold = (alpha: number, dl = 0) =>
      `hsla(${brandH}, ${brandS}%, ${Math.max(0, Math.min(100, brandL + dl))}%, ${Math.min(
        1,
        Math.max(0, alpha)
      )})`
    const hot = (alpha: number) =>
      `hsla(${brandH + 6}, 100%, 92%, ${Math.min(1, Math.max(0, alpha))})`

    const SPECTRAL = [
      { at: 0.0, h: orange.h, s: 100, l: 58 },
      { at: 0.3, h: brandH, s: brandS, l: 56 },
      { at: 0.56, h: 330, s: 88, l: 64 },
      { at: 0.82, h: 286, s: 82, l: 66 },
      { at: 1.0, h: 250, s: 80, l: 64 },
    ]
    let gMid: CanvasGradient | null = null
    let gBright: CanvasGradient | null = null
    let gFloor: CanvasGradient | null = null
    const buildGradients = () => {
      gMid = ctx.createLinearGradient(0, 0, W, 0)
      gBright = ctx.createLinearGradient(0, 0, W, 0)
      gFloor = ctx.createLinearGradient(0, 0, W, 0)
      for (const s of SPECTRAL) {
        gMid.addColorStop(s.at, `hsla(${s.h}, ${s.s}%, ${s.l}%, 0.5)`)
        gBright.addColorStop(s.at, `hsla(${s.h}, 100%, 80%, 0.95)`)
        gFloor.addColorStop(s.at, `hsla(${s.h}, ${s.s}%, ${s.l}%, 0.16)`)
      }
    }

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
      if (isBackground) buildGradients() // gradients span W → rebuild on resize
    }
    applySize()

    let observer: ResizeObserver | null = null
    if (isBackground && host && typeof ResizeObserver !== 'undefined') {
      observer = new ResizeObserver(applySize)
      observer.observe(host)
    }

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

      ctx.globalCompositeOperation = 'lighter'

      // floor glow — grounds the scene like the reference's reflection,
      // tinted across the full spectral ramp.
      if (gFloor) {
        ctx.globalAlpha = dim
        ctx.fillStyle = gFloor
        ctx.fillRect(0, cy, W, H - cy)
        ctx.globalAlpha = 1
      }

      // vertical light columns behind — energy radiates from the centre.
      const step = 18
      const barField = Math.min(H * 0.28, 260)
      for (let x = step / 2; x < W; x += step) {
        const i = Math.round(x / step)
        const distSlot = Math.min(SLOTS - 1, Math.floor((Math.abs(x - cx) / (W / 2)) * SLOTS))
        const v = distSlot === 0 ? Math.max(energy, hist[0]) : hist[distSlot]
        const idle = 0.08 + 0.12 * hash(i * 13) + 0.05 * Math.sin((animate ? t : 700) / 1200 + i)
        const amp = Math.min(1, idle + v * 1.0 + energyEnv * 0.22)
        const bh = amp * barField
        ctx.globalAlpha = 0.12 * dim
        ctx.strokeStyle = gMid ?? gold(0.12, 8)
        ctx.lineWidth = 6
        ctx.beginPath(); ctx.moveTo(x, cy - bh); ctx.lineTo(x, cy + bh * 0.85); ctx.stroke()
        ctx.globalAlpha = (v > 0.4 ? 0.5 : 0.28) * dim
        ctx.strokeStyle = v > 0.5 ? hot(1) : gBright ?? gold(0.5, 20)
        ctx.lineWidth = 1.4
        ctx.beginPath(); ctx.moveTo(x, cy - bh); ctx.lineTo(x, cy + bh * 0.85); ctx.stroke()
      }
      ctx.globalAlpha = 1

      // SILK RIBBON BUNDLES — the hero: many fine strands flowing through the
      // spectral ramp, a bright leading strand per bundle.
      const sway = 0.5 + 0.85 * energyEnv
      const bundles = [
        { amp: H * 0.055, k: 0.0062, sp: 1500, strands: 16, base: -H * 0.02 },
        { amp: H * 0.085, k: 0.0048, sp: 1950, strands: 18, base: H * 0.005 },
        { amp: H * 0.12, k: 0.0037, sp: 2500, strands: 20, base: H * 0.03 },
      ]
      for (let b = 0; b < bundles.length; b++) {
        const bd = bundles[b]
        const dir = b % 2 === 0 ? 1 : -1
        const A = Math.min(bd.amp, 150) * sway * (0.7 + 0.5 * dim)
        const speed = (animate ? t : 900) / bd.sp
        const lead = Math.floor(bd.strands / 2)
        for (let sIdx = 0; sIdx < bd.strands; sIdx++) {
          const off = sIdx - lead
          const ampJ = A * (0.82 + 0.36 * hash(b * 31 + sIdx))
          const ph = sIdx * 0.28 + hash(b * 7 + sIdx) * 2.2
          ctx.beginPath()
          for (let x = 0; x <= W; x += 7) {
            const env = 0.5 + 0.5 * Math.sin((x / W) * Math.PI)
            const y =
              cy +
              bd.base +
              Math.sin(x * bd.k + dir * speed * 2 + ph) * ampJ * env +
              Math.sin(x * bd.k * 2.3 + dir * speed * 3 + ph) * ampJ * 0.28 * env +
              off * 2.2
            if (x === 0) ctx.moveTo(x, y)
            else ctx.lineTo(x, y)
          }
          const isLead = sIdx === lead
          ctx.globalAlpha = (isLead ? 0.9 : 0.16) * dim
          ctx.strokeStyle = isLead ? gBright ?? hot(0.9) : gMid ?? gold(0.16, 12)
          ctx.lineWidth = isLead ? 2.0 : 0.9
          ctx.stroke()
        }
      }
      ctx.globalAlpha = 1

      // centre flare on Auto's voice
      const flareR = 24 + smoothAgent * 140
      const flare = ctx.createRadialGradient(cx, cy, 0, cx, cy, flareR)
      flare.addColorStop(0, hot(0.9 * dim * (0.25 + smoothAgent)))
      flare.addColorStop(0.4, gold(0.3 * dim, 8))
      flare.addColorStop(1, gold(0))
      ctx.fillStyle = flare
      ctx.beginPath()
      ctx.arc(cx, cy, flareR, 0, Math.PI * 2)
      ctx.fill()

      // particles — warm gold motes drifting through the field
      for (let p = 0; p < 32; p++) {
        const drift = (animate ? t / 1000 : 40) * (4 + hash(p) * 10)
        const px = (hash(p * 3) * W + drift * (p % 2 === 0 ? 1 : -1) + W * 8) % W
        const py = cy + (hash(p * 5) - 0.5) * Math.min(H * 0.7, 520)
        const tw = 0.3 + 0.6 * Math.abs(Math.sin((animate ? t : 500) / 900 + p * 1.7))
        ctx.globalAlpha = tw * 0.5 * dim
        ctx.fillStyle = gold(1, 18)
        ctx.beginPath()
        ctx.arc(px, py, p % 4 === 0 ? 1.7 : 1.05, 0, Math.PI * 2)
        ctx.fill()
      }
      ctx.globalAlpha = 1

      // thinking / connecting: a bright pulse crossing the scene
      if (st === 'thinking' || st === 'connecting') {
        const cycle = animate ? (t / 1600) % 1 : 0.35
        const xp = cycle * W
        const pg = ctx.createRadialGradient(xp, cy, 0, xp, cy, 28)
        pg.addColorStop(0, hot(0.9 * dim))
        pg.addColorStop(1, gold(0))
        ctx.fillStyle = pg
        ctx.beginPath()
        ctx.arc(xp, cy, 28, 0, Math.PI * 2)
        ctx.fill()
      }

      ctx.globalCompositeOperation = 'source-over'
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
