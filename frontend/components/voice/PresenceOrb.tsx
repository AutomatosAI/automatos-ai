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

// Silk ribbon bundles (background). yf/amp/thick are fractions of height; tw
// drives the TWIST — the bundle pinches thin then spreads wide along its
// length, so each sheet folds like fabric in wind (Gerard's reference).
const RIBBONS = [
  { yf: -0.03, amp: 0.15, k: 0.0038, k2: 0.0089, tw: 0.0023, sp: 0.0002, strands: 72, thick: 0.15, ph: 0.0 },
  { yf: 0.05, amp: 0.21, k: 0.003, k2: 0.0067, tw: 0.0017, sp: 0.00016, strands: 82, thick: 0.19, ph: 2.1 },
  { yf: -0.12, amp: 0.1, k: 0.0051, k2: 0.0111, tw: 0.003, sp: 0.00027, strands: 58, thick: 0.105, ph: 4.0 },
  { yf: 0.13, amp: 0.075, k: 0.0066, k2: 0.0139, tw: 0.0039, sp: 0.00033, strands: 44, thick: 0.075, ph: 5.4 },
]

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

    // Cap DPR in background mode — the silk is stroke-heavy and a full-screen
    // 3× canvas would waste fill rate for no visible gain.
    const dpr = Math.min(typeof window !== 'undefined' ? window.devicePixelRatio || 1 : 1, isBackground ? 1.5 : 2)
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
      { at: 0.0, h: orange.h, s: 100, l: 60 },
      { at: 0.28, h: brandH, s: brandS, l: 58 },
      { at: 0.54, h: 330, s: 92, l: 66 },
      { at: 0.78, h: 288, s: 86, l: 68 },
      { at: 1.0, h: 248, s: 88, l: 66 },
    ]
    let gMid: CanvasGradient | null = null
    let gBright: CanvasGradient | null = null
    let gDeep: CanvasGradient | null = null
    const buildGradients = () => {
      gMid = ctx.createLinearGradient(0, 0, W, 0)
      gBright = ctx.createLinearGradient(0, 0, W, 0)
      gDeep = ctx.createLinearGradient(0, 0, W, 0)
      for (const s of SPECTRAL) {
        gMid.addColorStop(s.at, `hsla(${s.h}, ${s.s}%, ${s.l}%, 0.85)`)
        gBright.addColorStop(s.at, `hsla(${s.h}, 100%, 82%, 1)`)
        gDeep.addColorStop(s.at, `hsla(${s.h}, ${s.s}%, ${s.l}%, 0.5)`)
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
    let degraded = false // set after sustained slow frames (halves the silk)
    let slow = 0

    // The silk: dense fine strands whose spread PINCHES then SPREADS along the
    // length (the twist), reflected below the horizon for the glossy floor.
    const drawSilkField = (t: number, floorY: number, dim: number, reflect: boolean) => {
      const sway = 0.6 + 0.95 * energyEnv
      const stride = degraded ? 2 : 1
      for (const r of RIBBONS) {
        if (degraded && r.strands < 60) continue // shed the faintest ribbons when slow
        const A = H * r.amp * sway * (0.7 + 0.5 * dim)
        const cy = floorY + H * r.yf
        for (let s = 0; s < r.strands; s += stride) {
          const f = s / (r.strands - 1) - 0.5
          const edge = 1 - Math.abs(f) * 1.35
          if (edge <= 0) continue
          const core = Math.abs(f) < 0.13
          ctx.globalAlpha = edge * (reflect ? 0.075 : 0.17) * dim * (stride === 2 ? 1.7 : 1)
          ctx.strokeStyle = core ? gBright ?? hot(0.9) : gMid ?? gold(0.6)
          ctx.lineWidth = core ? 1.7 : 1.0
          ctx.beginPath()
          for (let x = 0; x <= W; x += 6) {
            const flow =
              Math.sin(x * r.k + t * r.sp + r.ph) * A +
              Math.sin(x * r.k2 + t * r.sp * 1.7 + r.ph) * A * 0.34
            const twist = Math.sin(x * r.tw + t * r.sp * 0.5 + r.ph)
            const width = H * r.thick * (0.12 + 0.88 * Math.abs(twist))
            let y = cy + flow + f * width
            if (reflect) y = 2 * floorY - y
            if (x === 0) ctx.moveTo(x, y)
            else ctx.lineTo(x, y)
          }
          ctx.stroke()
        }
      }
      ctx.globalAlpha = 1
    }

    // Vertical light columns behind the silk, tinted by the same ramp.
    const drawBarsField = (t: number, floorY: number, dim: number, reflect: boolean) => {
      const step = 20
      const field = Math.min(H * 0.22, 220)
      for (let x = step / 2; x < W; x += step) {
        const i = Math.round(x / step)
        const idle = 0.06 + 0.14 * hash(i * 13) + 0.05 * Math.sin(t / 1200 + i)
        const amp = Math.min(1, idle + energyEnv * 0.5)
        const y2 = reflect ? floorY + amp * field : floorY - amp * field
        ctx.globalAlpha = (reflect ? 0.05 : 0.11) * dim
        ctx.strokeStyle = gDeep ?? gold(0.4, 6)
        ctx.lineWidth = 5
        ctx.beginPath(); ctx.moveTo(x, floorY); ctx.lineTo(x, y2); ctx.stroke()
        ctx.globalAlpha = (reflect ? 0.1 : 0.24) * dim
        ctx.strokeStyle = gBright ?? gold(0.6, 18)
        ctx.lineWidth = 1.3
        ctx.beginPath(); ctx.moveTo(x, floorY); ctx.lineTo(x, y2); ctx.stroke()
      }
      ctx.globalAlpha = 1
    }

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

      const floorY = H * horizonY
      ctx.globalCompositeOperation = 'lighter'

      // reflection first (below the horizon), then the real scene on top.
      // The mirror is dropped under load — it's the cheapest thing to lose.
      if (!degraded) {
        drawBarsField(t, floorY, dim, true)
        drawSilkField(t, floorY, dim, true)
      }
      drawBarsField(t, floorY, dim, false)
      drawSilkField(t, floorY, dim, false)

      // core flare on Auto's voice
      const flareR = 30 + smoothAgent * 170
      const flare = ctx.createRadialGradient(W / 2, floorY, 0, W / 2, floorY, flareR)
      flare.addColorStop(0, hot(0.95 * dim * (0.3 + smoothAgent)))
      flare.addColorStop(0.4, gold(0.35 * dim, 8))
      flare.addColorStop(1, gold(0))
      ctx.globalAlpha = 1
      ctx.fillStyle = flare
      ctx.beginPath()
      ctx.arc(W / 2, floorY, flareR, 0, Math.PI * 2)
      ctx.fill()

      // thinking / connecting: a bright pulse crossing the horizon
      if (st === 'thinking' || st === 'connecting') {
        const xp = (animate ? (t / 1600) % 1 : 0.35) * W
        const pg = ctx.createRadialGradient(xp, floorY, 0, xp, floorY, 30)
        pg.addColorStop(0, hot(0.9 * dim))
        pg.addColorStop(1, gold(0))
        ctx.fillStyle = pg
        ctx.beginPath()
        ctx.arc(xp, floorY, 30, 0, Math.PI * 2)
        ctx.fill()
      }

      ctx.globalCompositeOperation = 'source-over'

      // glossy floor edge — a soft bright horizon line
      if (gBright) {
        ctx.globalAlpha = 0.28 * dim
        ctx.fillStyle = gBright
        ctx.fillRect(0, floorY - 0.8, W, 1.6)
        ctx.globalAlpha = 1
      }

      // fade the reflection into the floor — a grounded mirror, not a full copy
      const rf = ctx.createLinearGradient(0, floorY, 0, H)
      rf.addColorStop(0, 'rgba(8,7,10,0)')
      rf.addColorStop(0.65, 'rgba(8,7,10,0.5)')
      rf.addColorStop(1, 'rgba(8,7,10,0.92)')
      ctx.fillStyle = rf
      ctx.fillRect(0, floorY, W, H - floorY)
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
      energyEnv = 0.4
      smoothAgent = 0.3 // a settled, mid-energy silk frame
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
      const t0 = typeof performance !== 'undefined' ? performance.now() : 0
      try {
        draw(t, true)
      } catch (err) {
        // A draw fault is cosmetic — log once, stop the loop, never crash React.
        dead = true
        console.error('[PresenceOrb] draw loop stopped', err)
        return
      }
      // Sustained slow frames → shed the reflection + half the strands, once.
      if (t0 && !degraded) {
        if (performance.now() - t0 > 18) {
          if (++slow >= 3) degraded = true
        } else {
          slow = 0
        }
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
