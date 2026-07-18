'use client'

/**
 * PresenceOrb v4 — the living background (PRD-207/PRD-208, Gerard's brief
 * 2026-07-18: "voice is not another channel — a visual and audio extra;
 * the BACKGROUND makes Auto feel alive").
 *
 * Full-bleed canvas behind the chat thread. The composition follows Gerard's
 * reference set — a luminous horizontal spectral wave: dense fine-grain
 * mirrored bars radiating speech history from the centre, flowing ribbon
 * waves, a hot horizon beam with a core flare, ember fog masses, drifting
 * particles, floor reflections — rendered entirely in the brand palette
 * (--warning gold + --primary orange), never the reference blues.
 *
 * Discipline (from the voice-UI field study, 2026-07-18):
 * - State and energy are separate axes: the OrbState picks the MODE of
 *   motion (thinking shimmers with zero RMS coupling; idle only breathes);
 *   RMS only modulates intensity within the mode.
 * - User and Auto get disjoint geometry: Auto is the horizon; the user is a
 *   top-edge rim light that answers the mic live — including while Auto
 *   speaks (barge-in preview).
 * - Text wins: a vignette keeps the top (thread) and bottom (composer)
 *   zones dark; hot cores live only in the horizon band.
 * - Envelopes are asymmetric (fast attack, slow release) and the agent
 *   envelope is floored while speaking so speech never looks dead.
 * - No per-frame gradient allocation: soft glows are pre-baked sprites and
 *   static gradients are rebuilt only on resize (per-frame
 *   createRadialGradient/shadowBlur/filter are the Safari killers).
 *
 * `prefers-reduced-motion` renders one static composed frame per state.
 * All per-element "randomness" is a deterministic index hash — stable
 * frame to frame.
 */

import { useEffect, useRef, type MutableRefObject } from 'react'
import type { VoiceLevels } from '@/hooks/use-retell-call'
import type { OrbState } from '@/lib/voice/orb-state'

interface PresenceOrbProps {
  state: OrbState
  levelsRef: MutableRefObject<VoiceLevels>
  /** Horizon line as a fraction of height (welcome ~0.56, thread ~0.74).
   * Changes glide — the wave sinks or rises over ~600ms. */
  horizon?: number
}

interface Hsl {
  h: number
  s: number
  l: number
}

function cssHsl(varName: string, fallback: Hsl): Hsl {
  if (typeof window !== 'undefined') {
    const raw = getComputedStyle(document.documentElement).getPropertyValue(varName).trim()
    const m = raw.match(/^([\d.]+)\s+([\d.]+)%\s+([\d.]+)%$/)
    if (m) return { h: Number(m[1]), s: Number(m[2]), l: Number(m[3]) }
  }
  return fallback
}

/** Deterministic 0..1 hash per index — stable "randomness", no Math.random. */
function hash(i: number): number {
  const x = Math.sin(i * 127.1 + 311.7) * 43758.5453
  return x - Math.floor(x)
}

/** Pre-baked soft radial sprite: colour stops fading to transparent. */
function bakeRadialSprite(size: number, stops: Array<[number, string]>): HTMLCanvasElement {
  const c = document.createElement('canvas')
  c.width = size
  c.height = size
  const sctx = c.getContext('2d')
  if (sctx) {
    const g = sctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2)
    for (const [at, col] of stops) g.addColorStop(at, col)
    sctx.fillStyle = g
    sctx.fillRect(0, 0, size, size)
  }
  return c
}

const BAR_SPACING = 2.6
const MAX_SLOTS = 420
const INTRO_MS = 400
const ENDED_DECAY_MS = 450
const HORIZON_GLIDE = 0.06 // per-frame lerp (~600ms settle at 60fps)

// State → master intensity. The MODE of motion is decided in drawFrame();
// this only scales how present the room feels.
const STATE_INTENSITY: Record<OrbState, number> = {
  speaking: 1,
  listening: 0.8,
  thinking: 0.9,
  connecting: 0.55,
  idle: 0.5,
  error: 0.32,
  ended: 0,
}

export function PresenceOrb({ state, levelsRef, horizon = 0.56 }: PresenceOrbProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const stateRef = useRef<OrbState>(state)
  stateRef.current = state
  const horizonTargetRef = useRef(horizon)
  horizonTargetRef.current = horizon

  useEffect(() => {
    const canvas = canvasRef.current
    const host = canvas?.parentElement
    if (!canvas || !host) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const GOLD = cssHsl('--warning', { h: 43, s: 96, l: 56 })
    const EMBER = cssHsl('--primary', { h: 16, s: 100, l: 60 })
    const clampL = (l: number) => Math.max(0, Math.min(100, l))
    const gold = (a: number, dl = 0) => `hsla(${GOLD.h},${GOLD.s}%,${clampL(GOLD.l + dl)}%,${a})`
    const ember = (a: number, dl = 0) => `hsla(${EMBER.h},${EMBER.s}%,${clampL(EMBER.l + dl)}%,${a})`
    const hot = (a: number) => `hsla(${GOLD.h + 6},100%,92%,${a})`

    // Pre-baked glow sprites — drawn scaled per frame, never re-created.
    const fogEmber = bakeRadialSprite(256, [[0, ember(0.85, -16)], [1, 'hsla(0,0%,0%,0)']])
    const fogGold = bakeRadialSprite(256, [[0, gold(0.85, -20)], [1, 'hsla(0,0%,0%,0)']])
    const flare = bakeRadialSprite(256, [
      [0, hot(0.85)],
      [0.35, gold(0.4, 10)],
      [1, 'hsla(0,0%,0%,0)'],
    ])

    let W = 0
    let H = 0
    let slots = 0
    let hist = new Float32Array(0)
    let haloGrad: CanvasGradient | null = null
    let beamGrad: CanvasGradient | null = null
    let horizonGrad: CanvasGradient | null = null
    let vignetteGrad: CanvasGradient | null = null
    let rimGrad: CanvasGradient | null = null
    let horizonY = 0

    const rebuildStatics = () => {
      haloGrad = ctx.createLinearGradient(0, -H * 0.26, 0, H * 0.26)
      haloGrad.addColorStop(0, 'hsla(0,0%,0%,0)')
      haloGrad.addColorStop(0.5, gold(0.12, -6))
      haloGrad.addColorStop(1, 'hsla(0,0%,0%,0)')
      beamGrad = ctx.createLinearGradient(0, -18, 0, 18)
      beamGrad.addColorStop(0, 'hsla(0,0%,0%,0)')
      beamGrad.addColorStop(0.5, gold(0.2, 8))
      beamGrad.addColorStop(1, 'hsla(0,0%,0%,0)')
      horizonGrad = ctx.createLinearGradient(0, 0, W, 0)
      horizonGrad.addColorStop(0, 'hsla(0,0%,0%,0)')
      horizonGrad.addColorStop(0.5, hot(0.55))
      horizonGrad.addColorStop(1, 'hsla(0,0%,0%,0)')
      vignetteGrad = ctx.createLinearGradient(0, 0, 0, H)
      vignetteGrad.addColorStop(0, 'rgba(7,6,6,0.55)')
      vignetteGrad.addColorStop(0.3, 'rgba(7,6,6,0)')
      vignetteGrad.addColorStop(0.76, 'rgba(7,6,6,0)')
      vignetteGrad.addColorStop(1, 'rgba(7,6,6,0.62)')
      rimGrad = ctx.createLinearGradient(0, 0, 0, 110)
      rimGrad.addColorStop(0, gold(0.5, 22))
      rimGrad.addColorStop(1, 'hsla(0,0%,0%,0)')
    }

    const resize = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 1.5)
      W = host.clientWidth
      H = host.clientHeight
      if (W === 0 || H === 0) return
      canvas.width = Math.round(W * dpr)
      canvas.height = Math.round(H * dpr)
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      slots = Math.min(MAX_SLOTS, Math.ceil(W / 2 / BAR_SPACING) + 2)
      const old = hist
      hist = new Float32Array(slots)
      hist.set(old.subarray(0, Math.min(old.length, slots)))
      if (horizonY === 0) horizonY = H * horizonTargetRef.current
      rebuildStatics()
    }
    resize()
    const ro = new ResizeObserver(resize)
    ro.observe(host)

    // Asymmetric envelopes (fast attack, slow release) over the raw RMS refs.
    let agentEnv = 0
    let userEnv = 0
    let masterAlpha = 0 // intro ramp / ended decay
    let smoothIntensity = STATE_INTENSITY[stateRef.current] ?? 0.5
    const envStep = (env: number, target: number, dt: number, attackMs: number, releaseMs: number) => {
      const tau = target > env ? attackMs : releaseMs
      return env + (target - env) * (1 - Math.exp(-dt / tau))
    }

    // Adaptive degrade: two slow frames inside a second halves the bar work.
    let degraded = false
    let slowFrames = 0
    let slowWindowStart = 0

    let raf = 0
    let lastT = 0
    let stopped = false

    const drawFrame = (t: number, animate: boolean) => {
      const dt = Math.min(50, lastT ? t - lastT : 16)
      lastT = t
      const st = stateRef.current
      const I0 = STATE_INTENSITY[st] ?? 0.5
      smoothIntensity += (I0 - smoothIntensity) * 0.08

      if (st === 'ended') {
        masterAlpha = Math.max(0, masterAlpha - dt / ENDED_DECAY_MS)
        if (masterAlpha === 0) stopped = true
      } else {
        masterAlpha = Math.min(1, masterAlpha + dt / INTRO_MS)
      }
      const I = smoothIntensity * masterAlpha
      horizonY += (H * horizonTargetRef.current - horizonY) * HORIZON_GLIDE

      const { agent, user } = levelsRef.current
      // The state picks the mode: idle/thinking/connecting ignore the mic
      // for the horizon; the rim always answers the user.
      const agentDrive =
        st === 'speaking' ? Math.max(0.25, agent * 1.6) : st === 'listening' ? agent * 0.4 : 0
      userEnv = envStep(userEnv, Math.min(1, user * 2.2), dt, 50, 200)
      agentEnv = envStep(agentEnv, Math.min(1, agentDrive), dt, 70, 300)
      const energy = Math.min(1, agentEnv + userEnv * 0.55)

      if (animate) {
        hist.copyWithin(1, 0) // history radiates outward from the centre
        hist[0] = energy
      }

      const cy = horizonY
      const breathe = 0.5 + 0.5 * Math.sin((t / 4200) * 2 * Math.PI)
      ctx.clearRect(0, 0, W, H)
      if (I <= 0.001) return

      // L1 — ember fog masses drifting on slow sines.
      const drawFog = (img: HTMLCanvasElement, x: number, y: number, r: number, a: number) => {
        ctx.globalAlpha = a
        ctx.drawImage(img, x - r, y - r, r * 2, r * 2)
      }
      drawFog(fogEmber, W * (0.3 + 0.1 * Math.sin(t / 17000)), cy - H * 0.16, W * 0.46, 0.2 * I)
      drawFog(fogGold, W * (0.72 + 0.08 * Math.sin(t / 23000 + 2)), cy + H * 0.1, W * 0.42, 0.14 * I)
      drawFog(fogEmber, W * (0.52 + 0.12 * Math.sin(t / 14000 + 4)), cy, W * 0.34, 0.12 * I)
      ctx.globalAlpha = 1

      ctx.globalCompositeOperation = 'lighter'

      // Wide soft halo hugging the horizon.
      if (haloGrad) {
        ctx.save()
        ctx.translate(0, cy)
        ctx.globalAlpha = Math.min(1, (0.75 + energy * 0.6) * I)
        ctx.fillStyle = haloGrad
        ctx.fillRect(0, -H * 0.26, W, H * 0.52)
        ctx.restore()
        ctx.globalAlpha = 1
      }

      // L2 — dense fine spectrum: neighbour-smoothed history + shimmer floor.
      const step = degraded ? 2 : 1
      for (let s = 0; s < slots; s += step) {
        const vRaw =
          (hist[Math.max(0, s - 2)] +
            hist[Math.max(0, s - 1)] * 2 +
            hist[s] * 3 +
            hist[Math.min(slots - 1, s + 1)] * 2 +
            hist[Math.min(slots - 1, s + 2)]) / 9
        const centerW = 1 - 0.5 * (s / slots)
        const floorH = (2.5 + 5 * hash(s * 3)) * (0.7 + 0.6 * Math.sin(t / 700 + s * 0.35))
        const hgt = (floorH + vRaw * H * 0.16 * centerW * (0.75 + 0.5 * hash(s))) * I
        const a = (0.1 + vRaw * 0.3) * centerW * I
        const colMain = vRaw > 0.65 ? hot(a) : gold(a * 1.3, 12)
        const colSoft = vRaw > 0.65 ? hot(a * 0.75) : gold(a, 4)
        for (const dir of [-1, 1]) {
          const x = W / 2 + dir * s * BAR_SPACING
          ctx.strokeStyle = colSoft
          ctx.lineWidth = 2.2
          ctx.beginPath()
          ctx.moveTo(x, cy - hgt)
          ctx.lineTo(x, cy + hgt * 0.7)
          ctx.stroke()
          ctx.strokeStyle = colMain
          ctx.lineWidth = 0.9
          ctx.beginPath()
          ctx.moveTo(x, cy - hgt)
          ctx.lineTo(x, cy + hgt * 0.7)
          ctx.stroke()
          if (!degraded && s % 2 === 0) {
            // Floor reflection — the reference set's mirrored glow.
            ctx.strokeStyle = gold(a * 0.36, -6)
            ctx.lineWidth = 1.4
            ctx.beginPath()
            ctx.moveTo(x, cy + hgt * 0.7)
            ctx.lineTo(x, cy + hgt * 0.7 + hgt * 0.48)
            ctx.stroke()
          }
        }
      }

      // L3 — ribbon waves flowing the full width (soft pass + bright core).
      const ribbonSpeed = st === 'thinking' ? 0.72 : 1
      const ribbons = [
        { wl: W / 2.1, sp: 2600, amp: 22, lw: 1.9, col: (a: number) => gold(a, 10) },
        { wl: W / 3.3, sp: 1900, amp: 15, lw: 1.3, col: (a: number) => ember(a, 8) },
        { wl: W / 5.2, sp: 1450, amp: 9, lw: 0.9, col: (a: number) => hot(a * 0.8) },
      ]
      for (let r = 0; r < ribbons.length; r++) {
        const rb = ribbons[r]
        const baseA = (0.16 + energy * 0.3) * I
        for (const pass of [
          { lw: rb.lw * 3.4, a: baseA * 0.35 },
          { lw: rb.lw, a: baseA },
        ]) {
          ctx.strokeStyle = rb.col(pass.a)
          ctx.lineWidth = pass.lw
          ctx.beginPath()
          for (let x = 0; x <= W; x += 4) {
            const cw = 1 - Math.min(1, Math.abs(x - W / 2) / (W / 2)) * 0.45
            const y =
              cy +
              Math.sin((x / rb.wl) * 2 * Math.PI + t / (rb.sp * ribbonSpeed) + r * 2.1) *
                (rb.amp + energy * 52 * cw) *
                cw *
                (0.6 + 0.4 * Math.sin(t / 3400 + r))
            if (x === 0) ctx.moveTo(x, y)
            else ctx.lineTo(x, y)
          }
          ctx.stroke()
        }
      }

      // L4 — horizon beam + core flare blooming with Auto's voice.
      if (beamGrad) {
        ctx.save()
        ctx.translate(0, cy)
        ctx.globalAlpha = I
        ctx.fillStyle = beamGrad
        ctx.fillRect(0, -18, W, 36)
        ctx.restore()
      }
      if (horizonGrad) {
        ctx.globalAlpha = I
        ctx.fillStyle = horizonGrad
        ctx.fillRect(0, cy - 0.8, W, 1.6)
        ctx.globalAlpha = 1
      }
      const flareR = (36 + agentEnv * 150 + 8 * breathe) * (0.6 + 0.4 * I)
      ctx.globalAlpha = 0.9 * I
      ctx.drawImage(flare, W / 2 - flareR, cy - flareR, flareR * 2, flareR * 2)
      ctx.globalAlpha = 1

      // L6 — drifting particles.
      const particleCount = degraded ? 14 : 30
      for (let i = 0; i < particleCount; i++) {
        const px = (((hash(i) * 1.3 + t / (30000 + 20000 * hash(i + 40))) % 1.1) - 0.05) * W
        const py = cy + (hash(i + 80) - 0.5) * H * 0.5
        const pa = (0.04 + 0.12 * hash(i + 21)) * I * (0.5 + 0.5 * Math.sin(t / 2400 + i))
        ctx.fillStyle = i % 3 ? gold(pa, 12) : ember(pa, 8)
        ctx.beginPath()
        ctx.arc(px, py, 0.8 + 1.4 * hash(i + 55), 0, 7)
        ctx.fill()
      }

      // L7 — thinking / connecting travelling pulse (volume-independent).
      if (st === 'thinking' || st === 'connecting') {
        const tx = W / 2 + Math.sin(t / 900) * W * 0.28
        ctx.globalAlpha = 0.55 * I
        ctx.drawImage(flare, tx - 26, cy - 26, 52, 52)
        ctx.globalAlpha = 1
      }

      ctx.globalCompositeOperation = 'source-over'

      // User rim — the top edge answers the mic (disjoint geometry from
      // Auto's horizon; lights while Auto speaks = barge-in preview).
      if (rimGrad && userEnv > 0.02 && st !== 'idle' && st !== 'ended' && st !== 'error') {
        ctx.globalAlpha = Math.min(0.5, userEnv * 0.45) * masterAlpha
        ctx.fillStyle = rimGrad
        ctx.fillRect(0, 0, W, 110)
        ctx.globalAlpha = 1
      }

      // Vignette — the thread (top) and composer (bottom) stay readable.
      if (vignetteGrad) {
        ctx.fillStyle = vignetteGrad
        ctx.fillRect(0, 0, W, H)
      }
    }

    const reduced = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches
    if (reduced) {
      // One static composed frame per state (and layout), no rAF loop.
      const staticDraw = () => {
        masterAlpha = 1
        lastT = 0
        agentEnv = stateRef.current === 'speaking' ? 0.45 : 0
        userEnv = stateRef.current === 'listening' ? 0.3 : 0
        for (let s = 0; s < slots; s++) {
          hist[s] = Math.max(0, 0.4 - s / slots) * (0.6 + 0.4 * hash(s))
        }
        horizonY = H * horizonTargetRef.current
        drawFrame(900, false)
      }
      staticDraw()
      const interval = window.setInterval(staticDraw, 1200)
      return () => {
        window.clearInterval(interval)
        ro.disconnect()
      }
    }

    const loop = (t: number) => {
      if (stopped) return
      const frameStart = performance.now()
      drawFrame(t, true)
      const frameMs = performance.now() - frameStart
      if (frameMs > 12) {
        if (t - slowWindowStart > 1000) {
          slowWindowStart = t
          slowFrames = 0
        }
        slowFrames += 1
        if (slowFrames >= 2) degraded = true
      }
      raf = requestAnimationFrame(loop)
    }
    raf = requestAnimationFrame(loop)

    return () => {
      cancelAnimationFrame(raf)
      ro.disconnect()
    }
  }, [levelsRef])

  return (
    <canvas
      ref={canvasRef}
      role="img"
      aria-hidden="true"
      data-orb-state={state}
      className="absolute inset-0 h-full w-full"
    />
  )
}

export default PresenceOrb
