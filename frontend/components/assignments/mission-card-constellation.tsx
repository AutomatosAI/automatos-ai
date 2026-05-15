'use client'

/**
 * Mission Hero Card — Constellation
 *
 * Animated SVG hero card for the Assignments page. Visualizes a Mission as a
 * constellation of agent orbs connected by traveling light pulses — evoking
 * field memory + parallel reasoning across many agents.
 *
 * The "X missions running" pill is wired to live data via useMissions.
 */

import { useId } from 'react'
import { motion } from 'framer-motion'
import { Card } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { useMissions } from '@/hooks/use-missions-api'

interface Orb {
  id: string
  cx: number
  cy: number
  r: number
  role: string
}

const ORBS: Orb[] = [
  { id: 'a', cx: 180, cy: 110, r: 10, role: 'Researcher' },
  { id: 'b', cx: 260, cy: 70, r: 8, role: 'Writer' },
  { id: 'c', cx: 340, cy: 130, r: 9, role: 'Analyst' },
  { id: 'd', cx: 130, cy: 200, r: 8, role: 'Critic' },
  { id: 'e', cx: 220, cy: 230, r: 11, role: 'Lead' },
  { id: 'f', cx: 320, cy: 220, r: 8, role: 'Builder' },
  { id: 'g', cx: 380, cy: 60, r: 7, role: 'Scout' },
]

const LINKS: Array<[string, string]> = [
  ['a', 'b'],
  ['b', 'c'],
  ['a', 'e'],
  ['c', 'f'],
  ['d', 'e'],
  ['e', 'f'],
  ['b', 'g'],
  ['c', 'g'],
  ['a', 'd'],
]

const ORB_BY_ID = Object.fromEntries(ORBS.map((o) => [o.id, o])) as Record<string, Orb>

// Deterministic dust positions (no Math.random — keeps SSR stable)
const DUST = Array.from({ length: 24 }, (_, i) => ({
  cx: (i * 53) % 480,
  cy: (i * 91) % 320,
  r: i % 7 === 0 ? 0.9 : 0.5,
  opacity: 0.2 + (i % 5) * 0.06,
}))

interface MissionConstellationCardProps {
  onStart: () => void
}

export function MissionConstellationCard({ onStart }: MissionConstellationCardProps) {
  // Unique gradient/filter IDs so multiple constellations on a page don't collide
  const uid = useId().replace(/:/g, '')
  const bgId = `cst-bg-${uid}`
  const orbId = `cst-orb-${uid}`
  const orbDimId = `cst-orb-dim-${uid}`
  const glowId = `cst-glow-${uid}`

  const { data } = useMissions({ state: 'running', limit: 20 })
  const runningCount = data?.missions?.length ?? 0

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.1 }}
      className="lg:col-span-2"
    >
      <Card className="relative overflow-hidden border-primary/20 bg-gradient-to-br from-[#14110f] via-[#0e0d12] to-[#0c0c10] min-h-[280px] hover:border-primary/40 transition-colors group">
        {/* ── Constellation visual (background, fills card) ───────── */}
        <svg
          viewBox="0 0 480 320"
          preserveAspectRatio="xMidYMid slice"
          className="absolute inset-0 w-full h-full"
          aria-hidden="true"
        >
          <defs>
            <radialGradient id={bgId} cx="55%" cy="55%" r="65%">
              <stop offset="0%" stopColor="#2a1810" stopOpacity="0.55" />
              <stop offset="60%" stopColor="#0e0d12" stopOpacity="0" />
            </radialGradient>
            <radialGradient id={orbId} cx="50%" cy="50%" r="50%">
              <stop offset="0%" stopColor="#fed7aa" stopOpacity="1" />
              <stop offset="55%" stopColor="#f97316" stopOpacity="0.95" />
              <stop offset="100%" stopColor="#7c2d12" stopOpacity="0" />
            </radialGradient>
            <radialGradient id={orbDimId} cx="50%" cy="50%" r="50%">
              <stop offset="0%" stopColor="#fde6c8" stopOpacity="0.9" />
              <stop offset="55%" stopColor="#cbd5e1" stopOpacity="0.55" />
              <stop offset="100%" stopColor="#1e293b" stopOpacity="0" />
            </radialGradient>
            <filter id={glowId} x="-50%" y="-50%" width="200%" height="200%">
              <feGaussianBlur stdDeviation="3.5" />
            </filter>
          </defs>

          {/* atmospheric glow */}
          <rect width="480" height="320" fill={`url(#${bgId})`} />

          {/* faint orbital rings */}
          <g stroke="rgba(249,115,22,0.10)" fill="none" strokeWidth="1">
            <ellipse cx="240" cy="160" rx="190" ry="115" />
            <ellipse cx="240" cy="160" rx="135" ry="80" />
            <ellipse
              cx="240"
              cy="160"
              rx="80"
              ry="48"
              strokeDasharray="2 4"
              stroke="rgba(255,255,255,0.07)"
            />
          </g>

          {/* connection arcs */}
          <g stroke="rgba(249,115,22,0.18)" fill="none" strokeWidth="1">
            {LINKS.map(([a, b], i) => {
              const A = ORB_BY_ID[a]
              const B = ORB_BY_ID[b]
              const mx = (A.cx + B.cx) / 2
              const my = (A.cy + B.cy) / 2 - 18
              return (
                <path
                  key={`${a}-${b}-${i}`}
                  d={`M${A.cx} ${A.cy} Q ${mx} ${my} ${B.cx} ${B.cy}`}
                />
              )
            })}
          </g>

          {/* traveling pulses */}
          <g fill="#fed7aa">
            <circle r="2.2">
              <animateMotion
                dur="3.2s"
                repeatCount="indefinite"
                path={`M${ORB_BY_ID.a.cx} ${ORB_BY_ID.a.cy} Q 220 60 ${ORB_BY_ID.b.cx} ${ORB_BY_ID.b.cy}`}
              />
              <animate
                attributeName="opacity"
                values="0;1;1;0"
                dur="3.2s"
                repeatCount="indefinite"
              />
            </circle>
            <circle r="2" fill="#fdba74">
              <animateMotion
                dur="4.1s"
                begin="0.6s"
                repeatCount="indefinite"
                path={`M${ORB_BY_ID.e.cx} ${ORB_BY_ID.e.cy} Q 280 180 ${ORB_BY_ID.f.cx} ${ORB_BY_ID.f.cy}`}
              />
              <animate
                attributeName="opacity"
                values="0;1;1;0"
                dur="4.1s"
                begin="0.6s"
                repeatCount="indefinite"
              />
            </circle>
            <circle r="2" fill="#fed7aa">
              <animateMotion
                dur="3.6s"
                begin="1.1s"
                repeatCount="indefinite"
                path={`M${ORB_BY_ID.c.cx} ${ORB_BY_ID.c.cy} Q 340 180 ${ORB_BY_ID.f.cx} ${ORB_BY_ID.f.cy}`}
              />
              <animate
                attributeName="opacity"
                values="0;1;1;0"
                dur="3.6s"
                begin="1.1s"
                repeatCount="indefinite"
              />
            </circle>
            <circle r="1.8" fill="#a5f3fc" opacity="0.7">
              <animateMotion
                dur="4.6s"
                begin="0.3s"
                repeatCount="indefinite"
                path={`M${ORB_BY_ID.d.cx} ${ORB_BY_ID.d.cy} Q 180 240 ${ORB_BY_ID.e.cx} ${ORB_BY_ID.e.cy}`}
              />
              <animate
                attributeName="opacity"
                values="0;0.9;0.9;0"
                dur="4.6s"
                begin="0.3s"
                repeatCount="indefinite"
              />
            </circle>
          </g>

          {/* orbs */}
          <g>
            {ORBS.map((o, i) => {
              const isLead = o.id === 'e'
              const fill = isLead
                ? `url(#${orbId})`
                : i % 2 === 0
                  ? `url(#${orbId})`
                  : `url(#${orbDimId})`
              return (
                <g key={o.id}>
                  {/* glow halo */}
                  <circle
                    cx={o.cx}
                    cy={o.cy}
                    r={o.r * 2.4}
                    fill={fill}
                    opacity="0.35"
                    filter={`url(#${glowId})`}
                  >
                    <animate
                      attributeName="opacity"
                      values="0.18;0.45;0.18"
                      dur={`${3 + (i % 4) * 0.6}s`}
                      begin={`${i * 0.3}s`}
                      repeatCount="indefinite"
                    />
                  </circle>
                  {/* core */}
                  <circle cx={o.cx} cy={o.cy} r={o.r} fill={fill} />
                  <circle
                    cx={o.cx - o.r * 0.3}
                    cy={o.cy - o.r * 0.3}
                    r={o.r * 0.35}
                    fill="rgba(255,255,255,0.55)"
                  />
                </g>
              )
            })}
          </g>

          {/* faint dust */}
          <g fill="rgba(255,255,255,0.45)">
            {DUST.map((d, i) => (
              <circle key={i} cx={d.cx} cy={d.cy} r={d.r} opacity={d.opacity} />
            ))}
          </g>
        </svg>

        {/* ── Bottom gradient mask for copy legibility ───────────── */}
        <div
          className="absolute inset-x-0 bottom-0 h-3/5 pointer-events-none"
          style={{
            background:
              'linear-gradient(180deg, rgba(14,13,18,0) 0%, rgba(14,13,18,0.78) 65%, rgba(14,13,18,0.95) 100%)',
          }}
        />

        {/* ── Live pill (top-left) ───────────────────────────────── */}
        {runningCount > 0 && (
          <div className="absolute top-4 left-4 z-10 flex items-center gap-2 px-2.5 py-1.5 rounded-full bg-background/60 border border-border/40 backdrop-blur-md font-mono text-[10.5px] tracking-[0.04em] text-muted-foreground lowercase">
            <span className="w-1.5 h-1.5 rounded-full bg-success shadow-[0_0_8px_rgba(34,197,94,0.9)]" />
            {runningCount} {runningCount === 1 ? 'mission' : 'missions'} running
          </div>
        )}

        {/* ── Copy + CTA (bottom-left) ───────────────────────────── */}
        <div className="absolute inset-x-0 bottom-0 z-10 p-6 flex items-end justify-between gap-4">
          <div className="min-w-0">
            <div className="inline-flex items-center gap-1.5 mb-3 px-2 py-0.5 rounded font-mono text-[10px] tracking-[0.12em] text-primary border border-primary/30 bg-primary/10">
              <span className="w-1 h-1 rounded-full bg-primary shadow-[0_0_6px_rgba(249,115,22,0.9)]" />
              MULTI-AGENT
            </div>
            <h3 className="text-3xl md:text-[2.25rem] font-semibold tracking-tight text-foreground leading-none">
              Mission
            </h3>
            <p className="text-sm text-muted-foreground mt-2 max-w-md leading-relaxed">
              Big, complex work. 6&ndash;9 agents, field memory, parallel reasoning.
            </p>
            <div className="hidden sm:flex items-center gap-2 mt-3 font-mono text-[10.5px] text-muted-foreground/80">
              <span>Researcher</span>
              <span className="text-muted-foreground/40">·</span>
              <span>Analyst</span>
              <span className="text-muted-foreground/40">·</span>
              <span>Builder</span>
              <span className="text-muted-foreground/40">+4</span>
            </div>
          </div>
          <Button
            onClick={onStart}
            className="shrink-0"
          >
            Start <span aria-hidden="true">&rarr;</span>
          </Button>
        </div>
      </Card>
    </motion.div>
  )
}
